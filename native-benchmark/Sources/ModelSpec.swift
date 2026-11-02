import CryptoKit
import Foundation

// Network specs produced by tools/export_models.py (build/models/<model>.json) and the
// weight manifests they point at (extension/public/weights for low/medium, build/weights for
// high/ultra).

struct ConvSpec: Codable {
    let weight: String
    let bias: String
    let logical_KCRS: [Int]
}

struct HeadSpec: Codable {
    let node_prefix: String
    let hidden: Int
    let outputs: Int
    let learned_shuffle: Int
    let baseline_scale: Int
    let phase_mapping: String
    let coefficients: [[Double]]
    let coefficient_source: String
}

struct ModelSpec: Codable {
    let model: String
    let family: String
    let fold_factor: Int
    let input_alignment: Int
    let input_channels: Int
    let encoder_channels: [Int]
    let head: HeadSpec
    let convs: [String: ConvSpec]
    let weights_bin: String
    let weights_manifest: String
    let variant_graph: String
    let variant_graph_sha256: String
    let source_dll_sha256: String
}

struct ManifestTensor: Codable {
    let offset: Int
    let size: Int
    let dtype: String
    let logical_shape: [Int]?
    let physical_shape: [Int]?
    let physical_layout: String?
}

struct WeightManifest: Codable {
    let model: String
    let total_size_bytes: Int
    let bundle_sha256: String
    let tensors: [String: ManifestTensor]
}

struct BenchError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

/// One convolution: FP16 weights in physical KRSC order (((k*R + r)*S + s)*C + c) and its bias.
struct ConvWeights {
    let k: Int
    let c: Int
    let r: Int
    var w: [Float16]
    var b: [Float16]
}

/// The four model ids the native benchmark supports.
let allModelIds = ["low", "medium", "high", "ultra"]

final class LoadedModel {
    let spec: ModelSpec
    let id: String
    let folded: Bool
    let widths: [Int]
    let inputChannels: Int
    let alignment: Int
    let shuffle: Int
    let hidden: Int
    let headOutputs: Int
    let coefficients: [[Float16]]
    private(set) var convs: [String: ConvWeights]
    let weightBytes: Int

    init(spec: ModelSpec, convs: [String: ConvWeights], weightBytes: Int) throws {
        self.spec = spec
        id = spec.model
        folded = spec.family == "folded"
        widths = spec.encoder_channels
        inputChannels = spec.input_channels
        alignment = spec.input_alignment
        shuffle = spec.head.learned_shuffle
        hidden = spec.head.hidden
        headOutputs = spec.head.outputs
        coefficients = spec.head.coefficients.map { $0.map { Float16($0) } }
        self.convs = convs
        self.weightBytes = weightBytes
        try validateTopology()
    }

    /// Loads build/models/<id>.json and its weight bundle; verifies bundle hash, tensor
    /// sizes, dtypes and that every conv shape matches the graph spec.
    static func load(id: String, modelsDir: URL) throws -> LoadedModel {
        let specURL = modelsDir.appendingPathComponent("\(id).json")
        guard let specData = try? Data(contentsOf: specURL) else {
            throw BenchError("cannot read model spec \(specURL.path) (run `make` to export it)")
        }
        let spec = try JSONDecoder().decode(ModelSpec.self, from: specData)
        let manifest = try JSONDecoder().decode(
            WeightManifest.self, from: Data(contentsOf: URL(fileURLWithPath: spec.weights_manifest)))
        let blob = try Data(contentsOf: URL(fileURLWithPath: spec.weights_bin))
        guard blob.count == manifest.total_size_bytes else {
            throw BenchError("\(spec.weights_bin): \(blob.count) bytes, manifest says \(manifest.total_size_bytes)")
        }
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        guard digest == manifest.bundle_sha256 else {
            throw BenchError("\(spec.weights_bin): SHA-256 \(digest) != manifest \(manifest.bundle_sha256)")
        }
        func halves(_ name: String, count: Int) throws -> [Float16] {
            guard let t = manifest.tensors[name] else { throw BenchError("tensor \(name) missing from manifest") }
            guard t.dtype == "float16", t.size == count * 2, t.offset + t.size <= blob.count else {
                throw BenchError("tensor \(name): dtype \(t.dtype) size \(t.size), expected \(count) halves")
            }
            var out = [Float16](repeating: 0, count: count)
            out.withUnsafeMutableBytes { dst in
                blob.copyBytes(to: dst.bindMemory(to: UInt8.self), from: t.offset..<(t.offset + t.size))
            }
            for v in out where !v.isFinite { throw BenchError("tensor \(name) has a non-finite value") }
            return out
        }
        var convs: [String: ConvWeights] = [:]
        for (cid, cs) in spec.convs {
            let shape = cs.logical_KCRS
            guard shape.count == 4, shape[2] == shape[3] else { throw BenchError("\(cid): bad shape \(shape)") }
            let (k, c, r) = (shape[0], shape[1], shape[2])
            if let t = manifest.tensors[cs.weight], let phys = t.physical_shape {
                guard phys == [k, r, r, c] else {
                    throw BenchError("\(cs.weight): physical shape \(phys) is not KRSC \([k, r, r, c])")
                }
            }
            let w = try halves(cs.weight, count: k * c * r * r)
            let b = try halves(cs.bias, count: k)
            convs[cid] = ConvWeights(k: k, c: c, r: r, w: w, b: b)
        }
        return try LoadedModel(spec: spec, convs: convs, weightBytes: blob.count)
    }

    func conv(_ id: String) -> ConvWeights {
        guard let c = convs[id] else { fatalError("conv \(id) missing (topology was validated at load)") }
        return c
    }

    /// Checks the U-Net chain the engines implement against the loaded shapes.
    private func validateTopology() throws {
        guard widths.count == 5 else { throw BenchError("\(id): expected 5 encoder widths") }
        guard spec.head.baseline_scale == 2, shuffle == 2 * spec.fold_factor,
              headOutputs == 3 * shuffle * shuffle else {
            throw BenchError("\(id): unsupported head p=\(shuffle) r=\(spec.head.baseline_scale)")
        }
        guard coefficients.count == 2, coefficients.allSatisfy({ $0.count == 4 }) else {
            throw BenchError("\(id): expected 2x4 baseline coefficients")
        }
        func want(_ cid: String, _ k: Int, _ c: Int, _ r: Int) throws {
            guard let cw = convs[cid] else { throw BenchError("\(id): conv \(cid) missing") }
            guard cw.k == k, cw.c == c, cw.r == r else {
                throw BenchError("\(id): \(cid) is [\(cw.k),\(cw.c),\(cw.r)], expected [\(k),\(c),\(r)]")
            }
            guard c % 8 == 0 else { throw BenchError("\(id): \(cid) input channels \(c) not a multiple of 8") }
        }
        var prev = inputChannels
        for i in 0..<5 {
            try want("rb\(i).c0", widths[i], prev, 3)
            try want("rb\(i).c1", widths[i], widths[i], 3)
            try want("rb\(i).projection", widths[i], prev, 1)
            prev = widths[i]
        }
        for j in 0..<4 {
            let k = widths[3 - j]
            try want("tb\(j).u", k, prev, 3)
            try want("tb\(j).c0", k, 2 * k, 3)
            try want("tb\(j).c1", k, k, 3)
            try want("tb\(j).projection", k, 2 * k, 1)
            prev = k
        }
        try want("head.c0", hidden, prev, 3)
        try want("head.c1", headOutputs, hidden, 1)
        // x_{l+1} = pool(s_l) has K_l channels and is stored in the U-half of cat_{l+1}.
        for l in 0..<3 where widths[l] > widths[l + 1] {
            throw BenchError("\(id): width K\(l)=\(widths[l]) exceeds K\(l + 1)=\(widths[l + 1])")
        }
        for (cid, cw) in convs where cid.hasSuffix(".projection") {
            guard cw.b.allSatisfy({ $0 == 0 }) else { throw BenchError("\(id): \(cid) bias is not zero") }
        }
    }

    /// Copy with every learned head weight zeroed and the given final 1x1 bias (oracle
    /// pattern of semantics/emulator/check_folded_heads.py: z == bias exactly).
    func zeroHeadCopy(bias1: [Float16]) throws -> LoadedModel {
        var c = convs
        guard var h0 = c["head.c0"], var h1 = c["head.c1"], bias1.count == h1.k else {
            throw BenchError("zeroHeadCopy: bad head")
        }
        h0.w = [Float16](repeating: 0, count: h0.w.count)
        h1.w = [Float16](repeating: 0, count: h1.w.count)
        h1.b = bias1
        c["head.c0"] = h0
        c["head.c1"] = h1
        return try LoadedModel(spec: spec, convs: c, weightBytes: weightBytes)
    }

    var parameterCount: Int { convs.values.reduce(0) { $0 + $1.w.count + $1.b.count } }
}

/// Frame geometry: model-aligned padded size and per-level feature grids.
struct Geometry {
    let srcW: Int, srcH: Int
    let alignedW: Int, alignedH: Int        // TW, TH
    let levelW: [Int], levelH: [Int]        // levels 0..4
    let pitch: [Int], rows: [Int]           // padded storage per level (16-pixel pitch, 8-row multiple)
    let outW: Int, outH: Int                // internal output 2W x 2H

    init(model: LoadedModel, width: Int, height: Int) throws {
        guard width >= 1, height >= 1 else { throw BenchError("bad input size \(width)x\(height)") }
        srcW = width; srcH = height
        let a = model.alignment
        alignedW = a * ((width + a - 1) / a)
        alignedH = a * ((height + a - 1) / a)
        let w0 = model.folded ? alignedW / 2 : alignedW
        let h0 = model.folded ? alignedH / 2 : alignedH
        guard w0 % 16 == 0, h0 % 16 == 0 else { throw BenchError("level-0 grid \(w0)x\(h0) not divisible by 16") }
        levelW = (0..<5).map { w0 >> $0 }
        levelH = (0..<5).map { h0 >> $0 }
        pitch = levelW.map { 16 * (($0 + 15) / 16) }
        rows = levelH.map { 8 * (($0 + 7) / 8) }
        outW = 2 * width; outH = 2 * height
    }

    func pixels(_ l: Int) -> Int { pitch[l] * rows[l] }
}

/// Preprocess LUT: I = H(2*(v/255) - 1) evaluated in f32 (notes section 3, step 2).
let preprocessLUT: [Float16] = (0..<256).map { v in
    let r = Float(v) / 255.0
    return Float16(2.0 * r - 1.0)
}
