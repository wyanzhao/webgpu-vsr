import Foundation
import Metal

// Host side of the Metal implementation: packs weights for simdgroup_matrix GEMM tiles,
// owns the activation buffers (aliasing plan of IMPLEMENTATION_NOTES.md section 4, padded
// to a 16-pixel pitch) and encodes one frame as a single serial compute pass.

struct PrepParams {
    var srcW: UInt32, srcH: UInt32, outW: UInt32, outH: UInt32, pitch: UInt32, fold: UInt32
}

struct ConvParams {
    var inW: UInt32 = 0, inH: UInt32 = 0, inPitch: UInt32 = 0, inStride: UInt32 = 0
    var outW: UInt32 = 0, outH: UInt32 = 0, outPitch: UInt32 = 0, outStride: UInt32 = 0
    var projStride: UInt32 = 0, poolPitch: UInt32 = 0, poolStride: UInt32 = 0, reserved: UInt32 = 0
}

struct HeadParams {
    var featW: UInt32, featH: UInt32, featPitch: UInt32, featStride: UInt32
    var srcW: UInt32, srcH: UInt32, outW: UInt32, outH: UInt32
    var coef: (Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16)
}

enum OutputFormat: String {
    case half      // planar FP16 RGB (the SDK's internal output; used for validation)
    case rgba8     // packed 8-bit RGBA, postProcessE store: clamp, trunc(v*255) (presentation path)
}

/// How one convolution's implicit GEMM is spread over simdgroups and threadgroups. A
/// threadgroup computes a 16x8 pixel tile (16 M-blocks of 8 pixels) for K/nsplit output
/// channels; each simdgroup owns mb M-blocks x nb 8-channel N-blocks, i.e. mb*nb f32 8x8
/// accumulators. Tuned on the M5 Max (docs/NATIVE_BENCHMARK.md): 4 accumulators per
/// simdgroup is the register sweet spot (8 accumulators run 2.6x slower, 16 run 9x slower),
/// and threadgroups of at most 16 simdgroups (512 threads) beat 8 or 32; wider K is split
/// across grid.z.
struct Tiling: Hashable, CustomStringConvertible {
    let mb: Int
    let nb: Int
    let nsplit: Int

    static let maxSimdgroups = 16

    var mgroups: Int { 16 / mb }
    func simdgroups(_ k: Int) -> Int { mgroups * (k / 8 / nb / nsplit) }
    func threads(_ k: Int) -> Int { 32 * simdgroups(k) }
    var description: String { "mb\(mb)nb\(nb)s\(nsplit)" }

    static func conv(k: Int) -> Tiling {
        let nb = min(k / 8, 2)
        let mb = 4 / nb
        var split = 1
        while (16 / mb) * (k / 8 / nb / split) > maxSimdgroups, (k / 8 / nb) % (2 * split) == 0 { split *= 2 }
        return Tiling(mb: mb, nb: nb, nsplit: split)
    }

    /// The head keeps all hidden channels in one threadgroup (its 1x1 needs every one):
    /// 4 M-blocks x 1 N-block per simdgroup (16 or 64 hidden -> 8 or 32 simdgroups).
    static let head = Tiling(mb: 4, nb: 1, nsplit: 1)
}

final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var cache: [String: MTLComputePipelineState] = [:]

    init(metallib: URL) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw BenchError("no Metal device") }
        device = dev
        guard let q = dev.makeCommandQueue() else { throw BenchError("cannot create command queue") }
        queue = q
        do {
            library = try dev.makeLibrary(URL: metallib)
        } catch {
            throw BenchError("cannot load \(metallib.path): \(error)")
        }
    }

    struct ConvVariant: Hashable {
        var cin: Int, k: Int, upsample: Bool, proj: Bool, projC: Int, pool: Bool, inPlace: Bool, tiling: Tiling
    }

    /// Every kernel gets a value for every function constant (indices 0..13 of
    /// shaders/vsr_kernels.metal), so no specialisation depends on an unset constant.
    private func constants(cin: Int, k: Int, tiling t: Tiling, upsample: Bool = false, proj: Bool = false,
                           projC: Int = 0, pool: Bool = false, inPlace: Bool = false, kz: Int = 0,
                           shuffle: Int = 0, outMode: Int = 0) -> MTLFunctionConstantValues {
        let fc = MTLFunctionConstantValues()
        let uints: [(Int, Int)] = [(0, cin), (1, k), (2, min(cin, 32)), (3, t.nb), (6, projC), (9, kz),
                                   (10, shuffle), (11, outMode), (12, t.mb), (13, t.nsplit)]
        let bools: [(Int, Bool)] = [(4, upsample), (5, proj), (7, pool), (8, inPlace)]
        for (i, v) in uints {
            var x = UInt32(v)
            fc.setConstantValue(&x, type: .uint, index: i)
        }
        for (i, v) in bools {
            var x = v
            fc.setConstantValue(&x, type: .bool, index: i)
        }
        return fc
    }

    private func build(_ name: String, _ fc: MTLFunctionConstantValues?, key: String, threads: Int) throws -> MTLComputePipelineState {
        if let p = cache[key] { return p }
        let fn: MTLFunction
        if let fc = fc { fn = try library.makeFunction(name: name, constantValues: fc) } else {
            guard let f = library.makeFunction(name: name) else { throw BenchError("missing kernel \(name)") }
            fn = f
        }
        let p = try device.makeComputePipelineState(function: fn)
        guard p.maxTotalThreadsPerThreadgroup >= threads else {
            throw BenchError("\(key): pipeline supports \(p.maxTotalThreadsPerThreadgroup) threads, needs \(threads)")
        }
        cache[key] = p
        return p
    }

    func prep() throws -> MTLComputePipelineState { try build("vsr_prep", nil, key: "prep", threads: 128) }

    func conv(_ v: ConvVariant) throws -> MTLComputePipelineState {
        let t = v.tiling
        guard v.k % 8 == 0, v.cin % 8 == 0, v.projC % 8 == 0, 16 % t.mb == 0, t.mb <= 8, t.nb <= 4,
              (v.k / 8) % (t.nb * t.nsplit) == 0, !(v.inPlace && t.nsplit > 1) else {
            throw BenchError("unsupported conv variant \(v)")
        }
        let fc = constants(cin: v.cin, k: v.k, tiling: t, upsample: v.upsample, proj: v.proj, projC: v.projC,
                           pool: v.pool, inPlace: v.inPlace)
        let key = "conv c\(v.cin) k\(v.k) up\(v.upsample) proj\(v.proj):\(v.projC) pool\(v.pool) inplace\(v.inPlace) \(t)"
        return try build("vsr_conv3x3", fc, key: key, threads: t.threads(v.k))
    }

    func head(cin: Int, hidden: Int, kz: Int, shuffle: Int, format: OutputFormat, tiling t: Tiling) throws -> MTLComputePipelineState {
        guard hidden <= 64, hidden % 8 == 0, kz <= 48, kz % 8 == 0, (hidden / 8) % t.nb == 0,
              16 % t.mb == 0, t.mb <= 8, t.nb <= 4 else {
            throw BenchError("unsupported head \(hidden)/\(kz) \(t)")
        }
        let fc = constants(cin: cin, k: hidden, tiling: t, kz: kz, shuffle: shuffle, outMode: format == .half ? 0 : 1)
        let key = "head c\(cin) h\(hidden) z\(kz) p\(shuffle) \(format.rawValue) \(t)"
        return try build("vsr_head", fc, key: key, threads: t.threads(hidden))
    }
}

/// Weights repacked for the GEMM tiles: 3x3 as [K/8][9][C/8][8 (c)][8 (k)], 1x1 as
/// [K/8][C/8][8][8], bias as an 8-row replicated f32 tile [8][Kpad] that initialises the
/// accumulators. K is zero-padded to a multiple of 8 (head 1x1: 12 -> 16).
struct PackedWeights {
    let buffer: MTLBuffer
    let offsets: [String: Int]       // byte offsets: "<conv>.w", "<conv>.b"

    static func roundUp(_ v: Int, _ m: Int) -> Int { (v + m - 1) / m * m }

    static func pack(_ cw: ConvWeights) -> [Float16] {
        let kp = roundUp(cw.k, 8)
        let cb = cw.c / 8
        let taps = cw.r * cw.r
        var out = [Float16](repeating: 0, count: kp * taps * cw.c)
        for k in 0..<cw.k {
            let nb = k / 8, n = k % 8
            for tap in 0..<taps {
                for c in 0..<cw.c {
                    let src = (k * taps + tap) * cw.c + c   // KRSC with (r, s) flattened to tap
                    let dst = (((nb * taps + tap) * cb + c / 8) * 8 + c % 8) * 8 + n
                    out[dst] = cw.w[src]
                }
            }
        }
        return out
    }

    static func biasTile(_ cw: ConvWeights) -> [Float] {
        let kp = roundUp(cw.k, 8)
        var out = [Float](repeating: 0, count: 8 * kp)
        for row in 0..<8 { for k in 0..<cw.k { out[row * kp + k] = Float(cw.b[k]) } }
        return out
    }

    init(device: MTLDevice, model: LoadedModel) throws {
        var blob = Data()
        var offsets: [String: Int] = [:]
        func append(_ bytes: UnsafeRawBufferPointer, _ key: String) {
            let aligned = Self.roundUp(blob.count, 256)
            blob.append(Data(count: aligned - blob.count))
            offsets[key] = blob.count
            blob.append(contentsOf: bytes)
        }
        for id in model.convs.keys.sorted() {
            let cw = model.conv(id)
            Self.pack(cw).withUnsafeBytes { append($0, "\(id).w") }
            Self.biasTile(cw).withUnsafeBytes { append($0, "\(id).b") }
        }
        guard let buf = blob.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared) }) else {
            throw BenchError("cannot allocate weight buffer")
        }
        buf.label = "weights.\(model.id)"
        buffer = buf
        self.offsets = offsets
    }

    func w(_ id: String) -> Int { offsets["\(id).w"]! }
    func b(_ id: String) -> Int { offsets["\(id).b"]! }
}

/// A view of channels [channelOffset, channelOffset + channels) of a padded level tensor
/// whose pixels are `stride` halves apart.
struct TensorRef {
    let buffer: MTLBuffer
    let channelOffset: Int
    let stride: Int
    let channels: Int
    let level: Int
    var byteOffset: Int { channelOffset * 2 }
}

final class VSREngine {
    let ctx: MetalContext
    let model: LoadedModel
    let geometry: Geometry
    let format: OutputFormat
    let weights: PackedWeights

    // Activations (FP16 NHWC, padded pitch).
    let x0: MTLBuffer
    let cat: [MTLBuffer]       // levels 0..3: [U | skip], 2*K_l channels
    let x4: MTLBuffer
    let b4: MTLBuffer
    let mid: MTLBuffer
    let dout: MTLBuffer?       // decoder output for stages whose K is split (no in-place write)
    let output: MTLBuffer
    private let dummy: MTLBuffer
    private let ownedBuffers: [MTLBuffer]

    private enum Kind { case prep, conv, head }
    private struct Step {
        let kind: Kind
        let pipeline: MTLComputePipelineState
        let buffers: [(MTLBuffer, Int)]   // slot index == array index
        let bytes: Data
        let grid: MTLSize
        let threads: MTLSize
        let label: String
        let macs: Double                  // multiply-accumulates per frame (GEMM work)
    }
    private var steps: [Step] = []
    private(set) var encoderStepCount = 0
    private var decoded: TensorRef?
    /// Where each decoder stage tb_j writes d_j (cat_L.U in place, or `dout` for split stages).
    private(set) var decoderOutputs: [TensorRef] = []
    /// Byte -> preprocessed half I used by the prep and head kernels (RGB channels). The default is
    /// dlpp_preProcessE's H(2*float32(v/255) - 1); validation replaces it to feed exact half sources.
    var lut: [Float16] = preprocessLUT {
        didSet { precondition(lut.count == 256, "the source LUT has 256 entries") }
    }

    var activationBytes: Int { ownedBuffers.reduce(0) { $0 + $1.length } }
    var weightBufferBytes: Int { weights.buffer.length }
    var allocatedBytes: Int { activationBytes + weightBufferBytes }

    init(ctx: MetalContext, model: LoadedModel, width: Int, height: Int, format: OutputFormat) throws {
        self.ctx = ctx
        self.model = model
        self.format = format
        geometry = try Geometry(model: model, width: width, height: height)
        weights = try PackedWeights(device: ctx.device, model: model)
        let g = geometry
        let K = model.widths
        x0 = try Self.alloc(ctx.device, g.pixels(0) * model.inputChannels * 2, "x0")
        cat = try (0..<4).map { l in try Self.alloc(ctx.device, g.pixels(l) * 2 * K[l] * 2, "cat\(l)") }
        x4 = try Self.alloc(ctx.device, g.pixels(4) * K[3] * 2, "x4")
        b4 = try Self.alloc(ctx.device, g.pixels(4) * K[4] * 2, "b4")
        let midElems = (0..<5).map { g.pixels($0) * K[$0] }.max()!
        mid = try Self.alloc(ctx.device, midElems * 2, "mid")
        let splitLevels = (0..<4).filter { Tiling.conv(k: K[$0]).nsplit > 1 }
        if let elems = splitLevels.map({ g.pixels($0) * K[$0] }).max() {
            dout = try Self.alloc(ctx.device, elems * 2, "dout")
        } else {
            dout = nil
        }
        let outBytes = format == .half ? 3 * g.outW * g.outH * 2 : g.outW * g.outH * 4
        output = try Self.alloc(ctx.device, outBytes, "output")
        dummy = try Self.alloc(ctx.device, 16, "dummy")
        ownedBuffers = [x0] + cat + [x4, b4, mid] + (dout.map { [$0] } ?? []) + [output]
        try buildSteps()
    }

    private static func alloc(_ device: MTLDevice, _ bytes: Int, _ label: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw BenchError("cannot allocate \(bytes) bytes for \(label)")
        }
        memset(b.contents(), 0, b.length)
        b.label = label
        return b
    }

    private func convStep(_ label: String, conv id: String, input: TensorRef, outLevel: Int, output out: TensorRef,
                          upsample: Bool = false, proj: (TensorRef, String)? = nil,
                          pool: TensorRef? = nil) throws -> Step {
        let g = geometry
        let cw = model.conv(id)
        let k = cw.k
        let tiling = Tiling.conv(k: k)
        // In place = the store overlaps channels the projection still reads at the same pixels.
        let inPlace = proj.map {
            $0.0.buffer === out.buffer && $0.0.channelOffset < out.channelOffset + out.channels
                && out.channelOffset < $0.0.channelOffset + $0.0.channels
        } ?? false
        guard out.channels == k, input.channels == cw.c, proj.map({ model.conv($0.1).c == $0.0.channels }) ?? true else {
            throw BenchError("\(label): tensor channels do not match \(id)")
        }
        let variant = MetalContext.ConvVariant(cin: input.channels, k: k, upsample: upsample, proj: proj != nil,
                                               projC: proj?.0.channels ?? 0, pool: pool != nil, inPlace: inPlace,
                                               tiling: tiling)
        let pso = try ctx.conv(variant)
        var p = ConvParams()
        p.inW = UInt32(g.levelW[input.level]); p.inH = UInt32(g.levelH[input.level])
        p.inPitch = UInt32(g.pitch[input.level]); p.inStride = UInt32(input.stride)
        p.outW = UInt32(g.levelW[outLevel]); p.outH = UInt32(g.levelH[outLevel])
        p.outPitch = UInt32(g.pitch[outLevel]); p.outStride = UInt32(out.stride)
        if let pr = proj { p.projStride = UInt32(pr.0.stride) }
        if let po = pool { p.poolPitch = UInt32(g.pitch[outLevel + 1]); p.poolStride = UInt32(po.stride) }
        let buffers: [(MTLBuffer, Int)] = [
            (input.buffer, input.byteOffset),
            (weights.buffer, weights.w(id)),
            (weights.buffer, weights.b(id)),
            proj.map { ($0.0.buffer, $0.0.byteOffset) } ?? (dummy, 0),
            proj.map { (weights.buffer, weights.w($0.1)) } ?? (dummy, 0),
            (out.buffer, out.byteOffset),
            pool.map { ($0.buffer, $0.byteOffset) } ?? (dummy, 0),
        ]
        let grid = MTLSize(width: (g.levelW[outLevel] + 15) / 16, height: (g.levelH[outLevel] + 7) / 8,
                           depth: tiling.nsplit)
        let macs = Double(g.levelW[outLevel] * g.levelH[outLevel] * k * (9 * input.channels + (proj?.0.channels ?? 0)))
        return Step(kind: .conv, pipeline: pso, buffers: buffers, bytes: withUnsafeBytes(of: &p) { Data($0) },
                    grid: grid, threads: MTLSize(width: tiling.threads(k), height: 1, depth: 1), label: label, macs: macs)
    }

    private func buildSteps() throws {
        let g = geometry
        let K = model.widths
        let cin0 = model.inputChannels
        var prep = PrepParams(srcW: UInt32(g.srcW), srcH: UInt32(g.srcH), outW: UInt32(g.levelW[0]),
                              outH: UInt32(g.levelH[0]), pitch: UInt32(g.pitch[0]), fold: model.folded ? 1 : 0)
        steps.append(Step(kind: .prep, pipeline: try ctx.prep(), buffers: [], bytes: withUnsafeBytes(of: &prep) { Data($0) },
                          grid: MTLSize(width: g.levelW[0], height: g.levelH[0], depth: 1),
                          threads: MTLSize(width: 16, height: 8, depth: 1), label: "prep", macs: 0))

        func catU(_ l: Int, _ c: Int) -> TensorRef {
            TensorRef(buffer: cat[l], channelOffset: 0, stride: 2 * K[l], channels: c, level: l)
        }
        func catSkip(_ l: Int) -> TensorRef {
            TensorRef(buffer: cat[l], channelOffset: K[l], stride: 2 * K[l], channels: K[l], level: l)
        }
        func dense(_ b: MTLBuffer, _ l: Int, _ c: Int) -> TensorRef {
            TensorRef(buffer: b, channelOffset: 0, stride: c, channels: c, level: l)
        }

        // Encoder. Input of level l: x0 (l = 0) or the U-half of cat_l holding x_l = pool(s_{l-1}).
        for l in 0..<4 {
            let input = l == 0 ? dense(x0, 0, cin0) : catU(l, K[l - 1])
            steps.append(try convStep("rb\(l).A", conv: "rb\(l).c0", input: input, outLevel: l, output: dense(mid, l, K[l])))
            let poolDst = l < 3 ? catU(l + 1, K[l]) : dense(x4, 4, K[3])
            steps.append(try convStep("rb\(l).B", conv: "rb\(l).c1", input: dense(mid, l, K[l]), outLevel: l,
                                      output: catSkip(l), proj: (input, "rb\(l).projection"), pool: poolDst))
        }
        steps.append(try convStep("rb4.A", conv: "rb4.c0", input: dense(x4, 4, K[3]), outLevel: 4, output: dense(mid, 4, K[4])))
        steps.append(try convStep("rb4.B", conv: "rb4.c1", input: dense(mid, 4, K[4]), outLevel: 4,
                                  output: dense(b4, 4, K[4]), proj: (dense(x4, 4, K[3]), "rb4.projection")))
        encoderStepCount = steps.count

        // Decoder tb_j at level L = 3 - j: up -> cat_L.U, A -> mid, B -> cat_L.U in place
        // (or dout when the stage splits K across threadgroups).
        var d = dense(b4, 4, K[4])
        for j in 0..<4 {
            let L = 3 - j
            steps.append(try convStep("tb\(j).up", conv: "tb\(j).u", input: d, outLevel: L, output: catU(L, K[L]),
                                      upsample: true))
            steps.append(try convStep("tb\(j).A", conv: "tb\(j).c0", input: catU(L, 2 * K[L]), outLevel: L,
                                      output: dense(mid, L, K[L])))
            let dst: TensorRef
            if Tiling.conv(k: K[L]).nsplit > 1, let buf = dout { dst = dense(buf, L, K[L]) } else { dst = catU(L, K[L]) }
            steps.append(try convStep("tb\(j).B", conv: "tb\(j).c1", input: dense(mid, L, K[L]), outLevel: L,
                                      output: dst, proj: (catU(L, 2 * K[L]), "tb\(j).projection")))
            d = dst
            decoderOutputs.append(dst)
        }
        decoded = d

        // Head on d_3.
        let h0 = model.conv("head.c0"), h1 = model.conv("head.c1")
        let kz = PackedWeights.roundUp(h1.k, 8)
        let c = model.coefficients
        var hp = HeadParams(featW: UInt32(g.levelW[0]), featH: UInt32(g.levelH[0]), featPitch: UInt32(g.pitch[0]),
                            featStride: UInt32(d.stride), srcW: UInt32(g.srcW), srcH: UInt32(g.srcH),
                            outW: UInt32(g.outW), outH: UInt32(g.outH),
                            coef: (c[0][0], c[0][1], c[0][2], c[0][3], c[1][0], c[1][1], c[1][2], c[1][3]))
        let ht = Tiling.head
        let headPSO = try ctx.head(cin: K[0], hidden: h0.k, kz: kz, shuffle: model.shuffle, format: format, tiling: ht)
        steps.append(Step(kind: .head, pipeline: headPSO,
                          buffers: [(d.buffer, d.byteOffset), (weights.buffer, weights.w("head.c0")),
                                    (weights.buffer, weights.b("head.c0")), (weights.buffer, weights.w("head.c1")),
                                    (weights.buffer, weights.b("head.c1"))],
                          bytes: withUnsafeBytes(of: &hp) { Data($0) },
                          grid: MTLSize(width: (g.levelW[0] + 15) / 16, height: (g.levelH[0] + 7) / 8, depth: 1),
                          threads: MTLSize(width: ht.threads(h0.k), height: 1, depth: 1), label: "head",
                          macs: Double(g.levelW[0] * g.levelH[0] * (h0.k * 9 * h0.c + h1.k * h1.c))))
    }

    /// Encodes one frame. `encoderOnly` stops after rb4 (used to read pooled encoder
    /// tensors before the decoder overwrites the U-halves).
    func encode(_ cb: MTLCommandBuffer, source: MTLBuffer, encoderOnly: Bool = false) {
        encode(cb, source: source, steps: 0..<(encoderOnly ? encoderStepCount : steps.count))
    }

    /// Encodes the given dispatch range (in order, `repeats` times) as one serial compute pass.
    func encode(_ cb: MTLCommandBuffer, source: MTLBuffer, steps range: Range<Int>, repeats: Int = 1) {
        guard let enc = cb.makeComputeCommandEncoder(dispatchType: .serial) else { fatalError("no compute encoder") }
        enc.label = "vsr.\(model.id)"
        let lutBytes = lut.withUnsafeBytes { Data($0) }
        func setBytes(_ d: Data, _ index: Int) {
            d.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
        }
        for _ in 0..<repeats {
            for i in range {
                let s = steps[i]
                enc.setComputePipelineState(s.pipeline)
                switch s.kind {
                case .prep:
                    enc.setBuffer(source, offset: 0, index: 0)
                    setBytes(lutBytes, 1)
                    enc.setBuffer(x0, offset: 0, index: 2)
                    setBytes(s.bytes, 3)
                    enc.dispatchThreads(s.grid, threadsPerThreadgroup: s.threads)
                case .conv:
                    for (slot, b) in s.buffers.enumerated() { enc.setBuffer(b.0, offset: b.1, index: slot) }
                    setBytes(s.bytes, 7)
                    enc.dispatchThreadgroups(s.grid, threadsPerThreadgroup: s.threads)
                case .head:
                    for (slot, b) in s.buffers.enumerated() { enc.setBuffer(b.0, offset: b.1, index: slot) }
                    enc.setBuffer(source, offset: 0, index: 5)
                    setBytes(lutBytes, 6)
                    enc.setBuffer(format == .half ? output : dummy, offset: 0, index: 7)
                    enc.setBuffer(format == .rgba8 ? output : dummy, offset: 0, index: 8)
                    setBytes(s.bytes, 9)
                    enc.dispatchThreadgroups(s.grid, threadsPerThreadgroup: s.threads)
                }
            }
        }
        enc.endEncoding()
    }

    var dispatchCount: Int { steps.count }

    /// Number of dispatches up to and including the one labelled `label` (see stepLabels).
    func stepCount(through label: String) -> Int {
        guard let i = steps.firstIndex(where: { $0.label == label }) else { fatalError("no dispatch named \(label)") }
        return i + 1
    }
    var stepLabels: [String] { steps.map { $0.label } }
    var stepMACs: [Double] { steps.map { $0.macs } }

    // MARK: readback (validation)

    func read(_ t: TensorRef) -> Tensor {
        let g = geometry
        let l = t.level
        var out = Tensor(w: g.levelW[l], h: g.levelH[l], c: t.channels)
        let p = t.buffer.contents().bindMemory(to: Float16.self, capacity: t.buffer.length / 2)
        for y in 0..<out.h {
            for x in 0..<out.w {
                let base = (y * g.pitch[l] + x) * t.stride + t.channelOffset
                for ch in 0..<t.channels { out.d[(y * out.w + x) * t.channels + ch] = p[base + ch] }
            }
        }
        return out
    }

    func readX0() -> Tensor {
        read(TensorRef(buffer: x0, channelOffset: 0, stride: model.inputChannels, channels: model.inputChannels, level: 0))
    }

    func readSkip(_ l: Int) -> Tensor {
        let k = model.widths[l]
        return read(TensorRef(buffer: cat[l], channelOffset: k, stride: 2 * k, channels: k, level: l))
    }

    /// x_{l+1} = pool(s_l); valid only after an encoder-only run for l < 3.
    func readPooled(_ l: Int) -> Tensor {
        let K = model.widths
        if l < 3 {
            return read(TensorRef(buffer: cat[l + 1], channelOffset: 0, stride: 2 * K[l + 1], channels: K[l], level: l + 1))
        }
        return read(TensorRef(buffer: x4, channelOffset: 0, stride: K[3], channels: K[3], level: 4))
    }

    /// The `mid` scratch tensor at `level` with `channels` channels (rb/tb part A output).
    func readMid(level: Int, channels: Int) -> Tensor {
        read(TensorRef(buffer: mid, channelOffset: 0, stride: channels, channels: channels, level: level))
    }

    /// Channels [offset, offset + channels) of cat_l (U-half at offset 0, skip-half at K_l).
    func readCat(_ l: Int, offset: Int, channels: Int) -> Tensor {
        read(TensorRef(buffer: cat[l], channelOffset: offset, stride: 2 * model.widths[l], channels: channels, level: l))
    }

    /// The bottleneck input x_4.
    func readX4() -> Tensor {
        read(TensorRef(buffer: x4, channelOffset: 0, stride: model.widths[3], channels: model.widths[3], level: 4))
    }

    func readBottleneck() -> Tensor {
        read(TensorRef(buffer: b4, channelOffset: 0, stride: model.widths[4], channels: model.widths[4], level: 4))
    }

    func readDecoded() -> Tensor {
        guard let d = decoded else { fatalError("decoder output location not built") }
        return read(d)
    }

    func readOutputHalf() -> [Float16] {
        precondition(format == .half)
        let n = 3 * geometry.outW * geometry.outH
        let p = output.contents().bindMemory(to: Float16.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    func readOutputRGBA() -> [UInt8] {
        precondition(format == .rgba8)
        let n = 4 * geometry.outW * geometry.outH
        let p = output.contents().bindMemory(to: UInt8.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
}

func makeSourceBuffer(_ device: MTLDevice, _ img: RGBAImage) throws -> MTLBuffer {
    guard let b = img.px.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else {
        throw BenchError("cannot allocate source buffer")
    }
    b.label = "source"
    return b
}

/// Runs the dispatches up to and including `label` synchronously (layer-level validation).
func runSteps(_ engine: VSREngine, source: MTLBuffer, through label: String) throws {
    guard let cb = engine.ctx.queue.makeCommandBuffer() else { throw BenchError("no command buffer") }
    engine.encode(cb, source: source, steps: 0..<engine.stepCount(through: label))
    cb.commit()
    cb.waitUntilCompleted()
    if let err = cb.error { throw BenchError("command buffer failed: \(err)") }
}

/// Runs one frame synchronously; returns GPU time in milliseconds.
@discardableResult
func runFrame(_ engine: VSREngine, source: MTLBuffer, encoderOnly: Bool = false) throws -> Double {
    guard let cb = engine.ctx.queue.makeCommandBuffer() else { throw BenchError("no command buffer") }
    engine.encode(cb, source: source, encoderOnly: encoderOnly)
    cb.commit()
    cb.waitUntilCompleted()
    if let err = cb.error { throw BenchError("command buffer failed: \(err)") }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000
}
