import Foundation
import Metal

// --validate: correctness of the Metal implementation against the independent CPU reference.
//
//  1. fold / preprocess   bit-exact (x0), unaligned sizes exercise edge replication
//  2. pool                bit-exact ordered half FMA on the GPU's own skip tensors, all levels
//  3. head oracle         zero learned head weights, z == bias: bit-exact baseline D,
//                         T and H(H(B+T)-0.5) for black / ramp / impulse sources
//  4. RGBA8 presentation  bit-exact postProcessE byte store of the half output: clamp, *255,
//                         truncation toward zero (the extension's direct canvas store)
//  5. research fixtures   low / medium: the folded head with zero learned weights against
//                         the research-emulator values of extension/tests/fixtures/head
//                         (tools/gen_head_fixtures.py), bit-exact
//  6. layers              every RB / TB convolution and the head, each evaluated by the CPU
//                         on the GPU's OWN input of that layer: at most LayerTolerance of the
//                         halves may differ, each by at most one rounding flip
//  7. full network        all four models at 64x64, 96x80, 100x60, 320x180, 960x540: max /
//                         mean |GPU - CPU| within tolerance, plus reversed-concat and
//                         shifted-upsample negative controls that must exceed it
// Every bit-exact check also runs a negative control that must produce mismatches, which
// proves the comparison can fail. The whole-network tolerance alone cannot see a broken
// rounding sequence (an unrounded projection before the HADD merge keeps max |err| at one
// half ulp); the layer check sees it in 10-60 % of a layer's halves.

struct CheckResult: Codable {
    let name: String
    let model: String
    let passed: Bool
    let detail: String
    var maxAbsError: Double? = nil
    var meanAbsError: Double? = nil
    var negativeMaxAbsError: Double? = nil
    var negativeMeanAbsError: Double? = nil
    var comparedValues: Int? = nil
    var mismatches: Int? = nil
    var negativeMismatches: Int? = nil
}

/// Whole-network tolerance on the [0,1]-range half output (see docs/NATIVE_BENCHMARK.md).
/// GPU and CPU differ only in f32 accumulation order inside convolutions, which flips an
/// occasional half rounding; those flips propagate through 19 layers.
let networkMaxTolerance = 1.0 / 256.0
let networkMeanTolerance = 1.0 / 8192.0

func errorStats(_ a: [Float16], _ b: [Float16]) -> (max: Double, mean: Double) {
    precondition(a.count == b.count)
    var mx = 0.0, sum = 0.0
    for i in 0..<a.count {
        let d = abs(Double(a[i]) - Double(b[i]))
        if d.isNaN { return (.infinity, .infinity) }
        mx = max(mx, d)
        sum += d
    }
    return (mx, sum / Double(max(a.count, 1)))
}

/// Layer-level parity (check 6). A layer computed from the GPU's own input differs from the
/// CPU only where the different f32 summation order (or exp) flips one rounding to half; that
/// happens for well under 1 % of the halves (measured maximum in docs/NATIVE_BENCHMARK.md).
/// A flip moves a value by one half ulp of what was rounded: the output, or e and p before the
/// merge H(e + H(p)). Where the summed terms cancel, the f32 error is absolute instead
/// (sumSlack). Rounding-policy violations make 10-60 % of a layer's halves differ.
enum LayerTolerance {
    static let mismatchFraction = 0.01
    static let headMismatchFraction = 0.001
    static let headMaxAbs = 1.0 / 1024.0
    static let sumSlack = 1.0 / 65536.0
}

/// One half ulp at the magnitude of x (2^-24 in the subnormal range).
func halfUlp(_ x: Float16) -> Double {
    let a = abs(Double(x))
    return a < 1.0 / 16384.0 ? 1.0 / 16777216.0 : pow(2.0, floor(log2(a)) - 10)
}

struct LayerDiff {
    var count = 0
    var mismatches = 0
    var outOfBound = 0
    var maxAbs = 0.0

    /// `operands`: the merge inputs (e, p) of a projection layer, which widen each value's bound.
    init(_ gpu: [Float16], _ ref: [Float16], operands: [[Float16]] = []) {
        precondition(gpu.count == ref.count && operands.allSatisfy { $0.count == ref.count })
        count = ref.count
        for i in 0..<ref.count where gpu[i].bitPattern != ref[i].bitPattern {
            mismatches += 1
            let d = abs(Double(gpu[i]) - Double(ref[i]))
            maxAbs = max(maxAbs, d)
            var bound = halfUlp(ref[i]) + LayerTolerance.sumSlack
            for o in operands { bound += halfUlp(o[i]) }
            if !(d <= bound) { outOfBound += 1 }
        }
    }

    var fraction: Double { Double(mismatches) / Double(max(count, 1)) }
}

func bitMismatches(_ a: [Float16], _ b: [Float16]) -> Int {
    precondition(a.count == b.count)
    var n = 0
    for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern { n += 1 }
    return n
}

final class Validator {
    let ctx: MetalContext
    let models: [LoadedModel]
    let verbose: Bool
    /// extension/tests/fixtures/head (research-emulator head values for the folded models).
    let headFixturesDir: URL
    private(set) var results: [CheckResult] = []

    init(ctx: MetalContext, models: [LoadedModel], verbose: Bool, headFixturesDir: URL) {
        self.ctx = ctx
        self.models = models
        self.verbose = verbose
        self.headFixturesDir = headFixturesDir
    }

    private func record(_ r: CheckResult) {
        results.append(r)
        let tag = r.passed ? "PASS" : "FAIL"
        print("[\(tag)] \(r.model.padding(toLength: 6, withPad: " ", startingAt: 0)) \(r.name): \(r.detail)")
        fflush(stdout)
    }

    func runAll() throws -> Bool {
        for m in models { try checkFold(m) }
        for m in models { try checkPool(m) }
        for m in models { try checkHeadOracle(m) }
        for m in models { try checkRGBA8(m) }
        for m in models where m.folded { try checkHeadFixtures(m) }
        for m in models { try checkLayers(m) }
        for m in models {
            for (w, h) in [(64, 64), (96, 80), (100, 60), (320, 180)] { try checkNetwork(m, w, h, controls: true) }
            // The smallest benchmark resolution itself; the controls are already proven above.
            try checkNetwork(m, 960, 540, controls: false)
        }
        let failed = results.filter { !$0.passed }
        print("validation: \(results.count - failed.count)/\(results.count) checks passed")
        return failed.isEmpty
    }

    // MARK: 1. fold / preprocess

    func checkFold(_ m: LoadedModel) throws {
        let ref = CPUReference(model: m)
        for (w, h) in [(64, 64), (100, 60)] {
            let img = syntheticFrame(width: w, height: h, seed: 11)
            let engine = try VSREngine(ctx: ctx, model: m, width: w, height: h, format: .half)
            try runFrame(engine, source: try makeSourceBuffer(ctx.device, img), encoderOnly: true)
            let gpu = engine.readX0()
            let padded = ref.paddedSource(img, engine.geometry)
            let cpu = ref.networkInput(padded)
            let mism = bitMismatches(gpu.d, cpu.d)
            // Negative control: zero padding instead of edge replication, combined with the
            // transposed fold order 8*dx + 4*dy (folded) or a sign-split C=8 input [I, -I]
            // (unfolded; FLOAT-RECONSTRUCTION.md: C=8 does not imply a sign split).
            var zeroPad = padded
            for y in 0..<padded.h {
                for x in 0..<padded.w where x >= w || y >= h {
                    for c in 0..<4 { zeroPad.d[(y * padded.w + x) * 4 + c] = 0 }
                }
            }
            let base = ref.networkInput(zeroPad)
            var neg = base
            for p in 0..<(neg.w * neg.h) {
                if m.folded {
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            for c in 0..<4 { neg.d[p * 16 + 8 * dx + 4 * dy + c] = base.d[p * 16 + 8 * dy + 4 * dx + c] }
                        }
                    }
                } else {
                    for c in 0..<4 { neg.d[p * 8 + 4 + c] = -base.d[p * 8 + c] }
                }
            }
            let negMism = bitMismatches(gpu.d, neg.d)
            var r = CheckResult(name: "fold/preprocess \(w)x\(h)", model: m.id, passed: mism == 0 && negMism > 0,
                                detail: "\(gpu.d.count) halves, \(mism) mismatches; zero-pad + \(m.folded ? "transposed-fold" : "sign-split") control \(negMism) mismatches")
            r.comparedValues = gpu.d.count
            r.mismatches = mism
            r.negativeMismatches = negMism
            record(r)
        }
    }

    // MARK: 2. pool

    func checkPool(_ m: LoadedModel) throws {
        let (w, h) = (96, 80)
        let img = syntheticFrame(width: w, height: h, seed: 5)
        let engine = try VSREngine(ctx: ctx, model: m, width: w, height: h, format: .half)
        try runFrame(engine, source: try makeSourceBuffer(ctx.device, img), encoderOnly: true)
        var total = 0, mism = 0, negMism = 0
        for l in 0..<4 {
            let s = engine.readSkip(l)
            let gpu = engine.readPooled(l)
            precondition(gpu.w == s.w / 2 && gpu.h == s.h / 2 && gpu.c == s.c)
            for y in 0..<gpu.h {
                for x in 0..<gpu.w {
                    for c in 0..<s.c {
                        let tl = s.at(2 * x, 2 * y, c), tr = s.at(2 * x + 1, 2 * y, c)
                        let bl = s.at(2 * x, 2 * y + 1, c), br = s.at(2 * x + 1, 2 * y + 1, c)
                        let g = gpu.at(x, y, c)
                        if orderedPool(tl, tr, bl, br).bitPattern != g.bitPattern { mism += 1 }
                        // Negative control: f32 sum / 4 rounded once.
                        let alt = Float16((Float(tl) + Float(tr) + Float(bl) + Float(br)) / 4)
                        if alt.bitPattern != g.bitPattern { negMism += 1 }
                        total += 1
                    }
                }
            }
        }
        var r = CheckResult(name: "pool (ordered half FMA) levels 0-3 @96x80", model: m.id,
                            passed: mism == 0 && negMism > 0,
                            detail: "\(total) halves, \(mism) mismatches; f32-mean control \(negMism) mismatches")
        r.comparedValues = total
        r.mismatches = mism
        r.negativeMismatches = negMism
        record(r)
    }

    // MARK: 3. head oracle (zero learned weights)

    func checkHeadOracle(_ m: LoadedModel) throws {
        let p = m.shuffle
        let kz = m.headOutputs
        // Unique-bias pattern of the research emulator: H((k-24)/128) for p4, H((k-6)/32) for p2.
        let center = p == 4 ? 24.0 : 6.0, denom = p == 4 ? 128.0 : 32.0
        let unique = (0..<kz).map { Float16((Double($0) - center) / denom) }
        let zero = [Float16](repeating: 0, count: kz)
        let fixture = m.folded ? 32 : 16    // research fixture source size (folded 32x32, 2x head 16x16)
        var cases: [(HeadPattern, Int, Int)] = HeadPattern.allCases.map { ($0, fixture, fixture) }
        cases.append((.ramp, 100, 60))
        var total = 0, mism = 0, negCoef = 0, negPhase = 0
        for (biasName, bias) in [("zero", zero), ("unique", unique)] {
            let zm = try m.zeroHeadCopy(bias1: bias)
            let ref = CPUReference(model: zm)
            for (pattern, w, h) in cases {
                let img = pattern.image(width: w, height: h)
                let engine = try VSREngine(ctx: ctx, model: zm, width: w, height: h, format: .half)
                try runFrame(engine, source: try makeSourceBuffer(ctx.device, img))
                let gpu = engine.readOutputHalf()
                let g = engine.geometry
                let padded = ref.paddedSource(img, g)
                let featW = g.levelW[0], featH = g.levelH[0]
                var z = [Float16](repeating: 0, count: featW * featH * kz)
                for i in 0..<(featW * featH) { for k in 0..<kz { z[i * kz + k] = bias[k] } }
                let expected = ref.composeOutput(z: z, featW: featW, kz: kz, padded: padded, g: g)
                let n = bitMismatches(gpu, expected)
                total += gpu.count
                mism += n
                if verbose && n > 0 { print("  head \(biasName) \(pattern.rawValue) \(w)x\(h): \(n) mismatches") }
                if pattern == .ramp {
                    let swapped = ref.composeOutput(z: z, featW: featW, kz: kz, padded: padded, g: g,
                                                    coefficients: [m.coefficients[1], m.coefficients[0]])
                    negCoef += bitMismatches(gpu, swapped)
                    if biasName == "unique" {
                        let transposed = ref.composeOutput(z: z, featW: featW, kz: kz, padded: padded, g: g,
                                                           swapPhaseAxes: true)
                        negPhase += bitMismatches(gpu, transposed)
                    }
                }
            }
        }
        var r = CheckResult(name: "head oracle (zero weights, p\(p)/r2, black/ramp/impulse + 100x60)", model: m.id,
                            passed: mism == 0 && negCoef > 0 && negPhase > 0,
                            detail: "\(total) halves, \(mism) mismatches; swapped-phase-coefficients control \(negCoef), transposed k-phase control \(negPhase) mismatches")
        r.comparedValues = total
        r.mismatches = mism
        r.negativeMismatches = min(negCoef, negPhase)
        record(r)
    }

    // MARK: 4. RGBA8 output

    func checkRGBA8(_ m: LoadedModel) throws {
        let (w, h) = (100, 60)
        let img = syntheticFrame(width: w, height: h, seed: 3)
        let src = try makeSourceBuffer(ctx.device, img)
        let eh = try VSREngine(ctx: ctx, model: m, width: w, height: h, format: .half)
        try runFrame(eh, source: src)
        let half = eh.readOutputHalf()
        let e8 = try VSREngine(ctx: ctx, model: m, width: w, height: h, format: .rgba8)
        try runFrame(e8, source: src)
        let rgba = e8.readOutputRGBA()
        let plane = eh.geometry.outW * eh.geometry.outH
        var mism = 0, negMism = 0
        for i in 0..<plane {
            for c in 0..<3 {
                let v = min(max(Float(half[c * plane + i]), 0), 1)
                let scaled = v * 255   // exact in Float: a half times an 8-bit integer
                if UInt8(scaled.rounded(.towardZero)) != rgba[i * 4 + c] { mism += 1 }
                // Negative control: the resample stage's round-half-up must not be what the direct path stores.
                if UInt8((scaled + 0.5).rounded(.down)) != rgba[i * 4 + c] { negMism += 1 }
            }
            if rgba[i * 4 + 3] != 255 { mism += 1 }
        }
        var r = CheckResult(name: "rgba8 presentation (postProcessE: clamp, trunc(v*255)) 100x60", model: m.id,
                            passed: mism == 0 && negMism > 0,
                            detail: "\(plane * 4) bytes, \(mism) mismatches; round-half-up control \(negMism) mismatches")
        r.comparedValues = plane * 4
        r.mismatches = mism
        r.negativeMismatches = negMism
        record(r)
    }

    // MARK: 5. research head fixtures (folded models)

    /// The fixture sources are exact half images I = H(2v - 1) whose values are not all of the
    /// form H(2*(b/255) - 1), so they are fed through the engine's source LUT: every distinct RGB
    /// value gets a byte code and the LUT maps the code back to the half. Prep (fold) and the head
    /// baseline both read the source through that LUT, exactly as they read v/255 in production.
    func checkHeadFixtures(_ m: LoadedModel) throws {
        let name = "research head fixtures (tools/gen_head_fixtures.py, zero weights)"
        let manifestURL = headFixturesDir.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let models = root["models"] as? [String: Any], let entry = models[m.id] as? [String: Any],
              let cases = entry["cases"] as? [[String: Any]], let coefs = entry["baseline_coefficients"] as? [[Double]],
              let size = root["source_size"] as? [Int], size.count == 2 else {
            record(CheckResult(name: name, model: m.id, passed: false, detail: "cannot read \(manifestURL.path)"))
            return
        }
        func halves(_ file: String) throws -> [Float16] {
            let d = try Data(contentsOf: headFixturesDir.appendingPathComponent(m.id).appendingPathComponent(file))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
        }
        let (w, h) = (size[0], size[1])
        let coefficientsMatch = coefs.map { $0.map { Float16($0) } } == m.coefficients
        var total = 0, mism = 0, negMism = 0
        var outputs: [String: [Float16]] = [:]
        var expected: [String: [Float16]] = [:]
        for c in cases {
            guard let caseName = c["name"] as? String else { continue }
            let src = try halves("\(caseName).source.bin")
            let bias = try halves("\(caseName).bias1.bin")
            let want = try halves("\(caseName).expected.bin")
            guard src.count == 4 * w * h, bias.count == m.headOutputs else {
                throw BenchError("\(m.id) \(caseName): fixture sizes do not match the model")
            }
            var codes: [UInt16: UInt8] = [:]
            var lut = [Float16](repeating: 0, count: 256)
            var px = [UInt8](repeating: 255, count: 4 * w * h)
            for i in 0..<(w * h) {
                for ch in 0..<3 {
                    let v = src[4 * i + ch]
                    let code: UInt8
                    if let known = codes[v.bitPattern] {
                        code = known
                    } else {
                        guard codes.count < 256 else { throw BenchError("\(m.id) \(caseName): more than 256 source values") }
                        code = UInt8(codes.count)
                        codes[v.bitPattern] = code
                        lut[Int(code)] = v
                    }
                    px[4 * i + ch] = code
                }
            }
            let zm = try m.zeroHeadCopy(bias1: bias)
            let engine = try VSREngine(ctx: ctx, model: zm, width: w, height: h, format: .half)
            engine.lut = lut
            try runFrame(engine, source: try makeSourceBuffer(ctx.device, RGBAImage(w: w, h: h, px: px)))
            let gpu = engine.readOutputHalf()
            guard gpu.count == want.count else { throw BenchError("\(m.id) \(caseName): output size mismatch") }
            let n = bitMismatches(gpu, want)
            if verbose && n > 0 { print("  fixture \(caseName): \(n) mismatches") }
            total += gpu.count
            mism += n
            outputs[caseName] = gpu
            expected[caseName] = want
        }
        // Negative control: each output against the fixture of the same source with the other
        // bias mode (zero vs unique learned bias) must differ.
        for (caseName, gpu) in outputs {
            let other = caseName.hasSuffix("-0") ? String(caseName.dropLast()) + "1" : String(caseName.dropLast()) + "0"
            if let alt = expected[other] { negMism += bitMismatches(gpu, alt) }
        }
        var r = CheckResult(name: "\(name), \(cases.count) cases \(w)x\(h)", model: m.id,
                            passed: coefficientsMatch && total > 0 && mism == 0 && negMism > 0,
                            detail: "\(total) halves, \(mism) mismatches against the research values; fixture coefficients \(coefficientsMatch ? "match" : "DIFFER"); other-bias control \(negMism) mismatches")
        r.comparedValues = total
        r.mismatches = mism
        r.negativeMismatches = negMism
        record(r)
    }

    // MARK: 6. layer parity on the GPU's own inputs

    func checkLayers(_ m: LoadedModel) throws {
        let (w, h) = (96, 80)
        let img = syntheticFrame(width: w, height: h, seed: 17)
        let src = try makeSourceBuffer(ctx.device, img)
        let engine = try VSREngine(ctx: ctx, model: m, width: w, height: h, format: .half)
        let ref = CPUReference(model: m)
        let K = m.widths
        var rows: [String] = []
        var failures: [String] = []
        var worst = 0.0
        var total = 0
        func check(_ label: String, _ d: LayerDiff) {
            rows.append(String(format: "%@ %.3f%%", label, 100 * d.fraction))
            worst = max(worst, d.fraction)
            total += d.count
            if d.fraction > LayerTolerance.mismatchFraction || d.outOfBound > 0 {
                failures.append(String(format: "%@: %d/%d differ, %d beyond one flip (max %.2e)", label, d.mismatches, d.count, d.outOfBound, d.maxAbs))
            }
        }
        /// H(e + H(p)) with the merge operands, as the GPU's part B computes it.
        func merge(_ e: Tensor, _ pSum: [Float]) -> (Tensor, [Float16]) {
            var t = e
            let p = pSum.map { Float16($0) }
            for i in 0..<t.d.count { t.d[i] = e.d[i] + p[i] }
            return (t, p)
        }
        // Encoder: after rb_l.B, mid still holds rb_l.A's output and the input x_l is intact.
        for l in 0...4 {
            try runSteps(engine, source: src, through: "rb\(l).B")
            let x = l == 0 ? engine.readX0() : (l < 4 ? engine.readCat(l, offset: 0, channels: K[l - 1]) : engine.readX4())
            let mid = engine.readMid(level: l, channels: K[l])
            check("rb\(l).A", LayerDiff(mid.d, ref.convELU(x, "rb\(l).c0").d))
            let e = ref.convELU(mid, "rb\(l).c1")
            let (s, p) = merge(e, ref.convSum(x, "rb\(l).projection"))
            let gpu = l < 4 ? engine.readSkip(l) : engine.readBottleneck()
            check("rb\(l).B", LayerDiff(gpu.d, s.d, operands: [e.d, p]))
        }
        // Decoder: after tb_j.A the cat holds [U | skip]; tb_j.B may overwrite U with d_j.
        for j in 0..<4 {
            let L = 3 - j
            try runSteps(engine, source: src, through: "tb\(j).A")
            let prev = j == 0 ? engine.readBottleneck() : engine.read(engine.decoderOutputs[j - 1])
            let u = engine.readCat(L, offset: 0, channels: K[L])
            check("tb\(j).up", LayerDiff(u.d, ref.convELU(ref.nearest2x(prev), "tb\(j).u").d))
            let cat = engine.readCat(L, offset: 0, channels: 2 * K[L])
            let mid = engine.readMid(level: L, channels: K[L])
            check("tb\(j).A", LayerDiff(mid.d, ref.convELU(cat, "tb\(j).c0").d))
            try runSteps(engine, source: src, through: "tb\(j).B")
            let e = ref.convELU(engine.readMid(level: L, channels: K[L]), "tb\(j).c1")
            let (d, p) = merge(e, ref.convSum(cat, "tb\(j).projection"))
            check("tb\(j).B", LayerDiff(engine.read(engine.decoderOutputs[j]).d, d.d, operands: [e.d, p]))
        }
        // Head on the GPU's own d3 (isolates the head from decoder drift).
        try runFrame(engine, source: src)
        let head = LayerDiff(engine.readOutputHalf(), ref.head(engine.readDecoded(), padded: ref.paddedSource(img, engine.geometry), g: engine.geometry))
        let headOK = head.fraction <= LayerTolerance.headMismatchFraction && head.maxAbs <= LayerTolerance.headMaxAbs
        if !headOK {
            failures.append(String(format: "head on GPU d3: %d/%d differ, max %.2e", head.mismatches, head.count, head.maxAbs))
        }
        if verbose { print("  layers \(m.id): \(rows.joined(separator: ", "))") }
        var r = CheckResult(
            name: "layers (CPU on the GPU's own layer input) @\(w)x\(h)", model: m.id, passed: failures.isEmpty,
            detail: failures.isEmpty
                ? String(format: "%d layers, worst %.3f%% of halves differ (limit %.1f%%), all within one flip; head on GPU d3 %d/%d differ, max %.6f",
                         rows.count, 100 * worst, 100 * LayerTolerance.mismatchFraction, head.mismatches, head.count, head.maxAbs)
                : failures.joined(separator: "; "))
        r.comparedValues = total + head.count
        r.mismatches = head.mismatches
        record(r)
    }

    // MARK: 7. full network

    func checkNetwork(_ m: LoadedModel, _ w: Int, _ h: Int, controls: Bool) throws {
        let img = syntheticFrame(width: w, height: h, seed: UInt32(w * 31 + h))
        let engine = try VSREngine(ctx: ctx, model: m, width: w, height: h, format: .half)
        try runFrame(engine, source: try makeSourceBuffer(ctx.device, img))
        let gpu = engine.readOutputHalf()
        let ref = CPUReference(model: m)
        let t0 = Date()
        let cpu = try ref.run(img)
        let cpuSeconds = Date().timeIntervalSince(t0)
        let none = (max: Double.infinity, mean: Double.infinity)
        let ne = controls ? errorStats(gpu, try ref.run(img, variant: .reversedConcat).output) : none
        let se = controls ? errorStats(gpu, try ref.run(img, variant: .shiftedUpsample).output) : none
        let e = errorStats(gpu, cpu.output)
        let b4 = errorStats(engine.readBottleneck().d, cpu.bottleneck.d)
        let gpuD3 = engine.readDecoded()
        let d3 = errorStats(gpuD3.d, cpu.decoded.d)
        // Head evaluated on the GPU's own d3 isolates head error from decoder drift.
        let headOnly = errorStats(gpu, ref.head(gpuD3, padded: ref.paddedSource(img, engine.geometry), g: engine.geometry))
        let ok = e.max <= networkMaxTolerance && e.mean <= networkMeanTolerance
        func controlFails(_ c: (max: Double, mean: Double)) -> Bool {
            c.max > networkMaxTolerance && c.mean > networkMeanTolerance
        }
        var r = CheckResult(
            name: "network \(w)x\(h) -> \(2 * w)x\(2 * h)", model: m.id, passed: ok && controlFails(ne) && controlFails(se),
            detail: String(format: "max %.6f mean %.7f (tol %.6f / %.7f); %@; b4 max %.4f, d3 max %.4f, head-on-GPU-d3 max %.4f; cpu %.2fs",
                           e.max, e.mean, networkMaxTolerance, networkMeanTolerance,
                           controls ? String(format: "controls: reversed-concat max %.4f mean %.5f, shifted-upsample max %.4f mean %.5f",
                                             ne.max, ne.mean, se.max, se.mean) : "controls run at the small sizes",
                           b4.max, d3.max, headOnly.max, cpuSeconds))
        r.maxAbsError = e.max
        r.meanAbsError = e.mean
        if controls {
            r.negativeMaxAbsError = min(ne.max, se.max)
            r.negativeMeanAbsError = min(ne.mean, se.mean)
        }
        r.comparedValues = gpu.count
        record(r)
    }
}
