import Foundation

// Independent CPU reference of the reconstructed float networks.
//
// Written as plain loops over dense (unpadded) NHWC tensors and the ORIGINAL KRSC weights,
// deliberately structured differently from the Metal engine: the source is first padded by
// edge replication into an explicit TW x TH half4 image, the fold is an explicit
// space-to-depth copy, nearest2x upsampling materialises the upsampled tensor before a
// zero-padded 3x3 convolution, the concat materialises a new tensor, and the head evaluates
// tanh in Double. Arithmetic uses Float accumulation and Float16 rounding at the same tensor
// boundaries as the GPU (one rounding per layer output; exact half sequences for fold, pool,
// baseline dot D, T and composition, using native Float16 multiply / fused addingProduct).

/// Dense NHWC FP16 tensor.
struct Tensor {
    let w: Int
    let h: Int
    let c: Int
    var d: [Float16]

    init(w: Int, h: Int, c: Int) {
        self.w = w; self.h = h; self.c = c
        d = [Float16](repeating: 0, count: w * h * c)
    }

    @inline(__always) func at(_ x: Int, _ y: Int, _ ch: Int) -> Float16 { d[(y * w + x) * c + ch] }
}

/// 8-bit RGBA source frame.
struct RGBAImage {
    let w: Int
    let h: Int
    var px: [UInt8]   // w*h*4

    @inline(__always) func rgb(_ x: Int, _ y: Int) -> (UInt8, UInt8, UInt8) {
        let i = (y * w + x) * 4
        return (px[i], px[i + 1], px[i + 2])
    }
}

/// Deliberately wrong network variants used as negative controls.
enum ReferenceVariant {
    case exact
    case reversedConcat      // cat = [skip, U] instead of [U, skip]
    case shiftedUpsample     // nearest2x source floor((o+1)/2) instead of floor(o/2) (half-pixel misalignment)
}

struct ReferenceResult {
    var x0: Tensor
    var skips: [Tensor]          // s_0..s_3
    var pooled: [Tensor]         // x_1..x_4
    var bottleneck: Tensor       // b = RB_4(x_4)
    var decoded: Tensor          // d_3
    var output: [Float16]        // planar RGB, 3 x outH x outW
}

/// Exact half dot D(v, w) (horizontal or vertical): a = H(v0*w0); a = H(fma(v2,w2,a));
/// b = H(v1*w1); b = H(fma(v3,w3,b)); D = H(a + b).
@inline(__always) func halfDotD(_ v: [Float16], _ w: [Float16]) -> Float16 {
    var a = v[0] * w[0]
    a = a.addingProduct(v[2], w[2])
    var b = v[1] * w[1]
    b = b.addingProduct(v[3], w[3])
    return a + b
}

/// Ordered half-FMA 2x2 pool: TL, TR, BL, BR.
@inline(__always) func orderedPool(_ tl: Float16, _ tr: Float16, _ bl: Float16, _ br: Float16) -> Float16 {
    var a = tl * Float16(0.25)
    a = a.addingProduct(tr, Float16(0.25))
    a = a.addingProduct(bl, Float16(0.25))
    a = a.addingProduct(br, Float16(0.25))
    return a
}

/// T = H((1 + tanh(z)) / 2), tanh evaluated in Double.
@inline(__always) func learnedT(_ z: Float16) -> Float16 {
    Float16((1.0 + tanh(Double(z))) / 2.0)
}

/// Normalised baseline source sample N = H(H(I + 1) * 0.5).
@inline(__always) func baselineNorm(_ i: Float16) -> Float16 {
    (i + Float16(1)) * Float16(0.5)
}

@inline(__always) func eluF(_ v: Float) -> Float { v >= 0 ? v : expf(v) - 1 }

final class CPUReference {
    let model: LoadedModel
    private var fw: [String: [Float]] = [:]   // weights promoted to Float (exact)
    private var fb: [String: [Float]] = [:]

    init(model: LoadedModel) {
        self.model = model
        for (id, cw) in model.convs.map({ ($0.key, $0.value) }) {
            fw[id] = cw.w.map { Float($0) }
            fb[id] = cw.b.map { Float($0) }
        }
    }

    /// Plain cross-correlation, stride 1, zero padding (r-1)/2. Returns f32 sums without bias.
    func convSum(_ x: Tensor, _ id: String) -> [Float] {
        let cw = model.conv(id)
        precondition(cw.c == x.c, "\(id): input has \(x.c) channels, conv expects \(cw.c)")
        let w = fw[id]!
        let (W, H, C, K, R) = (x.w, x.h, x.c, cw.k, cw.r)
        let pad = (R - 1) / 2
        var out = [Float](repeating: 0, count: W * H * K)
        let xs = x.d.map { Float($0) }
        out.withUnsafeMutableBufferPointer { op in
            let o = op
            xs.withUnsafeBufferPointer { xp in
                w.withUnsafeBufferPointer { wp in
                    DispatchQueue.concurrentPerform(iterations: H) { y in
                        for xx in 0..<W {
                            for k in 0..<K {
                                var s: Float = 0
                                for r in 0..<R {
                                    let iy = y + r - pad
                                    if iy < 0 || iy >= H { continue }
                                    for t in 0..<R {
                                        let ix = xx + t - pad
                                        if ix < 0 || ix >= W { continue }
                                        let wb = ((k * R + r) * R + t) * C
                                        let xb = (iy * W + ix) * C
                                        for c in 0..<C { s += wp[wb + c] * xp[xb + c] }
                                    }
                                }
                                o[(y * W + xx) * K + k] = s
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// H(ELU(C(x) + b))
    func convELU(_ x: Tensor, _ id: String) -> Tensor {
        let s = convSum(x, id)
        let b = fb[id]!
        var t = Tensor(w: x.w, h: x.h, c: b.count)
        for i in 0..<s.count { t.d[i] = Float16(eluF(s[i] + b[i % b.count])) }
        return t
    }

    /// RB(x) = H(e + p) with e = H(ELU(C3_1(H(ELU(C3_0(x)))))) and p = H(C1(x)), C1 biasless:
    /// each branch is rounded once, then merged by one half add.
    func residualBlock(_ x: Tensor, _ prefix: String) -> Tensor {
        let mid = convELU(x, "\(prefix).c0")
        let e = convELU(mid, "\(prefix).c1")
        let s1 = convSum(x, "\(prefix).projection")
        var t = Tensor(w: x.w, h: x.h, c: e.c)
        for i in 0..<s1.count { t.d[i] = e.d[i] + Float16(s1[i]) }
        return t
    }

    func pool(_ s: Tensor) -> Tensor {
        var t = Tensor(w: s.w / 2, h: s.h / 2, c: s.c)
        for y in 0..<t.h {
            for x in 0..<t.w {
                for c in 0..<s.c {
                    t.d[(y * t.w + x) * s.c + c] = orderedPool(
                        s.at(2 * x, 2 * y, c), s.at(2 * x + 1, 2 * y, c),
                        s.at(2 * x, 2 * y + 1, c), s.at(2 * x + 1, 2 * y + 1, c))
                }
            }
        }
        return t
    }

    func nearest2x(_ x: Tensor, shift: Int = 0) -> Tensor {
        var t = Tensor(w: 2 * x.w, h: 2 * x.h, c: x.c)
        for y in 0..<t.h {
            for xx in 0..<t.w {
                let sx = min((xx + shift) / 2, x.w - 1), sy = min((y + shift) / 2, x.h - 1)
                for c in 0..<x.c { t.d[(y * t.w + xx) * x.c + c] = x.at(sx, sy, c) }
            }
        }
        return t
    }

    func concat(_ a: Tensor, _ b: Tensor) -> Tensor {
        precondition(a.w == b.w && a.h == b.h)
        var t = Tensor(w: a.w, h: a.h, c: a.c + b.c)
        for p in 0..<(a.w * a.h) {
            for c in 0..<a.c { t.d[p * t.c + c] = a.d[p * a.c + c] }
            for c in 0..<b.c { t.d[p * t.c + a.c + c] = b.d[p * b.c + c] }
        }
        return t
    }

    /// Edge-replicated TW x TH preprocessed half4 image (channel 3 = 0).
    func paddedSource(_ img: RGBAImage, _ g: Geometry) -> Tensor {
        var t = Tensor(w: g.alignedW, h: g.alignedH, c: 4)
        for y in 0..<g.alignedH {
            for x in 0..<g.alignedW {
                let (r, gg, b) = img.rgb(min(x, img.w - 1), min(y, img.h - 1))
                let base = (y * t.w + x) * 4
                t.d[base] = preprocessLUT[Int(r)]
                t.d[base + 1] = preprocessLUT[Int(gg)]
                t.d[base + 2] = preprocessLUT[Int(b)]
                t.d[base + 3] = 0
            }
        }
        return t
    }

    /// Level-0 network input: fold (F[Y,X,8dy+4dx+c] = I[2Y+dy,2X+dx,c]) or the C=8 pad.
    func networkInput(_ padded: Tensor) -> Tensor {
        if model.folded {
            var f = Tensor(w: padded.w / 2, h: padded.h / 2, c: 16)
            for y in 0..<f.h {
                for x in 0..<f.w {
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            for c in 0..<4 {
                                f.d[(y * f.w + x) * 16 + 8 * dy + 4 * dx + c] = padded.at(2 * x + dx, 2 * y + dy, c)
                            }
                        }
                    }
                }
            }
            return f
        }
        var f = Tensor(w: padded.w, h: padded.h, c: 8)
        for p in 0..<(padded.w * padded.h) {
            for c in 0..<4 { f.d[p * 8 + c] = padded.d[p * 4 + c] }
        }
        return f
    }

    /// Head: h = H(ELU(C3(d))), z = H(C1(h) + b1), output (ox, oy, color) =
    /// H(H(B + T) - 0.5) with T from z[floor(oy/p), floor(ox/p), p*p*color + p*(oy%p) + ox%p].
    func head(_ d: Tensor, padded: Tensor, g: Geometry) -> [Float16] {
        let h = convELU(d, "head.c0")
        let zs = convSum(h, "head.c1")
        let b1 = fb["head.c1"]!
        let kz = b1.count
        let z = zs.enumerated().map { Float16($0.element + b1[$0.offset % kz]) }
        return composeOutput(z: z, featW: d.w, kz: kz, padded: padded, g: g)
    }

    /// Composition of learned logits z (feature grid featW wide, kz channels) with the scale-2
    /// source baseline. Exposed separately for the zero-weight head oracle.
    func composeOutput(z: [Float16], featW: Int, kz: Int, padded: Tensor, g: Geometry,
                       coefficients: [[Float16]]? = nil, swapPhaseAxes: Bool = false) -> [Float16] {
        let p = model.shuffle
        let coef = coefficients ?? model.coefficients
        let (OW, OH) = (g.outW, g.outH)
        var out = [Float16](repeating: 0, count: 3 * OW * OH)
        let TW = padded.w, TH = padded.h
        for oy in 0..<OH {
            for ox in 0..<OW {
                let fx = (Double(ox) + 0.5) / 2.0 - 0.5
                let fy = (Double(oy) + 0.5) / 2.0 - 0.5
                let ix = Int(floor(fx)), iy = Int(floor(fy))
                let wx = coef[ox % 2], wy = coef[oy % 2]
                for color in 0..<3 {
                    var rows = [Float16](repeating: 0, count: 4)
                    for t in 0..<4 {
                        let sy = min(max(iy - 1 + t, 0), TH - 1)
                        var v = [Float16](repeating: 0, count: 4)
                        for u in 0..<4 {
                            let sx = min(max(ix - 1 + u, 0), TW - 1)
                            v[u] = baselineNorm(padded.at(sx, sy, color))
                        }
                        rows[t] = halfDotD(v, wx)
                    }
                    let B = halfDotD(rows, wy)
                    let (dy, dx) = swapPhaseAxes ? (ox % p, oy % p) : (oy % p, ox % p)
                    let k = p * p * color + p * dy + dx
                    let zz = z[((oy / p) * featW + ox / p) * kz + k]
                    let T = learnedT(zz)
                    out[color * OW * OH + oy * OW + ox] = (B + T) - Float16(0.5)
                }
            }
        }
        return out
    }

    func run(_ img: RGBAImage, variant: ReferenceVariant = .exact) throws -> ReferenceResult {
        let g = try Geometry(model: model, width: img.w, height: img.h)
        let padded = paddedSource(img, g)
        let x0 = networkInput(padded)
        var x = x0
        var skips: [Tensor] = []
        var pooled: [Tensor] = []
        for i in 0..<4 {
            let s = residualBlock(x, "rb\(i)")
            skips.append(s)
            x = pool(s)
            pooled.append(x)
        }
        let b = residualBlock(x, "rb4")
        var d = b
        for j in 0..<4 {
            let level = 3 - j
            let u = convELU(nearest2x(d, shift: variant == .shiftedUpsample ? 1 : 0), "tb\(j).u")
            let cat = variant == .reversedConcat ? concat(skips[level], u) : concat(u, skips[level])
            d = residualBlock(cat, "tb\(j)")
        }
        let out = head(d, padded: padded, g: g)
        return ReferenceResult(x0: x0, skips: skips, pooled: pooled, bottleneck: b, decoded: d, output: out)
    }
}
