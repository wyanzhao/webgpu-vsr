import CoreGraphics
import Foundation
import ImageIO
import Metal

// Quality mode (`--quality DIR`): upscales every 960x540-class PNG in each input set of DIR
// (any subdirectory other than `ref`, e.g. `clean` and `h264` from tools/make_quality_set.sh)
// by 2x with every model and two CPU baselines, and scores the result against the reference of
// the same name in DIR/ref. Metrics are computed on BT.709 luma of the full-range RGB image:
// PSNR, SSIM (11x11 Gaussian, sigma 1.5, the usual constants) and the detail ratio, the RMS
// of the 3x3 Laplacian of the output over that of the reference (below 1 = softer than the
// reference, above 1 = more high-frequency energy, sharpening or artefacts).

struct QualityRow: Codable {
    let set: String
    let method: String
    let frames: Int
    let psnrY: Double
    let ssimY: Double
    let detailRatio: Double
    let perClip: [String: [Double]]   // clip -> [psnrY, ssimY, detailRatio]
}

func loadPNG(_ url: URL) throws -> RGBAImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw BenchError("cannot read \(url.path)") }
    // Raw samples, no colour management: the PNGs hold the video's own full-range code values.
    guard img.bitsPerComponent == 8, img.bitsPerPixel == 24 || img.bitsPerPixel == 32,
          let data = img.dataProvider?.data as Data? else { throw BenchError("\(url.lastPathComponent): need 8-bit RGB(A)") }
    let w = img.width, h = img.height, bpp = img.bitsPerPixel / 8, row = img.bytesPerRow
    var px = [UInt8](repeating: 255, count: w * h * 4)
    data.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
        for y in 0..<h {
            for x in 0..<w {
                let s = y * row + x * bpp, d = (y * w + x) * 4
                px[d] = p[s]; px[d + 1] = p[s + 1]; px[d + 2] = p[s + 2]
            }
        }
    }
    return RGBAImage(w: w, h: h, px: px)
}

func writePNG(_ img: RGBAImage, _ url: URL) throws {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let provider = CGDataProvider(data: Data(img.px) as CFData),
          let cg = CGImage(width: img.w, height: img.h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: img.w * 4,
                           space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dst = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else { throw BenchError("cannot write \(url.path)") }
    CGImageDestinationAddImage(dst, cg, nil)
    guard CGImageDestinationFinalize(dst) else { throw BenchError("cannot write \(url.path)") }
}

/// Separable 2x upscale with a symmetric kernel, sampling at the pixel-centre phases (o+0.5)/2-0.5.
func cpuUpscale2x(_ img: RGBAImage, radius: Int, kernel: (Double) -> Double) -> RGBAImage {
    let w = img.w, h = img.h, ow = 2 * w, oh = 2 * h
    func taps(_ o: Int, _ n: Int) -> [(Int, Double)] {
        let c = (Double(o) + 0.5) / 2 - 0.5
        let base = Int(floor(c))
        var t: [(Int, Double)] = []
        var sum = 0.0
        for i in (base - radius + 1)...(base + radius) {
            let wgt = kernel(c - Double(i))
            t.append((min(max(i, 0), n - 1), wgt))
            sum += wgt
        }
        return t.map { ($0.0, $0.1 / sum) }
    }
    let tx = (0..<ow).map { taps($0, w) }, ty = (0..<oh).map { taps($0, h) }
    var tmp = [Double](repeating: 0, count: ow * h * 3)
    for y in 0..<h {
        for x in 0..<ow {
            for c in 0..<3 {
                var a = 0.0
                for (sx, k) in tx[x] { a += k * Double(img.px[(y * w + sx) * 4 + c]) }
                tmp[(y * ow + x) * 3 + c] = a
            }
        }
    }
    var px = [UInt8](repeating: 255, count: ow * oh * 4)
    for y in 0..<oh {
        for x in 0..<ow {
            for c in 0..<3 {
                var a = 0.0
                for (sy, k) in ty[y] { a += k * tmp[(sy * ow + x) * 3 + c] }
                px[(y * ow + x) * 4 + c] = UInt8(max(0, min(255, a.rounded())))
            }
        }
    }
    return RGBAImage(w: ow, h: oh, px: px)
}

func catmullRom(_ x: Double) -> Double {
    let a = abs(x)
    if a < 1 { return 1.5 * a * a * a - 2.5 * a * a + 1 }
    if a < 2 { return -0.5 * a * a * a + 2.5 * a * a - 4 * a + 2 }
    return 0
}

func lanczos3(_ x: Double) -> Double {
    if x == 0 { return 1 }
    let a = abs(x)
    if a >= 3 { return 0 }
    let p = Double.pi * x
    return 3 * sin(p) * sin(p / 3) / (p * p)
}

func luma(_ img: RGBAImage) -> [Double] {
    (0..<(img.w * img.h)).map { i in
        0.2126 * Double(img.px[i * 4]) + 0.7152 * Double(img.px[i * 4 + 1]) + 0.0722 * Double(img.px[i * 4 + 2])
    }
}

func psnr(_ a: [Double], _ b: [Double]) -> Double {
    var se = 0.0
    for i in 0..<a.count { let d = a[i] - b[i]; se += d * d }
    let mse = se / Double(a.count)
    return mse == 0 ? 99 : 10 * log10(255 * 255 / mse)
}

func gaussianBlur(_ v: [Double], _ w: Int, _ h: Int) -> [Double] {
    let r = 5, sigma = 1.5
    var k = (-r...r).map { exp(-Double($0 * $0) / (2 * sigma * sigma)) }
    let s = k.reduce(0, +)
    k = k.map { $0 / s }
    var t = [Double](repeating: 0, count: w * h), o = t
    for y in 0..<h { for x in 0..<w {
        var a = 0.0
        for i in -r...r { a += k[i + r] * v[y * w + min(max(x + i, 0), w - 1)] }
        t[y * w + x] = a
    } }
    for y in 0..<h { for x in 0..<w {
        var a = 0.0
        for i in -r...r { a += k[i + r] * t[min(max(y + i, 0), h - 1) * w + x] }
        o[y * w + x] = a
    } }
    return o
}

func ssim(_ a: [Double], _ b: [Double], _ w: Int, _ h: Int) -> Double {
    let c1: Double = 6.5025, c2: Double = 58.5225   // (0.01*255)^2, (0.03*255)^2
    let ma = gaussianBlur(a, w, h), mb = gaussianBlur(b, w, h)
    let saa = gaussianBlur(a.map { v -> Double in v * v }, w, h), sbb = gaussianBlur(b.map { v -> Double in v * v }, w, h)
    let sab = gaussianBlur(zip(a, b).map { $0 * $1 }, w, h)
    var total = 0.0
    for i in 0..<a.count {
        let va = saa[i] - ma[i] * ma[i], vb = sbb[i] - mb[i] * mb[i], cov = sab[i] - ma[i] * mb[i]
        total += ((2 * ma[i] * mb[i] + c1) * (2 * cov + c2)) / ((ma[i] * ma[i] + mb[i] * mb[i] + c1) * (va + vb + c2))
    }
    return total / Double(a.count)
}

func laplacianRMS(_ v: [Double], _ w: Int, _ h: Int) -> Double {
    var s = 0.0
    for y in 1..<(h - 1) { for x in 1..<(w - 1) {
        let i = y * w + x
        let l = 4 * v[i] - v[i - 1] - v[i + 1] - v[i - w] - v[i + w]
        s += l * l
    } }
    return sqrt(s / Double((w - 2) * (h - 2)))
}

func runQuality(ctx: MetalContext, modelsDir: URL, dir: URL, outDir: URL?, models ids: [String]) throws -> [QualityRow] {
    let fm = FileManager.default
    let refDir = dir.appendingPathComponent("ref")
    let sets = try fm.contentsOfDirectory(atPath: dir.path).filter { name in
        var isDir: ObjCBool = false
        return name != "ref" && name != "src" && fm.fileExists(atPath: dir.appendingPathComponent(name).path, isDirectory: &isDir) && isDir.boolValue
    }.sorted()
    guard !sets.isEmpty else { throw BenchError("\(dir.path): no input sets next to ref/") }
    let models = try ids.map { try LoadedModel.load(id: $0, modelsDir: modelsDir) }
    var engines: [String: VSREngine] = [:]
    var rows: [QualityRow] = []
    for set in sets {
        let files = try fm.contentsOfDirectory(atPath: dir.appendingPathComponent(set).path).filter { $0.hasSuffix(".png") }.sorted()
        // method -> clip -> [(psnr, ssim, detail)]
        var acc: [String: [String: [(Double, Double, Double)]]] = [:]
        var methods: [String] = []
        for f in files {
            let input = try loadPNG(dir.appendingPathComponent(set).appendingPathComponent(f))
            let ref = try loadPNG(refDir.appendingPathComponent(f))
            guard ref.w == 2 * input.w, ref.h == 2 * input.h else { throw BenchError("\(f): reference is not 2x the input") }
            let clip = String(f[..<f.range(of: "_", options: .backwards)!.lowerBound])
            let refY = luma(ref), refDetail = laplacianRMS(refY, ref.w, ref.h)
            var outputs: [(String, RGBAImage)] = [
                ("catmull-rom", cpuUpscale2x(input, radius: 2, kernel: catmullRom)),
                ("lanczos3", cpuUpscale2x(input, radius: 3, kernel: lanczos3)),
            ]
            for m in models {
                let key = "\(m.id) \(input.w)x\(input.h)"
                if engines[key] == nil {
                    engines[key] = try VSREngine(ctx: ctx, model: m, width: input.w, height: input.h, format: .rgba8)
                }
                let e = engines[key]!
                try runFrame(e, source: try makeSourceBuffer(ctx.device, input))
                outputs.append((m.id, RGBAImage(w: e.geometry.outW, h: e.geometry.outH, px: e.readOutputRGBA())))
            }
            if methods.isEmpty { methods = outputs.map { $0.0 } }
            for (name, img) in outputs {
                let y = luma(img)
                acc[name, default: [:]][clip, default: []].append(
                    (psnr(y, refY), ssim(y, refY, img.w, img.h), laplacianRMS(y, img.w, img.h) / refDetail))
            }
            if let out = outDir, f.hasSuffix("_00.png") {
                let d = out.appendingPathComponent(set)
                try fm.createDirectory(at: d, withIntermediateDirectories: true)
                for (name, img) in outputs {
                    try writePNG(img, d.appendingPathComponent("\(clip)_\(name).png"))
                }
            }
            print("  \(set)/\(f)")
        }
        for name in methods {
            let clips = acc[name]!
            let all = clips.values.flatMap { $0 }
            func mean(_ v: [Double]) -> Double { v.reduce(0, +) / Double(v.count) }
            let per = clips.mapValues { v in [mean(v.map { $0.0 }), mean(v.map { $0.1 }), mean(v.map { $0.2 })] }
            rows.append(QualityRow(set: set, method: name, frames: all.count, psnrY: mean(all.map { $0.0 }),
                                   ssimY: mean(all.map { $0.1 }), detailRatio: mean(all.map { $0.2 }), perClip: per))
        }
    }
    return rows
}

func printQuality(_ rows: [QualityRow]) {
    print("\nset     method        frames  PSNR-Y dB  SSIM-Y   detail")
    for r in rows {
        print(String(format: "%-7@ %-12@ %6d  %9.3f  %.4f  %6.3f", r.set as NSString, r.method as NSString,
                     r.frames, r.psnrY, r.ssimY, r.detailRatio))
    }
    let clips = Set(rows.flatMap { $0.perClip.keys }).sorted()
    for set in Set(rows.map { $0.set }).sorted() {
        print("\n\(set): PSNR-Y per clip")
        let rs = rows.filter { $0.set == set }
        print("clip".padding(toLength: 16, withPad: " ", startingAt: 0) + rs.map { $0.method.padding(toLength: 12, withPad: " ", startingAt: 0) }.joined())
        for c in clips {
            print(c.padding(toLength: 16, withPad: " ", startingAt: 0) + rs.map {
                String(format: "%-12.3f", $0.perClip[c]?[0] ?? .nan)
            }.joined())
        }
    }
}
