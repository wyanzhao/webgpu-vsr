import Foundation

// Deterministic synthetic video frames: smooth gradients, a zone plate (all spatial
// frequencies / orientations), hard-edged blocks and hashed noise, so every network stage sees
// structured, non-symmetric content. Integer hashing keeps frames bit-identical across runs.

@inline(__always) func hash32(_ x: UInt32, _ y: UInt32, _ s: UInt32) -> UInt32 {
    var h = x &* 0x8da6_b343 ^ y &* 0xd816_3841 ^ s &* 0xcb1a_b31f
    h ^= h >> 15
    h = h &* 0x2c1b_3c6d
    h ^= h >> 12
    h = h &* 0x297a_2d39
    h ^= h >> 15
    return h
}

func syntheticFrame(width w: Int, height h: Int, seed: UInt32) -> RGBAImage {
    var px = [UInt8](repeating: 255, count: w * h * 4)
    let cx = Double(w) * (0.37 + 0.05 * Double(seed % 3)), cy = Double(h) * 0.55
    let k = Double.pi / Double(max(w, h)) * 0.9
    for y in 0..<h {
        for x in 0..<w {
            let fx = Double(x), fy = Double(y)
            let gradient = fx / Double(max(w - 1, 1))
            let r2 = (fx - cx) * (fx - cx) + (fy - cy) * (fy - cy)
            let zone = 0.5 + 0.5 * cos(k * r2 / 4.0)
            let block = ((x / 7 + y / 5 + Int(seed)) % 3 == 0) ? 1.0 : 0.0
            let n = hash32(UInt32(x), UInt32(y), seed)
            let noise = Double(n & 0xff) / 255.0
            let r = 0.45 * gradient + 0.35 * zone + 0.2 * noise
            let g = 0.5 * zone + 0.3 * block + 0.2 * Double((n >> 8) & 0xff) / 255.0
            let b = 0.6 * (fy / Double(max(h - 1, 1))) + 0.25 * block + 0.15 * Double((n >> 16) & 0xff) / 255.0
            let i = (y * w + x) * 4
            px[i] = UInt8(max(0, min(255, (r * 255).rounded())))
            px[i + 1] = UInt8(max(0, min(255, (g * 255).rounded())))
            px[i + 2] = UInt8(max(0, min(255, (b * 255).rounded())))
            px[i + 3] = 255
        }
    }
    return RGBAImage(w: w, h: h, px: px)
}

/// Patterns of semantics/emulator/check_folded_heads.py, quantised to 8-bit sources.
enum HeadPattern: String, CaseIterable {
    case black, ramp, impulse

    func image(width w: Int, height h: Int) -> RGBAImage {
        var px = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let rgb: [Double]
                switch self {
                case .black: rgb = [0, 0, 0]
                case .ramp: rgb = [Double(x) / Double(w - 1), Double(y) / Double(h - 1), Double((x + 3 * y) % 7) / 6]
                case .impulse:
                    rgb = [x == 0 && y == 0 ? 1 : 0, x == w / 2 - 1 && y == h / 2 - 1 ? 1 : 0, x == w - 1 && y == h - 1 ? 1 : 0]
                }
                let i = (y * w + x) * 4
                for c in 0..<3 { px[i + c] = UInt8((rgb[c] * 255).rounded()) }
            }
        }
        return RGBAImage(w: w, h: h, px: px)
    }
}
