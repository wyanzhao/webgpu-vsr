// Folded SR head (kernel ea1940: learned pixel-shuffle 4, original-source baseline scale 2).
//
//   h = H(ELU(C3_sr(d) + b3))           64 hidden channels, kept in workgroup memory
//   z = H(C1_sr(h) + b1)                48 channels, k = 16*colour + 4*(oy%4) + ox%4
//   T = H(0.5 * (t + 1)),  t = (2^(z log2 e) - 2^(-z log2 e)) / (2^(z log2 e) + 2^(-z log2 e))
//   B = vertical D of four horizontal D over the clamped original source, N = H(H(I+1)*0.5)
//   out = H(H(B + T) - 0.5)   (gain 1)   |   B + g*(T - 0.5) in f32 (other gains)
//
// Workgroup: an 8x8 tile of level-0 features -> 32x32 output pixels, 128 threads.
// Phase 1: thread = 2x2 feature block x 8 hidden channels; the horizontal baseline dots
// D(row, wx) of every (output column, source row) of the tile are computed once into
// workgroup memory. Phase 2: thread = 2 feature pixels x one output row phase (oy % 4) x
// 4 column phases x 3 colours (12 learned channels per pixel), producing 8 RGB pixels.
// Constants: K04 (= K0/4), FEAT_STRIDE4 (cat0 pixel stride), BASE_C0, BASE_C1 (coefficients).
// P.in_w/in_h = level-0 size, P.src_w/src_h = W x H, P.out_w/out_h = 2W x 2H.

const HT: u32 = 8u;
const NT: u32 = 128u;
const DW: u32 = HT + 2u;
const ST: u32 = 2u * HT + 4u;
const DTILE_N: u32 = DW * DW * K04;

@group(0) @binding(0) var<uniform> P: LayerParams;
@group(0) @binding(1) var<uniform> FR: FrameParams;
@group(0) @binding(2) var<storage, read> WT: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> FEAT: array<vec4<f16>>;

var<workgroup> dtile: array<vec4<f16>, DTILE_N>;
var<workgroup> hid: array<vec4<f16>, 1024>;
var<workgroup> ntile: array<vec4<f16>, 400>;
// horizontal baseline dots: [source row (ST)][output column of the tile (4*HT)]
var<workgroup> hrow: array<vec4<f16>, 640>;

// T = H(0.5 * (t + 1)) with t = tanh(z) evaluated from exp2(+-z log2 e) in f32.
fn tanh_unit(z: vec4<f32>) -> vec4<f32> {
  let zl = z * 1.4426950408889634;
  let ep = exp2(zl);
  let em = exp2(-zl);
  let t = (ep - em) / (ep + em);
  return hr4(0.5 * (t + vec4<f32>(1.0)));
}

@compute @workgroup_size(NT)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let fx0 = wg.x * HT;
  let fy0 = wg.y * HT;
  let in_w = i32(P.in_w);
  let in_h = i32(P.in_h);
  for (var i = li; i < DTILE_N; i += NT) {
    let c = i % K04;
    let p = i / K04;
    let x = i32(fx0) - 1 + i32(p % DW);
    let y = i32(fy0) - 1 + i32(p / DW);
    var v = vec4<f16>(0.0h);
    if (x >= 0 && x < in_w && y >= 0 && y < in_h) {
      v = FEAT[(u32(y) * P.in_w + u32(x)) * FEAT_STRIDE4 + c];
    }
    dtile[i] = v;
  }
  // Normalised original source, clamped to [0, W-1] x [0, H-1] (edge replication).
  let sx0 = i32(2u * fx0) - 2;
  let sy0 = i32(2u * fy0) - 2;
  let mx = i32(P.src_w) - 1;
  let my = i32(P.src_h) - 1;
  for (var i = li; i < ST * ST; i += NT) {
    let x = clamp(sx0 + i32(i % ST), 0, mx);
    let y = clamp(sy0 + i32(i / ST), 0, my);
    let s = src_i(x, y);
    ntile[i] = vec4<f16>(hr4(hr4(s + vec4<f32>(1.0)) * 0.5));
  }
  workgroupBarrier();

  // Phase 1: hidden = H(ELU(C3(d) + b3)).
  {
    let blk = li % 16u;
    let hg = li / 16u;
    let bx = blk % 4u;
    let by = blk / 4u;
    var acc: array<vec4<f32>, 8>;
    for (var kk = 0u; kk < 8u; kk++) {
      acc[kk] = vec4<f32>(0.0);
    }
    for (var c = 0u; c < K04; c++) {
      var pt: array<vec4<f16>, 16>;
      for (var r = 0u; r < 4u; r++) {
        for (var s = 0u; s < 4u; s++) {
          pt[r * 4u + s] = dtile[((2u * by + r) * DW + 2u * bx + s) * K04 + c];
        }
      }
      for (var kk = 0u; kk < 8u; kk++) {
        let wbase = P.w3 + (hg * 8u + kk) * 9u * K04 + c;
        var a = acc[kk];
        for (var r = 0u; r < 3u; r++) {
          for (var s = 0u; s < 3u; s++) {
            let w = vec4<f32>(WT[wbase + (r * 3u + s) * K04]);
            a += vec4<f32>(
              dot(vec4<f32>(pt[r * 4u + s]), w),
              dot(vec4<f32>(pt[r * 4u + s + 1u]), w),
              dot(vec4<f32>(pt[(r + 1u) * 4u + s]), w),
              dot(vec4<f32>(pt[(r + 1u) * 4u + s + 1u]), w));
          }
        }
        acc[kk] = a;
      }
    }
    for (var k4 = 0u; k4 < 2u; k4++) {
      let kc4 = hg * 2u + k4;
      let bias = vec4<f32>(WT[P.b3 + kc4]);
      for (var q = 0u; q < 4u; q++) {
        let px = 2u * bx + (q & 1u);
        let py = 2u * by + (q >> 1u);
        let v = vec4<f32>(acc[4u * k4][q], acc[4u * k4 + 1u][q], acc[4u * k4 + 2u][q], acc[4u * k4 + 3u][q]) + bias;
        hid[(py * HT + px) * 16u + kc4] = vec4<f16>(hr4(elu4(v)));
      }
    }
  }
  // Horizontal baseline D for each output column and source row of the tile.
  for (var i = li; i < ST * 4u * HT; i += NT) {
    let oxl = i % (4u * HT);
    let r = i / (4u * HT);
    let ox = 4u * fx0 + oxl;
    // taps ix-1..ix+2 with ix = floor((ox + 0.5)/2 - 0.5) = (ox - 1) >> 1
    let cx = u32(((i32(ox) - 1) >> 1u) - 1 - sx0);
    let wx = select(BASE_C0, BASE_C1, (ox & 1u) == 1u);
    let base = r * ST + cx;
    hrow[i] = vec4<f16>(half_dot(
      vec4<f32>(ntile[base]),
      vec4<f32>(ntile[base + 1u]),
      vec4<f32>(ntile[base + 2u]),
      vec4<f32>(ntile[base + 3u]),
      wx));
  }
  workgroupBarrier();

  // Phase 2: z = H(C1(h) + b1); channels k = 16*colour + 4*dy + dx for dx = 0..3.
  let dy = li / 32u;
  let pp = li % 32u;
  let py = pp / 4u;
  let px0 = 2u * (pp % 4u);
  let p0 = py * HT + px0;
  // z[2*colour + j]: the 4 column phases of colour `colour` at pixel px0 + j
  var z: array<vec4<f32>, 6>;
  for (var j = 0u; j < 6u; j++) {
    z[j] = vec4<f32>(0.0);
  }
  for (var j4 = 0u; j4 < 16u; j4++) {
    let h0 = vec4<f32>(hid[p0 * 16u + j4]);
    let h1 = vec4<f32>(hid[(p0 + 1u) * 16u + j4]);
    for (var col = 0u; col < 3u; col++) {
      let kb = 16u * col + 4u * dy;
      let w0 = vec4<f32>(WT[P.w1 + (kb + 0u) * 16u + j4]);
      let w1 = vec4<f32>(WT[P.w1 + (kb + 1u) * 16u + j4]);
      let w2 = vec4<f32>(WT[P.w1 + (kb + 2u) * 16u + j4]);
      let w3 = vec4<f32>(WT[P.w1 + (kb + 3u) * 16u + j4]);
      z[2u * col] += vec4<f32>(dot(h0, w0), dot(h0, w1), dot(h0, w2), dot(h0, w3));
      z[2u * col + 1u] += vec4<f32>(dot(h1, w0), dot(h1, w1), dot(h1, w2), dot(h1, w3));
    }
  }
  var tv: array<vec4<f32>, 6>;
  for (var col = 0u; col < 3u; col++) {
    let bias = vec4<f32>(WT[P.b1 + 4u * col + dy]);
    tv[2u * col] = tanh_unit(hr4(z[2u * col] + bias));
    tv[2u * col + 1u] = tanh_unit(hr4(z[2u * col + 1u] + bias));
  }
  let oy = 4u * (fy0 + py) + dy;
  let iy = (i32(oy) - 1) >> 1u;
  let cy = u32(iy - 1 - sy0);
  let wy = select(BASE_C0, BASE_C1, (oy & 1u) == 1u);
  for (var j = 0u; j < 2u; j++) {
    for (var dx = 0u; dx < 4u; dx++) {
      let oxl = 4u * (px0 + j) + dx;
      let ox = 4u * fx0 + oxl;
      if (ox >= P.out_w || oy >= P.out_h) {
        continue;
      }
      let t = vec4<f32>(tv[j][dx], tv[2u + j][dx], tv[4u + j][dx], 0.0);
      let b = half_dot(
        vec4<f32>(hrow[cy * 32u + oxl]),
        vec4<f32>(hrow[(cy + 1u) * 32u + oxl]),
        vec4<f32>(hrow[(cy + 2u) * 32u + oxl]),
        vec4<f32>(hrow[(cy + 3u) * 32u + oxl]),
        wy);
      var outv: vec4<f32>;
      if (FR.exact == 1u) {
        outv = hr4(hr4(b + t) - vec4<f32>(0.5));
      } else {
        outv = hr4(b + FR.gain * (t - vec4<f32>(0.5)));
      }
      store_out(ox, oy, outv);
    }
  }
}
