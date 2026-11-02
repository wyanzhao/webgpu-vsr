// Generic stride-1 3x3 convolution (cross-correlation, zero padding 1) over FP16 NHWC with
// f32 accumulation, bias + ELU in f32 and one rounding to half per output. Specialisations:
//
//   (default)  RB part A:  DST = H(ELU(C3(SRC) + b))
//   UPSAMPLE   decoder up: DST = H(ELU(C3u(nearest2x(SRC)) + b)); source coordinate
//              floor((out + tap - 1) / 2) per axis, out-of-range source -> 0
//   PROJ       RB part B:  e = H(ELU(C3(SRC) + b)); p = H(C1(X)) with the biasless 1x1
//              projection of X read at the thread's own pixels from DST (the read_write
//              buffer that also receives the result); DST = H(e + p)
//   POOL       (with PROJ) ordered TL,TR,BL,BR half-FMA 2x2 pool of the result, written
//              to POOLED (the next level's U-half or x4).
//
// Each workgroup covers a TX x TY output tile and ALL K output channels; a thread owns a
// 2x2 pixel block and KB channels (KG = K / KB channel groups). The input tile plus a
// 1-pixel halo is staged in workgroup memory in chunks of CCH channels. Because the whole
// channel range of a pixel belongs to one workgroup, PROJ may read X and write DST in the
// same buffer: all reads of X happen before the storageBarrier, all writes after it, and
// 3x3 neighbourhood reads only touch SRC (a different buffer).
//
// Constants (vec4 units end in 4): CIN4, IN_STRIDE4, IN_OFF4, K, KB, KG, TX, TY, CCH4,
// OUT_STRIDE4, OUT_OFF4, and with PROJ: CINX4, X_OFF4; with POOL: POOL_STRIDE4, POOL_OFF4.
// P.in_w/in_h = SRC level size, P.out_w/out_h = DST level size.

const BX: u32 = TX / 2u;
const BY: u32 = TY / 2u;
const NB: u32 = BX * BY;
const NT: u32 = NB * KG;
const KB4: u32 = KB / 4u;
#if UPSAMPLE
const SW: u32 = BX + 2u;
const SH: u32 = BY + 2u;
#else
const SW: u32 = TX + 2u;
const SH: u32 = TY + 2u;
#endif
const TILE_N: u32 = SW * SH * CCH4;

@group(0) @binding(0) var<uniform> P: LayerParams;
@group(0) @binding(1) var<storage, read> WT: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> SRC: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> DST: array<vec4<f16>>;
#if POOL
@group(0) @binding(4) var<storage, read_write> POOLED: array<vec4<f16>>;
#endif

var<workgroup> tile: array<vec4<f16>, TILE_N>;

@compute @workgroup_size(NT)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let kg = li / NB;
  let blk = li % NB;
  let bx = blk % BX;
  let by = blk / BX;
  let ox = wg.x * TX + 2u * bx;
  let oy = wg.y * TY + 2u * by;
#if UPSAMPLE
  let tx0 = i32(wg.x * BX) - 1;
  let ty0 = i32(wg.y * BY) - 1;
#else
  let tx0 = i32(wg.x * TX) - 1;
  let ty0 = i32(wg.y * TY) - 1;
#endif
  let in_w = i32(P.in_w);
  let in_h = i32(P.in_h);
  // pixel q of the block: q = 0 TL, 1 TR, 2 BL, 3 BR
  var pix: array<u32, 4>;
  var ok: array<bool, 4>;
  for (var q = 0u; q < 4u; q++) {
    let px = ox + (q & 1u);
    let py = oy + (q >> 1u);
    ok[q] = px < P.out_w && py < P.out_h;
    pix[q] = py * P.out_w + px;
  }

#if PROJ
  // Biasless 1x1 projection of X at the thread's own pixels, rounded to half.
  var proj: array<vec4<f32>, KB>;
  {
    var pacc: array<vec4<f32>, KB>;
    for (var kk = 0u; kk < KB; kk++) {
      pacc[kk] = vec4<f32>(0.0);
    }
    for (var c = 0u; c < CINX4; c++) {
      var xv: array<vec4<f32>, 4>;
      for (var q = 0u; q < 4u; q++) {
        xv[q] = vec4<f32>(0.0);
        if (ok[q]) {
          xv[q] = vec4<f32>(DST[pix[q] * OUT_STRIDE4 + X_OFF4 + c]);
        }
      }
      for (var kk = 0u; kk < KB; kk++) {
        let w = vec4<f32>(WT[P.w1 + (kg * KB + kk) * CINX4 + c]);
        pacc[kk] += vec4<f32>(dot(xv[0], w), dot(xv[1], w), dot(xv[2], w), dot(xv[3], w));
      }
    }
    for (var kk = 0u; kk < KB; kk++) {
      proj[kk] = hr4(pacc[kk]);
    }
  }
#endif

  var acc: array<vec4<f32>, KB>;
  for (var kk = 0u; kk < KB; kk++) {
    acc[kk] = vec4<f32>(0.0);
  }
  for (var c0 = 0u; c0 < CIN4; c0 += CCH4) {
    workgroupBarrier();
    for (var i = li; i < TILE_N; i += NT) {
      let c = i % CCH4;
      let p = i / CCH4;
      let sx = tx0 + i32(p % SW);
      let sy = ty0 + i32(p / SW);
      var v = vec4<f16>(0.0h);
      if (sx >= 0 && sx < in_w && sy >= 0 && sy < in_h) {
        v = SRC[(u32(sy) * P.in_w + u32(sx)) * IN_STRIDE4 + IN_OFF4 + c0 + c];
      }
      tile[i] = v;
    }
    workgroupBarrier();
    for (var c = 0u; c < CCH4; c++) {
#if UPSAMPLE
      // low-resolution 3x3 patch: source rows/cols m-1, m, m+1 of the 2x2 block (2m, 2m+1)
      var pt: array<vec4<f16>, 9>;
      for (var r = 0u; r < 3u; r++) {
        for (var s = 0u; s < 3u; s++) {
          pt[r * 3u + s] = tile[((by + r) * SW + bx + s) * CCH4 + c];
        }
      }
#else
      var pt: array<vec4<f16>, 16>;
      for (var r = 0u; r < 4u; r++) {
        for (var s = 0u; s < 4u; s++) {
          pt[r * 4u + s] = tile[((2u * by + r) * SW + 2u * bx + s) * CCH4 + c];
        }
      }
#endif
      for (var kk = 0u; kk < KB; kk++) {
        let wbase = P.w3 + (kg * KB + kk) * 9u * CIN4 + c0 + c;
        var a = acc[kk];
        for (var r = 0u; r < 3u; r++) {
          for (var s = 0u; s < 3u; s++) {
            let w = vec4<f32>(WT[wbase + (r * 3u + s) * CIN4]);
#if UPSAMPLE
            // patch index of floor((2m + d + tap - 1) / 2) - (m - 1), via arithmetic shift
            let r0 = u32(((i32(r) - 1) >> 1u) + 1);
            let r1 = u32(((i32(r)) >> 1u) + 1);
            let s0 = u32(((i32(s) - 1) >> 1u) + 1);
            let s1 = u32(((i32(s)) >> 1u) + 1);
            a += vec4<f32>(
              dot(vec4<f32>(pt[r0 * 3u + s0]), w),
              dot(vec4<f32>(pt[r0 * 3u + s1]), w),
              dot(vec4<f32>(pt[r1 * 3u + s0]), w),
              dot(vec4<f32>(pt[r1 * 3u + s1]), w));
#else
            a += vec4<f32>(
              dot(vec4<f32>(pt[r * 4u + s]), w),
              dot(vec4<f32>(pt[r * 4u + s + 1u]), w),
              dot(vec4<f32>(pt[(r + 1u) * 4u + s]), w),
              dot(vec4<f32>(pt[(r + 1u) * 4u + s + 1u]), w));
#endif
          }
        }
        acc[kk] = a;
      }
    }
  }

#if PROJ
  // Every read of X in this workgroup is complete before any result is written.
  storageBarrier();
#endif

  for (var k4 = 0u; k4 < KB4; k4++) {
    let kc4 = kg * KB4 + k4;  // output channel group (vec4 index within the K channels)
    let bias = vec4<f32>(WT[P.b3 + kc4]);
    var o: array<vec4<f32>, 4>;
    for (var q = 0u; q < 4u; q++) {
      let v = vec4<f32>(acc[4u * k4][q], acc[4u * k4 + 1u][q], acc[4u * k4 + 2u][q], acc[4u * k4 + 3u][q]);
      var h = hr4(elu4(v + bias));
#if PROJ
      h = hr4(h + vec4<f32>(proj[4u * k4][q], proj[4u * k4 + 1u][q], proj[4u * k4 + 2u][q], proj[4u * k4 + 3u][q]));
#endif
      o[q] = h;
      if (ok[q]) {
        DST[pix[q] * OUT_STRIDE4 + OUT_OFF4 + kc4] = vec4<f16>(h);
      }
    }
#if POOL
    if (ok[0]) {
      let pw = P.out_w / 2u;
      POOLED[((oy / 2u) * pw + ox / 2u) * POOL_STRIDE4 + POOL_OFF4 + kc4] = vec4<f16>(pool4(o[0], o[1], o[2], o[3]));
    }
#endif
  }
}
