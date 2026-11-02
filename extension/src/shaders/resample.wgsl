// Catmull-Rom (a = -0.5) resample-and-compose, dlpp_ResampleAndComposeFP16E
// (semantics/resample.ptx). Source: planar FP16 RGB (VSR internal image) or, for Quality 0,
// the layout-1 preprocess H(R), H(G), H(B) of the source frame evaluated per tap.
//
//   c = fma(u + 0.5, scale, -0.5), taps floor(c)-1 .. floor(c)+2 clamped to [0, valid-1]
//   t = half(c - floor(c)); P-1, P0, P1, P2 by the PTX half-FMA chains
//   weight(i, j) = half(wx_i * wy_j); acc = f32 FMA over 16 taps (rows outer, columns inner)
//   byte = floor_rz(sat(acc) * 255 + 0.5)
//
// The PTX chains round once per fma.rn.f16x2; an f32 fma followed by a rounding to half rounds
// twice and differs for one fraction (t = 0.022247314453125). The weights depend on the half t
// alone, so they come from CUBIC (src/pipeline/cubic_weights.ts): entry = half bit pattern of
// t (0 .. 0x3C00), word 0 = P-1 | P0 << 16, word 1 = P1 | P2 << 16, each a half bit pattern.
//
// P.in_w/in_h = valid source size, P.aux0 = planar row stride, P.aux1 = plane size,
// P.out_w/out_h = output size, P.f0/f1 = scale x/y.

@group(0) @binding(0) var<uniform> P: LayerParams;
@group(0) @binding(3) var<storage, read> CUBIC: array<u32>;

// Exact single-rounding weights [P-1, P0, P1, P2] of the half fraction t (t is exactly a
// half, so pack2x16float only re-encodes it).
fn cubic_weights(t: f32) -> vec4<f32> {
  let i = 2u * (pack2x16float(vec2<f32>(t, 0.0)) & 0xffffu);
  return vec4<f32>(unpack2x16float(CUBIC[i]), unpack2x16float(CUBIC[i + 1u]));
}

// round-half-up of s >= 0, identical to add.rz(s, 0.5) followed by truncation:
// floor(s) + (frac(s) >= 0.5), where s - floor(s) is exact.
fn round_half_up(s: vec3<f32>) -> vec3<f32> {
  let f = floor(s);
  return f + select(vec3<f32>(0.0), vec3<f32>(1.0), (s - f) >= vec3<f32>(0.5));
}

fn to_bytes(acc: vec3<f32>) -> vec3<f32> {
  return round_half_up(clamp(acc, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0);
}

@compute @workgroup_size(32, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.out_w || gid.y >= P.out_h) {
    return;
  }
  let cx = fma(f32(gid.x) + 0.5, P.f0, -0.5);
  let cy = fma(f32(gid.y) + 0.5, P.f1, -0.5);
  let flx = floor(cx);
  let fly = floor(cy);
  let ix = i32(flx);
  let iy = i32(fly);
  // half fractions (lanes: x, y) and their exact cubic weights
  let t = hr2(vec2<f32>(cx - flx, cy - fly));
  let wx = cubic_weights(t.x);
  let wy = cubic_weights(t.y);
  let mx = i32(P.in_w) - 1;
  let my = i32(P.in_h) - 1;
  var xs: array<i32, 4>;
  for (var i = 0; i < 4; i++) {
    xs[i] = clamp(ix - 1 + i, 0, mx);
  }
  var acc = vec3<f32>(0.0);
  for (var j = 0; j < 4; j++) {
    let y = clamp(iy - 1 + j, 0, my);
    for (var i = 0; i < 4; i++) {
      let w = hr1(wx[i] * wy[j]);
      acc = fma(vec3<f32>(w), load_src(xs[i], y), acc);
    }
  }
  store_out(gid.x, gid.y, to_bytes(acc));
}
