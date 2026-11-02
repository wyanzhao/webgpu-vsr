// Fused preprocess + edge-replicated padding to TW x TH + 2x2 space-to-depth fold.
//   F[Y, X, 8*dy + 4*dx + c] = I[2Y+dy, 2X+dx, c]   (PIXEL-FOLD.md)
// written into the U-half (channels 0..15) of cat0.
// Constants: CAT_STRIDE4 (cat0 pixel stride in vec4<f16>).
// P.out_w/out_h = level-0 size (TW/2 x TH/2), P.src_w/src_h = source W x H.

@group(0) @binding(0) var<uniform> P: LayerParams;
@group(0) @binding(1) var<storage, read_write> CAT0: array<vec4<f16>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.out_w || g.y >= P.out_h) {
    return;
  }
  let mx = i32(P.src_w) - 1;
  let my = i32(P.src_h) - 1;
  let x0 = i32(2u * g.x);
  let y0 = i32(2u * g.y);
  let base = (g.y * P.out_w + g.x) * CAT_STRIDE4;
  CAT0[base + 0u] = vec4<f16>(src_i(min(x0, mx), min(y0, my)));
  CAT0[base + 1u] = vec4<f16>(src_i(min(x0 + 1, mx), min(y0, my)));
  CAT0[base + 2u] = vec4<f16>(src_i(min(x0, mx), min(y0 + 1, my)));
  CAT0[base + 3u] = vec4<f16>(src_i(min(x0 + 1, mx), min(y0 + 1, my)));
}
