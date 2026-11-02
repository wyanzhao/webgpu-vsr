// Shared declarations for every webgpu-vsr compute kernel.
// shader_source.ts prepends `enable f16;` and the specialisation constants.

// Per-dispatch parameters (one 256-byte uniform slot per dispatch).
struct LayerParams {
  in_w: u32,   // input feature-map (or source) width
  in_h: u32,
  out_w: u32,  // output width
  out_h: u32,
  w3: u32,     // 3x3 weight offset (vec4<f16> units)
  b3: u32,     // 3x3 bias offset
  w1: u32,     // 1x1 weight offset
  b1: u32,     // 1x1 bias offset
  src_w: u32,  // original source width W
  src_h: u32,  // original source height H
  aux0: u32,   // kernel specific (planar stride)
  aux1: u32,   // kernel specific (planar plane size)
  f0: f32,     // kernel specific (resample scale x)
  f1: f32,     // kernel specific (resample scale y)
  f2: f32,
  f3: f32,
}

// Per-frame parameters (frame-ring slot uniform).
struct FrameParams {
  gain: f32,   // residual gain g = 2 * sharpness
  exact: u32,  // 1 when g == 1: use the exact half composition H(H(B+T)-0.5)
  frame: u32,  // frame counter (diagnostics)
  pad: u32,
}

// Rounding policy. The Metal compiler may evaluate f16 arithmetic at higher precision and
// may also drop f32 -> f16 -> f32 conversion round trips (including quantizeToF16) when it
// can fold them into neighbouring arithmetic. Kernels therefore compute in f32 and round to
// half explicitly with the integer bit-manipulation below, which no floating-point
// optimisation can elide. vec4<f16> is used only for storage and workgroup memory.

// Round each lane to the nearest IEEE binary16 value (ties to even), returned as f32.
// Normal range: round the f32 mantissa to 10 bits. Subnormal half range (|x| < 2^-14):
// multiples of 2^-24 (exact scaling, round() is ties-to-even). Overflow is not reachable
// for the network's value ranges; storage conversion saturates to infinity as usual.
fn hr4(x: vec4<f32>) -> vec4<f32> {
  let u = bitcast<vec4<u32>>(x);
  let sgn = u & vec4<u32>(0x80000000u);
  let a = u & vec4<u32>(0x7fffffffu);
  let lsb = (a >> vec4<u32>(13u)) & vec4<u32>(1u);
  let rn = (a + vec4<u32>(0x0fffu) + lsb) & vec4<u32>(0xffffe000u);
  let sub = round(bitcast<vec4<f32>>(a) * 16777216.0) * (1.0 / 16777216.0);
  let r = select(rn, bitcast<vec4<u32>>(sub), a < vec4<u32>(0x38800000u));
  return bitcast<vec4<f32>>(sgn | r);
}

fn hr3(x: vec3<f32>) -> vec3<f32> {
  return hr4(vec4<f32>(x, 0.0)).xyz;
}

fn hr2(x: vec2<f32>) -> vec2<f32> {
  return hr4(vec4<f32>(x, 0.0, 0.0)).xy;
}

fn hr1(x: f32) -> f32 {
  return hr4(vec4<f32>(x)).x;
}

// ELU (alpha = 1) evaluated in f32.
fn elu4(v: vec4<f32>) -> vec4<f32> {
  return select(exp(v) - vec4<f32>(1.0), v, v >= vec4<f32>(0.0));
}

// Preprocess (dlpp_preProcessE layout 0): I = (H(2R-1), H(2G-1), H(2B-1), 0) as exact halves.
// The PTX evaluates (R - 0.5) + (R - 0.5) in f32 and converts with round-to-nearest.
fn preprocess_i(rgb: vec3<f32>) -> vec4<f32> {
  let c = rgb - vec3<f32>(0.5);
  return vec4<f32>(hr3(c + c), 0.0);
}

// Ordered 2x2 average pool, half FMA chain TL, TR, BL, BR (semantics/HIGH-CLOSURE.md).
// Each f32 FMA of two halves is exact before its single rounding to half.
fn pool4(tl: vec4<f32>, tr: vec4<f32>, bl: vec4<f32>, br: vec4<f32>) -> vec4<f32> {
  let q = vec4<f32>(0.25);
  var a = hr4(tl * q);
  a = hr4(fma(tr, q, a));
  a = hr4(fma(bl, q, a));
  return hr4(fma(br, q, a));
}

// Exact baseline half dot D(v, w) applied lane-wise (lanes = colour channels):
// a = H(v0*w0); a = H(fma(v2,w2,a)); b = H(v1*w1); b = H(fma(v3,w3,b)); D = H(a+b).
fn half_dot(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, w: vec4<f32>) -> vec4<f32> {
  var a = hr4(v0 * w.x);
  a = hr4(fma(v2, vec4<f32>(w.z), a));
  var b = hr4(v1 * w.y);
  b = hr4(fma(v3, vec4<f32>(w.w), b));
  return hr4(a + b);
}
