/**
 * Reference operators for the folded low/medium RTX VSR network
 * (docs/IMPLEMENTATION_NOTES.md section 3). Feature maps are NHWC with every stored value a
 * half (Math.f16round). Convolutions accumulate in float64 (the exact sum; the GPU's float32
 * accumulation differs from it by far less than half an FP16 ulp of the result), add the
 * bias, apply ELU, then round to half once. Fold, pool, baseline D, T and composition use
 * the exact half sequences of the specification.
 */
import { F, H, elu, fmaF, fmaH } from './half';

export interface Tensor {
  w: number;
  h: number;
  c: number;
  data: Float32Array;
}

export function tensor(w: number, h: number, c: number, data?: Float32Array): Tensor {
  const d = data ?? new Float32Array(w * h * c);
  if (d.length !== w * h * c) throw new Error(`tensor data length ${d.length} != ${w}x${h}x${c}`);
  return { w, h, c, data: d };
}

/** Preprocess one 8-bit pixel: I = [H(2R-1), H(2G-1), H(2B-1), 0], R = float32(v/255). */
export function preprocessByte(v: number): number {
  const r = F(v / 255);
  const c = F(r - 0.5);
  return H(F(c + c));
}

/** Preprocess a float32 colour value already in [0, 1]. */
export function preprocessFloat(r: number): number {
  const c = F(F(r) - 0.5);
  return H(F(c + c));
}

/** RGBA8 (R in byte 0) -> half4 I image of W x H. */
export function preprocessRgba8(rgba: Uint8Array, w: number, h: number): Tensor {
  const out = tensor(w, h, 4);
  for (let p = 0; p < w * h; p++) {
    out.data[4 * p] = preprocessByte(rgba[4 * p]);
    out.data[4 * p + 1] = preprocessByte(rgba[4 * p + 1]);
    out.data[4 * p + 2] = preprocessByte(rgba[4 * p + 2]);
    out.data[4 * p + 3] = 0;
  }
  return out;
}

/** Edge-replicate the W x H source to tw x th. */
export function padReplicate(src: Tensor, tw: number, th: number): Tensor {
  const out = tensor(tw, th, src.c);
  for (let y = 0; y < th; y++) {
    const sy = Math.min(y, src.h - 1);
    for (let x = 0; x < tw; x++) {
      const sx = Math.min(x, src.w - 1);
      out.data.set(src.data.subarray((sy * src.w + sx) * src.c, (sy * src.w + sx + 1) * src.c), (y * tw + x) * src.c);
    }
  }
  return out;
}

/** F[Y, X, 8*dy + 4*dx + c] = I[2Y+dy, 2X+dx, c] (pure permutation). */
export function fold(src: Tensor): Tensor {
  if (src.c !== 4 || src.w % 2 || src.h % 2) throw new Error('fold expects an even-sized half4 image');
  const out = tensor(src.w / 2, src.h / 2, 16);
  for (let Y = 0; Y < out.h; Y++) {
    for (let X = 0; X < out.w; X++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          for (let c = 0; c < 4; c++) {
            out.data[(Y * out.w + X) * 16 + 8 * dy + 4 * dx + c] = src.data[((2 * Y + dy) * src.w + 2 * X + dx) * 4 + c];
          }
        }
      }
    }
  }
  return out;
}

export type Activation = 'elu' | 'none';

/**
 * 3x3 cross-correlation, stride 1, zero padding 1, KRSC weights, optional bias and ELU,
 * one rounding to half. With `upsample`, the input is the low-resolution map and the conv
 * is evaluated at (2*in.w) x (2*in.h) with source coordinate floor((out + tap - 1) / 2),
 * computed with an arithmetic right shift; out-of-range sources contribute zero.
 */
export function conv3x3(
  input: Tensor,
  weight: ArrayLike<number>,
  bias: ArrayLike<number> | null,
  k: number,
  act: Activation,
  upsample = false,
): Tensor {
  const cin = input.c;
  if (weight.length !== k * 9 * cin) throw new Error(`conv3x3 weight has ${weight.length} values, expected ${k * 9 * cin}`);
  const ow = upsample ? input.w * 2 : input.w;
  const oh = upsample ? input.h * 2 : input.h;
  const out = tensor(ow, oh, k);
  const acc = new Float64Array(k);
  const src = input.data;
  for (let oy = 0; oy < oh; oy++) {
    for (let ox = 0; ox < ow; ox++) {
      acc.fill(0);
      for (let r = 0; r < 3; r++) {
        const sy = upsample ? (oy + r - 1) >> 1 : oy + r - 1;
        if (sy < 0 || sy >= input.h) continue;
        for (let s = 0; s < 3; s++) {
          const sx = upsample ? (ox + s - 1) >> 1 : ox + s - 1;
          if (sx < 0 || sx >= input.w) continue;
          const ib = (sy * input.w + sx) * cin;
          const tap = r * 3 + s;
          for (let kk = 0; kk < k; kk++) {
            const wb = (kk * 9 + tap) * cin;
            let a = 0;
            for (let c = 0; c < cin; c++) a += src[ib + c] * weight[wb + c];
            acc[kk] += a;
          }
        }
      }
      const ob = (oy * ow + ox) * k;
      for (let kk = 0; kk < k; kk++) {
        const v = acc[kk] + (bias ? bias[kk] : 0);
        out.data[ob + kk] = H(act === 'elu' ? elu(v) : v);
      }
    }
  }
  return out;
}

/** 1x1 convolution (KRSC [K,1,1,C]), optional bias, one rounding to half. */
export function conv1x1(input: Tensor, weight: ArrayLike<number>, bias: ArrayLike<number> | null, k: number): Tensor {
  const cin = input.c;
  if (weight.length !== k * cin) throw new Error(`conv1x1 weight has ${weight.length} values, expected ${k * cin}`);
  const out = tensor(input.w, input.h, k);
  for (let p = 0; p < input.w * input.h; p++) {
    for (let kk = 0; kk < k; kk++) {
      let a = 0;
      for (let c = 0; c < cin; c++) a += input.data[p * cin + c] * weight[kk * cin + c];
      out.data[p * k + kk] = H(a + (bias ? bias[kk] : 0));
    }
  }
  return out;
}

/** Elementwise half add H(a + b). */
export function addHalf(a: Tensor, b: Tensor): Tensor {
  if (a.data.length !== b.data.length) throw new Error('addHalf shape mismatch');
  const out = tensor(a.w, a.h, a.c);
  for (let i = 0; i < a.data.length; i++) out.data[i] = H(a.data[i] + b.data[i]);
  return out;
}

/** Ordered 2x2 pool: a = H(TL/4); a = H(fma(TR,.25,a)); a = H(fma(BL,.25,a)); H(fma(BR,.25,a)). */
export function pool2x2(input: Tensor): Tensor {
  if (input.w % 2 || input.h % 2) throw new Error('pool2x2 expects even sizes');
  const out = tensor(input.w / 2, input.h / 2, input.c);
  const c = input.c;
  for (let y = 0; y < out.h; y++) {
    for (let x = 0; x < out.w; x++) {
      const tl = ((2 * y) * input.w + 2 * x) * c;
      const tr = tl + c;
      const bl = tl + input.w * c;
      const br = bl + c;
      for (let k = 0; k < c; k++) {
        let a = H(input.data[tl + k] * 0.25);
        a = fmaH(input.data[tr + k], 0.25, a);
        a = fmaH(input.data[bl + k], 0.25, a);
        out.data[(y * out.w + x) * c + k] = fmaH(input.data[br + k], 0.25, a);
      }
    }
  }
  return out;
}

/** Channel concatenation [a | b]. */
export function concat(a: Tensor, b: Tensor): Tensor {
  if (a.w !== b.w || a.h !== b.h) throw new Error('concat size mismatch');
  const out = tensor(a.w, a.h, a.c + b.c);
  for (let p = 0; p < a.w * a.h; p++) {
    out.data.set(a.data.subarray(p * a.c, (p + 1) * a.c), p * out.c);
    out.data.set(b.data.subarray(p * b.c, (p + 1) * b.c), p * out.c + a.c);
  }
  return out;
}

/** Stored scale-2 baseline coefficients, tap order [-1, 0, 1, 2], index = output parity. */
export const REF_BASELINE_COEFFICIENTS = [
  [-0.03515625, 0.26171875, 0.87890625, -0.10546875],
  [-0.10546875, 0.87890625, 0.26171875, -0.03515625],
];

/** Exact half dot D(v, w). */
export function halfDot(v: ArrayLike<number>, w: ArrayLike<number>): number {
  let a = H(v[0] * w[0]);
  a = fmaH(v[2], w[2], a);
  let b = H(v[1] * w[1]);
  b = fmaH(v[3], w[3], b);
  return H(a + b);
}

/**
 * Source baseline B for output (ox, oy), colour c, from the half4 source I (W x H; the
 * aligned padding is edge replication, so clamping to [0, TW-1] then replicating equals
 * clamping to [0, W-1]).
 */
export function baseline(src: Tensor, ox: number, oy: number, c: number, coefs = REF_BASELINE_COEFFICIENTS): number {
  const ix = Math.floor((ox + 0.5) / 2 - 0.5);
  const iy = Math.floor((oy + 0.5) / 2 - 0.5);
  const n = (x: number, y: number) => {
    const sx = Math.min(Math.max(x, 0), src.w - 1);
    const sy = Math.min(Math.max(y, 0), src.h - 1);
    return H(H(src.data[(sy * src.w + sx) * 4 + c] + 1) * 0.5);
  };
  const rows: number[] = [];
  for (let dy = -1; dy <= 2; dy++) {
    rows.push(halfDot([n(ix - 1, iy + dy), n(ix, iy + dy), n(ix + 1, iy + dy), n(ix + 2, iy + dy)], coefs[ox % 2]));
  }
  return halfDot(rows, coefs[oy % 2]);
}

const LOG2E_F32 = F(Math.LOG2E);

/** T = H(0.5*(t+1)), t = (ep-em)/(ep+em), ep = 2^(z log2e), em = 2^(-z log2e), in float32. */
export function learnedT(z: number): number {
  const zl = F(F(z) * LOG2E_F32);
  const ep = F(2 ** zl);
  const em = F(2 ** -zl);
  const t = F(F(ep - em) / F(ep + em));
  return H(F(0.5 * F(t + 1)));
}

/** Mathematical form T = H((1 + tanh z) / 2) evaluated in float64. */
export function learnedTanhT(z: number): number {
  return H((1 + Math.tanh(z)) / 2);
}

/** Composition with residual gain g = 2*sharpness: exact half form when g == 1. */
export function compose(b: number, t: number, gain = 1): number {
  if (gain === 1) return H(H(b + t) - 0.5);
  return H(F(b + F(gain * F(t - 0.5))));
}

/**
 * SR head on level-0 features d (K0 channels) and the half4 source I (W x H).
 * Returns planar RGB halves of 2W x 2H.
 */
export function srHead(
  d: Tensor,
  src: Tensor,
  w3: ArrayLike<number>,
  b3: ArrayLike<number>,
  w1: ArrayLike<number>,
  b1: ArrayLike<number>,
  gain = 1,
): Float32Array {
  const hidden = conv3x3(d, w3, b3, 64, 'elu');
  const z = conv1x1(hidden, w1, b1, 48);
  return headCompose(z, src, gain);
}

/** Pixel shuffle 4 of the 48-channel z with baseline composition, cropped to 2W x 2H. */
export function headCompose(z: Tensor, src: Tensor, gain = 1): Float32Array {
  const ow = 2 * src.w;
  const oh = 2 * src.h;
  const out = new Float32Array(3 * ow * oh);
  for (let oy = 0; oy < oh; oy++) {
    for (let ox = 0; ox < ow; ox++) {
      const fy = Math.floor(oy / 4);
      const fx = Math.floor(ox / 4);
      for (let c = 0; c < 3; c++) {
        const k = 16 * c + 4 * (oy % 4) + (ox % 4);
        const t = learnedT(z.data[(fy * z.w + fx) * 48 + k]);
        out[c * ow * oh + oy * ow + ox] = compose(baseline(src, ox, oy, c), t, gain);
      }
    }
  }
  return out;
}

/** Catmull-Rom (a = -0.5) weights for fraction t (a half), by the PTX half-FMA chains. */
export function cubicWeights(t: number): [number, number, number, number] {
  const r19 = fmaH(-0.5, t, 1);
  const r24 = fmaH(1.5, t, -2.5);
  const r29 = fmaH(-1.5, t, 2);
  const r34 = fmaH(0.5, t, -0.5);
  const r39 = fmaH(r19, t, -0.5);
  const r44 = fmaH(r24, t, 0);
  const r49 = fmaH(r29, t, 0.5);
  const r54 = fmaH(r34, t, 0);
  return [fmaH(r39, t, 0), fmaH(r44, t, 1), fmaH(r49, t, 0), fmaH(r54, t, 0)];
}

/**
 * dlpp_ResampleAndComposeFP16E on a planar source (3 planes of sw x sh halves), output
 * RGBA8 bytes (alpha 255) of ow x oh.
 */
export function resamplePlanar(
  src: Float32Array,
  sw: number,
  sh: number,
  ow: number,
  oh: number,
  scaleX: number,
  scaleY: number,
): Uint8Array {
  const out = new Uint8Array(4 * ow * oh);
  const plane = sw * sh;
  for (let v = 0; v < oh; v++) {
    const cy = fmaF(F(v + 0.5), scaleY, -0.5);
    const fly = Math.floor(cy);
    const wy = cubicWeights(H(cy - fly));
    for (let u = 0; u < ow; u++) {
      const cx = fmaF(F(u + 0.5), scaleX, -0.5);
      const flx = Math.floor(cx);
      const wx = cubicWeights(H(cx - flx));
      const acc = [0, 0, 0];
      for (let j = 0; j < 4; j++) {
        const y = Math.min(Math.max(fly - 1 + j, 0), sh - 1);
        for (let i = 0; i < 4; i++) {
          const x = Math.min(Math.max(flx - 1 + i, 0), sw - 1);
          const w = H(wx[i] * wy[j]);
          for (let c = 0; c < 3; c++) acc[c] = fmaF(w, src[c * plane + y * sw + x], acc[c]);
        }
      }
      for (let c = 0; c < 3; c++) {
        const s = F(Math.min(Math.max(acc[c], 0), 1) * 255);
        out[4 * (v * ow + u) + c] = Math.floor(s + 0.5);
      }
      out[4 * (v * ow + u) + 3] = 255;
    }
  }
  return out;
}

/** Quality 0 source: layout-1 preprocess H(v/255) planes of an RGBA8 image. */
export function planarFromRgba8(rgba: Uint8Array, w: number, h: number): Float32Array {
  const out = new Float32Array(3 * w * h);
  for (let p = 0; p < w * h; p++) {
    for (let c = 0; c < 3; c++) out[c * w * h + p] = H(F(rgba[4 * p + c] / 255));
  }
  return out;
}

/** postProcessE byte conversion of a planar half image (clamp, *255, truncate). */
export function planarToBytesTruncate(planar: Float32Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(4 * w * h);
  for (let p = 0; p < w * h; p++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.min(Math.max(planar[c * w * h + p], 0), 1);
      out[4 * p + c] = Math.trunc(F(v * 255));
    }
    out[4 * p + 3] = 255;
  }
  return out;
}
