/**
 * IEEE binary16 helpers for the reference implementation.
 *
 * H(x) = Math.f16round(x) (round-to-nearest-even). A half FMA fma(a, b, c) of halves is
 * Math.f16round(a*b + c): the product of two halves has at most 22 significant bits and
 * is exact in float64, and the sum is exact whenever the operands span at most 53 bits
 * (always the case for the value ranges used here), so the single final rounding matches a
 * fused half FMA.
 */

export const H = Math.f16round;
export const F = Math.fround;

export function fmaH(a: number, b: number, c: number): number {
  return Math.f16round(a * b + c);
}

/** float32 FMA (one rounding) for operands whose exact result fits in float64. */
export function fmaF(a: number, b: number, c: number): number {
  return Math.fround(a * b + c);
}

const f16 = new Float16Array(1);
const u16 = new Uint16Array(f16.buffer);

export function halfBits(v: number): number {
  f16[0] = v;
  return u16[0];
}

export function halfFromBits(bits: number): number {
  u16[0] = bits & 0xffff;
  return f16[0];
}

/** Decode a little-endian FP16 byte buffer into numbers. */
export function decodeHalves(bytes: Uint8Array): Float32Array {
  const n = bytes.byteLength / 2;
  const view = new Float16Array(bytes.buffer, bytes.byteOffset, n);
  return Float32Array.from(view);
}

/** Encode numbers (already representable as halves) to FP16 bytes. */
export function encodeHalves(values: ArrayLike<number>): Uint8Array {
  const out = new Float16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i];
  return new Uint8Array(out.buffer);
}

/** ELU (alpha = 1). */
export function elu(v: number): number {
  return v >= 0 ? v : Math.exp(v) - 1;
}
