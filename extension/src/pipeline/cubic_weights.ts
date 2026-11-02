/**
 * Exact Catmull-Rom (a = -0.5) tap weights of dlpp_ResampleAndComposeFP16E for every half
 * fraction t in [0, 1] (docs/IMPLEMENTATION_NOTES.md 3.11, semantics/resample.ptx).
 *
 * The PTX evaluates the weight polynomials with `fma.rn.f16x2`: every step is ONE rounding of
 * the exact a*b + c to half. A WGSL kernel cannot express that: an f32 `fma` followed by a
 * rounding to half rounds twice, and the two results differ for t = 0.022247314453125
 * (w1 = 0.9990234375 instead of 0.99853515625), a fraction that real sizes reach (for example
 * 1280 -> 854 wide, output column 194). The weights depend on t alone, and t is a half, so the
 * kernel looks them up in this table instead: 15,361 entries indexed by the half bit pattern of
 * t (0x0000 = 0 .. 0x3C00 = 1.0; t rounds up to 1.0 when c - floor(c) is just below 1).
 *
 * Evaluation here is exact single rounding: every operand is a half of magnitude < 4, so a
 * product a*b has at most 22 significant bits with an ulp of at least 2^-48, and a*b + c spans
 * at most 2^-48 .. 2^3, i.e. fewer than 53 bits. The float64 expression a*b + c is therefore
 * exact and {@link roundToHalf} applies the only rounding, as the fused half FMA does.
 */

/** Number of table entries: half bit patterns 0x0000 .. 0x3C00 (t = 0 .. 1). */
export const CUBIC_TABLE_ENTRIES = 0x3c00 + 1;
/** Table bytes: 4 weights x 2 bytes per entry (two u32 words of packed halves). */
export const CUBIC_TABLE_BYTES = CUBIC_TABLE_ENTRIES * 8;

const HALF_MAX = 65504;

/**
 * IEEE binary16 round-to-nearest-even of a float64 value.
 * Overflow gives infinity, as the hardware conversion does.
 */
export function roundToHalf(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const a = Math.abs(x);
  // ulp of the half binade containing a (subnormals: 2^-24)
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  const ulp = 2 ** Math.max(e - 10, -24);
  const q = a / ulp; // exact: division by a power of two
  const f = Math.floor(q);
  const d = q - f; // exact: q and f share the binade
  const n = d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
  const r = n * ulp;
  if (r > HALF_MAX) return x > 0 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  return x < 0 ? -r : r;
}

/** Value of a binary16 bit pattern (finite patterns only). */
export function halfFromBits(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0x1f) throw new RangeError(`0x${bits.toString(16)} is not a finite half`);
  return sign * (exp === 0 ? mant * 2 ** -24 : (1 + mant / 1024) * 2 ** (exp - 15));
}

/** Bit pattern of a value that is exactly representable as a finite half. */
export function halfToBits(v: number): number {
  if (roundToHalf(v) !== v || !Number.isFinite(v)) throw new RangeError(`${v} is not a finite half`);
  const sign = v < 0 || Object.is(v, -0) ? 0x8000 : 0;
  const a = Math.abs(v);
  if (a < 2 ** -14) return sign | (a * 2 ** 24);
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  return sign | ((e + 15) << 10) | (a / 2 ** e - 1) * 1024;
}

/** fma.rn.f16: one rounding of the exact a*b + c (see the module comment for exactness). */
function fmaHalf(a: number, b: number, c: number): number {
  return roundToHalf(a * b + c);
}

/** The four tap weights [P-1, P0, P1, P2] for the half fraction t, literal PTX FMA order. */
export function cubicWeightsForFraction(t: number): [number, number, number, number] {
  const r19 = fmaHalf(-0.5, t, 1);
  const r24 = fmaHalf(1.5, t, -2.5);
  const r29 = fmaHalf(-1.5, t, 2);
  const r34 = fmaHalf(0.5, t, -0.5);
  const r39 = fmaHalf(r19, t, -0.5);
  const r44 = fmaHalf(r24, t, 0);
  const r49 = fmaHalf(r29, t, 0.5);
  const r54 = fmaHalf(r34, t, 0);
  return [fmaHalf(r39, t, 0), fmaHalf(r44, t, 1), fmaHalf(r49, t, 0), fmaHalf(r54, t, 0)];
}

let table: Uint32Array | null = null;

/**
 * The GPU table: entry i (half bit pattern of t) is two u32 words, word 0 = P-1 | P0 << 16 and
 * word 1 = P1 | P2 << 16 (half bit patterns; unpack2x16float in WGSL). Built once per script.
 */
export function cubicWeightTable(): Uint32Array {
  if (table) return table;
  const out = new Uint32Array(2 * CUBIC_TABLE_ENTRIES);
  for (let i = 0; i < CUBIC_TABLE_ENTRIES; i++) {
    const w = cubicWeightsForFraction(halfFromBits(i)).map(halfToBits);
    out[2 * i] = ((w[0] as number) | ((w[1] as number) << 16)) >>> 0;
    out[2 * i + 1] = ((w[2] as number) | ((w[3] as number) << 16)) >>> 0;
  }
  table = out;
  return out;
}
