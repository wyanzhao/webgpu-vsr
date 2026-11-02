/**
 * Exact resample weights (NUM-1): the table that replaces the in-shader f32-FMA emulation of
 * the PTX `fma.rn.f16x2` chains, and the production resampler on Dawn at a size whose fraction
 * the double-rounded emulation got wrong.
 */
import { describe, expect, it } from 'vitest';
import {
  CUBIC_TABLE_BYTES,
  CUBIC_TABLE_ENTRIES,
  cubicWeightTable,
  cubicWeightsForFraction,
  halfFromBits,
  halfToBits,
  roundToHalf,
} from '../../src/pipeline/cubic_weights';
import { BicubicPipeline } from '../../src/pipeline/resample_pipeline';
import { resampleScale } from '../../src/pipeline/geometry';
import { cubicWeights, planarFromRgba8, resamplePlanar } from '../reference/ops';
import { halfFromBits as refHalfFromBits } from '../reference/half';
import { lcg, makeRgba8 } from '../reference/patterns';
import { nodeGpu, nodeGpuContext, readBack } from '../harness/node_gpu';

const F = Math.fround;

/** The former WGSL evaluation: f32 FMA (one f32 rounding), then a rounding to half. */
function doubleRoundedWeights(t: number): number[] {
  const fma = (a: number, b: number, c: number) => Math.f16round(F(a * b + c));
  const r19 = fma(-0.5, t, 1);
  const r24 = fma(1.5, t, -2.5);
  const r29 = fma(-1.5, t, 2);
  const r34 = fma(0.5, t, -0.5);
  const r39 = fma(r19, t, -0.5);
  const r44 = fma(r24, t, 0);
  const r49 = fma(r29, t, 0.5);
  const r54 = fma(r34, t, 0);
  return [fma(r39, t, 0), fma(r44, t, 1), fma(r49, t, 0), fma(r54, t, 0)];
}

describe('half conversion helpers', () => {
  it('roundToHalf equals IEEE binary16 round-to-nearest-even (Math.f16round)', () => {
    const rnd = lcg(7);
    for (let i = 0; i < 200000; i++) {
      const x = (rnd() - 0.5) * 2 ** Math.floor(rnd() * 44 - 28);
      expect(roundToHalf(x)).toBe(Math.f16round(x));
    }
    // every half, every midpoint between neighbours and the values just around it
    for (let b = 0; b < 0x7bff; b++) {
      const lo = refHalfFromBits(b);
      const hi = refHalfFromBits(b + 1);
      const mid = (lo + hi) / 2;
      for (const x of [lo, mid, mid * (1 - 2 ** -40), mid * (1 + 2 ** -40), -mid]) expect(roundToHalf(x)).toBe(Math.f16round(x));
    }
    for (const x of [65504, 65519.99, 65520, 1e6, -70000, 0, -0]) expect(Object.is(roundToHalf(x), Math.f16round(x))).toBe(true);
  });

  it('halfFromBits / halfToBits round-trip every finite half', () => {
    for (let b = 0; b < 0x10000; b++) {
      if (((b >> 10) & 0x1f) === 0x1f) continue;
      const v = halfFromBits(b);
      expect(v).toBe(refHalfFromBits(b));
      expect(halfToBits(v)).toBe(b);
    }
    expect(() => halfToBits(1 + 2 ** -12)).toThrow(RangeError);
  });
});

describe('cubic weight table', () => {
  it('holds the single-rounding PTX weights for every half fraction 0 .. 1', () => {
    const table = cubicWeightTable();
    expect(CUBIC_TABLE_ENTRIES).toBe(15361);
    expect(table.byteLength).toBe(CUBIC_TABLE_BYTES);
    for (let i = 0; i < CUBIC_TABLE_ENTRIES; i++) {
      const t = halfFromBits(i);
      const want = cubicWeights(t);
      expect(cubicWeightsForFraction(t)).toEqual(want);
      const got = [table[2 * i]! & 0xffff, table[2 * i]! >>> 16, table[2 * i + 1]! & 0xffff, table[2 * i + 1]! >>> 16].map(halfFromBits);
      expect(got).toEqual(want);
    }
    expect(cubicWeightsForFraction(0)).toEqual([0, 1, 0, 0]);
    expect(cubicWeightsForFraction(1)).toEqual([0, 0, 1, 0]);
  });

  it('differs from the double-rounded f32 emulation at exactly one fraction', () => {
    const differing: number[] = [];
    for (let i = 0; i < CUBIC_TABLE_ENTRIES; i++) {
      const t = halfFromBits(i);
      if (doubleRoundedWeights(t).some((w, k) => w !== cubicWeightsForFraction(t)[k])) differing.push(t);
    }
    expect(differing).toEqual([0.022247314453125]);
    expect(cubicWeightsForFraction(0.022247314453125)[1]).toBe(0.99853515625);
    expect(doubleRoundedWeights(0.022247314453125)[1]).toBe(0.9990234375);
    // the fraction is reached in practice: 1280 -> 854 wide, output column 194
    const s = resampleScale(1280, 720, 854, 480).x;
    const c = F(F(194 + 0.5) * s - 0.5);
    expect(Math.f16round(c - Math.floor(c))).toBe(0.022247314453125);
  });
});

describe('Quality 0 resampler on Dawn (exact weights)', () => {
  it('is byte-exact against the reference at 1280x720 -> 854x480 (the double-rounded fraction)', async () => {
    const ctx = await nodeGpuContext();
    const [w, h, ow, oh] = [1280, 720, 854, 480];
    const rgba = makeRgba8(w, h, 1234);
    const pipe = new BicubicPipeline({ storageFormat: ctx.storageFormat, source: 'rgba8', output: 'buffer' });
    await pipe.initialize(ctx.device, w, h, ow, oh);
    const { device } = ctx;
    const src = device.createBuffer({ size: rgba.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(src, 0, rgba);
    const out = device.createBuffer({ size: 4 * ow * oh, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 4 * ow * oh, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    pipe.encodeFrame(enc, { kind: 'rgba8', buffer: src }, { kind: 'buffer', buffer: out });
    enc.copyBufferToBuffer(out, 0, rb, 0, 4 * ow * oh);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const got = new Uint8Array(rb.getMappedRange().slice(0));
    rb.unmap();
    const sc = resampleScale(w, h, ow, oh);
    const expected = resamplePlanar(planarFromRgba8(rgba, w, h), w, h, ow, oh, sc.x, sc.y);
    let mismatches = 0;
    for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) mismatches++;
    expect(mismatches).toBe(0);
    for (const b of [src, out, rb]) b.destroy();
    pipe.destroy();
  });
});

describe('Quality 0 without shader-f16 on Dawn', () => {
  it('compiles texture and rgba8 pipelines and matches the f16 device byte-for-byte', async () => {
    const ctx = await nodeGpuContext();
    expect(ctx.hasF16).toBe(true);
    const adapter = await nodeGpu().requestAdapter();
    expect(adapter).not.toBeNull();
    const plain = await adapter!.requestDevice({ requiredFeatures: [] });
    expect(plain.features.has('shader-f16')).toBe(false);
    const [w, h, ow, oh] = [67, 43, 101, 65];
    const rgba = makeRgba8(w, h, 1234);
    const texture = new BicubicPipeline({ storageFormat: 'rgba8unorm', source: 'texture' });
    const render = async (device: GPUDevice): Promise<Uint8Array> => {
      const pipe = new BicubicPipeline({ storageFormat: 'rgba8unorm', source: 'rgba8', output: 'buffer' });
      const src = device.createBuffer({ size: rgba.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const out = device.createBuffer({ size: 4 * ow * oh, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      try {
        await pipe.initialize(device, w, h, ow, oh);
        device.queue.writeBuffer(src, 0, rgba);
        const enc = device.createCommandEncoder();
        pipe.encodeFrame(enc, { kind: 'rgba8', buffer: src }, { kind: 'buffer', buffer: out });
        device.queue.submit([enc.finish()]);
        return await readBack(device, out);
      } finally {
        pipe.destroy();
        src.destroy();
        out.destroy();
      }
    };
    try {
      await texture.initialize(plain, w, h, ow, oh);
      const got = await render(plain);
      const expected = await render(ctx.device);
      expect(got).toEqual(expected);
      const sc = resampleScale(w, h, ow, oh);
      expect(got).toEqual(resamplePlanar(planarFromRgba8(rgba, w, h), w, h, ow, oh, sc.x, sc.y));
    } finally {
      texture.destroy();
      plain.destroy();
    }
  });
});
