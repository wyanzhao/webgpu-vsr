/**
 * GPU parity suite: the production WGSL kernels on the real WebGPU (Metal) device versus the
 * TypeScript reference (tests/reference) and the research oracle fixtures (tests/fixtures).
 * Requires `npm run build:harness`.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gpuTest as test, expect, TEST_ORIGIN, FIXTURES_DIR } from './fixtures';
import { bytesFromHalves, call, diffStats, halvesFromBytes, loadBundle, openHarness, refWeights } from '../harness/client';
import { fromBase64, makeRgba8, toBase64 } from '../reference/patterns';
import {
  addHalf,
  conv1x1,
  conv3x3,
  fold,
  padReplicate,
  planarFromRgba8,
  planarToBytesTruncate,
  pool2x2,
  preprocessRgba8,
  resamplePlanar,
  srHead,
  tensor,
  type Tensor,
} from '../reference/ops';
import { runNetwork } from '../reference/network';
import { H } from '../reference/half';
import { alignedSize, levelSizes, resampleScale } from '../../src/pipeline/geometry';
import { BASELINE_COEFFICIENTS, type ModelId } from '../../src/pipeline/types';
import { MODEL_CHANNELS, type LayerName } from '../../src/weights/types';

const MODELS: ModelId[] = ['low', 'medium'];

/**
 * Whole-network tolerance derived from the FP16 policy (docs/IMPLEMENTATION_NOTES.md 3):
 * activations are rounded to half after every layer; the GPU accumulates in f32 and uses
 * the Metal exp(), the reference sums exactly in float64. Either difference can flip one
 * rounding of an intermediate half (relative 2^-11); flips propagate through the decoder
 * and the bounded tanh head. Outputs are halves in about [-0.5, 1.5] (ulp <= 2^-10 there).
 *   max |d| <= 4 ulp(1) = 2^-8,  p99.9 <= 2 ulp(1) = 2^-9,  mean |d| <= 2^-12.
 */
const TOL = { maxAbs: 2 ** -8, p999: 2 ** -9, meanAbs: 2 ** -12 };

function halfImage(rgba: Uint8Array, w: number, h: number): Tensor {
  return preprocessRgba8(rgba, w, h);
}

/**
 * Per-layer bounds (FP16 numerics policy, docs/IMPLEMENTATION_NOTES.md 3). Each layer is
 * evaluated by the reference on the GPU's OWN input of that layer, so differences cannot
 * accumulate: what remains is the f32 (GPU) vs exact (reference) accumulation and the Metal
 * exp() in ELU, which flip a rounding to half only when the exact result lies within a few f32
 * ulps of a half rounding boundary. A flipped half is off by one half ulp of the value that was
 * rounded: the result itself, or e and p before the projection merge H(e + p). Where the terms
 * of a sum cancel (results far smaller than the terms) the f32 accumulation error is absolute,
 * not relative to the result: measured up to 2.7e-6 (2^-18.5), allowed 2^-16.
 * Measured on the M5 Max (160x96, low / medium): at most 0.21 % of a layer's halves differ
 * (rb0A), most layers below 0.1 %. Contract violations, measured by mutating the shaders
 * (unrounded projection p, unrounded e, half-precision partial sums, exp() rounded to half),
 * make 11 - 63 % of the affected layers' halves differ, while the whole-network test (e) still
 * passes. The SR head on the GPU's own d3 differs in 21 of 184,320 outputs (0.011 %); without
 * the z rounding 521 / 919 differ (0.28 / 0.50 %), with half-precision hidden sums 6,242 /
 * 4,279 (3.4 / 2.3 %).
 */
const LAYER_TOL = { mismatchFraction: 0.01 };
const HEAD_TOL = { mismatchFraction: 0.001, maxAbs: 2 ** -10 };
const F32_SUM_SLACK = 2 ** -16;

/** One half ulp at the magnitude of x (2^-24 in the subnormal range). */
function halfUlp(x: number): number {
  const a = Math.abs(x);
  return a < 2 ** -14 ? 2 ** -24 : 2 ** (Math.floor(Math.log2(a)) - 10);
}

interface LayerDiff {
  count: number;
  mismatches: number;
  /** Values off by more than the flip bound of their rounding (see LAYER_TOL). */
  outOfBound: number;
  maxAbs: number;
}

/**
 * Compares channels [offset, offset + ref.c) of an interleaved GPU tensor with the reference.
 * `operands` (the merge inputs e and p of a projection layer) widen the per-value bound.
 */
function layerDiff(gpu: ArrayLike<number>, stride: number, offset: number, ref: Tensor, operands: Tensor[] = []): LayerDiff {
  let mismatches = 0;
  let outOfBound = 0;
  let maxAbs = 0;
  for (let p = 0; p < ref.w * ref.h; p++) {
    for (let c = 0; c < ref.c; c++) {
      const i = p * ref.c + c;
      const g = gpu[p * stride + offset + c] as number;
      const r = ref.data[i] as number;
      if (Object.is(g, r)) continue;
      mismatches++;
      const d = Math.abs(g - r);
      maxAbs = Math.max(maxAbs, d);
      let bound = halfUlp(r) + F32_SUM_SLACK;
      for (const t of operands) bound += halfUlp(t.data[i] as number);
      if (!(d <= bound)) outOfBound++;
    }
  }
  return { count: ref.w * ref.h * ref.c, mismatches, outOfBound, maxAbs };
}

/** Channels [offset, offset + c) of an interleaved GPU buffer as a dense tensor. */
function gather(gpu: ArrayLike<number>, w: number, h: number, stride: number, offset: number, c: number): Tensor {
  const t = tensor(w, h, c);
  for (let p = 0; p < w * h; p++) for (let k = 0; k < c; k++) t.data[p * c + k] = gpu[p * stride + offset + k] as number;
  return t;
}

test.describe('GPU parity (WebGPU / Metal)', () => {
  test.beforeEach(async ({ gpuPage }) => {
    await openHarness(gpuPage, TEST_ORIGIN);
  });

  test('device exposes the features the pipeline relies on', async ({ gpuPage }) => {
    const info = await call(gpuPage, 'info');
    console.log(`[gpu] ${info.vendor} ${info.architecture} ${info.description} storage=${info.storageFormat}`);
    expect(info.hasF16).toBe(true);
    expect(info.hasTimestamps).toBe(true);
    expect(info.storageFormat).toBe('bgra8unorm');
    expect(info.maxComputeWorkgroupStorageSize).toBeGreaterThanOrEqual(32768);
  });

  test('(a) preprocess + edge padding + fold is bit-exact for half4 and RGBA8 sources', async ({ gpuPage }) => {
    for (const model of MODELS) {
      for (const [w, h] of [
        [64, 64],
        [100, 60],
        [37, 19],
      ]) {
        const rgba = makeRgba8(w, h, 11 + w);
        const I = halfImage(rgba, w, h);
        const a = alignedSize(w, h);
        const expected = fold(padReplicate(I, a.width, a.height));
        const stride = 2 * MODEL_CHANNELS[model][0];
        for (const source of ['half4', 'rgba8'] as const) {
          const data = source === 'half4' ? toBase64(bytesFromHalves(I.data)) : toBase64(rgba);
          const r = await call(gpuPage, 'runNetwork', { model, width: w, height: h, source, data, stopAfter: 'fold', read: ['cat0'] });
          const cat0 = halvesFromBytes(fromBase64(r.buffers.cat0));
          let mismatches = 0;
          for (let p = 0; p < expected.w * expected.h; p++) {
            for (let c = 0; c < 16; c++) {
              if (!Object.is(cat0[p * stride + c], expected.data[p * 16 + c])) mismatches++;
            }
          }
          expect(mismatches, `${model} ${w}x${h} ${source}`).toBe(0);
        }
      }
    }
  });

  test('(b) ordered half-FMA 2x2 pool is bit-exact (and order sensitive)', async ({ gpuPage }) => {
    let naiveDiffers = 0;
    let compared = 0;
    for (const model of MODELS) {
      const K = MODEL_CHANNELS[model];
      const [w, h] = [160, 96];
      const rgba = makeRgba8(w, h, 5);
      const I = halfImage(rgba, w, h);
      const lv = levelSizes(w, h);
      for (const level of [0, 1, 2, 3]) {
        const stop = `rb${level}B`;
        const read = level < 3 ? [`cat${level}`, `cat${level + 1}`] : ['cat3', 'x4b4'];
        const r = await call(gpuPage, 'runNetwork', {
          model,
          width: w,
          height: h,
          source: 'half4',
          data: toBase64(bytesFromHalves(I.data)),
          stopAfter: stop,
          read: read as ('cat0' | 'cat1' | 'cat2' | 'cat3' | 'x4b4')[],
        });
        const full = halvesFromBytes(fromBase64(r.buffers[read[0]]));
        const pooled = halvesFromBytes(fromBase64(r.buffers[read[1]]));
        const k = K[level];
        const stride = 2 * k;
        const skip = tensor(lv[level].width, lv[level].height, k);
        for (let p = 0; p < skip.w * skip.h; p++) for (let c = 0; c < k; c++) skip.data[p * k + c] = full[p * stride + k + c];
        const expected = pool2x2(skip);
        const pstride = level < 3 ? 2 * K[level + 1] : K[3] + K[4];
        let mismatches = 0;
        for (let p = 0; p < expected.w * expected.h; p++) {
          const y = Math.floor(p / expected.w);
          const x = p % expected.w;
          for (let c = 0; c < k; c++) {
            const got = pooled[p * pstride + c];
            const want = expected.data[p * k + c];
            if (!Object.is(got, want)) mismatches++;
            const at = (dy: number, dx: number) => skip.data[((2 * y + dy) * skip.w + 2 * x + dx) * k + c];
            const naive = H((at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1)) / 4);
            if (naive !== want) naiveDiffers++;
            compared++;
          }
        }
        expect(mismatches, `${model} level ${level}`).toBe(0);
      }
    }
    console.log(`[pool] ${compared} pooled halves exact; naive f32 average differs in ${naiveDiffers}`);
    expect(naiveDiffers).toBeGreaterThan(0);
  });

  test('(c) decoder up-convolution source mapping is exact on impulses', async ({ gpuPage }) => {
    const lowW = 9;
    const lowH = 7;
    const cin = 16;
    const k = 16;
    const src = tensor(lowW, lowH, cin);
    const impulses: [number, number, number][] = [
      [0, 0, 0],
      [lowW - 1, lowH - 1, 5],
      [4, 3, 9],
      [1, 5, 15],
      [7, 0, 2],
    ];
    for (const [x, y, c] of impulses) src.data[(y * lowW + x) * cin + c] = 1;
    // distinct positive weights per (k, r, s, c): outputs stay positive (ELU = identity) and
    // every output is a sum of at most a few exactly representable terms
    const weights = new Float32Array(k * 9 * cin);
    for (let kk = 0; kk < k; kk++)
      for (let t = 0; t < 9; t++) for (let c = 0; c < cin; c++) weights[(kk * 9 + t) * cin + c] = H((1 + ((kk * 7 + t * 3 + c) % 29)) / 64);
    const bias = new Float32Array(k);
    const expected = conv3x3(src, weights, bias, k, 'elu', true);
    const outW = 2 * lowW;
    const outH = 2 * lowH;
    const wbytes = bytesFromHalves([...weights, ...bias]);
    const r = await call(gpuPage, 'runConv', {
      spec: { kind: 'conv', cin, inStride: cin, inOffset: 0, k, kb: 8, tx: 16, ty: 16, cch: 16, outStride: k, outOffset: 0, upsample: true },
      params: { inW: lowW, inH: lowH, outW, outH, w3: 0, b3: (k * 9 * cin) / 4 },
      groups: [Math.ceil(outW / 16), Math.ceil(outH / 16)],
      weights: toBase64(wbytes),
      src: toBase64(bytesFromHalves(src.data)),
      dst: toBase64(new Uint8Array(outW * outH * k * 2)),
    });
    const got = halvesFromBytes(fromBase64(r.dst));
    let mismatches = 0;
    let nonzero = 0;
    for (let i = 0; i < expected.data.length; i++) {
      if (got[i] !== expected.data[i]) mismatches++;
      if (expected.data[i] !== 0) nonzero++;
    }
    expect(mismatches).toBe(0);
    expect(nonzero).toBeGreaterThan(100);
    // sensitivity: a half-pixel-centred mapping floor((out + tap) / 2) gives different results
    const shifted = tensor(outW, outH, k);
    for (let oy = 0; oy < outH; oy++)
      for (let ox = 0; ox < outW; ox++)
        for (let kk = 0; kk < k; kk++) {
          let a = 0;
          for (let r0 = 0; r0 < 3; r0++)
            for (let s0 = 0; s0 < 3; s0++) {
              const sy = (oy + r0) >> 1;
              const sx = (ox + s0) >> 1;
              if (sy >= lowH || sx >= lowW) continue;
              for (let c = 0; c < cin; c++) a += src.data[(sy * lowW + sx) * cin + c] * weights[(kk * 9 + r0 * 3 + s0) * cin + c];
            }
          shifted.data[(oy * outW + ox) * k + kk] = H(a);
        }
    expect(diffStats(got, shifted.data).maxAbs).toBeGreaterThan(0.01);
  });

  test('(d) folded SR head reproduces the 147,456-value research oracle exactly', async ({ gpuPage }) => {
    const manifest = JSON.parse(readFileSync(join(FIXTURES_DIR, 'head', 'manifest.json'), 'utf8'));
    let checked = 0;
    for (const model of MODELS) {
      const m = manifest.models[model];
      expect(m.feature_channels).toBe(MODEL_CHANNELS[model][0]);
      // the fixture constant bank coefficients equal the table baked into the shader
      expect(m.baseline_coefficients).toEqual(BASELINE_COEFFICIENTS.map((r) => [...r]));
      for (const c of m.cases) {
        const file = (kind: string) => readFileSync(join(FIXTURES_DIR, 'head', model, `${c.name}.${kind}.bin`));
        const r = await call(gpuPage, 'runHeadFixture', {
          model,
          source: toBase64(new Uint8Array(file('source'))),
          bias1: toBase64(new Uint8Array(file('bias1'))),
        });
        expect(r.direct).toBe(true);
        const got = new Uint16Array(fromBase64(r.output).buffer);
        const exp = new Uint16Array(new Uint8Array(file('expected')).buffer);
        expect(got.length).toBe(exp.length);
        let mismatches = 0;
        let first = '';
        for (let i = 0; i < exp.length; i++) {
          if (got[i] !== exp[i]) {
            if (!first) first = `index ${i}: got 0x${got[i].toString(16)} want 0x${exp[i].toString(16)}`;
            mismatches++;
          }
        }
        expect(mismatches, `${model} ${c.name} ${first}`).toBe(0);
        checked += exp.length;
      }
    }
    expect(checked).toBe(147456);
    console.log(`[head] ${checked} half values bit-exact`);
  });

  test('(e) full low/medium network matches the reference within the FP16 tolerance', async ({ gpuPage }) => {
    test.setTimeout(600000);
    const sizes: [number, number][] = [
      [64, 64],
      [100, 60],
      [160, 96],
      [256, 144],
    ];
    for (const model of MODELS) {
      const bundle = await loadBundle(model);
      const weights = refWeights(bundle);
      for (const [w, h] of sizes) {
        const rgba = makeRgba8(w, h, w * 31 + h);
        const I = halfImage(rgba, w, h);
        const r = await call(gpuPage, 'runNetwork', { model, width: w, height: h, source: 'half4', data: toBase64(bytesFromHalves(I.data)) });
        const gpu = halvesFromBytes(fromBase64(r.output));
        const ref = runNetwork(model, weights, I);
        const s = diffStats(gpu, ref.output);
        console.log(
          `[net] ${model} ${w}x${h}: max ${s.maxAbs.toExponential(3)} mean ${s.meanAbs.toExponential(3)} p99.9 ${s.p999.toExponential(3)} exact ${(100 * s.exactFraction).toFixed(2)}%`,
        );
        expect(s.maxAbs).toBeLessThanOrEqual(TOL.maxAbs);
        expect(s.p999).toBeLessThanOrEqual(TOL.p999);
        expect(s.meanAbs).toBeLessThanOrEqual(TOL.meanAbs);
        if (w === 100) {
          for (const control of ['swap-concat', 'no-projection'] as const) {
            const bad = runNetwork(model, weights, I, { negativeControl: control });
            const n = diffStats(gpu, bad.output);
            console.log(`[net] ${model} ${w}x${h} negative control ${control}: max ${n.maxAbs.toFixed(4)} mean ${n.meanAbs.toExponential(3)}`);
            expect(n.meanAbs).toBeGreaterThan(20 * TOL.meanAbs);
            expect(n.maxAbs).toBeGreaterThan(10 * TOL.maxAbs);
          }
        }
      }
    }
  });

  test('(e2) residual gain (sharpness != 0.5) and the VSR resample stage', async ({ gpuPage }) => {
    test.setTimeout(300000);
    const model: ModelId = 'low';
    const bundle = await loadBundle(model);
    const [w, h] = [100, 60];
    const rgba = makeRgba8(w, h, 77);
    const I = halfImage(rgba, w, h);
    const data = toBase64(bytesFromHalves(I.data));
    for (const sharpness of [0.2, 0.9]) {
      const r = await call(gpuPage, 'runNetwork', { model, width: w, height: h, source: 'half4', data, sharpness });
      const ref = runNetwork(model, refWeights(bundle), I, { gain: 2 * sharpness });
      const s = diffStats(halvesFromBytes(fromBase64(r.output)), ref.output);
      expect(s.maxAbs).toBeLessThanOrEqual(TOL.maxAbs);
      expect(s.meanAbs).toBeLessThanOrEqual(TOL.meanAbs);
    }
    // resample internal (2W x 2H) -> 150x90: the GPU resample of the GPU internal image is exact
    const [ow, oh] = [150, 90];
    const r = await call(gpuPage, 'runNetwork', { model, width: w, height: h, outWidth: ow, outHeight: oh, source: 'half4', data, read: ['internal'] });
    expect(r.direct).toBe(false);
    const internal = halvesFromBytes(fromBase64(r.buffers.internal)).subarray(0, 3 * 4 * w * h);
    const sc = resampleScale(2 * w, 2 * h, ow, oh);
    const expected = resamplePlanar(internal, 2 * w, 2 * h, ow, oh, sc.x, sc.y);
    const got = fromBase64(r.output);
    let mismatches = 0;
    for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) mismatches++;
    expect(mismatches).toBe(0);
    // and the internal image itself is the network output
    const ref = runNetwork(model, refWeights(bundle), I);
    expect(diffStats(internal, ref.output).maxAbs).toBeLessThanOrEqual(TOL.maxAbs);
  });

  test('(f) production path: VideoFrame -> importExternalTexture -> canvas equals the buffer variant', async ({ gpuPage }) => {
    test.setTimeout(300000);
    const cases: { pipeline: ModelId | 'q0'; w: number; h: number; ow: number; oh: number }[] = [
      { pipeline: 'low', w: 100, h: 60, ow: 200, oh: 120 },
      { pipeline: 'medium', w: 64, h: 48, ow: 128, oh: 96 },
      { pipeline: 'low', w: 100, h: 60, ow: 150, oh: 90 },
      { pipeline: 'q0', w: 100, h: 60, ow: 237, oh: 141 },
    ];
    for (const c of cases) {
      const rgba = makeRgba8(c.w, c.h, c.w + c.oh);
      const r = await call(gpuPage, 'runProduction', { pipeline: c.pipeline, width: c.w, height: c.h, outWidth: c.ow, outHeight: c.oh, rgba: toBase64(rgba) });
      const canvas = fromBase64(r.canvas);
      const buffer = fromBase64(r.buffer);
      const expected = r.direct ? planarToBytesTruncate(halvesFromBytes(buffer), c.ow, c.oh) : buffer;
      let mismatches = 0;
      let first = '';
      for (let i = 0; i < expected.length; i++) {
        if (canvas[i] !== expected[i]) {
          if (!first) first = `byte ${i}: canvas ${canvas[i]} buffer ${expected[i]}`;
          mismatches++;
        }
      }
      expect(mismatches, `${c.pipeline} ${c.w}x${c.h}->${c.ow}x${c.oh} (${r.format}) ${first}`).toBe(0);
    }
  });

  test('(g) Quality 0 Catmull-Rom resampler is bit-exact against the reference', async ({ gpuPage }) => {
    const cases: [number, number, number, number][] = [
      [100, 60, 237, 141],
      [64, 36, 1920, 1080],
      [160, 96, 100, 60],
      [33, 17, 50, 51],
      // column 194 has t = 0.022247314453125, the fraction whose weight an f32-FMA emulation of
      // the half FMA chain double-rounds (src/pipeline/cubic_weights.ts)
      [1280, 720, 854, 480],
    ];
    for (const [w, h, ow, oh] of cases) {
      const rgba = makeRgba8(w, h, w * h);
      const r = await call(gpuPage, 'runBicubic', { width: w, height: h, outWidth: ow, outHeight: oh, data: toBase64(rgba) });
      const s = resampleScale(w, h, ow, oh);
      const expected = resamplePlanar(planarFromRgba8(rgba, w, h), w, h, ow, oh, s.x, s.y);
      const got = fromBase64(r.output);
      let mismatches = 0;
      for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) mismatches++;
      expect(mismatches, `${w}x${h}->${ow}x${oh}`).toBe(0);
      expect(r.scaleFactor).toBeCloseTo(ow / w, 12);
    }
  });

  test('(h) every RB / TB convolution with the real weights matches the reference on its own GPU input', async ({ gpuPage }) => {
    test.setTimeout(600000);
    const [w, h] = [160, 96];
    const rgba = makeRgba8(w, h, 4242);
    const I = halfImage(rgba, w, h);
    const data = toBase64(bytesFromHalves(I.data));
    const lv = levelSizes(w, h);
    const results: string[] = [];
    for (const model of MODELS) {
      const K = MODEL_CHANNELS[model];
      const wt = refWeights(await loadBundle(model));
      type Buf = 'cat0' | 'cat1' | 'cat2' | 'cat3' | 'x4b4' | 'mid';
      const run = async (stopAfter: string, read: Buf[]) => {
        const r = await call(gpuPage, 'runNetwork', { model, width: w, height: h, source: 'half4', data, stopAfter, read });
        return Object.fromEntries(read.map((b) => [b, halvesFromBytes(fromBase64(r.buffers[b]))])) as Record<Buf, Float32Array>;
      };
      const check = (label: string, d: LayerDiff) => {
        const frac = d.mismatches / d.count;
        results.push(`${model} ${label}: ${d.mismatches}/${d.count} (${(100 * frac).toFixed(3)} %) differ, max |d| ${d.maxAbs.toExponential(2)}`);
        expect(frac, `${model} ${label}: fraction of halves that differ`).toBeLessThanOrEqual(LAYER_TOL.mismatchFraction);
        expect(d.outOfBound, `${model} ${label}: halves off by more than one rounding flip`).toBe(0);
      };
      // Encoder: after rb_iB the U-half still holds x_i and mid still holds rb_iA's output.
      for (let i = 0; i <= 4; i++) {
        const layer = `rb${i}` as LayerName;
        const { width: lw, height: lh } = lv[i];
        const cin = i === 0 ? 16 : K[i - 1];
        const k = K[i];
        const buf = (i < 4 ? `cat${i}` : 'x4b4') as Buf;
        const g = await run(`${layer}B`, [buf, 'mid']);
        const stride = i < 4 ? 2 * K[i] : K[3] + K[4];
        const x = gather(g[buf], lw, lh, stride, 0, cin);
        const mid = gather(g.mid, lw, lh, k, 0, k);
        check(`${layer}A`, layerDiff(g.mid, k, 0, conv3x3(x, wt.get(layer, 'weight', '3x3_0'), wt.get(layer, 'bias', '3x3_0'), k, 'elu')));
        const e = conv3x3(mid, wt.get(layer, 'weight', '3x3_1'), wt.get(layer, 'bias', '3x3_1'), k, 'elu');
        const p = conv1x1(x, wt.get(layer, 'weight', '1x1'), null, k);
        check(`${layer}B (3x3 + projection)`, layerDiff(g[buf], stride, i < 4 ? K[i] : K[3], addHalf(e, p), [e, p]));
      }
      // Decoder: tb_jA leaves U | skip in cat_L; tb_jB overwrites U with d_j (mid unchanged).
      for (let j = 0; j < 4; j++) {
        const L = 3 - j;
        const layer = `tb${j}` as LayerName;
        const { width: lw, height: lh } = lv[L];
        const k = K[L];
        const cat = `cat${L}` as Buf;
        const prevBuf = (j === 0 ? 'x4b4' : `cat${L + 1}`) as Buf;
        const a = await run(`${layer}A`, [prevBuf, cat, 'mid']);
        const prev =
          j === 0 ? gather(a.x4b4, lv[4].width, lv[4].height, K[3] + K[4], K[3], K[4]) : gather(a[prevBuf], lv[L + 1].width, lv[L + 1].height, 2 * K[L + 1], 0, K[L + 1]);
        check(`${layer}U`, layerDiff(a[cat], 2 * k, 0, conv3x3(prev, wt.get(layer, 'weight', '3x3_u'), wt.get(layer, 'bias', '3x3_u'), k, 'elu', true)));
        const catT = gather(a[cat], lw, lh, 2 * k, 0, 2 * k);
        check(`${layer}A`, layerDiff(a.mid, k, 0, conv3x3(catT, wt.get(layer, 'weight', '3x3_0'), wt.get(layer, 'bias', '3x3_0'), k, 'elu')));
        const b = await run(`${layer}B`, [cat, 'mid']);
        const mid = gather(b.mid, lw, lh, k, 0, k);
        const e = conv3x3(mid, wt.get(layer, 'weight', '3x3_1'), wt.get(layer, 'bias', '3x3_1'), k, 'elu');
        const p = conv1x1(catT, wt.get(layer, 'weight', '1x1'), null, k);
        check(`${layer}B (3x3 + projection)`, layerDiff(b[cat], 2 * k, 0, addHalf(e, p), [e, p]));
      }
    }
    console.log(`[layers]\n  ${results.join('\n  ')}`);
  });

  test('(i) SR head with the real weights matches the reference on the GPU decoder output', async ({ gpuPage }) => {
    test.setTimeout(300000);
    const [w, h] = [160, 96];
    for (const model of MODELS) {
      const bundle = await loadBundle(model);
      const K0 = MODEL_CHANNELS[model][0];
      const rgba = makeRgba8(w, h, 99);
      const I = halfImage(rgba, w, h);
      const r = await call(gpuPage, 'runNetwork', { model, width: w, height: h, source: 'half4', data: toBase64(bytesFromHalves(I.data)), read: ['cat0'] });
      const L0 = levelSizes(w, h)[0];
      const d3 = gather(halvesFromBytes(fromBase64(r.buffers.cat0)), L0.width, L0.height, 2 * K0, 0, K0);
      const t = (kind: 'weight' | 'bias', op: '3x3' | '1x1') => bundle.tensor('sr', kind, op).data;
      const expected = srHead(d3, I, t('weight', '3x3'), t('bias', '3x3'), t('weight', '1x1'), t('bias', '1x1'));
      const got = halvesFromBytes(fromBase64(r.output));
      expect(got.length).toBe(expected.length);
      const d = layerDiff(got, 1, 0, tensor(expected.length, 1, 1, expected));
      console.log(`[head] ${model} on GPU d3: ${d.mismatches}/${d.count} differ, max |d| ${d.maxAbs.toExponential(2)}`);
      expect(d.mismatches / d.count).toBeLessThanOrEqual(HEAD_TOL.mismatchFraction);
      // a flipped hidden value moves z, then T by at most one ulp: |d| <= one ulp of B + T in [1, 2)
      expect(d.maxAbs).toBeLessThanOrEqual(HEAD_TOL.maxAbs);
      // the learned path is really exercised: most outputs differ from the zero-weight head
      const zeroHead = srHead(d3, I, new Float32Array(t('weight', '3x3').length), t('bias', '3x3'), new Float32Array(t('weight', '1x1').length), t('bias', '1x1'));
      expect(diffStats(got, zeroHead).exactFraction).toBeLessThan(0.5);
    }
  });

  test('device loss triggers listeners and re-acquisition with backoff', async ({ gpuPage }) => {
    const r = await call(gpuPage, 'deviceRecovery');
    expect(r.lost).toEqual(['destroyed']);
    expect(r.sameDevice).toBe(false);
    expect(r.back).toEqual([1, 2, 3, 4]);
    expect(r.elapsed).toBeGreaterThanOrEqual(100);
    expect(r.hasF16).toBe(true);
    expect(r.refs).toBe(0);
    expect(r.current).toBe(false);
  });
});
