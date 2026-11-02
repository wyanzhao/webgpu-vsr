/**
 * Tests of the TypeScript reference itself (tests/reference): hand-computed cases and
 * structural invariants, so the GPU parity suite compares against a trustworthy oracle.
 */
import { describe, expect, it } from 'vitest';
import { H, fmaH, halfBits, halfFromBits } from '../reference/half';
import {
  REF_BASELINE_COEFFICIENTS,
  baseline,
  compose,
  concat,
  conv1x1,
  conv3x3,
  cubicWeights,
  fold,
  halfDot,
  learnedT,
  learnedTanhT,
  padReplicate,
  planarFromRgba8,
  planarToBytesTruncate,
  pool2x2,
  preprocessByte,
  resamplePlanar,
  tensor,
} from '../reference/ops';
import { runNetwork, type RefWeights } from '../reference/network';
import { expectedTensors } from '../../src/weights/types';
import { BASELINE_COEFFICIENTS } from '../../src/pipeline/types';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

describe('half helpers', () => {
  it('round to nearest even and fused half FMA', () => {
    expect(H(1 + 2 ** -11)).toBe(1); // tie -> even
    expect(H(1 + 3 * 2 ** -11)).toBe(1 + 2 ** -9); // tie -> even (up)
    expect(halfBits(1)).toBe(0x3c00);
    expect(halfFromBits(0x7bff)).toBe(65504);
    // fused: one rounding of a*b + c (separately rounding a*b first gives a different result)
    // a*a = 1 + 2^-5 + 2^-9 + 2^-12 + 2^-15 + 2^-20: the fused result keeps 2^-12 + 2^-15
    const a = 1 + 2 ** -6 + 2 ** -10;
    expect(fmaH(a, a, -1)).toBe(2 ** -5 + 2 ** -9 + 2 ** -12 + 2 ** -15);
    expect(H(H(a * a) - 1)).toBe(2 ** -5 + 2 ** -9);
  });
});

describe('preprocess and fold', () => {
  it('maps bytes to H(2*float32(v/255) - 1)', () => {
    expect(preprocessByte(0)).toBe(-1);
    expect(preprocessByte(255)).toBe(1);
    expect(preprocessByte(128)).toBe(H(2 * Math.fround(128 / 255) - 1));
    expect(preprocessByte(128)).toBe(0.0039215087890625);
  });

  it('fold is the permutation F[Y,X,8dy+4dx+c] = I[2Y+dy, 2X+dx, c]', () => {
    const I = tensor(4, 2, 4);
    for (let i = 0; i < I.data.length; i++) I.data[i] = i;
    const F = fold(I);
    expect([F.w, F.h, F.c]).toEqual([2, 1, 16]);
    // folded pixel (0,0): TL = I(0,0), TR = I(1,0), BL = I(0,1), BR = I(1,1)
    expect(Array.from(F.data.subarray(0, 16))).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]);
    expect(Array.from(F.data.subarray(16, 20))).toEqual([8, 9, 10, 11]);
    expect(new Set(F.data).size).toBe(32);
  });

  it('edge replication pads with the last row/column', () => {
    const I = tensor(2, 1, 4, Float32Array.from([1, 2, 3, 0, 4, 5, 6, 0]));
    const P = padReplicate(I, 4, 2);
    expect(Array.from(P.data.subarray(12, 16))).toEqual([4, 5, 6, 0]); // (3, 0)
    expect(Array.from(P.data.subarray(16, 20))).toEqual([1, 2, 3, 0]); // (0, 1)
  });
});

describe('convolutions', () => {
  it('KRSC indexing: a single non-zero weight shifts one input channel into one output channel', () => {
    const cin = 8;
    const k = 4;
    const input = tensor(5, 4, cin);
    for (let i = 0; i < input.data.length; i++) input.data[i] = H((i % 97) / 16 - 2);
    const [K0, R0, S0, C0] = [2, 0, 2, 5];
    const w = new Float32Array(k * 9 * cin);
    w[((K0 * 3 + R0) * 3 + S0) * cin + C0] = 1;
    const out = conv3x3(input, w, null, k, 'none');
    for (let y = 0; y < 4; y++)
      for (let x = 0; x < 5; x++)
        for (let kk = 0; kk < k; kk++) {
          const sy = y + R0 - 1;
          const sx = x + S0 - 1;
          const inside = sy >= 0 && sy < 4 && sx >= 0 && sx < 5;
          const want = kk === K0 && inside ? input.data[(sy * 5 + sx) * cin + C0] : 0;
          expect(out.data[(y * 5 + x) * k + kk]).toBe(want);
        }
  });

  it('applies bias then ELU in float, rounding once', () => {
    const input = tensor(1, 1, 4, Float32Array.from([1, 0, 0, 0]));
    const w = new Float32Array(9 * 4);
    w[4 * 4] = -0.75; // centre tap, channel 0
    const out = conv3x3(input, w, [-0.5], 1, 'elu');
    expect(out.data[0]).toBe(H(Math.exp(-1.25) - 1));
    const pos = conv3x3(input, w, [2], 1, 'elu');
    expect(pos.data[0]).toBe(1.25);
  });

  it('nearest-2x up-conv uses source floor((out + tap - 1) / 2) with zero padding', () => {
    // low-res 3x1 impulse at x = 1, a single centre-column tap s weighting channel 0
    const input = tensor(3, 1, 4);
    input.data[1 * 4] = 1;
    for (let s = 0; s < 3; s++) {
      const w = new Float32Array(9 * 4);
      w[(1 * 3 + s) * 4] = 1;
      const out = conv3x3(input, w, null, 1, 'none', true);
      expect([out.w, out.h]).toEqual([6, 2]);
      const row = Array.from({ length: 6 }, (_, x) => out.data[x]);
      // output x reads source floor((x + s - 1) / 2): s=0 -> [-1,0,0,1,1,2], s=1 -> [0,0,1,1,2,2], s=2 -> [0,1,1,2,2,3]
      const src = [0, 1, 2, 3, 4, 5].map((x) => (x + s - 1) >> 1);
      expect(row).toEqual(src.map((v) => (v === 1 ? 1 : 0)));
    }
  });

  it('1x1 projection is a per-pixel matrix product', () => {
    const input = tensor(2, 1, 4, Float32Array.from([1, 2, 3, 4, -1, 0, 1, 0]));
    const w = Float32Array.from([1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]);
    const out = conv1x1(input, w, null, 2);
    expect(Array.from(out.data)).toEqual([1, 5, -1, 0]);
  });
});

describe('ordered pool', () => {
  it('is the TL, TR, BL, BR half-FMA chain', () => {
    const t = tensor(2, 2, 1, Float32Array.from([1, 2, 3, 4]));
    expect(pool2x2(t).data[0]).toBe(2.5);
  });

  it('is order sensitive where a float average is not', () => {
    // TL=1, TR=1, BL=1+2^-10, BR=0.5+2^-11:
    //   a = 0.25; a = H(0.25 + 0.25) = 0.5; a = H(0.75 + 2^-12) = 0.75 (tie to even);
    //   out = H(0.875 + 2^-13) = 0.875, while the exact mean 0.875 + 2^-12 + 2^-13 rounds up.
    const v = [1, 1, 1 + 2 ** -10, 0.5 + 2 ** -11];
    const chain = pool2x2(tensor(2, 2, 1, Float32Array.from(v))).data[0];
    expect(chain).toBe(0.875);
    expect(H((v[0] + v[1] + v[2] + v[3]) / 4)).toBe(0.875 + 2 ** -11);
    const reversed = pool2x2(tensor(2, 2, 1, Float32Array.from([v[3], v[2], v[1], v[0]]))).data[0];
    expect(reversed).toBe(0.875 + 2 ** -11);
  });
});

describe('SR head pieces', () => {
  it('uses the stored coefficient table of the shader', () => {
    expect(REF_BASELINE_COEFFICIENTS).toEqual(BASELINE_COEFFICIENTS.map((r) => [...r]));
    for (const row of REF_BASELINE_COEFFICIENTS) {
      expect(row.reduce((a, b) => a + b, 0)).toBe(1);
      for (const c of row) expect(H(c)).toBe(c);
    }
  });

  it('half dot uses the two-accumulator (0,2) / (1,3) order', () => {
    const w = REF_BASELINE_COEFFICIENTS[0];
    const v = [0.5, 0.5, 0.3330078125, 0.199951171875];
    // exact products: v0*w0 = -0.017578125, v1*w1 = 0.130859375, v2*w2 = 0.29268646240234375,
    // v3*w3 = -0.021088600158691406.
    // a = H(0.29268646240234375 - 0.017578125) = 1127 * 2^-12 = 0.275146484375
    // b = H(0.130859375 - 0.021088600158691406) = 1798 * 2^-14 = 0.1097412109375
    // D = H(a + b) = H(1576.5 * 2^-12) = 1576 * 2^-12 (tie to even) = 0.384765625
    expect(fmaH(v[2], w[2], H(v[0] * w[0]))).toBe(0.275146484375);
    expect(fmaH(v[3], w[3], H(v[1] * w[1]))).toBe(0.1097412109375);
    expect(halfDot(v, w)).toBe(0.384765625);
    // a sequential four-term half-FMA chain rounds differently for this input:
    // -0.017578125 -> 0.11328125 -> 1663 * 2^-12 -> H(0.38491725921630859) = 1577 * 2^-12
    let s = H(v[0] * w[0]);
    for (let i = 1; i < 4; i++) s = fmaH(v[i], w[i], s);
    expect(s).toBe(0.385009765625);
  });

  it('baseline of a constant image is the constant and taps clamp at borders', () => {
    const src = tensor(5, 3, 4);
    for (let p = 0; p < 15; p++) src.data.set([0, -0.5, 0.5, 0], 4 * p); // N = 0.5, 0.25, 0.75
    for (const [ox, oy] of [
      [0, 0],
      [9, 5],
      [4, 3],
    ]) {
      expect(baseline(src, ox, oy, 0)).toBe(0.5);
      expect(baseline(src, ox, oy, 1)).toBe(0.25);
      expect(baseline(src, ox, oy, 2)).toBe(0.75);
    }
  });

  it('exp2-form T agrees with the tanh form on the fixture biases and for |z| <= 2', () => {
    for (let k = 0; k < 48; k++) {
      const z = H((k - 24) / 128);
      expect(learnedT(z)).toBe(learnedTanhT(z));
    }
    // The float32 form (as the SASS evaluates it) loses precision in t + 1 for strongly
    // negative z; for |z| <= 2 it differs from the float64 tanh in < 0.1% of halves.
    let disagreements = 0;
    for (let i = -4096; i <= 4096; i++) if (learnedT(H(i / 2048)) !== learnedTanhT(H(i / 2048))) disagreements++;
    expect(disagreements).toBeLessThan(9);
    let farNegative = 0;
    for (let i = 1; i <= 512; i++) if (learnedT(H(-6 - i / 256)) !== learnedTanhT(H(-6 - i / 256))) farNegative++;
    expect(farNegative).toBeGreaterThan(0);
    expect(learnedT(0)).toBe(0.5);
  });

  it('composition H(H(B + T) - 0.5) and the gain form', () => {
    expect(compose(0.25, 0.75)).toBe(0.5);
    expect(compose(0.1, 0.6)).toBe(H(H(0.1 + 0.6) - 0.5));
    expect(compose(0.25, 0.75, 0)).toBe(0.25);
    expect(compose(0.25, 0.75, 2)).toBe(0.75);
  });
});

describe('Catmull-Rom resample', () => {
  it('weights interpolate (t = 0 -> centre tap) and sum to one', () => {
    expect(cubicWeights(0)).toEqual([0, 1, 0, 0]);
    expect(cubicWeights(0.5)).toEqual([-0.0625, 0.5625, 0.5625, -0.0625]);
    for (let t = 0; t < 1; t += 1 / 64) {
      const w = cubicWeights(H(t));
      expect(Math.abs(w.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(2 ** -9);
    }
  });

  it('identity scale reproduces the source bytes', () => {
    const rgba = new Uint8Array(4 * 7 * 5);
    for (let i = 0; i < rgba.length; i++) rgba[i] = (i * 37) & 255;
    for (let p = 0; p < 35; p++) rgba[4 * p + 3] = 255;
    const out = resamplePlanar(planarFromRgba8(rgba, 7, 5), 7, 5, 7, 5, 1, 1);
    expect(Array.from(out)).toEqual(Array.from(rgba));
  });

  it('2x upscale of a step edge overshoots (a = -0.5 lobes) and clamps', () => {
    const rgba = new Uint8Array(4 * 4);
    for (let x = 0; x < 4; x++) rgba.set(x < 2 ? [40, 40, 40, 255] : [200, 200, 200, 255], 4 * x);
    const out = resamplePlanar(planarFromRgba8(rgba, 4, 1), 4, 1, 8, 1, 0.5, 1);
    const r = Array.from({ length: 8 }, (_, i) => out[4 * i]);
    expect(Math.min(...r)).toBeLessThan(40);
    expect(Math.max(...r)).toBeGreaterThan(200);
  });

  it('postProcess byte conversion truncates', () => {
    const planar = Float32Array.from([0.5, 1.5, -0.25]);
    expect(Array.from(planarToBytesTruncate(planar, 1, 1))).toEqual([127, 255, 0, 255]);
  });
});

describe('network', () => {
  /** Synthetic weights: everything zero except the SR head biases. */
  function zeroWeights(model: 'low' | 'medium', b1: number[]): RefWeights {
    const shapes = new Map(expectedTensors(model).map((t) => [`${t.layer}/${t.kind}/${t.op}`, t.physicalShape]));
    return {
      get: (layer, kind, op) => {
        const shape = shapes.get(`${layer}/${kind}/${op}`)!;
        const n = shape.reduce((a, b) => a * b, 1);
        return layer === 'sr' && kind === 'bias' && op === '1x1' ? Float32Array.from(b1) : new Float32Array(n);
      },
    };
  }

  it('with zero weights the output is the baseline plus the learned-bias residual', () => {
    const src = tensor(8, 6, 4);
    for (let p = 0; p < 48; p++) src.data.set([H((p % 7) / 4 - 0.75), H((p % 5) / 3 - 0.5), H(-((p % 3) / 2)), 0], 4 * p);
    const b1 = Array.from({ length: 48 }, (_, k) => H((k - 24) / 64));
    const out = runNetwork('low', zeroWeights('low', b1), src).output;
    expect(out.length).toBe(3 * 16 * 12);
    for (const [ox, oy, c] of [
      [0, 0, 0],
      [5, 7, 1],
      [15, 11, 2],
    ]) {
      const k = 16 * c + 4 * (oy % 4) + (ox % 4);
      expect(out[c * 192 + oy * 16 + ox]).toBe(compose(baseline(src, ox, oy, c), learnedT(b1[k])));
    }
  });

  it('negative controls change the result on real weights', () => {
    const dir = join(__dirname, '..', '..', 'public', 'weights');
    const manifest = JSON.parse(readFileSync(join(dir, 'low_manifest.json'), 'utf8'));
    const bytes = readFileSync(join(dir, 'low.bin'));
    const w: RefWeights = {
      get: (layer, kind, op) => {
        const t = manifest.tensors[`Flatten_DLPP_low_2x_${layer}_${kind}_${op}`];
        return new Float16Array(bytes.buffer.slice(bytes.byteOffset + t.offset, bytes.byteOffset + t.offset + t.size));
      },
    };
    const src = tensor(32, 32, 4);
    for (let p = 0; p < 1024; p++) src.data.set([H(Math.sin(p * 0.1)), H(Math.cos(p * 0.07)), H(((p * 13) % 17) / 8.5 - 1), 0], 4 * p);
    const good = runNetwork('low', w, src);
    expect(good.skips.map((s) => [s.w, s.h, s.c])).toEqual([
      [16, 16, 16],
      [8, 8, 32],
      [4, 4, 32],
      [2, 2, 64],
    ]);
    expect([good.bottleneck.w, good.bottleneck.c]).toEqual([1, 64]);
    for (const control of ['swap-concat', 'no-projection'] as const) {
      const bad = runNetwork('low', w, src, { negativeControl: control }).output;
      let diff = 0;
      for (let i = 0; i < bad.length; i++) diff += Math.abs(bad[i] - good.output[i]);
      expect(diff / bad.length).toBeGreaterThan(1e-3);
    }
    // concat order matters: [U | skip] differs from [skip | U]
    const u = tensor(1, 1, 2, Float32Array.from([1, 2]));
    const s = tensor(1, 1, 2, Float32Array.from([3, 4]));
    expect(Array.from(concat(u, s).data)).toEqual([1, 2, 3, 4]);
  });
});
