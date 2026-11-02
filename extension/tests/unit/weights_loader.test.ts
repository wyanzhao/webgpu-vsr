import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import {
  WeightValidationError,
  createWeightBundle,
  loadWeightBundle,
  sha256Hex,
  uploadWeightBundle,
  validateBundleStructure,
  vec4Offset,
  type FetchLike,
} from '../../src/weights/loader';
import { MODEL_CHANNELS, expectedTensors, tensorName, type WeightManifest } from '../../src/weights/types';
import type { ModelId } from '../../src/pipeline/types';
import { nodeGpuContext, readBack, readBundleFiles } from '../harness/node_gpu';

const MODELS: ModelId[] = ['low', 'medium'];

function clone(model: ModelId): { manifest: WeightManifest; bytes: ArrayBuffer } {
  const { manifest, bytes } = readBundleFiles(model);
  return { manifest: JSON.parse(JSON.stringify(manifest)), bytes: bytes.slice(0) };
}

describe('expected topology', () => {
  it('lists 99 tensors (66 FP16 + 33 opaque8) with KRSC shapes', () => {
    for (const model of MODELS) {
      const t = expectedTensors(model);
      expect(t).toHaveLength(99);
      expect(t.filter((x) => x.dtype === 'float16')).toHaveLength(66);
      expect(t.filter((x) => x.dtype === 'opaque8')).toHaveLength(33);
      expect(new Set(t.map((x) => x.name)).size).toBe(99);
    }
    const low = new Map(expectedTensors('low').map((t) => [t.name, t.physicalShape]));
    expect(low.get(tensorName('low', 'rb0', 'weight', '3x3_0'))).toEqual([16, 3, 3, 16]);
    expect(low.get(tensorName('low', 'tb0', 'weight', '3x3_u'))).toEqual([64, 3, 3, 64]);
    expect(low.get(tensorName('low', 'tb0', 'weight', '3x3_0'))).toEqual([64, 3, 3, 128]);
    expect(low.get(tensorName('low', 'tb3', 'weight', '1x1'))).toEqual([16, 1, 1, 32]);
    expect(low.get(tensorName('low', 'sr', 'weight', '1x1'))).toEqual([48, 1, 1, 64]);
    const med = new Map(expectedTensors('medium').map((t) => [t.name, t.physicalShape]));
    expect(med.get(tensorName('medium', 'rb0', 'weight', '3x3_0'))).toEqual([32, 3, 3, 16]);
    expect(med.get(tensorName('medium', 'tb3', 'weight', '3x3_u'))).toEqual([32, 3, 3, 64]);
    expect(med.get(tensorName('medium', 'sr', 'weight', '3x3'))).toEqual([64, 3, 3, 32]);
  });
});

describe('shipped bundles', () => {
  for (const model of MODELS) {
    it(`${model}: validates, verifies SHA-256 and exposes typed views`, async () => {
      const { manifest, bytes } = readBundleFiles(model);
      expect(bytes.byteLength).toBe(model === 'low' ? 884816 : 1687728);
      const bundle = await createWeightBundle(model, manifest, bytes);
      expect(bundle.channels).toEqual(MODEL_CHANNELS[model]);
      // WebCrypto digest agrees with Node's independent implementation
      expect(await sha256Hex(bytes)).toBe(createHash('sha256').update(new Uint8Array(bytes)).digest('hex'));
      const v = bundle.tensor('rb2', 'weight', '3x3_1');
      const t = manifest.tensors[v.name];
      expect(v.shape).toEqual(t.physical_shape);
      expect(v.elementOffset).toBe(t.offset / 2);
      expect(v.data.length).toBe(t.size / 2);
      const raw = new DataView(bytes, t.offset, 8);
      const f16 = new Float16Array(new Uint16Array([raw.getUint16(0, true), raw.getUint16(2, true)]).buffer);
      expect(v.data[0]).toBe(f16[0]);
      expect(v.data[1]).toBe(f16[1]);
      expect(vec4Offset(v)).toBe(t.offset / 8);
      expect(bundle.tensor('sr', 'bias', '1x1').data.some((x) => x !== 0)).toBe(true);
      expect(() => bundle.tensor('rb0', 'weight', '3x3_u')).toThrow(WeightValidationError);
    });
  }
});

describe('validation failures', () => {
  const expectInvalid = async (mutate: (m: WeightManifest, b: ArrayBuffer) => ArrayBuffer | void, pattern: RegExp) => {
    const { manifest, bytes } = clone('low');
    const b = mutate(manifest, bytes) ?? bytes;
    await expect(createWeightBundle('low', manifest, b)).rejects.toThrow(pattern);
  };

  it('rejects a truncated bundle', async () => {
    await expectInvalid((_m, b) => b.slice(0, b.byteLength - 16), /bundle is 884800 B/);
  });
  it('rejects a digest mismatch', async () => {
    await expectInvalid((_m, b) => {
      new Uint8Array(b)[1000] ^= 1;
    }, /SHA-256/);
  });
  it('rejects a misaligned offset', async () => {
    await expectInvalid((m) => {
      m.tensors[tensorName('low', 'rb1', 'weight', '3x3_0')].offset += 2;
    }, /not 16-byte aligned/);
  });
  it('rejects a wrong physical shape', async () => {
    await expectInvalid((m) => {
      m.tensors[tensorName('low', 'rb1', 'weight', '3x3_0')].physical_shape = [32, 3, 3, 32];
    }, /physical shape/);
  });
  it('rejects a non-KRSC layout', async () => {
    await expectInvalid((m) => {
      m.tensors[tensorName('low', 'rb1', 'weight', '3x3_0')].physical_layout = 'KCRS';
    }, /layout/);
  });
  it('rejects overlapping tensors', async () => {
    await expectInvalid((m) => {
      const a = m.tensors[tensorName('low', 'rb0', 'bias', '3x3_0')];
      m.tensors[tensorName('low', 'rb0', 'bias', '3x3_1')].offset = a.offset;
    }, /overlaps/);
  });
  it('rejects a missing tensor', async () => {
    await expectInvalid((m) => {
      delete m.tensors[tensorName('low', 'tb2', 'bias', '3x3_u')];
      m.tensor_count = 99;
    }, /tensors, expected 99|missing tensor/);
  });
  it('rejects a non-zero skip-projection bias (the shaders never add it)', () => {
    const { manifest, bytes } = clone('low');
    const t = manifest.tensors[tensorName('low', 'rb3', 'bias', '1x1')];
    new Uint16Array(bytes, t.offset, 1)[0] = 0x3c00;
    expect(() => validateBundleStructure('low', manifest, bytes)).toThrow(/skip bias must be zero/);
  });
  it('rejects the manifest of another model', () => {
    const { manifest, bytes } = clone('medium');
    expect(() => validateBundleStructure('low', manifest, bytes)).toThrow(/expected 'low'/);
  });
});

describe('fetching and upload', () => {
  it('fetches {model}.bin and {model}_manifest.json from the injected base URL', async () => {
    const { manifest, bytes } = readBundleFiles('medium');
    const urls: string[] = [];
    const fetchFn: FetchLike = async (url) => {
      urls.push(url);
      return {
        ok: true,
        status: 200,
        json: async () => JSON.parse(JSON.stringify(manifest)),
        arrayBuffer: async () => bytes.slice(0),
      };
    };
    const bundle = await loadWeightBundle('medium', { baseUrl: 'chrome-extension://abc/weights', fetchFn, verifyTensorDigests: false });
    expect(bundle.model).toBe('medium');
    expect(urls.sort()).toEqual(['chrome-extension://abc/weights/medium.bin', 'chrome-extension://abc/weights/medium_manifest.json']);
  });

  it('reports HTTP failures', async () => {
    const fetchFn: FetchLike = async () => ({ ok: false, status: 404, json: async () => ({}), arrayBuffer: async () => new ArrayBuffer(0) });
    await expect(loadWeightBundle('low', { baseUrl: '/weights/', fetchFn })).rejects.toThrow(/HTTP 404/);
  });

  it('uploads the bundle verbatim as one storage buffer', async () => {
    const { device } = await nodeGpuContext();
    const { manifest, bytes } = readBundleFiles('low');
    const bundle = await createWeightBundle('low', manifest, bytes, false);
    const buf = uploadWeightBundle(device, bundle, 'weights-test', GPUBufferUsage.COPY_SRC);
    expect(buf.size).toBe(bytes.byteLength);
    expect(buf.usage & GPUBufferUsage.STORAGE).toBeTruthy();
    expect(buf.usage & GPUBufferUsage.COPY_DST).toBeTruthy();
    const back = await readBack(device, buf);
    expect(Buffer.from(back).equals(Buffer.from(new Uint8Array(bytes)))).toBe(true);
    buf.destroy();
  });
});
