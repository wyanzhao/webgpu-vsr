/**
 * Weight bundle loader: fetches {model}.bin + {model}_manifest.json, validates them against
 * the expected low/medium topology, verifies SHA-256 digests with WebCrypto and exposes
 * typed FP16 tensor views plus element offsets. The bundle is uploaded verbatim as ONE
 * STORAGE | COPY_DST buffer; shaders address tensors by element offset.
 */
import type { ModelId } from '../pipeline/types';
import {
  MODEL_CHANNELS,
  expectedTensors,
  tensorName,
  type LayerName,
  type ManifestTensor,
  type TensorOp,
  type TensorView,
  type WeightBundle,
  type WeightManifest,
} from './types';

export class WeightValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WeightValidationError';
  }
}

function check(cond: boolean, message: string): asserts cond {
  if (!cond) throw new WeightValidationError(message);
}

export type FetchLike = (url: string) => Promise<{ ok: boolean; status: number; arrayBuffer(): Promise<ArrayBuffer>; json(): Promise<unknown> }>;

export interface LoadOptions {
  /** URL prefix ending in '/', e.g. chrome.runtime.getURL('weights/') or '/weights/'. */
  baseUrl: string;
  /** Injectable fetch (defaults to globalThis.fetch). */
  fetchFn?: FetchLike;
  /** Verify every per-tensor SHA-256 in addition to the bundle digest (default true). */
  verifyTensorDigests?: boolean;
}

export async function sha256Hex(data: ArrayBuffer | ArrayBufferView): Promise<string> {
  const view = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const digest = await crypto.subtle.digest('SHA-256', view as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

function product(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function sameShape(a: readonly number[] | null, b: readonly number[] | null): boolean {
  if (a === null || b === null) return a === b;
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/**
 * Structural validation of a manifest + bundle pair (no hashing). Throws
 * WeightValidationError describing the first violation.
 */
export function validateBundleStructure(model: ModelId, manifest: WeightManifest, bytes: ArrayBuffer): void {
  check(manifest.model === model, `manifest is for model '${manifest.model}', expected '${model}'`);
  check(manifest.byte_order === 'little_endian', `unsupported byte order ${manifest.byte_order}`);
  check(sameShape(manifest.encoder_channels, MODEL_CHANNELS[model]), `encoder channels ${JSON.stringify(manifest.encoder_channels)} do not match ${model}`);
  check(bytes.byteLength === manifest.total_size_bytes, `bundle is ${bytes.byteLength} B, manifest says ${manifest.total_size_bytes} B`);
  check(bytes.byteLength % 4 === 0, `bundle size ${bytes.byteLength} is not a multiple of 4`);
  const expected = expectedTensors(model);
  const names = Object.keys(manifest.tensors);
  check(names.length === expected.length && manifest.tensor_count === expected.length, `manifest has ${names.length} tensors, expected ${expected.length}`);
  const ranges: Array<[number, number, string]> = [];
  for (const e of expected) {
    const t: ManifestTensor | undefined = manifest.tensors[e.name];
    check(t !== undefined, `missing tensor ${e.name}`);
    check(t.layer === e.layer && t.kind === e.kind && t.op === e.op, `${e.name}: layer/kind/op metadata mismatch`);
    check(t.dtype === e.dtype, `${e.name}: dtype ${t.dtype}, expected ${e.dtype}`);
    check(Number.isInteger(t.offset) && t.offset >= 0 && t.offset % 16 === 0, `${e.name}: offset ${t.offset} is not 16-byte aligned`);
    check(Number.isInteger(t.size) && t.size > 0 && t.offset + t.size <= bytes.byteLength, `${e.name}: [${t.offset}, +${t.size}) exceeds the bundle`);
    ranges.push([t.offset, t.offset + t.size, e.name]);
    if (e.physicalShape === null) continue;
    check(sameShape(t.physical_shape, e.physicalShape), `${e.name}: physical shape ${JSON.stringify(t.physical_shape)}, expected ${JSON.stringify(e.physicalShape)}`);
    check(t.size === product(e.physicalShape) * 2, `${e.name}: ${t.size} B does not hold ${product(e.physicalShape)} halves`);
    if (e.physicalShape.length === 4) {
      const [k, r, s, c] = e.physicalShape;
      check(t.physical_layout === 'KRSC', `${e.name}: layout ${t.physical_layout}, expected KRSC`);
      check(sameShape(t.logical_shape, [k, c, r, s]), `${e.name}: logical shape ${JSON.stringify(t.logical_shape)} is not [K,C,R,S] of ${JSON.stringify(e.physicalShape)}`);
    } else {
      check(sameShape(t.logical_shape, e.physicalShape), `${e.name}: logical shape mismatch`);
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < ranges.length; i++) {
    check(ranges[i][0] >= ranges[i - 1][1], `${ranges[i][2]} overlaps ${ranges[i - 1][2]}`);
  }
  // RB/TB 1x1 projections are biasless (notes 3.4): the shaders never read these biases,
  // which is only correct if the stored payload is exactly zero.
  for (const e of expected) {
    if (e.kind !== 'bias' || e.op !== '1x1' || e.layer === 'sr') continue;
    const t = manifest.tensors[e.name];
    const v = new Uint16Array(bytes, t.offset, t.size / 2);
    check(v.every((h) => (h & 0x7fff) === 0), `${e.name}: 1x1 skip bias must be zero`);
  }
}

class Bundle implements WeightBundle {
  readonly channels: readonly number[];
  constructor(
    readonly model: ModelId,
    readonly manifest: WeightManifest,
    readonly bytes: ArrayBuffer,
  ) {
    this.channels = MODEL_CHANNELS[model];
  }

  tensor(layer: LayerName, kind: 'weight' | 'bias', op: TensorOp): TensorView {
    const name = tensorName(this.model, layer, kind, op);
    const t = this.manifest.tensors[name];
    if (!t || t.dtype !== 'float16' || !t.physical_shape) throw new WeightValidationError(`no FP16 tensor ${name}`);
    return {
      name,
      shape: t.physical_shape,
      elementOffset: t.offset / 2,
      data: new Float16Array(this.bytes, t.offset, t.size / 2),
    };
  }
}

/** Validate digests and structure of an already fetched bundle. */
export async function createWeightBundle(
  model: ModelId,
  manifest: WeightManifest,
  bytes: ArrayBuffer,
  verifyTensorDigests = true,
): Promise<WeightBundle> {
  validateBundleStructure(model, manifest, bytes);
  const digest = await sha256Hex(bytes);
  check(digest === manifest.bundle_sha256, `bundle SHA-256 ${digest} does not match manifest ${manifest.bundle_sha256}`);
  if (verifyTensorDigests) {
    for (const [name, t] of Object.entries(manifest.tensors)) {
      const d = await sha256Hex(new Uint8Array(bytes, t.offset, t.size));
      check(d === t.sha256, `${name}: SHA-256 mismatch`);
    }
  }
  return new Bundle(model, manifest, bytes);
}

/** Fetch, validate and wrap the weight bundle of a model. */
export async function loadWeightBundle(model: ModelId, options: LoadOptions): Promise<WeightBundle> {
  const fetchFn: FetchLike = options.fetchFn ?? ((url) => fetch(url));
  const base = options.baseUrl.endsWith('/') ? options.baseUrl : `${options.baseUrl}/`;
  const [mRes, bRes] = await Promise.all([fetchFn(`${base}${model}_manifest.json`), fetchFn(`${base}${model}.bin`)]);
  check(mRes.ok, `manifest fetch failed with HTTP ${mRes.status}`);
  check(bRes.ok, `bundle fetch failed with HTTP ${bRes.status}`);
  const manifest = (await mRes.json()) as WeightManifest;
  const bytes = await bRes.arrayBuffer();
  return createWeightBundle(model, manifest, bytes, options.verifyTensorDigests ?? true);
}

/** Upload the whole bundle as a single STORAGE | COPY_DST buffer (`extraUsage` e.g. COPY_SRC for tests). */
export function uploadWeightBundle(device: GPUDevice, bundle: WeightBundle, label = `weights-${bundle.model}`, extraUsage = 0): GPUBuffer {
  const buffer = device.createBuffer({ label, size: bundle.bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage });
  device.queue.writeBuffer(buffer, 0, bundle.bytes);
  return buffer;
}

/** Offset of a tensor in vec4<f16> (8-byte) units, as consumed by the shaders. */
export function vec4Offset(view: TensorView): number {
  if (view.elementOffset % 4 !== 0) throw new WeightValidationError(`${view.name}: element offset ${view.elementOffset} is not vec4-aligned`);
  return view.elementOffset / 4;
}
