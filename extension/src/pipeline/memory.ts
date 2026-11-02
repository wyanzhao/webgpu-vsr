/**
 * GPU memory plan (docs/IMPLEMENTATION_NOTES.md section 4) and exact allocation accounting.
 *
 * planVsrMemory() is THE allocation plan: VsrPipeline allocates exactly the buffers it lists
 * (through a GpuMemoryLedger), so memoryBytes() and the plan total are the same numbers.
 */
import type { ModelId } from './types';
import { alignedSize, isDirectOutput, levelSizes, type Size } from './geometry';
import { FOLD_CHANNELS, MODEL_CHANNELS } from '../weights/types';
import { CUBIC_TABLE_BYTES } from './cubic_weights';

/** One uniform slot per dispatch; 256 = WebGPU minUniformBufferOffsetAlignment. */
export const UNIFORM_SLOT_BYTES = 256;
/** Bytes per FP16 element. */
const F16 = 2;

/** Round a buffer size up to a multiple of 4 bytes (WebGPU requirement). */
export function bufferSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

export type VsrBufferName = 'cat0' | 'cat1' | 'cat2' | 'cat3' | 'x4b4' | 'mid' | 'params' | 'frame' | 'cubic';

/**
 * The internal planar FP16 image (3 x 2W x 2H, resample path only) is written by the head
 * after the last read of `mid` (tb3B) and read only by the resample pass, so it aliases
 * `mid`, which the plan enlarges to max(mid, internal) when resampling.
 */
export const INTERNAL_ALIAS: VsrBufferName = 'mid';

export interface BufferPlan {
  name: VsrBufferName;
  bytes: number;
}

export interface VsrMemoryPlan {
  model: ModelId;
  input: Size;
  aligned: Size;
  output: Size;
  direct: boolean;
  levels: Size[];
  /** Pixel stride (channels) of cat_l = 2*K_l for l = 0..3. */
  catStride: number[];
  /** Pixel stride of the bottleneck buffer x4b4 = K3 + K4 ([x4 | b4]). */
  x4b4Stride: number;
  buffers: BufferPlan[];
  /** Bytes of the internal planar image (0 on the direct path); stored in INTERNAL_ALIAS. */
  internalBytes: number;
  /** Activation, uniform and table buffers (excludes weights and the frame ring). */
  activationBytes: number;
}

/** Number of compute dispatches the VSR pipeline records per frame (one params slot each). */
export function vsrDispatchCount(direct: boolean): number {
  // fold + 4 x (rbA, rbB) + (rb4A, rb4B) + 4 x (up, tbA, tbB) + head [+ resample]
  return 1 + 8 + 2 + 12 + 1 + (direct ? 0 : 1);
}

export function planVsrMemory(model: ModelId, inWidth: number, inHeight: number, outWidth: number, outHeight: number): VsrMemoryPlan {
  const K = MODEL_CHANNELS[model];
  const levels = levelSizes(inWidth, inHeight);
  const aligned = alignedSize(inWidth, inHeight);
  const direct = isDirectOutput(inWidth, inHeight, outWidth, outHeight);
  const px = (l: number) => levels[l].width * levels[l].height;
  const catStride = [0, 1, 2, 3].map((l) => 2 * K[l]);
  const x4b4Stride = K[3] + K[4];
  // mid holds ELU(C3_0(.)) of every block: K_l channels at level l (rb4 uses K4 at level 4).
  let midElems = px(4) * K[4];
  for (let l = 0; l < 4; l++) midElems = Math.max(midElems, px(l) * K[l]);
  if (2 * K[0] < FOLD_CHANNELS) throw new Error('cat0 U-half cannot hold the 16-channel fold');
  const internalBytes = direct ? 0 : 3 * 2 * inWidth * 2 * inHeight * F16;
  const buffers: BufferPlan[] = [
    { name: 'cat0', bytes: bufferSize(px(0) * catStride[0] * F16) },
    { name: 'cat1', bytes: bufferSize(px(1) * catStride[1] * F16) },
    { name: 'cat2', bytes: bufferSize(px(2) * catStride[2] * F16) },
    { name: 'cat3', bytes: bufferSize(px(3) * catStride[3] * F16) },
    { name: 'x4b4', bytes: bufferSize(px(4) * x4b4Stride * F16) },
    { name: 'mid', bytes: bufferSize(Math.max(midElems * F16, internalBytes)) },
  ];
  buffers.push({ name: 'params', bytes: vsrDispatchCount(direct) * UNIFORM_SLOT_BYTES });
  buffers.push({ name: 'frame', bytes: UNIFORM_SLOT_BYTES });
  // exact Catmull-Rom weights of the internal -> target resample (cubic_weights.ts)
  if (!direct) buffers.push({ name: 'cubic', bytes: CUBIC_TABLE_BYTES });
  return {
    model,
    input: { width: inWidth, height: inHeight },
    aligned,
    output: { width: outWidth, height: outHeight },
    direct,
    levels,
    catStride,
    x4b4Stride,
    buffers,
    internalBytes,
    activationBytes: buffers.reduce((a, b) => a + b.bytes, 0),
  };
}

/** True when every buffer of `next` fits in the corresponding allocation of `current`. */
export function planFits(current: VsrMemoryPlan, next: VsrMemoryPlan): boolean {
  if (current.model !== next.model || current.direct !== next.direct) return false;
  return next.buffers.every((b) => {
    const have = current.buffers.find((c) => c.name === b.name);
    return have !== undefined && have.bytes >= b.bytes;
  });
}

/** Buffers of the Quality 0 bicubic pipeline: one params slot and the exact cubic weight table. */
export function planBicubicMemory(): { name: 'params' | 'cubic'; bytes: number }[] {
  return [
    { name: 'params', bytes: UNIFORM_SLOT_BYTES },
    { name: 'cubic', bytes: CUBIC_TABLE_BYTES },
  ];
}

/** Bytes of one timestamp query result. */
export const QUERY_BYTES = 8;

interface LedgerEntry {
  label: string;
  bytes: number;
  destroy(): void;
}

/**
 * Tracks every GPU allocation of an owner so memoryBytes() is exact: buffers by their
 * requested size, query sets by count * 8 bytes.
 */
export class GpuMemoryLedger {
  private readonly entries = new Map<object, LedgerEntry>();

  createBuffer(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    this.entries.set(buffer, { label: descriptor.label ?? 'buffer', bytes: descriptor.size, destroy: () => buffer.destroy() });
    return buffer;
  }

  createQuerySet(device: GPUDevice, descriptor: GPUQuerySetDescriptor): GPUQuerySet {
    const qs = device.createQuerySet(descriptor);
    this.entries.set(qs, { label: descriptor.label ?? 'querySet', bytes: descriptor.count * QUERY_BYTES, destroy: () => qs.destroy() });
    return qs;
  }

  /** Track an allocation made elsewhere (e.g. the uploaded weight buffer). */
  adopt(buffer: GPUBuffer, label: string): GPUBuffer {
    this.entries.set(buffer, { label, bytes: buffer.size, destroy: () => buffer.destroy() });
    return buffer;
  }

  release(resource: object): void {
    const e = this.entries.get(resource);
    if (!e) return;
    e.destroy();
    this.entries.delete(resource);
  }

  releaseAll(): void {
    for (const e of this.entries.values()) e.destroy();
    this.entries.clear();
  }

  get totalBytes(): number {
    let total = 0;
    for (const e of this.entries.values()) total += e.bytes;
    return total;
  }

  breakdown(): { label: string; bytes: number }[] {
    return Array.from(this.entries.values(), ({ label, bytes }) => ({ label, bytes }));
  }
}
