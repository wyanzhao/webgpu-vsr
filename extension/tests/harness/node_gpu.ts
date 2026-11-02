/**
 * Real WebGPU in Node/vitest through Dawn (the `webgpu` package, Metal backend on macOS).
 * The GPU instance is kept at module scope: if it were garbage collected while a device is
 * alive Dawn aborts the process.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { create, globals } from 'webgpu';
import { createGpuContext } from '../../src/pipeline/device';
import type { GpuContext, ModelId } from '../../src/pipeline/types';
import { createWeightBundle } from '../../src/weights/loader';
import type { WeightBundle, WeightManifest } from '../../src/weights/types';

Object.assign(globalThis, globals);

let instance: GPU | null = null;
let ctx: Promise<GpuContext> | null = null;

export function nodeGpu(): GPU {
  if (!instance) instance = create([]) as unknown as GPU;
  return instance;
}

export function nodeGpuContext(): Promise<GpuContext> {
  if (!ctx) ctx = createGpuContext(nodeGpu());
  return ctx;
}

const WEIGHTS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'public', 'weights');

export function readBundleFiles(model: ModelId): { manifest: WeightManifest; bytes: ArrayBuffer } {
  const manifest = JSON.parse(readFileSync(join(WEIGHTS_DIR, `${model}_manifest.json`), 'utf8')) as WeightManifest;
  const raw = readFileSync(join(WEIGHTS_DIR, `${model}.bin`));
  return { manifest, bytes: raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer };
}

export async function nodeBundle(model: ModelId): Promise<WeightBundle> {
  const { manifest, bytes } = readBundleFiles(model);
  return createWeightBundle(model, manifest, bytes);
}

export async function readBack(device: GPUDevice, buffer: GPUBuffer, size = buffer.size): Promise<Uint8Array> {
  const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, rb, 0, size);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}
