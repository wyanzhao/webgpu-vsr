/**
 * Browser-side GPU harness for the parity and benchmark suites (bundled to
 * dist-harness/harness.js by `npm run build:harness`, loaded by tests/harness/index.html).
 * Every entry point runs the production modules (src/pipeline, src/weights) on the real
 * WebGPU device and returns raw results as base64 for comparison in Node.
 */
import { GpuDeviceManager, configureCanvas, createGpuContext } from '../../src/pipeline/device';
import { FrameRing } from '../../src/pipeline/ring';
import { BicubicPipeline } from '../../src/pipeline/resample_pipeline';
import { VsrPipeline } from '../../src/pipeline/vsr_pipeline';
import { encodeLayerParams, getComputePipeline, type ConvSpec, type LayerParamsData } from '../../src/pipeline/shader_source';
import type { VsrBufferName } from '../../src/pipeline/memory';
import type { GpuContext, ModelId } from '../../src/pipeline/types';
import { createWeightBundle, loadWeightBundle, sha256Hex } from '../../src/weights/loader';
import { tensorName, type WeightBundle, type WeightManifest } from '../../src/weights/types';
import { fromBase64, toBase64 } from '../reference/patterns';

let ctxPromise: Promise<GpuContext> | null = null;
function gpuContext(): Promise<GpuContext> {
  if (!ctxPromise) ctxPromise = createGpuContext(navigator.gpu);
  return ctxPromise;
}

const bundles = new Map<ModelId, Promise<WeightBundle>>();
function realBundle(model: ModelId): Promise<WeightBundle> {
  let b = bundles.get(model);
  if (!b) {
    b = loadWeightBundle(model, { baseUrl: '/weights/' });
    bundles.set(model, b);
  }
  return b;
}

function storageBuffer(device: GPUDevice, bytes: Uint8Array): GPUBuffer {
  const size = Math.max(16, Math.ceil(bytes.byteLength / 4) * 4);
  const b = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const padded = new Uint8Array(size);
  padded.set(bytes);
  device.queue.writeBuffer(b, 0, padded);
  return b;
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size = buffer.size): Promise<Uint8Array> {
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

async function withValidation<T>(device: GPUDevice, fn: () => Promise<T>): Promise<T> {
  device.pushErrorScope('validation');
  const r = await fn();
  const err = await device.popErrorScope();
  if (err) throw new Error(`WebGPU validation error: ${err.message}`);
  return r;
}

/**
 * Copy of the real bundle with some FP16 tensors replaced; digests are recomputed so the
 * synthetic bundle passes the loader's full validation.
 */
async function syntheticBundle(model: ModelId, replace: Record<string, Uint8Array | 'zero'>): Promise<WeightBundle> {
  const real = await realBundle(model);
  const bytes = real.bytes.slice(0);
  const manifest: WeightManifest = JSON.parse(JSON.stringify(real.manifest));
  for (const [name, value] of Object.entries(replace)) {
    const t = manifest.tensors[name];
    if (!t) throw new Error(`unknown tensor ${name}`);
    const target = new Uint8Array(bytes, t.offset, t.size);
    if (value === 'zero') target.fill(0);
    else {
      if (value.byteLength !== t.size) throw new Error(`${name}: replacement is ${value.byteLength} B, tensor is ${t.size} B`);
      target.set(value);
    }
    t.sha256 = await sha256Hex(target);
  }
  manifest.bundle_sha256 = await sha256Hex(bytes);
  return createWeightBundle(model, manifest, bytes);
}

export interface RunNetworkArgs {
  model: ModelId;
  width: number;
  height: number;
  outWidth?: number;
  outHeight?: number;
  source: 'half4' | 'rgba8';
  data: string;
  sharpness?: number;
  stopAfter?: string;
  read?: (VsrBufferName | 'internal')[];
  /** Replace tensors by name (base64 FP16 payload or 'zero'). */
  replace?: Record<string, string | 'zero'>;
}

async function runNetwork(args: RunNetworkArgs) {
  const ctx = await gpuContext();
  const { device } = ctx;
  const outW = args.outWidth ?? 2 * args.width;
  const outH = args.outHeight ?? 2 * args.height;
  const replace = args.replace
    ? Object.fromEntries(Object.entries(args.replace).map(([k, v]) => [k, v === 'zero' ? ('zero' as const) : fromBase64(v)]))
    : null;
  const bundle = replace ? await syntheticBundle(args.model, replace) : await realBundle(args.model);
  return withValidation(device, async () => {
    const pipe = new VsrPipeline({ model: args.model, storageFormat: ctx.storageFormat, weights: bundle, source: args.source, output: 'buffer' });
    await pipe.initialize(device, args.width, args.height, outW, outH);
    if (args.sharpness !== undefined) pipe.setSharpness(args.sharpness);
    const src = storageBuffer(device, fromBase64(args.data));
    const outBytes = pipe.direct ? 3 * outW * outH * 2 : 4 * outW * outH;
    const out = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const enc = device.createCommandEncoder();
    pipe.encodeFrame(enc, { kind: args.source, buffer: src }, { kind: 'buffer', buffer: out }, undefined, undefined, args.stopAfter);
    device.queue.submit([enc.finish()]);
    const buffers: Record<string, string> = {};
    for (const name of args.read ?? []) {
      const b = name === 'internal' ? pipe.internalBuffer() : pipe.activationBuffer(name);
      if (!b) throw new Error(`no buffer ${name}`);
      buffers[name] = toBase64(await readBuffer(device, b));
    }
    const output = toBase64(await readBuffer(device, out));
    const result = {
      output,
      direct: pipe.direct,
      buffers,
      stepLabels: pipe.stepLabels,
      memoryBytes: pipe.memoryBytes(),
      memoryBreakdown: pipe.memoryBreakdown(),
    };
    pipe.destroy();
    src.destroy();
    out.destroy();
    return result;
  });
}

/** Folded SR-head oracle: zero sr weights, fixture learned biases, fixture source. */
async function runHeadFixture(args: { model: ModelId; source: string; bias1: string }) {
  return runNetwork({
    model: args.model,
    width: 32,
    height: 32,
    source: 'half4',
    data: args.source,
    replace: {
      [tensorName(args.model, 'sr', 'weight', '3x3')]: 'zero',
      [tensorName(args.model, 'sr', 'weight', '1x1')]: 'zero',
      [tensorName(args.model, 'sr', 'bias', '1x1')]: args.bias1,
    },
  });
}

export interface RunConvArgs {
  spec: ConvSpec;
  params: LayerParamsData;
  groups: [number, number];
  weights: string;
  src: string;
  dst: string;
  pool?: string;
}

/** Run one conv specialisation (the production conv.wgsl template) on synthetic data. */
async function runConv(args: RunConvArgs) {
  const { device } = await gpuContext();
  return withValidation(device, async () => {
    const pipeline = await getComputePipeline(device, args.spec);
    const params = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, encodeLayerParams(args.params));
    const w = storageBuffer(device, fromBase64(args.weights));
    const src = storageBuffer(device, fromBase64(args.src));
    const dst = storageBuffer(device, fromBase64(args.dst));
    const pool = args.pool !== undefined ? storageBuffer(device, fromBase64(args.pool)) : null;
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: w } },
      { binding: 2, resource: { buffer: src } },
      { binding: 3, resource: { buffer: dst } },
    ];
    if (pool) entries.push({ binding: 4, resource: { buffer: pool } });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(args.groups[0], args.groups[1]);
    pass.end();
    device.queue.submit([enc.finish()]);
    const result = { dst: toBase64(await readBuffer(device, dst)), pool: pool ? toBase64(await readBuffer(device, pool)) : null };
    for (const b of [params, w, src, dst, pool]) b?.destroy();
    return result;
  });
}

/** Quality 0 buffer variant: RGBA8 source -> packed RGBA8 output. */
async function runBicubic(args: { width: number; height: number; outWidth: number; outHeight: number; data: string }) {
  const ctx = await gpuContext();
  const { device } = ctx;
  return withValidation(device, async () => {
    const pipe = new BicubicPipeline({ storageFormat: ctx.storageFormat, source: 'rgba8', output: 'buffer' });
    await pipe.initialize(device, args.width, args.height, args.outWidth, args.outHeight);
    const src = storageBuffer(device, fromBase64(args.data));
    const out = device.createBuffer({ size: 4 * args.outWidth * args.outHeight, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const enc = device.createCommandEncoder();
    pipe.encodeFrame(enc, { kind: 'rgba8', buffer: src }, { kind: 'buffer', buffer: out });
    device.queue.submit([enc.finish()]);
    const output = toBase64(await readBuffer(device, out));
    const result = { output, memoryBytes: pipe.memoryBytes(), scaleFactor: pipe.scaleFactor };
    pipe.destroy();
    src.destroy();
    out.destroy();
    return result;
  });
}

function makeFrame(rgba: Uint8Array, width: number, height: number): VideoFrame {
  return new VideoFrame(rgba, { format: 'RGBA', codedWidth: width, codedHeight: height, timestamp: 0 });
}

/** Read a canvas swap-chain texture back as RGBA-ordered bytes. */
async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Uint8Array> {
  const w = texture.width;
  const h = texture.height;
  const bytesPerRow = Math.ceil((4 * w) / 256) * 256;
  const buf = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, { width: w, height: h });
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(buf.getMappedRange());
  const out = new Uint8Array(4 * w * h);
  const bgra = texture.format === 'bgra8unorm';
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = y * bytesPerRow + 4 * x;
      const d = 4 * (y * w + x);
      out[d] = raw[s + (bgra ? 2 : 0)];
      out[d + 1] = raw[s + 1];
      out[d + 2] = raw[s + (bgra ? 0 : 2)];
      out[d + 3] = raw[s + 3];
    }
  }
  buf.unmap();
  buf.destroy();
  return out;
}

export interface ProductionArgs {
  pipeline: ModelId | 'q0';
  width: number;
  height: number;
  outWidth: number;
  outHeight: number;
  rgba: string;
}

/**
 * Production path: VideoFrame(RGBA) -> importExternalTexture -> pipeline.execute -> canvas
 * swap-chain texture, compared with the buffer variant (RGBA8 buffer source) of the same
 * pipeline. Returns both as RGBA-ordered bytes plus the buffer variant's raw output.
 */
async function runProduction(args: ProductionArgs) {
  const ctx = await gpuContext();
  const { device } = ctx;
  const rgba = fromBase64(args.rgba);
  return withValidation(device, async () => {
    const canvas = document.createElement('canvas');
    canvas.width = args.outWidth;
    canvas.height = args.outHeight;
    document.body.appendChild(canvas);
    const gctx = canvas.getContext('webgpu');
    if (!gctx) throw new Error('no webgpu canvas context');
    configureCanvas(gctx, ctx, GPUTextureUsage.COPY_SRC);
    const isQ0 = args.pipeline === 'q0';
    const prod = isQ0
      ? new BicubicPipeline({ storageFormat: ctx.storageFormat })
      : new VsrPipeline({ model: args.pipeline as ModelId, storageFormat: ctx.storageFormat, weights: await realBundle(args.pipeline as ModelId) });
    await prod.initialize(device, args.width, args.height, args.outWidth, args.outHeight);
    const frame = makeFrame(rgba, args.width, args.height);
    let canvasBytes: Uint8Array;
    try {
      const ext = device.importExternalTexture({ source: frame });
      const tex = gctx.getCurrentTexture();
      const enc = device.createCommandEncoder();
      prod.execute(enc, ext, tex.createView());
      device.queue.submit([enc.finish()]);
      canvasBytes = await readTexture(device, tex);
    } finally {
      frame.close();
    }
    const ref = isQ0
      ? new BicubicPipeline({ storageFormat: ctx.storageFormat, source: 'rgba8', output: 'buffer' })
      : new VsrPipeline({
          model: args.pipeline as ModelId,
          storageFormat: ctx.storageFormat,
          weights: await realBundle(args.pipeline as ModelId),
          source: 'rgba8',
          output: 'buffer',
        });
    await ref.initialize(device, args.width, args.height, args.outWidth, args.outHeight);
    const src = storageBuffer(device, rgba);
    const direct = !isQ0 && (ref as VsrPipeline).direct;
    const outBytes = direct ? 3 * args.outWidth * args.outHeight * 2 : 4 * args.outWidth * args.outHeight;
    const out = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const enc = device.createCommandEncoder();
    if (isQ0) (ref as BicubicPipeline).encodeFrame(enc, { kind: 'rgba8', buffer: src }, { kind: 'buffer', buffer: out });
    else (ref as VsrPipeline).encodeFrame(enc, { kind: 'rgba8', buffer: src }, { kind: 'buffer', buffer: out });
    device.queue.submit([enc.finish()]);
    const buffer = await readBuffer(device, out);
    prod.destroy();
    ref.destroy();
    src.destroy();
    out.destroy();
    canvas.remove();
    return { canvas: toBase64(canvasBytes), buffer: toBase64(buffer), direct, format: ctx.storageFormat };
  });
}

/** Device-loss recovery: destroy the managed device externally and wait for re-acquisition. */
async function deviceRecovery() {
  const manager = new GpuDeviceManager({ baseDelayMs: 100 });
  const first = await manager.acquire();
  const lost: string[] = [];
  manager.onLost((i) => lost.push(i.reason));
  const restored = new Promise<GpuContext>((resolve) => manager.onRestored(resolve));
  const t0 = performance.now();
  first.device.destroy();
  const second = await restored;
  const elapsed = performance.now() - t0;
  // the new device must be usable
  const b = second.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  second.device.queue.writeBuffer(b, 0, new Uint32Array([1, 2, 3, 4]));
  const back = new Uint32Array((await readBuffer(second.device, b)).buffer);
  b.destroy();
  const sameDevice = first.device === second.device;
  manager.release();
  return { lost, elapsed, sameDevice, back: Array.from(back), refs: manager.referenceCount, hasF16: second.hasF16, current: manager.current !== null };
}

async function info() {
  const ctx = await gpuContext();
  const a = ctx.adapter.info;
  return {
    vendor: a.vendor,
    architecture: a.architecture,
    description: a.description,
    features: [...ctx.device.features].sort(),
    hasF16: ctx.hasF16,
    hasTimestamps: ctx.hasTimestamps,
    storageFormat: ctx.storageFormat,
    maxComputeWorkgroupStorageSize: ctx.device.limits.maxComputeWorkgroupStorageSize,
  };
}

export interface BenchArgs {
  pipeline: ModelId | 'q0';
  width: number;
  height: number;
  outWidth: number;
  outHeight: number;
  frames: number;
  warmup: number;
}

/**
 * Production-path benchmark: a VideoFrame is imported every frame, the pipeline renders to
 * a canvas, and up to 3 frames are in flight through the FrameRing. Returns per-frame GPU
 * times (timestamp queries when available).
 */
async function bench(args: BenchArgs) {
  const ctx = await gpuContext();
  const { device } = ctx;
  const canvas = document.createElement('canvas');
  canvas.width = args.outWidth;
  canvas.height = args.outHeight;
  document.body.appendChild(canvas);
  const gctx = canvas.getContext('webgpu')!;
  configureCanvas(gctx, ctx);
  const pipe =
    args.pipeline === 'q0'
      ? new BicubicPipeline({ storageFormat: ctx.storageFormat })
      : new VsrPipeline({ model: args.pipeline, storageFormat: ctx.storageFormat, weights: await realBundle(args.pipeline) });
  await pipe.initialize(device, args.width, args.height, args.outWidth, args.outHeight);
  const ring = new FrameRing(device, ctx.hasTimestamps);
  const rgba = new Uint8Array(4 * args.width * args.height);
  for (let i = 0; i < rgba.length; i++) rgba[i] = (i * 2654435761) >>> 24;
  const times: number[] = [];
  const sources = new Set<string>();
  const pending: Promise<void>[] = [];
  let dropped = 0;
  const total = args.warmup + args.frames;
  const t0 = performance.now();
  for (let f = 0; f < total; f++) {
    let slot = ring.acquire();
    while (!slot) {
      dropped++;
      await Promise.race(pending);
      slot = ring.acquire();
    }
    const frame = makeFrame(rgba, args.width, args.height);
    try {
      const ext = device.importExternalTexture({ source: frame });
      const enc = device.createCommandEncoder();
      if (pipe instanceof VsrPipeline) pipe.executeSlot(enc, ext, gctx.getCurrentTexture().createView(), slot);
      else pipe.executeSlot(enc, ext, gctx.getCurrentTexture().createView(), slot);
      ring.encodeResolve(enc, slot);
      const done = ring.submit(slot, [enc.finish()]).then((t) => {
        if (f >= args.warmup) {
          times.push(t.gpuTimeMs);
          sources.add(t.timingSource);
        }
      });
      const tracked: Promise<void> = done.finally(() => pending.splice(pending.indexOf(tracked), 1));
      pending.push(tracked);
    } finally {
      frame.close();
    }
  }
  await Promise.all(pending);
  const wall = performance.now() - t0;
  const result = {
    times,
    timingSources: [...sources],
    memoryBytes: pipe.memoryBytes() + ring.memoryBytes(),
    pipelineBytes: pipe.memoryBytes(),
    ringBytes: ring.memoryBytes(),
    ringDropped: ring.droppedFrames,
    waits: dropped,
    wallMs: wall,
    frames: total,
  };
  ring.destroy();
  pipe.destroy();
  canvas.remove();
  return result;
}

const api = { info, runNetwork, runHeadFixture, runConv, runBicubic, runProduction, deviceRecovery, bench };
export type HarnessApi = typeof api;
(globalThis as unknown as { mvsr: HarnessApi }).mvsr = api;
document.title = 'webgpu-vsr harness ready';
