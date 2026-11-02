import { afterAll, describe, expect, it } from 'vitest';
import { GpuMemoryLedger, UNIFORM_SLOT_BYTES, bufferSize, planBicubicMemory, planFits, planVsrMemory, vsrDispatchCount } from '../../src/pipeline/memory';
import { CUBIC_TABLE_BYTES } from '../../src/pipeline/cubic_weights';
import { VsrPipeline } from '../../src/pipeline/vsr_pipeline';
import { BicubicPipeline } from '../../src/pipeline/resample_pipeline';
import { FrameRing } from '../../src/pipeline/ring';
import type { ModelId } from '../../src/pipeline/types';
import { nodeBundle, nodeGpuContext } from '../harness/node_gpu';

const MB = 1e6;
const WEIGHT_BYTES: Record<ModelId, number> = { low: 884816, medium: 1687728 };

describe('allocation plan', () => {
  it('lists every activation buffer of the aliasing plan', () => {
    const p = planVsrMemory('low', 960, 540, 1920, 1080);
    expect(p.direct).toBe(true);
    expect(p.buffers.map((b) => b.name)).toEqual(['cat0', 'cat1', 'cat2', 'cat3', 'x4b4', 'mid', 'params', 'frame']);
    expect(p.catStride).toEqual([32, 64, 64, 128]);
    expect(p.x4b4Stride).toBe(128);
    const r = planVsrMemory('low', 1280, 720, 1920, 1080);
    expect(r.direct).toBe(false);
    // the internal planar image (3 x 2W x 2H halves) aliases the enlarged mid buffer
    expect(r.internalBytes).toBe(3 * 2560 * 1440 * 2);
    expect(r.buffers.find((b) => b.name === 'mid')!.bytes).toBe(r.internalBytes);
    expect(p.internalBytes).toBe(0);
    expect(p.buffers.find((b) => b.name === 'mid')!.bytes).toBe(480 * 272 * 16 * 2);
    expect(r.buffers.find((b) => b.name === 'params')!.bytes).toBe(vsrDispatchCount(false) * UNIFORM_SLOT_BYTES);
    // the resample path also owns the exact cubic weight table (15,361 x 4 halves)
    expect(r.buffers.map((b) => b.name)).toEqual(['cat0', 'cat1', 'cat2', 'cat3', 'x4b4', 'mid', 'params', 'frame', 'cubic']);
    expect(r.buffers.find((b) => b.name === 'cubic')!.bytes).toBe(CUBIC_TABLE_BYTES);
    expect(CUBIC_TABLE_BYTES).toBe(122888);
  });

  it('low 960x540 -> 1920x1080 total (activations, weights, frame ring) matches the documented footprint', async () => {
    const plan = planVsrMemory('low', 960, 540, 1920, 1080);
    const { device, hasTimestamps } = await nodeGpuContext();
    const ring = new FrameRing(device, hasTimestamps);
    const total = plan.activationBytes + WEIGHT_BYTES.low + ring.memoryBytes();
    ring.destroy();
    console.log(`[memory] low 960x540->1920x1080: activations ${(plan.activationBytes / MB).toFixed(3)} MB, total ${(total / MB).toFixed(3)} MB`);
    // pins the figure reported in docs/BENCHMARKS.md (there is no VRAM ceiling; the total is tracked, not bounded)
    expect(total).toBe(19_301_088);
  });

  it('re-anchoring fits smaller inputs into the existing allocation', () => {
    const big = planVsrMemory('medium', 1280, 720, 2560, 1440);
    const small = planVsrMemory('medium', 640, 360, 1280, 720);
    expect(planFits(big, small)).toBe(true);
    expect(planFits(small, big)).toBe(false);
    // the resample path needs the internal image: never satisfied by a direct allocation
    expect(planFits(big, planVsrMemory('medium', 640, 360, 1000, 600))).toBe(false);
    expect(planFits(big, planVsrMemory('low', 640, 360, 1280, 720))).toBe(false);
  });

  it('rounds buffer sizes to multiples of 4', () => {
    expect(bufferSize(1)).toBe(4);
    expect(bufferSize(6)).toBe(8);
    expect(bufferSize(16)).toBe(16);
  });
});

describe('ledger', () => {
  it('counts buffers, query sets and adopted buffers exactly, and releases them', async () => {
    const { device } = await nodeGpuContext();
    const ledger = new GpuMemoryLedger();
    const a = ledger.createBuffer(device, { size: 1024, usage: GPUBufferUsage.STORAGE });
    ledger.createQuerySet(device, { type: 'timestamp', count: 2 });
    ledger.adopt(device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM }), 'adopted');
    expect(ledger.totalBytes).toBe(1024 + 16 + 64);
    ledger.release(a);
    expect(ledger.totalBytes).toBe(80);
    ledger.releaseAll();
    expect(ledger.totalBytes).toBe(0);
  });
});

describe('real pipeline allocations (Dawn)', () => {
  const pipes: VsrPipeline[] = [];
  afterAll(() => pipes.forEach((p) => p.destroy()));

  it('memoryBytes() equals the plan plus the weight buffer, for both models and paths', async () => {
    const ctx = await nodeGpuContext();
    for (const model of ['low', 'medium'] as ModelId[]) {
      for (const [w, h, ow, oh] of [
        [960, 540, 1920, 1080],
        [1280, 720, 1920, 1080],
      ]) {
        const pipe = new VsrPipeline({ model, storageFormat: ctx.storageFormat, weights: await nodeBundle(model) });
        pipes.push(pipe);
        await pipe.initialize(ctx.device, w, h, ow, oh);
        const plan = planVsrMemory(model, w, h, ow, oh);
        expect(pipe.memoryBytes()).toBe(plan.activationBytes + WEIGHT_BYTES[model]);
        const labels = pipe.memoryBreakdown().map((b) => b.label);
        expect(labels).toContain(`weights-${model}`);
        expect(labels.filter((l) => l.startsWith('vsr-'))).toHaveLength(plan.buffers.length);
        console.log(`[memory] ${model} ${w}x${h}->${ow}x${oh}: ${(pipe.memoryBytes() / MB).toFixed(3)} MB (${plan.direct ? 'direct' : 'resample'})`);
        pipe.destroy();
      }
    }
  });

  it('re-anchoring reuses buffers when the new size fits and reallocates otherwise', async () => {
    const ctx = await nodeGpuContext();
    const pipe = new VsrPipeline({ model: 'low', storageFormat: ctx.storageFormat, weights: await nodeBundle('low') });
    pipes.push(pipe);
    await pipe.initialize(ctx.device, 960, 540, 1920, 1080);
    const cat0 = pipe.activationBuffer('cat0');
    const bytes = pipe.memoryBytes();
    await pipe.initialize(ctx.device, 640, 360, 1280, 720);
    expect(pipe.activationBuffer('cat0')).toBe(cat0);
    expect(pipe.memoryBytes()).toBe(bytes);
    expect(pipe.memoryPlan!.input).toEqual({ width: 640, height: 360 });
    await pipe.initialize(ctx.device, 1280, 720, 1920, 1080);
    expect(pipe.activationBuffer('cat0')).not.toBe(cat0);
    expect(pipe.internalBuffer()).not.toBeNull();
    expect(pipe.memoryBytes()).toBe(planVsrMemory('low', 1280, 720, 1920, 1080).activationBytes + WEIGHT_BYTES.low);
  });

  it('Quality 0 owns a single uniform slot and the cubic weight table', async () => {
    const ctx = await nodeGpuContext();
    const q0 = new BicubicPipeline({ storageFormat: ctx.storageFormat });
    await q0.initialize(ctx.device, 960, 540, 1920, 1080);
    expect(planBicubicMemory()).toEqual([
      { name: 'params', bytes: UNIFORM_SLOT_BYTES },
      { name: 'cubic', bytes: CUBIC_TABLE_BYTES },
    ]);
    expect(q0.memoryBytes()).toBe(UNIFORM_SLOT_BYTES + CUBIC_TABLE_BYTES);
    expect(q0.scaleFactor).toBe(2);
    await q0.initialize(ctx.device, 1280, 720, 1600, 900); // same device: nothing reallocated
    expect(q0.memoryBytes()).toBe(UNIFORM_SLOT_BYTES + CUBIC_TABLE_BYTES);
    q0.destroy();
    expect(q0.memoryBytes()).toBe(0);
  });
});
