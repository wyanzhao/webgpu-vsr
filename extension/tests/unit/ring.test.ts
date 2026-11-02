/**
 * Frame ring (triple buffering + timing) on a real Dawn device, and the device manager's
 * acquisition / loss / backoff behaviour with an injected GPU.
 */
import { describe, expect, it } from 'vitest';
import { FrameRing, RING_SLOTS } from '../../src/pipeline/ring';
import { UNIFORM_SLOT_BYTES } from '../../src/pipeline/memory';
import { GpuDeviceManager, RECOVERY_BASE_DELAY_MS, allowedQualities, effectiveQuality } from '../../src/pipeline/device';
import { nodeGpuContext } from '../harness/node_gpu';

/** A compute pass that does measurable work (so timestamps differ). */
async function busyPipeline(device: GPUDevice): Promise<{ pipeline: GPUComputePipeline; bindGroup: GPUBindGroup }> {
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  var x = f32(g.x);
  for (var i = 0u; i < 2000u; i++) { x = fma(x, 1.0001, 0.5); }
  b[g.x] = x;
}`,
  });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const buf = device.createBuffer({ size: 4 * 65536, usage: GPUBufferUsage.STORAGE });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  return { pipeline, bindGroup };
}

describe('FrameRing', () => {
  it('hands out 3 slots, then drops (and counts) frames until one completes', async () => {
    const { device, hasTimestamps } = await nodeGpuContext();
    const ring = new FrameRing(device, hasTimestamps);
    expect(ring.size).toBe(RING_SLOTS);
    const slots = [ring.acquire(), ring.acquire(), ring.acquire()];
    expect(slots.every((s) => s !== null)).toBe(true);
    expect(new Set(slots.map((s) => s!.index)).size).toBe(3);
    expect(ring.inFlight).toBe(3);
    expect(ring.acquire()).toBeNull();
    expect(ring.acquire()).toBeNull();
    expect(ring.droppedFrames).toBe(2);
    const timing = await ring.submit(slots[1]!, [device.createCommandEncoder().finish()]);
    expect(timing.timingSource).toBe('cpu'); // nothing resolved: falls back to submit->done time
    expect(timing.gpuTimeMs).toBeGreaterThanOrEqual(0);
    expect(timing.latencyMs).toBe(timing.gpuTimeMs);
    expect(ring.inFlight).toBe(2);
    const again = ring.acquire();
    expect(again?.index).toBe(slots[1]!.index);
    ring.abandon(again!);
    ring.abandon(slots[0]!);
    ring.abandon(slots[2]!);
    expect(ring.inFlight).toBe(0);
    ring.destroy();
    expect(() => ring.acquire()).toThrow(/after destroy/);
  });

  it('measures GPU time with the slot timestamp query set', async () => {
    const { device, hasTimestamps } = await nodeGpuContext();
    expect(hasTimestamps).toBe(true);
    const ring = new FrameRing(device, true);
    const { pipeline, bindGroup } = await busyPipeline(device);
    const frame = () => {
      const slot = ring.acquire()!;
      expect(slot.timestampWrites?.beginningOfPassWriteIndex).toBe(0);
      expect(slot.timestampWrites?.endOfPassWriteIndex).toBe(1);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass({ timestampWrites: slot.timestampWrites });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1024);
      pass.end();
      ring.encodeResolve(enc, slot);
      return ring.submit(slot, [enc.finish()]);
    };
    // three frames in flight at once, then one more after they completed
    const timings = await Promise.all([frame(), frame(), frame()]);
    timings.push(await frame());
    for (const t of timings) {
      expect(t.timingSource).toBe('timestamp');
      expect(t.gpuTimeMs).toBeGreaterThan(0);
      expect(t.gpuTimeMs).toBeLessThan(1000);
      // submit -> done includes the execution (plus queueing behind the other frames)
      expect(t.latencyMs).toBeGreaterThanOrEqual(t.gpuTimeMs * 0.5);
    }
    expect(ring.completedFrames).toBe(4);
    expect(ring.inFlight).toBe(0);
    ring.destroy();
  });

  it('accounts its memory exactly: uniform + query set + resolve + readback per slot', async () => {
    const { device } = await nodeGpuContext();
    const withTs = new FrameRing(device, true);
    expect(withTs.memoryBytes()).toBe(RING_SLOTS * (UNIFORM_SLOT_BYTES + 2 * 8 + 16 + 16));
    withTs.destroy();
    expect(withTs.memoryBytes()).toBe(0);
    const noTs = new FrameRing(device, false);
    expect(noTs.memoryBytes()).toBe(RING_SLOTS * UNIFORM_SLOT_BYTES);
    expect(noTs.acquire()!.timestampWrites).toBeUndefined();
    noTs.destroy();
  });

  it('rejects slots it did not hand out', async () => {
    const { device } = await nodeGpuContext();
    const a = new FrameRing(device, false);
    const b = new FrameRing(device, false);
    const slot = a.acquire()!;
    await expect(b.submit(slot, [])).rejects.toThrow(/foreign/);
    a.destroy();
    b.destroy();
  });
});

/** Fake WebGPU objects for deterministic device-manager tests. */
function fakeGpu(opts: { failAdapters?: number; features?: string[]; beforeDeviceReturn?: (index: number) => Promise<void> } = {}) {
  let failures = opts.failAdapters ?? 0;
  const devices: { lose: (reason: GPUDeviceLostReason) => void; destroyed: boolean }[] = [];
  const gpu = {
    async requestAdapter() {
      if (failures > 0) {
        failures--;
        return null;
      }
      const features = new Set(opts.features ?? ['shader-f16', 'timestamp-query', 'bgra8unorm-storage']);
      return {
        features,
        limits: { maxComputeWorkgroupStorageSize: 32768, maxStorageBuffersPerShaderStage: 10, maxComputeInvocationsPerWorkgroup: 1024 },
        async requestDevice(desc: GPUDeviceDescriptor) {
          let lose!: (info: GPUDeviceLostInfo) => void;
          const lost = new Promise<GPUDeviceLostInfo>((r) => (lose = r));
          const rec = { lose: (reason: GPUDeviceLostReason) => lose({ reason, message: 'fake' } as GPUDeviceLostInfo), destroyed: false };
          devices.push(rec);
          await opts.beforeDeviceReturn?.(devices.length - 1);
          return {
            features: new Set(desc.requiredFeatures ?? []),
            lost,
            destroy() {
              rec.destroyed = true;
              rec.lose('destroyed');
            },
          };
        },
      };
    },
  };
  return { gpu: gpu as unknown as GPU, devices, setFailures: (n: number) => (failures = n) };
}

describe('GpuDeviceManager', () => {
  it('shares one device, requests the optional features and destroys on last release', async () => {
    const f = fakeGpu();
    const m = new GpuDeviceManager({ gpu: f.gpu });
    const [a, b] = await Promise.all([m.acquire(), m.acquire()]);
    expect(a).toBe(b);
    expect(f.devices).toHaveLength(1);
    expect(a.hasF16 && a.hasTimestamps).toBe(true);
    expect(a.storageFormat).toBe('bgra8unorm');
    m.release();
    expect(f.devices[0].destroyed).toBe(false);
    m.release();
    expect(f.devices[0].destroyed).toBe(true);
    expect(m.current).toBeNull();
  });

  it('falls back to rgba8unorm storage and Quality 0 without bgra8unorm-storage / shader-f16', async () => {
    const m = new GpuDeviceManager({ gpu: fakeGpu({ features: ['timestamp-query'] }).gpu });
    const ctx = await m.acquire();
    expect(ctx.storageFormat).toBe('rgba8unorm');
    expect(ctx.hasF16).toBe(false);
    expect(allowedQualities(ctx)).toEqual(['quality0_bicubic']);
    expect(effectiveQuality(ctx, 'medium')).toBe('quality0_bicubic');
    expect(effectiveQuality({ hasF16: true }, 'medium')).toBe('medium');
  });

  it('re-acquires after an unexpected loss with exponential backoff from 1000 ms', async () => {
    const f = fakeGpu();
    const delays: number[] = [];
    const m = new GpuDeviceManager({ gpu: f.gpu, sleep: async (ms) => void delays.push(ms) });
    const first = await m.acquire();
    const lost: string[] = [];
    const restored: unknown[] = [];
    m.onLost((i) => lost.push(i.reason));
    m.onRestored((c) => restored.push(c));
    f.setFailures(2); // the first two re-acquisition attempts find no adapter
    f.devices[0].lose('unknown');
    await Promise.resolve();
    await m.whenRecovered();
    expect(lost).toEqual(['unknown']);
    expect(delays).toEqual([RECOVERY_BASE_DELAY_MS, 2 * RECOVERY_BASE_DELAY_MS, 4 * RECOVERY_BASE_DELAY_MS]);
    expect(restored).toHaveLength(1);
    expect(m.current).not.toBe(first);
    expect(m.current).toBe(restored[0]);
  });

  it('does not recover a device it destroyed itself or when nobody holds it', async () => {
    const f = fakeGpu();
    const delays: number[] = [];
    const m = new GpuDeviceManager({ gpu: f.gpu, sleep: async (ms) => void delays.push(ms) });
    await m.acquire();
    m.release();
    await Promise.resolve();
    await m.whenRecovered();
    expect(delays).toEqual([]);
    expect(f.devices).toHaveLength(1);
  });

  it.each([false, true])('destroys an unreferenced device acquired while requestDevice is pending (recovery=%s)', async (recovery) => {
    let finish!: () => void;
    let started!: () => void;
    const held = new Promise<void>((r) => (finish = r));
    const requested = new Promise<void>((r) => (started = r));
    const f = fakeGpu({
      beforeDeviceReturn: async (index) => {
        if (index === (recovery ? 1 : 0)) {
          started();
          await held;
        }
      },
    });
    const m = new GpuDeviceManager({ gpu: f.gpu, sleep: async () => {} });
    const restored: unknown[] = [];
    m.onRestored((ctx) => restored.push(ctx));
    const acquisition = m.acquire();
    if (recovery) {
      await acquisition;
      f.devices[0].lose('unknown');
    }
    await requested;
    m.release();
    finish();
    if (recovery) await m.whenRecovered();
    else await acquisition;
    expect(f.devices[recovery ? 1 : 0].destroyed).toBe(true);
    expect(m.current).toBeNull();
    expect(m.referenceCount).toBe(0);
    expect(restored).toEqual([]);
  });

  it('reports a missing adapter', async () => {
    const m = new GpuDeviceManager({ gpu: fakeGpu({ failAdapters: 1 }).gpu });
    await expect(m.acquire()).rejects.toThrow(/no WebGPU adapter/);
    expect(m.referenceCount).toBe(0);
  });
});
