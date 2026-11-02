import { afterEach, describe, expect, it, vi } from 'vitest';
import { GpuDeviceManager, RECOVERY_STABILITY_WINDOW_MS } from '../../src/pipeline/device';

class FakeDevice {
  readonly features = new Set<string>();
  lose!: (info: GPUDeviceLostInfo) => void;
  readonly lost = new Promise<GPUDeviceLostInfo>((resolve) => { this.lose = resolve; });
  readonly destroy = vi.fn();
}

function fakeGpu() {
  const devices: FakeDevice[] = [];
  const requestDevice = vi.fn(async () => {
    const device = new FakeDevice();
    devices.push(device);
    return device;
  });
  const gpu = { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU;
  return { gpu, devices, requestDevice };
}

function lose(device: FakeDevice): void {
  device.lose({ reason: 'unknown', message: 'device lost' } as GPUDeviceLostInfo);
}

async function recover(manager: GpuDeviceManager, device: FakeDevice) {
  lose(device);
  await Promise.resolve();
  return manager.whenRecovered();
}

afterEach(() => {
  vi.useRealTimers();
});

describe('GpuDeviceManager recovery state machine', () => {
  it('keeps one budget for losses during notification and reports failure once to all waiters', async () => {
    const f = fakeGpu();
    const sleep = vi.fn(async (_ms: number) => undefined);
    const manager = new GpuDeviceManager({ gpu: f.gpu, sleep, maxAttempts: 2 });
    const failed = vi.fn();
    manager.onFailed(failed);
    manager.onRestored((ctx) => lose(ctx.device as unknown as FakeDevice));
    await manager.acquire();
    const waits: Array<ReturnType<GpuDeviceManager['whenRecovered']>> = [];
    manager.onLost(() => { waits.push(manager.whenRecovered()); });
    const result = await recover(manager, f.devices[0]!);
    expect(result).toEqual({ generation: 3, success: false });
    expect(await Promise.all(waits)).toEqual([result, result, result]);
    expect(manager.phase).toBe('failed');
    expect(manager.attempt).toBe(2);
    expect(manager.current).toBeNull();
    expect(f.requestDevice).toHaveBeenCalledTimes(3);
    expect(sleep.mock.calls).toEqual([[1000], [2000]]);
    expect(failed).toHaveBeenCalledExactlyOnceWith(result);
    await expect(manager.whenRecovered()).resolves.toEqual(result);
    await expect(manager.acquire()).rejects.toThrow('GPU device recovery exhausted');
    expect(f.requestDevice).toHaveBeenCalledTimes(3);
    expect(failed).toHaveBeenCalledTimes(1);
    manager.release();
  });

  it('resets the budget only after a replacement stays alive for the 10 s stability window', async () => {
    vi.useFakeTimers();
    const f = fakeGpu();
    const sleep = vi.fn(async (_ms: number) => undefined);
    const setTimer = vi.fn((callback: () => void, ms: number) => setTimeout(callback, ms));
    const clearTimer = vi.fn((timer: ReturnType<typeof setTimeout>) => clearTimeout(timer));
    const manager = new GpuDeviceManager({ gpu: f.gpu, sleep, maxAttempts: 2, now: () => Date.now(), setTimer, clearTimer });
    try {
      await manager.acquire();
      expect(await recover(manager, f.devices[0]!)).toEqual({ generation: 1, success: true });
      expect(manager.attempt).toBe(1);
      await vi.advanceTimersByTimeAsync(RECOVERY_STABILITY_WINDOW_MS - 1);
      expect(manager.attempt).toBe(1);
      expect(await recover(manager, f.devices[1]!)).toEqual({ generation: 2, success: true });
      expect(manager.attempt).toBe(2);
      expect(clearTimer).toHaveBeenCalledTimes(1);
      // A cancelled callback from B cannot reset C's budget or timer.
      setTimer.mock.calls[0]![0]();
      expect(manager.attempt).toBe(2);
      await vi.advanceTimersByTimeAsync(RECOVERY_STABILITY_WINDOW_MS - 1);
      expect(manager.attempt).toBe(2);
      await vi.advanceTimersByTimeAsync(1);
      expect(manager.attempt).toBe(0);
      expect(await recover(manager, f.devices[2]!)).toEqual({ generation: 3, success: true });
      expect(sleep.mock.calls).toEqual([[1000], [2000], [1000]]);
      expect(manager.phase).toBe('idle');
      expect(manager.current?.device).toBe(f.devices[3]);
    } finally {
      manager.release();
    }
    expect(vi.getTimerCount()).toBe(0);
  });

  it('fails the burst if the last successful replacement is lost before becoming stable', async () => {
    vi.useFakeTimers();
    const f = fakeGpu();
    const sleep = vi.fn(async (_ms: number) => undefined);
    const manager = new GpuDeviceManager({ gpu: f.gpu, sleep, maxAttempts: 1, now: () => Date.now() });
    const failed = vi.fn();
    manager.onFailed(failed);
    try {
      await manager.acquire();
      await recover(manager, f.devices[0]!);
      await vi.advanceTimersByTimeAsync(RECOVERY_STABILITY_WINDOW_MS - 1);
      expect(await recover(manager, f.devices[1]!)).toEqual({ generation: 2, success: false });
      expect(manager.phase).toBe('failed');
      expect(sleep.mock.calls).toEqual([[1000]]);
      expect(f.requestDevice).toHaveBeenCalledTimes(2);
      expect(failed).toHaveBeenCalledTimes(1);
    } finally {
      manager.release();
    }
  });

  it('resets the budget after release and fresh acquisition when the new context survives 10 s', async () => {
    vi.useFakeTimers();
    const f = fakeGpu();
    const sleep = vi.fn(async (_ms: number) => undefined);
    const manager = new GpuDeviceManager({ gpu: f.gpu, sleep, maxAttempts: 1, now: () => Date.now() });
    const failed = vi.fn();
    manager.onFailed(failed);
    try {
      await manager.acquire(); // A
      expect(await recover(manager, f.devices[0]!)).toEqual({ generation: 1, success: true }); // B
      expect(manager.attempt).toBe(1);
      manager.release();
      expect(manager.current).toBeNull();
      const fresh = await manager.acquire(); // C
      expect(fresh.device).toBe(f.devices[2]);
      expect(manager.attempt).toBe(1);
      await vi.advanceTimersByTimeAsync(RECOVERY_STABILITY_WINDOW_MS);
      expect(await recover(manager, f.devices[2]!)).toEqual({ generation: 2, success: true });
      expect(f.requestDevice).toHaveBeenCalledTimes(4);
      expect(sleep.mock.calls).toEqual([[1000], [1000]]);
      expect(manager.current?.device).toBe(f.devices[3]);
      expect(manager.phase).toBe('idle');
      expect(failed).not.toHaveBeenCalled();
    } finally {
      manager.release();
    }
    expect(vi.getTimerCount()).toBe(0);
  });

  it('lets an acquisition during recovery share its backoff and replacement', async () => {
    const f = fakeGpu();
    let resume!: () => void;
    const sleep = vi.fn(() => new Promise<void>((resolve) => { resume = resolve; }));
    const manager = new GpuDeviceManager({ gpu: f.gpu, sleep });
    await manager.acquire();
    lose(f.devices[0]!);
    await Promise.resolve();
    const acquired = manager.acquire();
    await Promise.resolve();
    expect(f.requestDevice).toHaveBeenCalledTimes(1);
    resume();
    const ctx = await acquired;
    expect(ctx).toBe(manager.current);
    expect(f.requestDevice).toHaveBeenCalledTimes(2);
    expect(sleep).toHaveBeenCalledExactlyOnceWith(1000);
    manager.release();
    manager.release();
  });
});
