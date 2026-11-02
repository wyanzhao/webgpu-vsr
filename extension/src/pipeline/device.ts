/**
 * Singleton GPUDevice management: high-performance adapter, optional features
 * (shader-f16, timestamp-query, bgra8unorm-storage), device-loss detection and automatic
 * re-acquisition with exponential backoff, plus the canvas configuration helper.
 */
import type { GpuContext, QualityTier } from './types';

export const OPTIONAL_FEATURES = ['shader-f16', 'timestamp-query', 'bgra8unorm-storage'] as const;
export const RECOVERY_BASE_DELAY_MS = 1000;
export const RECOVERY_MAX_ATTEMPTS = 6;
export const RECOVERY_STABILITY_WINDOW_MS = 10_000;

export class WebGpuUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGpuUnavailableError';
  }
}

export type LostListener = (info: { reason: string; message: string; generation: number }) => void;
export type RestoredListener = (ctx: GpuContext, generation: number) => void;
export interface RecoveryResult {
  generation: number;
  success: boolean;
}

export interface DeviceManagerOptions {
  /** navigator.gpu (injectable for tests and workers). */
  gpu?: GPU;
  baseDelayMs?: number;
  maxAttempts?: number;
  /** Timer used for the backoff (injectable). */
  sleep?: (ms: number) => Promise<void>;
  /** Clock and timers used for the 10 s stability window (injectable). */
  now?: () => number;
  setTimer?: (callback: () => void, ms: number) => ReturnType<typeof setTimeout>;
  clearTimer?: (timer: ReturnType<typeof setTimeout>) => void;
}

/** Limits raised to the adapter's capability (the shaders use up to 17.8 KB of workgroup memory). */
const RAISED_LIMITS = ['maxComputeWorkgroupStorageSize', 'maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup'] as const;

export async function createGpuContext(gpu: GPU): Promise<GpuContext> {
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGpuUnavailableError('no WebGPU adapter');
  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  const requiredLimits: Record<string, number> = {};
  for (const name of RAISED_LIMITS) requiredLimits[name] = adapter.limits[name];
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  return {
    adapter,
    device,
    hasF16: device.features.has('shader-f16'),
    hasTimestamps: device.features.has('timestamp-query'),
    storageFormat: device.features.has('bgra8unorm-storage') ? 'bgra8unorm' : 'rgba8unorm',
  };
}

/** Without shader-f16 the neural tiers cannot compile: only Quality 0 (f32 bicubic) runs. */
export function allowedQualities(ctx: Pick<GpuContext, 'hasF16'>): QualityTier[] {
  return ctx.hasF16 ? ['quality0_bicubic', 'low', 'medium'] : ['quality0_bicubic'];
}

export function effectiveQuality(ctx: Pick<GpuContext, 'hasF16'>, requested: QualityTier): QualityTier {
  return allowedQualities(ctx).includes(requested) ? requested : 'quality0_bicubic';
}

/**
 * Configure a canvas so its swap-chain texture can be written by compute shaders.
 * `extraUsage` adds usages such as COPY_SRC (read-back in tests).
 */
export function configureCanvas(context: GPUCanvasContext, ctx: GpuContext, extraUsage = 0): GPUCanvasConfiguration {
  const config: GPUCanvasConfiguration = {
    device: ctx.device,
    format: ctx.storageFormat,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | extraUsage,
    alphaMode: 'opaque',
  };
  context.configure(config);
  return config;
}

export class GpuDeviceManager {
  private ctx: GpuContext | null = null;
  private pending: Promise<GpuContext> | null = null;
  private refs = 0;
  private releasing = new WeakSet<GPUDevice>();
  private readonly lostListeners = new Set<LostListener>();
  private readonly restoredListeners = new Set<RestoredListener>();
  private readonly failedListeners = new Set<(result: RecoveryResult) => void>();
  private readonly recoveryWaiters = new Set<(result: RecoveryResult) => void>();
  private lossGeneration = 0;
  private recoveryAttempt = 0;
  private recoveryPhase: 'idle' | 'recovering' | 'failed' = 'idle';
  private stableSince: number | null = null;
  private stabilityTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly baseDelayMs: number;
  private readonly maxAttempts: number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly now: () => number;
  private readonly setTimer: NonNullable<DeviceManagerOptions['setTimer']>;
  private readonly clearTimer: NonNullable<DeviceManagerOptions['clearTimer']>;

  constructor(private readonly options: DeviceManagerOptions = {}) {
    this.baseDelayMs = options.baseDelayMs ?? RECOVERY_BASE_DELAY_MS;
    this.maxAttempts = options.maxAttempts ?? RECOVERY_MAX_ATTEMPTS;
    this.sleep = options.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.now = options.now ?? Date.now;
    this.setTimer = options.setTimer ?? ((callback, ms) => setTimeout(callback, ms));
    this.clearTimer = options.clearTimer ?? ((timer) => clearTimeout(timer));
  }

  private gpu(): GPU {
    const gpu = this.options.gpu ?? (typeof navigator !== 'undefined' ? navigator.gpu : undefined);
    if (!gpu) throw new WebGpuUnavailableError('navigator.gpu is not available (WebGPU requires a secure context)');
    return gpu;
  }

  /** Current context, if a device is alive. */
  get current(): GpuContext | null {
    return this.ctx;
  }

  get referenceCount(): number {
    return this.refs;
  }

  get generation(): number {
    return this.lossGeneration;
  }

  get attempt(): number {
    return this.recoveryAttempt;
  }

  get phase(): 'idle' | 'recovering' | 'failed' {
    return this.recoveryPhase;
  }

  /** Acquire the shared context (creating the device on first use). */
  async acquire(): Promise<GpuContext> {
    this.refs++;
    try {
      while (this.recoveryPhase === 'recovering') await this.whenRecovered();
      const ctx = await this.ensure();
      if (this.refs === 0 && !this.releasing.has(ctx.device)) {
        this.releasing.add(ctx.device);
        ctx.device.destroy();
      }
      return ctx;
    } catch (e) {
      if (this.refs > 0) this.refs--;
      throw e;
    }
  }

  /** Release one reference; the device is destroyed when the last holder releases. */
  release(): void {
    if (this.refs === 0) return;
    this.refs--;
    if (this.refs === 0) this.cancelStabilityWindow();
    if (this.refs === 0 && this.ctx) {
      const device = this.ctx.device;
      this.ctx = null;
      this.releasing.add(device);
      device.destroy();
    }
  }

  onLost(listener: LostListener): () => void {
    this.lostListeners.add(listener);
    return () => this.lostListeners.delete(listener);
  }

  onRestored(listener: RestoredListener): () => void {
    this.restoredListeners.add(listener);
    return () => this.restoredListeners.delete(listener);
  }

  onFailed(listener: (result: RecoveryResult) => void): () => void {
    this.failedListeners.add(listener);
    return () => this.failedListeners.delete(listener);
  }

  /** Resolves once an in-progress recovery has finished (successfully or not). */
  whenRecovered(): Promise<RecoveryResult> {
    if (this.recoveryPhase !== 'recovering') {
      return Promise.resolve({ generation: this.generation, success: this.ctx !== null && this.recoveryPhase === 'idle' });
    }
    return new Promise((resolve) => this.recoveryWaiters.add(resolve));
  }

  private ensure(): Promise<GpuContext> {
    if (this.recoveryPhase === 'failed') return Promise.reject(new WebGpuUnavailableError('GPU device recovery exhausted'));
    if (this.ctx) return Promise.resolve(this.ctx);
    if (!this.pending) {
      this.pending = createGpuContext(this.gpu())
        .then((ctx) => {
          this.adopt(ctx);
          return ctx;
        })
        .finally(() => {
          this.pending = null;
        });
    }
    return this.pending;
  }

  private adopt(ctx: GpuContext): void {
    if (this.refs === 0) {
      this.releasing.add(ctx.device);
      ctx.device.destroy();
      return;
    }
    this.ctx = ctx;
    void ctx.device.lost.then((info) => this.handleLost(ctx, info));
    if (this.recoveryAttempt > 0) {
      const generation = this.generation;
      this.stableSince = this.now();
      this.stabilityTimer = this.setTimer(() => {
        if (this.ctx !== ctx || this.generation !== generation || this.recoveryPhase !== 'idle') return;
        this.recoveryAttempt = 0;
        this.stabilityTimer = null;
        this.stableSince = null;
      }, RECOVERY_STABILITY_WINDOW_MS);
    }
  }

  private handleLost(ctx: GpuContext, info: GPUDeviceLostInfo): void {
    if (this.releasing.has(ctx.device)) return; // destroyed by release(): not a failure
    if (this.ctx !== ctx) return;
    if (this.stableSince !== null && this.now() - this.stableSince >= RECOVERY_STABILITY_WINDOW_MS) this.recoveryAttempt = 0;
    this.cancelStabilityWindow();
    this.ctx = null;
    this.lossGeneration++;
    // Start recovering before notifying, so listeners can await whenRecovered().
    if (this.refs > 0 && this.recoveryPhase === 'idle') {
      this.recoveryPhase = 'recovering';
      void this.recover();
    }
    const generation = this.generation;
    for (const l of this.lostListeners) l({ reason: info.reason, message: info.message, generation });
  }

  private async recover(): Promise<void> {
    // Even an exhausted burst must notify loss listeners before completing their waiters.
    await Promise.resolve();
    while (this.recoveryAttempt < this.maxAttempts && this.refs > 0) {
      await this.sleep(this.baseDelayMs * 2 ** this.recoveryAttempt++);
      if (this.refs === 0) break;
      try {
        const ctx = await this.ensure();
        if (this.refs === 0) {
          if (!this.releasing.has(ctx.device)) {
            this.releasing.add(ctx.device);
            ctx.device.destroy();
          }
          break;
        }
        // The replacement may have been lost before ensure() settled.
        if (this.ctx !== ctx) continue;
        const generation = this.generation;
        for (const l of this.restoredListeners) {
          if (this.ctx !== ctx || this.generation !== generation) break;
          l(ctx, generation);
          // Let device.lost handlers run before another listener can bind this context.
          await Promise.resolve();
        }
        if (this.ctx !== ctx || this.generation !== generation) continue;
        this.recoveryPhase = 'idle';
        this.completeRecovery(true);
        return;
      } catch {
        // retry with a doubled delay
      }
    }
    this.recoveryPhase = this.refs > 0 ? 'failed' : 'idle';
    const result = this.completeRecovery(false);
    if (this.recoveryPhase === 'failed') {
      for (const l of this.failedListeners) l(result);
    }
  }

  private completeRecovery(success: boolean): RecoveryResult {
    const result = { generation: this.generation, success };
    const waiters = [...this.recoveryWaiters];
    this.recoveryWaiters.clear();
    for (const resolve of waiters) resolve(result);
    return result;
  }

  private cancelStabilityWindow(): void {
    if (this.stabilityTimer !== null) this.clearTimer(this.stabilityTimer);
    this.stabilityTimer = null;
    this.stableSince = null;
  }
}

/** Process-wide device manager. */
export const gpuDeviceManager = new GpuDeviceManager();
