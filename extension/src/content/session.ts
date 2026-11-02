/**
 * VideoSRSession: the per-video GPU session behind the content controller's
 * {@link VideoSessionHandle} contract, and the factory the controller is bootstrapped with.
 *
 * Lifecycle: attach() acquires the shared GPUDevice (gpuDeviceManager), creates the sibling
 * {@link CanvasOverlay}, configures its WebGPU context for compute writes, creates the 3-slot
 * {@link FrameRing} and starts the requestVideoFrameCallback loop. The pipeline is created lazily
 * from the first VideoFrame: Quality 0 (Catmull-Rom) or RTX VSR low / medium, chosen from
 * settings.qualityLevel through effectiveQuality() (no shader-f16 -> Quality 0) and lowered by the
 * {@link BudgetController}. It is initialised with the VideoFrame's display size and the overlay's
 * canvas backing store, which the overlay sizes by the same rule as decideOutput() (1080p ceiling;
 * VSR presents its 2W x 2H image directly when the target covers it).
 *
 * Per frame (strict try/finally): ring slot or drop ->
 * `new VideoFrame(video)` -> size check (re-initialise on change) -> importExternalTexture ->
 * one compute pass into the canvas swap-chain texture -> ring submit -> `frame.close()` in finally.
 * Completed frames present only after clean GPU error scopes, first-frame pipeline validation and
 * a matching overlay presentation epoch, so stale or broken frames never hide the video. No CPU pixel read-backs:
 * the only GPU -> CPU traffic is the ring's 16-byte timestamp read-back per frame.
 *
 * Failure handling: SecurityError (tainted frame) latches taint and stops (never retried);
 * DRM events / mediaKeys stop at once; device loss restores the video, drops GPU objects and
 * re-binds when gpuDeviceManager restores a device; a budget downgrade swaps pipelines without a
 * gap (the old pipeline renders until the new one is initialised); a throttle (Quality 0 still
 * over budget or timing out) stops processing and asks the controller to re-run the gate, whose
 * point 9 then fails. Every stop restores the native video immediately.
 */

import { LOG_PREFIX, hasWebGpuApi } from '../shared/env';
import type { SessionTelemetry } from '../shared/messages';
import { QUALITY_PIPELINE_IDS, type QualityLevel, type UserSettings } from '../shared/settings';
import { defaultDiagnostics, syntheticLatencyFor, type Diagnostics } from '../shared/diagnostics';
import { GpuDeviceManager, WebGpuUnavailableError, configureCanvas, effectiveQuality, gpuDeviceManager } from '../pipeline/device';
import { decideOutput, type OutputDecision, type Size } from '../pipeline/geometry';
import { BicubicPipeline } from '../pipeline/resample_pipeline';
import { FrameRing, type FrameSlot } from '../pipeline/ring';
import { BudgetController, FrameRateEstimator, type BudgetEvent } from '../pipeline/timing';
import type { FrameTiming, GpuContext, ModelId, QualityTier, SRComputePipeline } from '../pipeline/types';
import { VsrPipeline, defaultWeightsBaseUrl } from '../pipeline/vsr_pipeline';
import { loadWeightBundle } from '../weights/loader';
import type { WeightBundle } from '../weights/types';
import { CanvasOverlay, type OverlayEnvironment } from './overlay';
import { isHdrTransfer } from './eligibility';
import type { VideoSessionContext, VideoSessionFactory, VideoSessionHandle } from './index';

// ---------------------------------------------------------------------------------------------
// Pure policy helpers
// ---------------------------------------------------------------------------------------------

/** settings.qualityLevel -> pipeline tier. */
export const QUALITY_TIERS: Readonly<Record<QualityLevel, QualityTier>> = Object.freeze({
  0: 'quality0_bicubic',
  1: 'low',
  2: 'medium',
});

const TIER_LEVEL: Readonly<Record<QualityTier, QualityLevel>> = Object.freeze({ quality0_bicubic: 0, low: 1, medium: 2 });

export function requestedTier(settings: Pick<UserSettings, 'qualityLevel'>): QualityTier {
  return QUALITY_TIERS[settings.qualityLevel];
}

/** The tier the user asked for, limited by the device (no shader-f16 -> Quality 0). */
export function selectTier(settings: Pick<UserSettings, 'qualityLevel'>, gpu: Pick<GpuContext, 'hasF16'>): QualityTier {
  return effectiveQuality(gpu, requestedTier(settings));
}

/** 'bicubic-q0' | 'rtx-vsr-low' | 'rtx-vsr-medium' */
export function pipelineIdForTier(tier: QualityTier): string {
  return QUALITY_PIPELINE_IDS[TIER_LEVEL[tier]];
}

/** The overlay's internal scale: 2 for the VSR models, none (render at the target) for Quality 0. */
export function internalScaleForTier(tier: QualityTier): number | null {
  return tier === 'quality0_bicubic' ? null : 2;
}

/** Output size of a pipeline for a source and the overlay's device-pixel target (notes 3.12). */
export function planOutput(inWidth: number, inHeight: number, target: Size, tier: QualityTier): OutputDecision {
  return decideOutput(inWidth, inHeight, target.width, target.height, tier === 'quality0_bicubic');
}

export type FrameErrorKind = 'tainted' | 'unavailable' | 'fatal';

/**
 * Classifies an exception from `new VideoFrame(video)`: a SecurityError means the frame is
 * cross-origin tainted (latched, never retried); an InvalidStateError means there is no decoded
 * frame right now (skip this callback); anything else is a session failure.
 */
export function classifyVideoFrameError(err: unknown): FrameErrorKind {
  if (typeof err !== 'object' || err === null) return 'fatal';
  const e = err as { name?: unknown; message?: unknown };
  if (e.name === 'SecurityError' || (typeof e.message === 'string' && /tainted|cross-origin/i.test(e.message))) return 'tainted';
  if (e.name === 'InvalidStateError') return 'unavailable';
  return 'fatal';
}

/** A gap in `presentedFrames` larger than this (or older than {@link GAP_WINDOW_MS}) is a seek or pause, not drops. */
export const MAX_COUNTED_GAP = 10;
export const GAP_WINDOW_MS = 500;

/**
 * Presentation cadence of the rVFC loop: callbacks per second over the last second, and frames the
 * compositor presented without a callback reaching us (`presentedFrames` gaps).
 */
export class PresentationTracker {
  private readonly stamps: number[] = [];
  private lastPresented: number | null = null;
  private lastNow = 0;
  private dropped = 0;
  private rate = 0;

  /** Records one callback; returns the number of frames newly counted as dropped. */
  record(now: number, presentedFrames?: number): number {
    this.stamps.push(now);
    while (this.stamps.length > 1 && now - (this.stamps[0] as number) > 1000) this.stamps.shift();
    if (this.stamps.length >= 2) {
      const span = now - (this.stamps[0] as number);
      this.rate = span > 0 ? ((this.stamps.length - 1) * 1000) / span : this.rate;
    }
    let gap = 0;
    if (typeof presentedFrames === 'number' && Number.isFinite(presentedFrames)) {
      if (this.lastPresented !== null) {
        const delta = presentedFrames - this.lastPresented;
        if (delta > 1 && delta <= MAX_COUNTED_GAP && now - this.lastNow < GAP_WINDOW_MS) gap = delta - 1;
      }
      this.lastPresented = presentedFrames;
      this.lastNow = now;
    }
    this.dropped += gap;
    return gap;
  }

  get fps(): number {
    return this.rate;
  }

  get droppedFrames(): number {
    return this.dropped;
  }

  reset(): void {
    this.stamps.length = 0;
    this.lastPresented = null;
    this.lastNow = 0;
    this.dropped = 0;
    this.rate = 0;
  }
}

/** Allows one event per interval. */
export class RateLimiter {
  private last = Number.NEGATIVE_INFINITY;

  constructor(readonly intervalMs: number) {}

  ready(now: number): boolean {
    return now - this.last >= this.intervalMs;
  }

  /** Milliseconds until {@link ready} turns true. */
  delay(now: number): number {
    return Math.max(0, this.last + this.intervalMs - now);
  }

  mark(now: number): void {
    this.last = now;
  }
}

/** Telemetry reaches the controller (and the HUD) at most twice per second. */
export const TELEMETRY_INTERVAL_MS = 500;

// ---------------------------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------------------------

/** A production pipeline: records a frame with the timestamp writes of a ring slot. */
export type RenderPipeline = SRComputePipeline & {
  executeSlot(encoder: GPUCommandEncoder, external: GPUExternalTexture, view: GPUTextureView, slot: FrameSlot): void;
};

const weightCache = new Map<ModelId, Promise<WeightBundle>>();

/** Loads (once per content-script lifetime) and validates the bundled weights of a model. */
function cachedWeights(model: ModelId): Promise<WeightBundle> {
  let pending = weightCache.get(model);
  if (!pending) {
    pending = loadWeightBundle(model, { baseUrl: defaultWeightsBaseUrl() });
    pending.catch(() => weightCache.delete(model));
    weightCache.set(model, pending);
  }
  return pending;
}

export function createPipeline(tier: QualityTier, gpu: GpuContext): RenderPipeline {
  if (tier === 'quality0_bicubic') return new BicubicPipeline({ storageFormat: gpu.storageFormat });
  return new VsrPipeline({ model: tier, storageFormat: gpu.storageFormat, weights: () => cachedWeights(tier) });
}

// ---------------------------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------------------------

export interface SessionServices {
  deviceManager: GpuDeviceManager;
  createPipeline(tier: QualityTier, gpu: GpuContext): RenderPipeline;
  diagnostics(): Diagnostics;
  /** WebGPU cannot be used in this frame (gate point 8 fails from now on). */
  markGpuUnavailable(): void;
  now(): number;
  /** DOM services of the overlay (the browser's by default; injectable for tests). */
  overlayEnvironment?: OverlayEnvironment;
}

interface PipelineSpec {
  tier: QualityTier;
  inWidth: number;
  inHeight: number;
  outWidth: number;
  outHeight: number;
}

interface CompletedFrame {
  token: number;
  presentationEpoch: number;
  validated: boolean;
  presented: boolean;
}

interface ActivePipeline extends PipelineSpec {
  latestCompleted: CompletedFrame | null;
  impl: RenderPipeline;
  /** The first submitted frame of this pipeline passed its GPU error scopes. */
  validated: boolean;
  /** Token of the first frame this pipeline submitted (the one that validates it). */
  firstToken: number | null;
}

type StopReason = 'throttled' | 'tainted' | 'drm' | 'failed' | 'gpu-unavailable' | 'hdr' | 'unsupported-layout';

const DRM_EVENTS = ['encrypted', 'waitingforkey'] as const;
const HAVE_CURRENT_DATA = 2;

function sameSpec(a: PipelineSpec, b: PipelineSpec): boolean {
  return a.tier === b.tier && a.inWidth === b.inWidth && a.inHeight === b.inHeight && a.outWidth === b.outWidth && a.outHeight === b.outHeight;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export class VideoSRSession implements VideoSessionHandle {
  private readonly video: HTMLVideoElement;
  private settings: UserSettings;
  private readonly budget: BudgetController;
  private readonly frameRate = new FrameRateEstimator();
  private readonly presentation = new PresentationTracker();
  private readonly telemetryLimiter = new RateLimiter(TELEMETRY_INTERVAL_MS);

  private started = false;
  private destroyed = false;
  private detached = false;
  private deviceLost = false;
  private deviceGeneration = 0;
  private stopReason: StopReason | null = null;
  private holdsDevice = false;

  private gpu: GpuContext | null = null;
  /** Incremented whenever the GPU objects are replaced; stale async results compare against it. */
  private gpuGeneration = 0;
  private overlay: CanvasOverlay | null = null;
  private canvasContext: GPUCanvasContext | null = null;
  private ring: FrameRing | null = null;
  private pipeline: ActivePipeline | null = null;
  private pendingSpec: PipelineSpec | null = null;
  private configRevision = 0;

  private rvfcHandle: number | null = null;
  private watchdogTimer: ReturnType<typeof setTimeout> | null = null;
  private telemetryTimer: ReturnType<typeof setTimeout> | null = null;
  private frameToken = 0;
  private pendingStill = false;
  private lastOverlaySync = 0;
  private presentedFrames = 0;
  private pipelineInits = 0;
  private ringDrops = 0;
  private rejectedFrames = 0;
  private readonly unsubscribers: Array<() => void> = [];

  constructor(
    private readonly context: VideoSessionContext,
    private readonly services: SessionServices,
  ) {
    this.video = context.video;
    this.settings = context.getSettings();
    this.budget = new BudgetController(requestedTier(this.settings));
  }

  // ---- VideoSessionHandle -----------------------------------------------------------------

  attach(): void {
    if (this.destroyed) return;
    if (this.started) {
      this.ensureLoop();
      return;
    }
    this.started = true;
    for (const type of DRM_EVENTS) this.video.addEventListener(type, this.onDrmEvent);
    this.video.addEventListener('seeked', this.onStillFrameEvent);
    this.video.addEventListener('loadeddata', this.onStillFrameEvent);
    this.unsubscribers.push(
      () => {
        for (const type of DRM_EVENTS) this.video.removeEventListener(type, this.onDrmEvent);
        this.video.removeEventListener('seeked', this.onStillFrameEvent);
        this.video.removeEventListener('loadeddata', this.onStillFrameEvent);
      },
      this.services.deviceManager.onLost(this.onDeviceLost),
      this.services.deviceManager.onRestored(this.onDeviceRestored),
    );
    void this.start();
  }

  reattach(): void {
    if (this.destroyed || !this.started) return;
    this.detached = false;
    this.overlay?.reattach();
    if (this.stopReason !== null) return;
    this.ensureLoop();
    this.renderFrame();
  }

  detach(): void {
    if (this.destroyed || !this.started) return;
    this.detached = true;
    this.cancelLoop();
    this.overlay?.restoreVideo();
  }

  updateSettings(settings: UserSettings): void {
    if (this.destroyed) return;
    const previous = this.settings;
    this.settings = settings;
    if (settings.sharpness !== previous.sharpness) {
      this.pipeline?.impl.setSharpness(settings.sharpness);
      this.kick();
    }
    if (settings.qualityLevel !== previous.qualityLevel) {
      this.budget.reset(this.gpu ? selectTier(settings, this.gpu) : requestedTier(settings));
      if (this.pendingSpec && this.pendingSpec.tier !== this.budget.tier) {
        this.pendingSpec = null;
        this.configRevision++;
      }
      // A new quality is the user's cue to retry a throttled session.
      if (this.stopReason === 'throttled') this.resume();
      this.kick();
    }
  }

  sourceChanged(): void {
    if (this.destroyed) return;
    this.presentation.reset();
    this.frameRate.reset();
    this.budget.setFrameRate(null);
    this.ringDrops = 0;
    this.budget.reset(this.gpu ? selectTier(this.settings, this.gpu) : this.budget.ceiling);
    this.overlay?.restoreVideo();
    if (this.stopReason === 'throttled') this.resume();
    this.kick();
  }

  /** Periodic controller evaluation also refreshes paused videos with no frame callbacks. */
  refreshGeometry(): void {
    if (!this.canRender()) return;
    this.lastOverlaySync = this.services.now();
    this.overlay?.sync(); // onLayout queues a still frame when the layout changed or the canvas remounted.
    this.checkLayout();
  }

  isLayoutSupported(): boolean {
    return this.overlay?.layoutSupported ?? true;
  }

  hasInsertionLayoutFailure(): boolean {
    return this.overlay?.hasInsertionLayoutFailure ?? false;
  }

  readSiteStyle<T>(read: () => T): T {
    return this.overlay ? this.overlay.readSiteStyle(read) : read();
  }

  private checkLayout(): boolean {
    if (this.isLayoutSupported()) return true;
    this.stop('unsupported-layout');
    this.context.requestEvaluation();
    return false;
  }

  isBudgetHealthy(): boolean {
    return this.stopReason !== 'throttled';
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.cancelLoop();
    this.clearTimers();
    for (const fn of this.unsubscribers.splice(0)) fn();
    this.configRevision++;
    this.pendingSpec = null;
    this.pipeline?.impl.destroy();
    this.pipeline = null;
    this.ring?.destroy();
    this.ring = null;
    this.unconfigureCanvas();
    this.overlay?.destroy();
    this.overlay = null;
    this.canvasContext = null;
    this.gpu = null;
    if (this.holdsDevice) {
      this.holdsDevice = false;
      this.services.deviceManager.release();
    }
  }

  // ---- setup ------------------------------------------------------------------------------

  private async start(): Promise<void> {
    let gpu: GpuContext;
    try {
      gpu = await this.services.deviceManager.acquire();
    } catch (err) {
      if (this.destroyed) return;
      if (err instanceof WebGpuUnavailableError) {
        this.services.markGpuUnavailable();
        this.stop('gpu-unavailable');
        this.context.requestEvaluation();
        return;
      }
      this.fail(err);
      return;
    }
    if (this.destroyed) {
      this.services.deviceManager.release();
      return;
    }
    this.holdsDevice = true;
    if (this.deviceLost) {
      const current = this.services.deviceManager.current;
      if (current) this.onDeviceRestored(current, this.services.deviceManager.generation);
      return;
    }
    this.budget.reset(selectTier(this.settings, gpu));
    this.bindDevice(gpu);
  }

  /** (Re)creates everything that belongs to one GPUDevice; the overlay and canvas are kept. */
  private bindDevice(gpu: GpuContext): void {
    try {
      this.gpu = gpu;
      this.gpuGeneration++;
      if (!this.overlay) {
        this.overlay = new CanvasOverlay(this.video, {
          internalScale: internalScaleForTier(this.budget.tier),
          env: this.services.overlayEnvironment,
          onLayout: () => {
            if (!this.video.paused) return;
            this.pendingStill = true;
            queueMicrotask(() => this.retryStill());
          },
        });
      }
      this.lastOverlaySync = this.services.now();
      if (!this.canvasContext) {
        const ctx = this.overlay.canvas.getContext('webgpu');
        if (!ctx) throw new Error('the overlay canvas has no WebGPU context');
        this.canvasContext = ctx;
      }
      configureCanvas(this.canvasContext, gpu);
      this.ring = new FrameRing(gpu.device, gpu.hasTimestamps);
    } catch (err) {
      this.fail(err);
      return;
    }
    if (!this.checkLayout()) return;
    this.ensureLoop();
    // Creates the pipeline from the current frame right away (and shows it when paused).
    this.renderFrame();
  }

  // ---- frame loop -------------------------------------------------------------------------

  private canRender(): boolean {
    return (
      this.started &&
      !this.destroyed &&
      !this.detached &&
      !this.deviceLost &&
      this.stopReason === null &&
      this.gpu !== null &&
      this.ring !== null &&
      this.overlay !== null &&
      this.canvasContext !== null
    );
  }

  private ensureLoop(): void {
    if (this.rvfcHandle !== null || !this.canRender()) return;
    this.rvfcHandle = this.video.requestVideoFrameCallback(this.onVideoFrame);
  }

  private cancelLoop(): void {
    if (this.rvfcHandle === null) return;
    this.video.cancelVideoFrameCallback(this.rvfcHandle);
    this.rvfcHandle = null;
  }

  private readonly onVideoFrame = (now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata): void => {
    this.rvfcHandle = null;
    if (!this.canRender()) return;
    this.presentation.record(now, metadata.presentedFrames);
    this.frameRate.record(metadata, this.video.playbackRate);
    this.budget.setFrameRate(this.frameRate.frameRate);
    this.ensureLoop();
    this.renderFrame();
  };

  /** Paused-MSE workaround: show the current frame on `seeked` / `loadeddata` while paused. */
  private readonly onStillFrameEvent = (): void => {
    if (this.video.paused) this.kick();
  };

  /** Makes a paused video pick up a tier change now (a playing one does on its next frame). */
  private kick(): void {
    if (this.destroyed || !this.video.paused) return;
    this.pendingStill = true;
    this.renderFrame();
  }

  /** Processes the video's current frame (rVFC callback, still frame, or after a pipeline swap). */
  private renderFrame(): void {
    if (!this.canRender() || this.video.readyState < HAVE_CURRENT_DATA) return;
    if (this.video.mediaKeys) {
      this.onDrmEvent();
      return;
    }
    const overlay = this.overlay as CanvasOverlay;
    const ring = this.ring as FrameRing;
    // Re-parented without a detach event, or the page removed / moved the canvas (a player that
    // re-renders its container but keeps the <video>): mount it next to the video again.
    const now = this.services.now();
    if (!overlay.isMounted) {
      this.lastOverlaySync = now;
      overlay.reattach();
    } else if (now - this.lastOverlaySync >= 500) {
      this.lastOverlaySync = now;
      overlay.sync();
    }
    if (!this.checkLayout()) return;
    const layout = overlay.layout;
    if (!layout || layout.geometry.visibleWidth <= 0 || layout.geometry.visibleHeight <= 0) return;
    this.handleBudgetEvent(this.budget.checkWatchdog(now));
    if (!this.canRender()) return;

    const slot = ring.acquire();
    if (!slot) {
      this.ringDrops++;
      this.handleBudgetEvent(this.budget.ringFull(now));
      return;
    }
    this.budget.ringAvailable();
    let slotHeld = true;
    let frame: VideoFrame | null = null;
    try {
      try {
        frame = new VideoFrame(this.video);
      } catch (err) {
        const kind = classifyVideoFrameError(err);
        if (kind === 'fatal') throw err;
        ring.abandon(slot);
        slotHeld = false;
        if (kind === 'tainted') this.onTainted();
        return;
      }
      if (isHdrTransfer(frame.colorSpace?.transfer)) {
        ring.abandon(slot);
        slotHeld = false;
        this.context.latches.latchHdr(this.video);
        this.stop('hdr');
        this.context.requestEvaluation();
        return;
      }
      const inWidth = frame.displayWidth;
      const inHeight = frame.displayHeight;
      const canvas = overlay.canvas;
      const active = this.pipeline;
      const wanted = this.budget.tier;
      if (
        !active ||
        active.inWidth !== inWidth ||
        active.inHeight !== inHeight ||
        active.outWidth !== canvas.width ||
        active.outHeight !== canvas.height
      ) {
        // First frame, source resolution change or canvas backing change: the current pipeline
        // cannot draw this frame. Keep the native video visible until the new one presents.
        ring.abandon(slot);
        slotHeld = false;
        overlay.restoreVideo();
        this.requestPipeline({ tier: wanted, inWidth, inHeight, outWidth: canvas.width, outHeight: canvas.height });
        return;
      }
      if (active.tier !== wanted) {
        // Downgrade (or quality change): build the new tier while the current one keeps drawing.
        const out = planOutput(inWidth, inHeight, layout.target, wanted);
        this.requestPipeline({ tier: wanted, inWidth, inHeight, outWidth: out.width, outHeight: out.height });
      }
      slotHeld = false;
      this.encodeAndSubmit(frame, slot, active, now);
    } catch (err) {
      if (slotHeld) ring.abandon(slot);
      this.fail(err);
    } finally {
      frame?.close();
    }
  }

  private encodeAndSubmit(frame: VideoFrame, slot: FrameSlot, active: ActivePipeline, now: number): void {
    const gpu = this.gpu as GpuContext;
    const ring = this.ring as FrameRing;
    const device = gpu.device;
    const generation = this.gpuGeneration;
    const presentationEpoch = (this.overlay as CanvasOverlay).presentationEpoch;
    const token = ++this.frameToken;
    const completion: CompletedFrame = { token, presentationEpoch, validated: false, presented: false };
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let submitted = false;
    try {
      const external = device.importExternalTexture({ source: frame });
      const view = (this.canvasContext as GPUCanvasContext).getCurrentTexture().createView();
      const encoder = device.createCommandEncoder({ label: 'webgpu-vsr-frame' });
      active.impl.executeSlot(encoder, external, view, slot);
      ring.encodeResolve(encoder, slot);
      const done = ring.submit(slot, [encoder.finish()]);
      submitted = true;
      this.pendingStill = false;
      if (active.firstToken === null) active.firstToken = token;
      this.budget.frameSubmitted(token, active.tier, now);
      done.then(
        (timing) => this.onFrameDone(completion, generation, active, timing),
        (err: unknown) => this.onFrameRejected(token, generation, err),
      );
      this.armWatchdog();
    } finally {
      if (!submitted) ring.abandon(slot);
      const scopes = Promise.all([device.popErrorScope(), device.popErrorScope()]);
      scopes.then(
        ([validation, memory]) => this.onFrameValidated(generation, completion, active, submitted, validation ?? memory),
        () => undefined, // device lost: the loss handler takes over
      );
    }
  }

  private onFrameValidated(generation: number, completion: CompletedFrame, active: ActivePipeline, submitted: boolean, error: GPUError | null): void {
    if (this.destroyed || generation !== this.gpuGeneration) return;
    if (error) {
      this.fail(new Error(`GPU error in ${active.impl.id}: ${error.message}`));
      return;
    }
    if (!submitted) return;
    completion.validated = true;
    // Only the first frame's scopes validate the pipeline, even if later scopes resolve first.
    if (completion.token === active.firstToken) active.validated = true;
    this.presentCompleted(active);
  }

  private presentCompleted(active: ActivePipeline): void {
    const completion = active.latestCompleted;
    const overlay = this.overlay;
    if (this.pipeline !== active || !this.canRender() || !overlay || !completion) return;
    if (completion.presentationEpoch !== overlay.presentationEpoch) {
      if (completion.token === this.frameToken) this.kick();
      return;
    }
    if (active.validated && completion.validated && !completion.presented) {
      completion.presented = true;
      this.present();
    }
  }

  private present(): void {
    const overlay = this.overlay as CanvasOverlay;
    const wasHidden = overlay.videoHidden;
    overlay.markFramePresented();
    this.presentedFrames++;
    const nowHidden = !wasHidden && overlay.videoHidden;
    // The first frame on the canvas is a state change: its telemetry goes out as soon as allowed.
    this.reportTelemetry(nowHidden);
    if (nowHidden) this.context.reportActive();
  }

  private onFrameDone(completion: CompletedFrame, generation: number, active: ActivePipeline, timing: FrameTiming): void {
    if (this.destroyed || generation !== this.gpuGeneration) return;
    // Synthetic latency (developer diagnostics) acts like slower GPU work: it lengthens both the
    // execution and the submit -> done latency the controller sees.
    const token = completion.token;
    if (!active.latestCompleted || token > active.latestCompleted.token) active.latestCompleted = completion;
    this.presentCompleted(active);
    const extra = syntheticLatencyFor(this.services.diagnostics(), active.impl.id);
    const event = this.budget.frameCompleted(token, this.services.now(), {
      gpuMs: timing.gpuTimeMs + extra,
      latencyMs: timing.latencyMs + extra,
      source: timing.timingSource,
    });
    this.handleBudgetEvent(event);
    this.reportTelemetry(false);
    this.retryStill();
  }

  private retryStill(): void {
    if (this.pendingStill) this.renderFrame();
  }

  private onFrameRejected(token: number, generation: number, err: unknown): void {
    if (this.destroyed || generation !== this.gpuGeneration) return;
    this.budget.frameCompleted(token, this.services.now(), null);
    if (this.rejectedFrames++ === 0) console.warn(`${LOG_PREFIX} frame completion failed`, err);
    this.retryStill();
  }

  // ---- pipelines --------------------------------------------------------------------------

  private requestPipeline(spec: PipelineSpec): void {
    if (this.pendingSpec && sameSpec(this.pendingSpec, spec)) return;
    if (this.pipeline && sameSpec(this.pipeline, spec)) {
      // Back to what is already running: drop any build in progress.
      this.pendingSpec = null;
      this.configRevision++;
      return;
    }
    this.pendingSpec = spec;
    const revision = ++this.configRevision;
    void this.buildPipeline(spec, this.gpu as GpuContext, this.gpuGeneration, revision);
  }

  private async buildPipeline(spec: PipelineSpec, gpu: GpuContext, generation: number, revision: number): Promise<void> {
    const stale = (): boolean => this.destroyed || revision !== this.configRevision || generation !== this.gpuGeneration || this.stopReason !== null;
    let impl: RenderPipeline | null = null;
    try {
      impl = this.services.createPipeline(spec.tier, gpu);
      await impl.initialize(gpu.device, spec.inWidth, spec.inHeight, spec.outWidth, spec.outHeight);
      impl.setSharpness(this.settings.sharpness);
    } catch (err) {
      impl?.destroy();
      if (stale()) return;
      this.pendingSpec = null;
      this.fail(err);
      return;
    }
    if (stale()) {
      impl.destroy();
      return;
    }
    this.pendingSpec = null;
    const previous = this.pipeline;
    this.pipeline = { ...spec, impl, validated: false, firstToken: null, latestCompleted: null };
    this.pipelineInits++;
    // VSR <-> Quality 0 changes the canvas backing store (decideOutput); in-flight frames of the
    // previous pipeline keep their buffers alive until the GPU finished them.
    this.overlay?.setInternalScale(internalScaleForTier(spec.tier));
    previous?.impl.destroy();
    this.reportTelemetry(true);
    this.pendingStill = true;
    this.renderFrame();
  }

  // ---- budget -----------------------------------------------------------------------------

  private handleBudgetEvent(event: BudgetEvent | null): void {
    if (!event || this.destroyed) return;
    if (event.type === 'downgrade') {
      console.warn(`${LOG_PREFIX} ${event.cause === 'budget' ? 'frame budget exceeded' : event.cause === 'timeout' ? 'GPU frame timed out' : 'GPU queue full'}: ${pipelineIdForTier(event.from)} -> ${pipelineIdForTier(event.to)}`);
      // Not synchronously: this may run inside renderFrame().
      queueMicrotask(() => this.kick());
      return;
    }
    console.warn(`${LOG_PREFIX} ${pipelineIdForTier(event.tier)} still exceeds the frame budget (${event.cause}); processing throttled`);
    this.stop('throttled');
    this.context.requestEvaluation();
  }

  /**
   * Watchdog heartbeat: while frames are in flight the controller is observed at least every
   * watchdogHeartbeatMs (frame callbacks observe too). Only observed time counts towards the
   * 250 ms timeout, so a main-thread stall is not mistaken for a hung GPU (timing.ts).
   */
  private armWatchdog(): void {
    if (this.watchdogTimer !== null) return;
    const next = this.budget.nextWatchdogCheck();
    if (next === null) return;
    this.watchdogTimer = setTimeout(
      () => {
        this.watchdogTimer = null;
        if (this.destroyed || this.stopReason !== null || this.deviceLost) return;
        this.handleBudgetEvent(this.budget.checkWatchdog(this.services.now()));
        this.armWatchdog();
      },
      Math.max(0, next - this.services.now()) + 1,
    );
  }

  // ---- telemetry --------------------------------------------------------------------------

  private telemetry(): SessionTelemetry {
    const stats = this.budget.stats();
    const p = this.pipeline;
    const round = (v: number, digits: number): number => (Number.isFinite(v) && v > 0 ? Number(v.toFixed(digits)) : 0);
    return {
      inputWidth: p?.inWidth ?? 0,
      inputHeight: p?.inHeight ?? 0,
      outputWidth: p?.outWidth ?? 0,
      outputHeight: p?.outHeight ?? 0,
      fps: round(this.presentation.fps, 1),
      gpuTimeMeanMs: round(stats.meanMs, 3),
      gpuTimeP95Ms: round(stats.p95Ms, 3),
      droppedFrames: this.ringDrops + this.presentation.droppedFrames,
      activeModel: p ? p.impl.id : pipelineIdForTier(this.budget.tier),
      timingSource: stats.timingSource ?? (this.gpu?.hasTimestamps ? 'timestamp' : 'cpu'),
      presentedFrames: this.presentedFrames,
      pipelineInits: this.pipelineInits,
    };
  }

  /** Sends telemetry if the 2 Hz limit allows; `soon` schedules it for the earliest allowed time. */
  private reportTelemetry(soon: boolean): void {
    // Nothing is presented while detached; reattach() resumes the reports.
    if (this.destroyed || this.detached) return;
    const now = this.services.now();
    if (this.telemetryLimiter.ready(now)) {
      this.telemetryLimiter.mark(now);
      this.context.reportTelemetry(this.telemetry());
      return;
    }
    if (!soon || this.telemetryTimer !== null) return;
    this.telemetryTimer = setTimeout(() => {
      this.telemetryTimer = null;
      this.reportTelemetry(true);
    }, this.telemetryLimiter.delay(now) + 1);
  }

  // ---- failure paths ----------------------------------------------------------------------

  /** Stops processing for good (until destroyed or, when throttled, a quality/source change). */
  private stop(reason: StopReason): void {
    if (this.stopReason !== null) return;
    this.stopReason = reason;
    this.cancelLoop();
    if (this.watchdogTimer !== null) clearTimeout(this.watchdogTimer);
    this.watchdogTimer = null;
    this.overlay?.restoreVideo();
  }

  private resume(): void {
    this.stopReason = null;
    // Builds discarded while stopped must not block a new request for the same spec.
    this.pendingSpec = null;
    this.configRevision++;
    this.ensureLoop();
  }

  private fail(err: unknown): void {
    if (this.destroyed || this.stopReason !== null) return;
    console.error(`${LOG_PREFIX} video session failed`, err);
    this.stop('failed');
    this.context.reportError(errorMessage(err));
  }

  private readonly onDrmEvent = (): void => {
    if (this.destroyed) return;
    this.stop('drm');
    this.context.requestEvaluation();
  };

  private onTainted(): void {
    this.context.latches.latchTaint(this.video);
    this.stop('tainted');
    this.context.requestEvaluation();
  }

  private readonly onDeviceLost = (info: { generation: number }): void => {
    const manager = this.services.deviceManager;
    const generation = info.generation;
    if (this.destroyed || this.stopReason !== null || generation !== manager.generation || generation < this.deviceGeneration) return;
    this.deviceGeneration = generation;
    this.deviceLost = true;
    this.gpuGeneration++;
    this.cancelLoop();
    if (this.watchdogTimer !== null) clearTimeout(this.watchdogTimer);
    this.watchdogTimer = null;
    this.overlay?.restoreVideo();
    this.configRevision++;
    this.pendingSpec = null;
    this.pipeline?.impl.destroy();
    this.pipeline = null;
    this.ring?.destroy();
    this.ring = null;
    this.unconfigureCanvas();
    this.gpu = null;
    this.budget.abandonInFlight();
    void manager.whenRecovered().then((result) => {
      if (this.destroyed || this.stopReason !== null || !this.deviceLost || generation !== this.deviceGeneration ||
          result.generation !== generation || manager.generation !== generation || result.success || manager.current !== null) return;
      this.services.markGpuUnavailable();
      this.stop('gpu-unavailable');
      this.context.requestEvaluation();
    });
  };

  private readonly onDeviceRestored = (gpu: GpuContext, generation: number): void => {
    if (this.destroyed || !this.holdsDevice || !this.deviceLost || this.stopReason !== null) return;
    const manager = this.services.deviceManager;
    if (generation !== this.deviceGeneration || generation !== manager.generation || manager.current !== gpu) return;
    this.deviceLost = false;
    if (TIER_LEVEL[this.budget.tier] > TIER_LEVEL[selectTier(this.settings, gpu)]) this.budget.reset(selectTier(this.settings, gpu));
    this.bindDevice(gpu);
  };

  private unconfigureCanvas(): void {
    try {
      this.canvasContext?.unconfigure();
    } catch {
      // the context's device is already gone
    }
  }

  private clearTimers(): void {
    if (this.watchdogTimer !== null) clearTimeout(this.watchdogTimer);
    if (this.telemetryTimer !== null) clearTimeout(this.telemetryTimer);
    this.watchdogTimer = null;
    this.telemetryTimer = null;
  }
}

// ---------------------------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------------------------

export interface GpuSessionFactoryOptions {
  deviceManager?: GpuDeviceManager;
  createPipeline?: (tier: QualityTier, gpu: GpuContext) => RenderPipeline;
  now?: () => number;
  overlayEnvironment?: OverlayEnvironment;
}

/** Creates a {@link VideoSRSession} per eligible video; owns the frame-wide GPU availability verdict. */
export class GpuSessionFactory implements VideoSessionFactory {
  private gpuUnavailable = false;
  private diagnostics: Diagnostics = defaultDiagnostics();
  private readonly services: SessionServices;

  constructor(options: GpuSessionFactoryOptions = {}) {
    this.services = {
      deviceManager: options.deviceManager ?? gpuDeviceManager,
      createPipeline: options.createPipeline ?? createPipeline,
      diagnostics: () => this.diagnostics,
      markGpuUnavailable: () => {
        this.gpuUnavailable = true;
      },
      now: options.now ?? (() => performance.now()),
      overlayEnvironment: options.overlayEnvironment,
    };
  }

  /** Gate point 8: WebGPU exists in this frame and no device acquisition / recovery has failed. */
  isGpuAvailable(): boolean {
    return !this.gpuUnavailable && hasWebGpuApi();
  }

  /** Developer diagnostics (synthetic GPU latency), see src/shared/diagnostics.ts. */
  setDiagnostics(diagnostics: Diagnostics): void {
    this.diagnostics = diagnostics;
  }

  create(context: VideoSessionContext): VideoSessionHandle {
    return new VideoSRSession(context, this.services);
  }
}
