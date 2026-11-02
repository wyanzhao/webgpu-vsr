/**
 * Frame-budget accounting and the automatic quality downgrade controller
 * (docs/IMPLEMENTATION_NOTES.md section 6).
 *
 * Two independent protections decide when the session must step down one quality tier
 * (medium -> low -> quality0_bicubic) and, below Quality 0, stop processing altogether
 * (`throttle`, which fails eligibility point 9 and restores the native video):
 *
 * 1. Sustained budget violation. A completed frame is LATE when it missed two frame slots
 *    (submit -> GPU completion latency above `lateLatencyRatio` (2) frame budgets) while its own
 *    GPU execution used more than `thresholdRatio` (0.70) of the budget. The rolling window holds the late flags of
 *    the last `windowSize` (60) frames; while more than `lateFraction` (half) of them are late a
 *    violation counter rises by one per frame, otherwise it decays by one ("consecutive-ish").
 *    At `sustainFrames` (60) the controller downgrades. The frame budget is
 *    1000 / fps where fps is the video's measured cadence ({@link FrameRateEstimator}), 60 fps
 *    until known.
 *
 *    Why not the mean GPU time against 0.70 x budget: under paced playback (one frame per video
 *    frame) Apple GPUs lower their clock until the work fills about 75-80 % of the interval, so
 *    the timestamp time of a frame follows the clock governor, not the workload (low at
 *    720p60 -> 1600x900: 3.4 ms back to back, 12.5 ms paced, docs/IMPLEMENTATION_NOTES.md 6.7).
 *    A mean-time threshold of 70 % then fires on pipelines with several times the needed
 *    capacity. A frame that completes inside its slot is on time at whatever clock the GPU
 *    chose; a GPU that cannot keep up even at full clock misses the slots. The 0.70 GPU-time
 *    condition keeps frames that were late for reasons other than GPU work (a busy main thread
 *    delaying the completion callback of a light frame) from counting.
 *
 *    Why two slots and not one: with the clock governor filling 75-80 % of each interval, the
 *    paced GPU time always passes the 0.70 condition and completion jitter puts 20-50 % of
 *    on-time frames just past one interval (low 1080p60 -> 3200x1800 on an M5 Max: latency
 *    p50 16 ms, max 20 ms against a 16.7 ms slot, 0 drops), which stepped a pipeline with
 *    twice the needed capacity down to Quality 0. A GPU that cannot keep up backs up the
 *    3-slot ring, so its frames wait behind the ones queued ahead and exceed two slots.
 * 2. Timeout watchdog. Submitted frames are tracked in queue (FIFO) order. A frame's clock starts
 *    when it reaches the head of the GPU queue (its submission, or the completion of the frame
 *    before it). It advances only by time the page's main thread was observed running: the
 *    session observes on every frame callback and on a `watchdogHeartbeatMs` (50 ms) timer, and a
 *    gap between observations longer than `maxObservationGapMs` (100 ms) is a main-thread stall
 *    (page jank, GC, devtools) during which the completion callback could not run either; the
 *    gap is not charged to the GPU. If the head frame has waited `watchdogTimeoutMs` (250 ms) of
 *    observed time, or the frame ring was full (3 frames in flight) for `ringFullLimit`
 *    consecutive frames, the controller downgrades at once.
 *
 * Hysteresis: every tier change (and {@link BudgetController.reset}) clears the window, the
 * violation and ring-full counters; timeout and ring-full steps also restart the watchdog clock.
 * The first `cooldownSamples` frames of a new tier are warm-up and do not count towards a budget
 * decision. Samples rendered by any other tier (frames still in flight from before a switch) are
 * ignored.
 *
 * The controller never upgrades by itself: its tier starts at, and only ever moves below, the
 * ceiling chosen by the user (`reset(ceiling)` is the only way up).
 */
import type { QualityTier, TimingSource } from './types';

/** Lightest first. */
export const TIER_ORDER: readonly QualityTier[] = ['quality0_bicubic', 'low', 'medium'];

export function tierRank(tier: QualityTier): number {
  return TIER_ORDER.indexOf(tier);
}

/** The next lighter tier, or null below Quality 0. */
export function lowerTier(tier: QualityTier): QualityTier | null {
  const rank = tierRank(tier);
  return rank > 0 ? (TIER_ORDER[rank - 1] as QualityTier) : null;
}

export const DEFAULT_FRAME_RATE = 60;

export interface BudgetControllerOptions {
  /** Rolling window length (frames). */
  windowSize?: number;
  /** Fraction of the frame budget a late frame's own GPU execution must exceed to count. */
  thresholdRatio?: number;
  /** Frame budgets the submit -> completion latency must exceed for a frame to be late. */
  lateLatencyRatio?: number;
  /** Fraction of late frames in the window above which the violation counter rises. */
  lateFraction?: number;
  /** Violation count that triggers a downgrade. */
  sustainFrames?: number;
  /** Warm-up frames after a tier change that are ignored by the budget decision. */
  cooldownSamples?: number;
  /** Hard per-frame completion timeout (ms of observed main-thread time). */
  watchdogTimeoutMs?: number;
  /** Interval of the session's watchdog observations while frames are in flight (ms). */
  watchdogHeartbeatMs?: number;
  /** A longer gap between two observations is a main-thread stall and is not charged (ms). */
  maxObservationGapMs?: number;
  /** Consecutive ring-full drops that trigger a downgrade. */
  ringFullLimit?: number;
  /** Frame rate assumed until the cadence is known. */
  defaultFrameRate?: number;
}

export const BUDGET_DEFAULTS: Readonly<Required<BudgetControllerOptions>> = Object.freeze({
  windowSize: 60,
  thresholdRatio: 0.7,
  lateLatencyRatio: 2,
  lateFraction: 0.5,
  sustainFrames: 60,
  cooldownSamples: 30,
  watchdogTimeoutMs: 250,
  watchdogHeartbeatMs: 50,
  maxObservationGapMs: 100,
  ringFullLimit: 15,
  defaultFrameRate: DEFAULT_FRAME_RATE,
});

/** One completed frame as the controller sees it. */
export interface FrameSample {
  /** GPU execution time (timestamp query), or the CPU submit -> done time without timestamps. */
  gpuMs: number;
  /** Submit -> GPU completion latency (ms); equals gpuMs for CPU timing. */
  latencyMs: number;
  source: TimingSource | null;
}

export type BudgetCause = 'budget' | 'timeout' | 'ring-full';

export type BudgetEvent =
  | { type: 'downgrade'; from: QualityTier; to: QualityTier; cause: BudgetCause }
  | { type: 'throttle'; tier: QualityTier; cause: BudgetCause };

export interface TimingStats {
  /** Samples in the statistics window. */
  samples: number;
  /** GPU time statistics (telemetry only: paced GPU times follow the clock governor). */
  meanMs: number;
  /** Nearest-rank 95th percentile. */
  p95Ms: number;
  /** Fraction of late frames in the decision window (0 during warm-up). */
  lateFraction: number;
  /** Source of the most recent sample (null before the first one). */
  timingSource: TimingSource | null;
}

/** Nearest-rank percentile (p in (0, 1]) of unsorted values; 0 for an empty list. */
export function nearestRankPercentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length, Math.max(1, Math.ceil(p * sorted.length)));
  return sorted[rank - 1] as number;
}

/** Fixed-capacity window of the most recent values. */
export class RollingWindow {
  private readonly values: number[] = [];
  private sum = 0;

  constructor(readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError(`window capacity must be a positive integer, got ${capacity}`);
  }

  push(value: number): void {
    this.values.push(value);
    this.sum += value;
    if (this.values.length > this.capacity) this.sum -= this.values.shift() as number;
  }

  clear(): void {
    this.values.length = 0;
    this.sum = 0;
  }

  get length(): number {
    return this.values.length;
  }

  get mean(): number {
    return this.values.length === 0 ? 0 : this.sum / this.values.length;
  }

  percentile(p: number): number {
    return nearestRankPercentile(this.values, p);
  }
}

interface PendingFrame {
  token: number;
  tier: QualityTier;
  submittedAt: number;
}

function positiveOr(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : fallback;
}

export class BudgetController {
  private readonly opts: Required<BudgetControllerOptions>;
  private ceilingTier: QualityTier;
  private currentTier: QualityTier;
  private isThrottled = false;
  private frameRate: number | null = null;
  /** Decision window: 1 per late frame, 0 per frame on time (warm-up excluded). */
  private readonly window: RollingWindow;
  /** GPU-time statistics window for telemetry (warm-up included). */
  private readonly display: RollingWindow;
  private lastSource: TimingSource | null = null;
  private violations = 0;
  private warmupLeft: number;
  private ringFullStreak = 0;
  private pending: PendingFrame[] = [];
  /** Observed main-thread time the queue-head frame has been waiting (watchdog clock). */
  private waitedMs = 0;
  private lastObservedAt = Number.NEGATIVE_INFINITY;

  constructor(ceiling: QualityTier, options: BudgetControllerOptions = {}) {
    const lateFraction = options.lateFraction;
    this.opts = {
      windowSize: Math.round(positiveOr(options.windowSize, BUDGET_DEFAULTS.windowSize)),
      thresholdRatio: positiveOr(options.thresholdRatio, BUDGET_DEFAULTS.thresholdRatio),
      lateLatencyRatio: positiveOr(options.lateLatencyRatio, BUDGET_DEFAULTS.lateLatencyRatio),
      lateFraction: lateFraction !== undefined && lateFraction >= 0 && lateFraction < 1 ? lateFraction : BUDGET_DEFAULTS.lateFraction,
      sustainFrames: Math.round(positiveOr(options.sustainFrames, BUDGET_DEFAULTS.sustainFrames)),
      cooldownSamples: Math.max(0, Math.round(options.cooldownSamples ?? BUDGET_DEFAULTS.cooldownSamples)),
      watchdogTimeoutMs: positiveOr(options.watchdogTimeoutMs, BUDGET_DEFAULTS.watchdogTimeoutMs),
      watchdogHeartbeatMs: positiveOr(options.watchdogHeartbeatMs, BUDGET_DEFAULTS.watchdogHeartbeatMs),
      maxObservationGapMs: positiveOr(options.maxObservationGapMs, BUDGET_DEFAULTS.maxObservationGapMs),
      ringFullLimit: Math.round(positiveOr(options.ringFullLimit, BUDGET_DEFAULTS.ringFullLimit)),
      defaultFrameRate: positiveOr(options.defaultFrameRate, BUDGET_DEFAULTS.defaultFrameRate),
    };
    if (this.opts.watchdogHeartbeatMs >= this.opts.maxObservationGapMs) {
      throw new RangeError('watchdogHeartbeatMs must be shorter than maxObservationGapMs');
    }
    this.window = new RollingWindow(this.opts.windowSize);
    this.display = new RollingWindow(this.opts.windowSize);
    this.ceilingTier = ceiling;
    this.currentTier = ceiling;
    this.warmupLeft = this.opts.cooldownSamples;
  }

  get options(): Readonly<Required<BudgetControllerOptions>> {
    return this.opts;
  }

  /** The tier the session should be running. */
  get tier(): QualityTier {
    return this.currentTier;
  }

  /** The user's choice; the controller never goes above it. */
  get ceiling(): QualityTier {
    return this.ceilingTier;
  }

  /** True once processing must stop (Quality 0 still over budget or timing out). */
  get throttled(): boolean {
    return this.isThrottled;
  }

  get violationCount(): number {
    return this.violations;
  }

  /** Frame budget in ms from the measured cadence (or the default frame rate). */
  get frameBudgetMs(): number {
    return 1000 / (this.frameRate ?? this.opts.defaultFrameRate);
  }

  /** GPU time a late frame must exceed to count as a budget violation. */
  get thresholdMs(): number {
    return this.opts.thresholdRatio * this.frameBudgetMs;
  }

  /** Measured frames per (wall-clock) second of the video, or null when unknown. */
  setFrameRate(fps: number | null): void {
    this.frameRate = fps !== null && Number.isFinite(fps) && fps > 0 ? fps : null;
  }

  /**
   * Back to the user's tier (settings or source change). Clears every counter, the watchdog and
   * the throttled state.
   */
  reset(ceiling: QualityTier = this.ceilingTier): void {
    this.ceilingTier = ceiling;
    this.currentTier = ceiling;
    this.isThrottled = false;
    this.pending = [];
    this.lastSource = null;
    this.lastObservedAt = Number.NEGATIVE_INFINITY;
    this.restartTier(true);
  }

  /** Telemetry statistics of the current tier's recent frames. */
  stats(): TimingStats {
    return {
      samples: this.display.length,
      meanMs: this.display.mean,
      p95Ms: this.display.percentile(0.95),
      lateFraction: this.window.mean,
      timingSource: this.lastSource,
    };
  }

  /** Whether a completed frame missed its slots because of its own GPU work (see the module comment). */
  isLate(sample: Pick<FrameSample, 'gpuMs' | 'latencyMs'>): boolean {
    return sample.latencyMs > this.opts.lateLatencyRatio * this.frameBudgetMs && sample.gpuMs > this.thresholdMs;
  }

  /**
   * One completed frame rendered by `tier`. Returns the decision it triggered. Samples from
   * another tier, non-finite or negative times, and samples after throttling are ignored.
   */
  recordSample(tier: QualityTier, sample: FrameSample): BudgetEvent | null {
    const { gpuMs, latencyMs, source } = sample;
    if (this.isThrottled || tier !== this.currentTier) return null;
    if (!Number.isFinite(gpuMs) || gpuMs < 0 || !Number.isFinite(latencyMs) || latencyMs < 0) return null;
    this.display.push(gpuMs);
    if (source !== null) this.lastSource = source;
    if (this.warmupLeft > 0) {
      this.warmupLeft--;
      return null;
    }
    this.window.push(this.isLate(sample) ? 1 : 0);
    if (this.window.mean > this.opts.lateFraction) this.violations++;
    else this.violations = Math.max(0, this.violations - 1);
    if (this.violations >= this.opts.sustainFrames) return this.stepDown('budget');
    return null;
  }

  /** A frame rendered by `tier` was submitted to the GPU queue at `nowMs`. */
  frameSubmitted(token: number, tier: QualityTier, nowMs: number): void {
    if (this.isThrottled) return;
    this.observe(nowMs);
    if (this.pending.length === 0) this.waitedMs = 0;
    this.pending.push({ token, tier, submittedAt: nowMs });
  }

  /**
   * The frame `token` completed (its callback ran at `nowMs`) with `sample` (null when no time
   * could be measured). Advances the watchdog and records the sample.
   */
  frameCompleted(token: number, nowMs: number, sample: FrameSample | null): BudgetEvent | null {
    const index = this.pending.findIndex((p) => p.token === token);
    if (index < 0) return null;
    const [frame] = this.pending.splice(index, 1) as [PendingFrame];
    this.observe(nowMs);
    // Progress: the next frame in the queue starts its own clock now.
    if (index === 0) this.waitedMs = 0;
    return sample === null ? null : this.recordSample(frame.tier, sample);
  }

  /**
   * Forgets every in-flight frame (their device is gone) and the ring-full streak; the tier,
   * the statistics and the throttled state are kept.
   */
  abandonInFlight(): void {
    this.pending = [];
    this.ringFullStreak = 0;
    this.waitedMs = 0;
  }

  /** Frames submitted and not yet completed. */
  get inFlight(): number {
    return this.pending.length;
  }

  /** Observed time the queue-head frame has waited for completion (ms). */
  get watchdogWaitedMs(): number {
    return this.pending.length > 0 ? this.waitedMs : 0;
  }

  /**
   * When the session should next observe for the watchdog (absolute ms), or null when nothing is
   * in flight: one heartbeat after the last observation, or earlier when the head frame's
   * remaining time is shorter.
   */
  nextWatchdogCheck(): number | null {
    if (this.pending.length === 0 || this.isThrottled || !Number.isFinite(this.lastObservedAt)) return null;
    const remaining = Math.max(0, this.opts.watchdogTimeoutMs - this.waitedMs);
    return this.lastObservedAt + Math.min(this.opts.watchdogHeartbeatMs, remaining);
  }

  /** Observes the main thread at `nowMs` and checks the head frame against the hard timeout. */
  checkWatchdog(nowMs: number): BudgetEvent | null {
    this.observe(nowMs);
    if (this.pending.length === 0 || this.isThrottled) return null;
    if (this.waitedMs <= this.opts.watchdogTimeoutMs) return null;
    return this.stepDown('timeout');
  }

  /** A frame was dropped because every ring slot was in flight. */
  ringFull(nowMs: number): BudgetEvent | null {
    if (this.isThrottled) return null;
    this.observe(nowMs);
    this.ringFullStreak++;
    if (this.ringFullStreak >= this.opts.ringFullLimit) return this.stepDown('ring-full');
    return null;
  }

  /** A ring slot was available (ends a ring-full streak). */
  ringAvailable(): void {
    this.ringFullStreak = 0;
  }

  get ringFullCount(): number {
    return this.ringFullStreak;
  }

  /**
   * Advances the watchdog clock by the time since the previous observation, unless that gap is a
   * main-thread stall (longer than maxObservationGapMs): during a stall the completion callback
   * could not run, so the GPU is not charged for it.
   */
  private observe(nowMs: number): void {
    if (!Number.isFinite(nowMs)) return;
    if (Number.isFinite(this.lastObservedAt)) {
      const dt = nowMs - this.lastObservedAt;
      if (dt <= 0) return;
      if (this.pending.length > 0 && dt <= this.opts.maxObservationGapMs) this.waitedMs += dt;
    }
    this.lastObservedAt = nowMs;
  }

  private stepDown(cause: BudgetCause): BudgetEvent {
    const from = this.currentTier;
    const to = lowerTier(from);
    if (to === null) {
      this.isThrottled = true;
      this.pending = [];
      return { type: 'throttle', tier: from, cause };
    }
    this.currentTier = to;
    // Frames of the previous tier may still be queued ahead; they keep their place in the FIFO.
    // After a timeout or ring-full step the next one needs another full period without progress.
    this.restartTier(cause !== 'budget');
    return { type: 'downgrade', from, to, cause };
  }

  private restartTier(restartWatchdog: boolean): void {
    this.window.clear();
    this.display.clear();
    this.violations = 0;
    this.ringFullStreak = 0;
    this.warmupLeft = this.opts.cooldownSamples;
    if (restartWatchdog) this.waitedMs = 0;
  }
}

/**
 * Estimates the video's frame rate from requestVideoFrameCallback metadata: the median of
 * `mediaTime` deltas divided by the `presentedFrames` delta (frames the compositor presented in
 * between), divided by the playback rate so the budget follows wall-clock time. Seeks, pauses
 * and discontinuities (non-positive deltas, or a per-frame interval outside 1/240 .. 1/5 s) are
 * skipped.
 */
export class FrameRateEstimator {
  private last: { mediaTime: number; presentedFrames: number | null } | null = null;
  private readonly intervals: number[] = [];

  constructor(
    private readonly capacity = 32,
    private readonly minSamples = 4,
  ) {}

  record(metadata: { mediaTime?: number; presentedFrames?: number }, playbackRate = 1): void {
    const mediaTime = metadata.mediaTime;
    if (typeof mediaTime !== 'number' || !Number.isFinite(mediaTime)) return;
    const presented = typeof metadata.presentedFrames === 'number' && Number.isFinite(metadata.presentedFrames) ? metadata.presentedFrames : null;
    const prev = this.last;
    this.last = { mediaTime, presentedFrames: presented };
    if (!prev) return;
    const frames = presented !== null && prev.presentedFrames !== null ? presented - prev.presentedFrames : 1;
    const dMedia = mediaTime - prev.mediaTime;
    if (!(frames >= 1) || !(dMedia > 0)) return;
    const perFrame = dMedia / frames;
    if (perFrame < 1 / 240 || perFrame > 1 / 5) return;
    const rate = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
    this.intervals.push(perFrame / rate);
    if (this.intervals.length > this.capacity) this.intervals.shift();
  }

  reset(): void {
    this.last = null;
    this.intervals.length = 0;
  }

  /** Frames per wall-clock second, or null until enough intervals were seen. */
  get frameRate(): number | null {
    if (this.intervals.length < this.minSamples) return null;
    const sorted = [...this.intervals].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    const median = sorted.length % 2 === 1 ? (sorted[mid] as number) : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
    return 1 / median;
  }
}
