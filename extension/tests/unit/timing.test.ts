import { describe, expect, it } from 'vitest';
import {
  BUDGET_DEFAULTS,
  BudgetController,
  FrameRateEstimator,
  RollingWindow,
  TIER_ORDER,
  lowerTier,
  nearestRankPercentile,
  tierRank,
  type BudgetEvent,
  type FrameSample,
} from '../../src/pipeline/timing';
import type { QualityTier } from '../../src/pipeline/types';

/**
 * A completed frame: GPU execution `gpuMs`, submit -> done `latencyMs` (default: twice the GPU
 * time, a frame that waited behind one queued frame of the same cost).
 */
const S = (gpuMs: number, latencyMs = 2 * gpuMs): FrameSample => ({ gpuMs, latencyMs, source: 'timestamp' });

/** Records `n` samples rendered by `tier`; returns the non-null events. */
function feed(c: BudgetController, tier: QualityTier, gpuMs: number, n: number, latencyMs = 2 * gpuMs): BudgetEvent[] {
  const events: BudgetEvent[] = [];
  for (let i = 0; i < n; i++) {
    const e = c.recordSample(tier, S(gpuMs, latencyMs));
    if (e) events.push(e);
  }
  return events;
}

/** Watchdog observations (frame callbacks / heartbeat timer) every `step` ms in (from, to]. */
function observe(c: BudgetController, from: number, to: number, step = 50): BudgetEvent[] {
  const events: BudgetEvent[] = [];
  for (let t = from + step; t <= to; t += step) {
    const e = c.checkWatchdog(t);
    if (e) events.push(e);
  }
  return events;
}

const T60 = (0.7 * 1000) / 60; // 11.666... ms
const B60 = 1000 / 60; // 16.666... ms

describe('tier helpers', () => {
  it('orders tiers lightest first and steps down one at a time', () => {
    expect(TIER_ORDER).toEqual(['quality0_bicubic', 'low', 'medium']);
    expect(tierRank('medium')).toBe(2);
    expect(lowerTier('medium')).toBe('low');
    expect(lowerTier('low')).toBe('quality0_bicubic');
    expect(lowerTier('quality0_bicubic')).toBeNull();
  });
});

describe('statistics', () => {
  it('nearest-rank p95', () => {
    expect(nearestRankPercentile([], 0.95)).toBe(0);
    expect(nearestRankPercentile([7], 0.95)).toBe(7);
    // 20 values: rank ceil(0.95 * 20) = 19
    expect(nearestRankPercentile([20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 0.95)).toBe(19);
    // 100 values: rank 95
    expect(nearestRankPercentile(Array.from({ length: 100 }, (_, i) => 100 - i), 0.95)).toBe(95);
    // 10 values: rank ceil(9.5) = 10 -> the maximum
    expect(nearestRankPercentile([1, 1, 1, 1, 1, 1, 1, 1, 1, 50], 0.95)).toBe(50);
  });

  it('rolling window keeps the most recent values', () => {
    const w = new RollingWindow(3);
    for (const v of [1, 2, 3, 4]) w.push(v);
    expect(w.length).toBe(3);
    expect(w.mean).toBe(3); // (2 + 3 + 4) / 3
    expect(w.percentile(0.95)).toBe(4);
    w.clear();
    expect([w.length, w.mean]).toEqual([0, 0]);
    expect(() => new RollingWindow(0)).toThrow(RangeError);
  });

  it('reports mean, p95, late fraction and timing source of the current tier, warm-up included', () => {
    const c = new BudgetController('low');
    for (let i = 1; i <= 20; i++) c.recordSample('low', { gpuMs: i, latencyMs: i, source: i === 20 ? 'cpu' : 'timestamp' });
    expect(c.stats()).toEqual({ samples: 20, meanMs: 10.5, p95Ms: 19, lateFraction: 0, timingSource: 'cpu' });
    feed(c, 'low', 30, 10); // warm-up ends after 30 samples
    feed(c, 'low', 30, 3);
    feed(c, 'low', 2, 1);
    expect(c.stats().lateFraction).toBe(0.75);
    c.reset();
    expect(c.stats()).toEqual({ samples: 0, meanMs: 0, p95Ms: 0, lateFraction: 0, timingSource: null });
  });
});

describe('BudgetController: frame budget', () => {
  it('uses 60 fps until the cadence is known; the GPU-time threshold is 0.70 x 1000 / fps', () => {
    const c = new BudgetController('medium');
    expect(c.frameBudgetMs).toBeCloseTo(B60, 9);
    expect(c.thresholdMs).toBeCloseTo(T60, 9);
    c.setFrameRate(30);
    expect(c.thresholdMs).toBeCloseTo(23.333333, 5);
    c.setFrameRate(24);
    expect(c.thresholdMs).toBeCloseTo(29.166667, 5);
    for (const bad of [null, 0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      c.setFrameRate(bad);
      expect(c.thresholdMs).toBeCloseTo(T60, 9);
    }
  });

  it('a frame is late only if it missed two slots AND its own GPU work used more than 70 % of one', () => {
    const c = new BudgetController('medium');
    expect(BUDGET_DEFAULTS.lateLatencyRatio).toBe(2);
    expect(c.isLate(S(12.5, 34))).toBe(true); // over two slots, GPU-heavy
    expect(c.isLate(S(12.5, 33.3))).toBe(false); // under two slots
    expect(c.isLate(S(13.2, 20))).toBe(false); // paced clock plus completion jitter past one slot
    expect(c.isLate(S(12.5, 13.7))).toBe(false); // paced at a low clock, but on time
    expect(c.isLate(S(16, 16.6))).toBe(false); // fills the slot, still on time
    expect(c.isLate(S(2, 40))).toBe(false); // late, but a light frame: the main thread was busy
    expect(c.isLate(S(11.6, 40))).toBe(false); // GPU time at or under 70 %
    c.setFrameRate(24);
    expect(c.isLate(S(25, 50))).toBe(false); // the same frame is on time at 24 fps
    expect(new BudgetController('low', { lateLatencyRatio: 1 }).isLate(S(12.5, 17))).toBe(true);
  });

  it('the same GPU time is over budget at 60 fps and within it at 24 fps', () => {
    const at24 = new BudgetController('medium');
    at24.setFrameRate(24);
    expect(feed(at24, 'medium', 25, 1000)).toEqual([]);
    const at60 = new BudgetController('medium');
    expect(feed(at60, 'medium', 25, 1000)[0]).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'budget' });
  });
});

describe('BudgetController: sustained violation', () => {
  it('downgrades after the warm-up plus exactly 60 late frames', () => {
    const c = new BudgetController('medium');
    expect(BUDGET_DEFAULTS.cooldownSamples).toBe(30);
    expect(feed(c, 'medium', 20, 30)).toEqual([]); // warm-up
    expect(c.violationCount).toBe(0);
    expect(feed(c, 'medium', 20, 59)).toEqual([]);
    expect(c.violationCount).toBe(59);
    expect(c.recordSample('medium', S(20))).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'budget' });
    expect(c.tier).toBe('low');
  });

  it('never downgrades paced frames that finish inside their slot, whatever their GPU time (PERF-1)', () => {
    // Measured on the M5 Max: low 1280x720 -> 1600x900 at 60 fps costs 3.4 ms back to back, but
    // the paced timestamp time is 12.5 ms (the GPU clock drops) and submit -> done about 13.7 ms.
    // The former mean-GPU-time rule stepped this down to Quality 0 within seconds.
    for (const tier of ['medium', 'low'] as QualityTier[]) {
      const c = new BudgetController(tier);
      c.setFrameRate(60);
      expect(feed(c, tier, 12.5, 3000, 13.7)).toEqual([]);
      expect(feed(c, tier, 15.9, 3000, 16.5)).toEqual([]);
      expect(c.violationCount).toBe(0);
      expect(c.stats().meanMs).toBeCloseTo(15.9, 9); // still reported as telemetry
      // low 1080p60 -> 3200x1800 on the M5 Max, paced: GPU 13.2 ms, latency 16-20 ms, 0 drops;
      // the one-slot rule stepped it down to Quality 0 in two of three real-site runs.
      expect(feed(c, tier, 13.2, 3000, 19.9)).toEqual([]);
      expect(c.tier).toBe(tier);
    }
  });

  it('downgrades when most frames miss their slot with heavy GPU work (deadline misses)', () => {
    const c = new BudgetController('low');
    c.setFrameRate(60);
    feed(c, 'low', 12.5, 30, 13.7); // warm-up
    const events: BudgetEvent[] = [];
    for (let i = 0; i < 500 && events.length === 0; i++) {
      const late = i % 3 !== 0; // two of three frames late
      const e = c.recordSample('low', late ? S(15, 40) : S(12.5, 13.7));
      if (e) events.push(e);
    }
    expect(events).toEqual([{ type: 'downgrade', from: 'low', to: 'quality0_bicubic', cause: 'budget' }]);
  });

  it('half or fewer late frames in the window never accumulate violations', () => {
    const c = new BudgetController('medium');
    feed(c, 'medium', 2, 30); // warm-up
    for (let i = 0; i < 5000; i++) expect(c.recordSample('medium', i % 2 === 0 ? S(30) : S(5))).toBeNull();
    expect(c.violationCount).toBeLessThanOrEqual(1);
    expect(c.tier).toBe('medium');
    // isolated late frames between healthy ones do not count either
    for (let i = 0; i < 2000; i++) expect(c.recordSample('medium', i % 10 === 0 ? S(40) : S(6))).toBeNull();
    expect(c.violationCount).toBe(0);
  });

  it('light frames delayed by a busy main thread never count (GPU time under 70 %)', () => {
    const c = new BudgetController('quality0_bicubic');
    expect(feed(c, 'quality0_bicubic', 1.8, 2000, 45)).toEqual([]);
    expect(c.throttled).toBe(false);
  });

  it('counts consecutive-ish: on-time frames decay the violation count', () => {
    const c = new BudgetController('medium');
    feed(c, 'medium', 20, 30); // warm-up
    feed(c, 'medium', 20, 30);
    expect(c.violationCount).toBe(30);
    // the window still holds 30 late flags: it stays more than half late for 29 on-time frames
    feed(c, 'medium', 5, 29);
    expect(c.violationCount).toBe(59);
    feed(c, 'medium', 5, 1); // 30 of 60 late: not more than half
    expect(c.violationCount).toBe(58);
    feed(c, 'medium', 5, 58);
    expect(c.violationCount).toBe(0);
    expect(c.tier).toBe('medium');
  });

  it('applies hysteresis after a step: window reset, other-tier frames ignored, fresh warm-up', () => {
    const c = new BudgetController('medium');
    feed(c, 'medium', 30, 90);
    expect(c.tier).toBe('low');
    expect(c.stats().samples).toBe(0);
    expect(c.violationCount).toBe(0);
    // frames still in flight from the medium pipeline do not count for low
    expect(feed(c, 'medium', 500, 200)).toEqual([]);
    expect(c.stats().samples).toBe(0);
    // low's own warm-up is ignored by the decision, then 60 frames are needed again
    expect(feed(c, 'low', 30, 30)).toEqual([]);
    expect(c.violationCount).toBe(0);
    expect(c.stats().samples).toBe(30);
    expect(feed(c, 'low', 30, 59)).toEqual([]);
    expect(feed(c, 'low', 30, 1)).toEqual([{ type: 'downgrade', from: 'low', to: 'quality0_bicubic', cause: 'budget' }]);
  });

  it('throttles when Quality 0 is still over budget, and then ignores everything', () => {
    const c = new BudgetController('quality0_bicubic');
    expect(feed(c, 'quality0_bicubic', 30, 89)).toEqual([]);
    expect(c.recordSample('quality0_bicubic', S(30))).toEqual({ type: 'throttle', tier: 'quality0_bicubic', cause: 'budget' });
    expect(c.throttled).toBe(true);
    expect(c.tier).toBe('quality0_bicubic');
    expect(feed(c, 'quality0_bicubic', 30, 200)).toEqual([]);
    c.frameSubmitted(1, 'quality0_bicubic', 0);
    expect(c.inFlight).toBe(0);
    expect(observe(c, 0, 10_000)).toEqual([]);
    expect(c.ringFull(0)).toBeNull();
  });

  it('ignores invalid samples', () => {
    const c = new BudgetController('low', { cooldownSamples: 0 });
    for (const bad of [S(Number.NaN), S(-1), S(Number.POSITIVE_INFINITY), S(5, Number.NaN), S(5, -1)]) expect(c.recordSample('low', bad)).toBeNull();
    expect(c.stats().samples).toBe(0);
    expect(() => new BudgetController('low', { watchdogHeartbeatMs: 100, maxObservationGapMs: 100 })).toThrow(RangeError);
  });
});

describe('BudgetController: timeout watchdog', () => {
  it('downgrades when the head frame waited 250 ms of observed time, once per timeout period', () => {
    const c = new BudgetController('medium');
    expect(c.nextWatchdogCheck()).toBeNull();
    c.frameSubmitted(1, 'medium', 1000);
    expect(c.nextWatchdogCheck()).toBe(1050); // one heartbeat
    expect(observe(c, 1000, 1250)).toEqual([]);
    expect(c.watchdogWaitedMs).toBe(250);
    expect(c.nextWatchdogCheck()).toBe(1250); // nothing left of the 250 ms
    expect(c.checkWatchdog(1251)).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'timeout' });
    // the stuck frame keeps its place; the next step needs another full period without progress
    expect(c.watchdogWaitedMs).toBe(0);
    expect(observe(c, 1251, 1501)).toEqual([]);
    expect(c.checkWatchdog(1502)).toEqual({ type: 'downgrade', from: 'low', to: 'quality0_bicubic', cause: 'timeout' });
    expect(observe(c, 1502, 1752)).toEqual([]);
    expect(c.checkWatchdog(1753)).toEqual({ type: 'throttle', tier: 'quality0_bicubic', cause: 'timeout' });
    expect(c.throttled).toBe(true);
    expect(c.nextWatchdogCheck()).toBeNull();
  });

  it('a main-thread stall is not charged to the GPU (RT-2)', () => {
    const c = new BudgetController('medium');
    c.frameSubmitted(1, 'medium', 0);
    expect(observe(c, 0, 40, 20)).toEqual([]);
    // a 400 ms busy loop: the frame callback that runs first after it sees no completion yet
    expect(c.checkWatchdog(440)).toBeNull();
    expect(c.watchdogWaitedMs).toBe(40);
    expect(c.frameCompleted(1, 442, S(6, 8))).toBeNull(); // the completion task runs right after
    expect(c.tier).toBe('medium');
    // four such stalls 1.5 s apart during 60 fps playback (frames done 5 ms after submit)
    let now = 1000;
    let token = 1;
    for (let stall = 0; stall < 4; stall++) {
      for (let f = 0; f < 90; f++) {
        expect(c.checkWatchdog(now)).toBeNull();
        c.frameSubmitted(++token, 'medium', now);
        if (f === 89) break;
        expect(c.frameCompleted(token, now + 5, S(6, 5))).toBeNull();
        now += 1000 / 60;
      }
      now += 400; // the frame submitted last is in flight during the stall
      expect(c.checkWatchdog(now)).toBeNull();
      expect(c.frameCompleted(token, now + 1, S(6, 405))).toBeNull();
      now += 1000 / 60;
    }
    expect(c.tier).toBe('medium');
  });

  it('a GPU that is really stuck still times out after a stall', () => {
    const c = new BudgetController('medium');
    c.frameSubmitted(1, 'medium', 0);
    expect(c.checkWatchdog(400)).toBeNull(); // stall: not charged
    expect(observe(c, 400, 650)).toEqual([]); // 250 ms observed
    expect(c.checkWatchdog(651)).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'timeout' });
    // gaps up to maxObservationGapMs are ordinary (e.g. a 24 fps callback cadence) and count
    const d = new BudgetController('medium');
    d.frameSubmitted(1, 'medium', 0);
    expect(observe(d, 0, 200, 100)).toEqual([]);
    expect(d.watchdogWaitedMs).toBe(200);
    expect(d.checkWatchdog(300)).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'timeout' });
  });

  it("starts a frame's clock when it reaches the head of the queue", () => {
    const c = new BudgetController('medium');
    c.frameSubmitted(1, 'medium', 0);
    c.frameSubmitted(2, 'medium', 10);
    c.frameSubmitted(3, 'medium', 20);
    expect(c.inFlight).toBe(3);
    expect(observe(c, 20, 200, 45)).toEqual([]);
    expect(c.watchdogWaitedMs).toBe(200);
    expect(c.frameCompleted(1, 200, S(4))).toBeNull();
    expect(c.watchdogWaitedMs).toBe(0);
    expect(observe(c, 200, 450)).toEqual([]);
    c.frameCompleted(2, 450, S(4));
    expect(observe(c, 450, 600)).toEqual([]);
    expect(c.watchdogWaitedMs).toBe(150);
    c.frameCompleted(3, 600, S(4));
    expect(c.nextWatchdogCheck()).toBeNull();
    expect(observe(c, 600, 100_000, 1000)).toEqual([]);
    expect(c.tier).toBe('medium');
    expect(c.stats().samples).toBe(3);
  });

  it('records completions without a time as progress only; unknown tokens are ignored', () => {
    const c = new BudgetController('low');
    c.frameSubmitted(7, 'low', 0);
    expect(c.frameCompleted(99, 10, S(5))).toBeNull();
    expect(c.inFlight).toBe(1);
    expect(c.frameCompleted(7, 10, null)).toBeNull();
    expect(c.inFlight).toBe(0);
    expect(c.stats().samples).toBe(0);
  });

  it('a budget step keeps the watchdog clock of frames still queued', () => {
    const c = new BudgetController('medium', { cooldownSamples: 0, sustainFrames: 1 });
    c.frameSubmitted(1, 'medium', 0);
    c.frameSubmitted(2, 'medium', 0);
    expect(observe(c, 0, 200)).toEqual([]);
    expect(c.frameCompleted(2, 200, S(30))).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'budget' });
    expect(c.watchdogWaitedMs).toBe(200); // frame 1 is still the head and keeps its clock
    expect(c.checkWatchdog(250)).toBeNull();
    expect(c.checkWatchdog(251)).toEqual({ type: 'downgrade', from: 'low', to: 'quality0_bicubic', cause: 'timeout' });
  });

  it('downgrades after 15 consecutive ring-full drops; a free slot resets the streak', () => {
    const c = new BudgetController('medium');
    for (let i = 0; i < 14; i++) expect(c.ringFull(i)).toBeNull();
    c.ringAvailable();
    expect(c.ringFullCount).toBe(0);
    for (let i = 0; i < 14; i++) expect(c.ringFull(i)).toBeNull();
    expect(c.ringFull(14)).toEqual({ type: 'downgrade', from: 'medium', to: 'low', cause: 'ring-full' });
    expect(c.ringFullCount).toBe(0);
    for (let i = 0; i < 14; i++) c.ringFull(i);
    expect(c.ringFull(20)).toEqual({ type: 'downgrade', from: 'low', to: 'quality0_bicubic', cause: 'ring-full' });
    for (let i = 0; i < 14; i++) c.ringFull(i);
    expect(c.ringFull(30)).toEqual({ type: 'throttle', tier: 'quality0_bicubic', cause: 'ring-full' });
  });

  it('abandonInFlight forgets frames of a lost device but keeps the tier', () => {
    const c = new BudgetController('medium');
    c.frameSubmitted(1, 'medium', 0);
    feed(c, 'medium', 30, 90);
    expect(c.tier).toBe('low');
    c.abandonInFlight();
    expect(c.inFlight).toBe(0);
    expect(observe(c, 0, 10_000)).toEqual([]);
    expect(c.tier).toBe('low');
  });
});

describe('BudgetController: never above the user ceiling', () => {
  it('no sequence of events raises the tier; only reset() (a user action) does', () => {
    let seed = 12345;
    const rand = (): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    for (const ceiling of TIER_ORDER) {
      const c = new BudgetController(ceiling, { sustainFrames: 20, cooldownSamples: 5, ringFullLimit: 4 });
      let rank = tierRank(c.tier);
      let now = 0;
      let token = 0;
      const pending: number[] = [];
      for (let step = 0; step < 20000; step++) {
        now += rand() * 20;
        const r = rand();
        const anyTier = TIER_ORDER[Math.floor(rand() * 3)] as QualityTier;
        if (r < 0.4) c.recordSample(anyTier, S(rand() * 60, rand() * 60));
        else if (r < 0.6) {
          c.frameSubmitted(++token, anyTier, now);
          pending.push(token);
        } else if (r < 0.75 && pending.length) c.frameCompleted(pending.shift() as number, now, rand() < 0.2 ? null : S(rand() * 30, rand() * 40));
        else if (r < 0.85) c.checkWatchdog(now);
        else if (r < 0.95) c.ringFull(now);
        else c.ringAvailable();
        const next = tierRank(c.tier);
        expect(next).toBeLessThanOrEqual(rank);
        expect(next).toBeLessThanOrEqual(tierRank(ceiling));
        rank = next;
        if (c.throttled) {
          c.reset();
          expect(c.tier).toBe(ceiling);
          rank = tierRank(ceiling);
          pending.length = 0;
        }
      }
    }
  });

  it('reset() returns to the (possibly new) ceiling and clears throttling', () => {
    const c = new BudgetController('medium');
    feed(c, 'medium', 30, 90);
    feed(c, 'low', 30, 90);
    feed(c, 'quality0_bicubic', 30, 90);
    expect(c.throttled).toBe(true);
    c.reset();
    expect([c.tier, c.ceiling, c.throttled, c.violationCount]).toEqual(['medium', 'medium', false, 0]);
    c.reset('low');
    expect([c.tier, c.ceiling]).toEqual(['low', 'low']);
    expect(feed(c, 'medium', 30, 500)).toEqual([]); // a tier above the ceiling never counts
  });
});

describe('FrameRateEstimator', () => {
  const run = (est: FrameRateEstimator, fps: number, n: number, step = 1, rate = 1, t0 = 0, f0 = 0): void => {
    for (let i = 0; i < n; i++) est.record({ mediaTime: t0 + (i * step) / fps, presentedFrames: f0 + i * step }, rate);
  };

  it('needs a few intervals, then reports the median frame rate', () => {
    const est = new FrameRateEstimator();
    run(est, 30, 4);
    expect(est.frameRate).toBeNull(); // 3 intervals
    est.record({ mediaTime: 4 / 30, presentedFrames: 4 });
    expect(est.frameRate).toBeCloseTo(30, 6);
  });

  it('divides media-time deltas by the presented-frame delta and the playback rate', () => {
    const skipped = new FrameRateEstimator();
    run(skipped, 24, 10, 2); // every other callback missed
    expect(skipped.frameRate).toBeCloseTo(24, 6);
    const fast = new FrameRateEstimator();
    run(fast, 30, 10, 1, 2);
    expect(fast.frameRate).toBeCloseTo(60, 6);
    const noCount = new FrameRateEstimator();
    for (let i = 0; i < 10; i++) noCount.record({ mediaTime: i / 25 });
    expect(noCount.frameRate).toBeCloseTo(25, 6);
  });

  it('skips seeks, pauses and outliers', () => {
    const est = new FrameRateEstimator();
    run(est, 60, 20);
    est.record({ mediaTime: 0.1, presentedFrames: 21 }); // backwards seek
    est.record({ mediaTime: 30, presentedFrames: 22 }); // forward jump (5 fps limit)
    run(est, 60, 20, 1, 1, 30 + 1 / 60, 23);
    est.record({ mediaTime: 50, presentedFrames: 100 }); // single long interval inside the bound
    est.record({ mediaTime: 50 + 1 / 24, presentedFrames: 101 });
    expect(est.frameRate).toBeCloseTo(60, 6);
    est.record({ mediaTime: Number.NaN });
    est.record({});
    expect(est.frameRate).toBeCloseTo(60, 6);
    est.reset();
    expect(est.frameRate).toBeNull();
  });
});
