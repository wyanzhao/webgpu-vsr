import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  GAP_WINDOW_MS,
  GpuSessionFactory,
  MAX_COUNTED_GAP,
  PresentationTracker,
  QUALITY_TIERS,
  RateLimiter,
  classifyVideoFrameError,
  createPipeline,
  internalScaleForTier,
  pipelineIdForTier,
  planOutput,
  requestedTier,
  selectTier,
} from '../../src/content/session';
import { ContentController, type VideoSessionContext } from '../../src/content/index';
import { FakeEnv, FakeNode } from './fake_dom';
import { EligibilityLatches } from '../../src/content/eligibility';
import { computeCanvasResolution, computeTargetResolution } from '../../src/content/overlay';
import { WebGpuUnavailableError, type GpuDeviceManager } from '../../src/pipeline/device';
import type { GpuContext, QualityTier } from '../../src/pipeline/types';
import { QUALITY_PIPELINE_IDS, defaultSettings, type QualityLevel } from '../../src/shared/settings';
import { defaultSiteRule } from '../../src/content/site_rules';

beforeEach(() => {
  vi.stubGlobal('chrome', { runtime: { id: 'test-extension', sendMessage: async () => undefined } });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('tier selection', () => {
  it('maps quality levels to tiers and pipeline ids', () => {
    expect(QUALITY_TIERS).toEqual({ 0: 'quality0_bicubic', 1: 'low', 2: 'medium' });
    for (const level of [0, 1, 2] as QualityLevel[]) {
      const tier = requestedTier({ qualityLevel: level });
      expect(pipelineIdForTier(tier)).toBe(QUALITY_PIPELINE_IDS[level]);
    }
  });

  it('locks to Quality 0 without shader-f16', () => {
    for (const level of [0, 1, 2] as QualityLevel[]) {
      expect(selectTier({ qualityLevel: level }, { hasF16: false })).toBe('quality0_bicubic');
      expect(selectTier({ qualityLevel: level }, { hasF16: true })).toBe(QUALITY_TIERS[level]);
    }
  });

  it('creates the pipeline class of each tier with the matching id', () => {
    const gpu = { storageFormat: 'bgra8unorm' } as GpuContext;
    for (const tier of ['quality0_bicubic', 'low', 'medium'] as QualityTier[]) {
      const p = createPipeline(tier, gpu);
      expect(p.id).toBe(pipelineIdForTier(tier));
      expect(typeof p.executeSlot).toBe('function');
    }
  });
});

describe('output planning', () => {
  it('presents the VSR image directly when the target covers 2W x 2H, else resamples', () => {
    expect(planOutput(960, 540, { width: 1920, height: 1080 }, 'low')).toEqual({ width: 1920, height: 1080, mode: 'direct' });
    expect(planOutput(960, 540, { width: 1200, height: 675 }, 'medium')).toEqual({ width: 1200, height: 675, mode: 'resample' });
    expect(planOutput(640, 272, { width: 1280, height: 544 }, 'low')).toEqual({ width: 1280, height: 544, mode: 'direct' });
    expect(planOutput(960, 540, { width: 1920, height: 1080 }, 'quality0_bicubic')).toEqual({ width: 1920, height: 1080, mode: 'resample' });
    expect(internalScaleForTier('quality0_bicubic')).toBeNull();
    expect(internalScaleForTier('low')).toBe(2);
    expect(internalScaleForTier('medium')).toBe(2);
  });

  it("matches the overlay's canvas backing store for every tier (decideOutput == computeCanvasResolution)", () => {
    const sources = [
      [960, 540],
      [1280, 720],
      [640, 272],
      [854, 480],
      [720, 404],
      [1440, 1080],
      [480, 640],
    ];
    const displays = [
      [1200, 675],
      [1920, 1080],
      [2560, 1440],
      [1000, 1333.5],
      [800, 340],
      [3456, 1944],
    ];
    let checked = 0;
    for (const [w, h] of sources as [number, number][]) {
      for (const [dw, dh] of displays as [number, number][]) {
        const target = computeTargetResolution(dw, dh);
        for (const tier of ['quality0_bicubic', 'low', 'medium'] as QualityTier[]) {
          const plan = planOutput(w, h, target, tier);
          const canvas = computeCanvasResolution(target, w, h, internalScaleForTier(tier));
          expect([plan.width, plan.height]).toEqual([canvas.width, canvas.height]);
          expect(plan.mode === 'direct').toBe(canvas.presentsInternal);
          checked++;
        }
      }
    }
    expect(checked).toBe(7 * 6 * 3);
  });
});

describe('VideoFrame errors', () => {
  it('tainted frames, missing frames and real failures are told apart', () => {
    expect(classifyVideoFrameError({ name: 'SecurityError', message: 'x' })).toBe('tainted');
    expect(classifyVideoFrameError(new Error('The video element contains cross-origin data'))).toBe('tainted');
    expect(classifyVideoFrameError({ name: 'InvalidStateError', message: 'no frame' })).toBe('unavailable');
    expect(classifyVideoFrameError(new TypeError('bad'))).toBe('fatal');
    expect(classifyVideoFrameError('boom')).toBe('fatal');
    expect(classifyVideoFrameError(null)).toBe('fatal');
  });
});

describe('PresentationTracker', () => {
  it('measures the rVFC cadence over the last second', () => {
    const p = new PresentationTracker();
    expect(p.fps).toBe(0);
    for (let i = 0; i <= 120; i++) p.record(i * (1000 / 60), i);
    expect(p.fps).toBeCloseTo(60, 1);
    const q = new PresentationTracker();
    for (let i = 0; i <= 48; i++) q.record(i * (1000 / 24), i);
    expect(q.fps).toBeCloseTo(24, 1);
  });

  it('counts presentedFrames gaps as drops, but not seeks, pauses or missing metadata', () => {
    const p = new PresentationTracker();
    p.record(0, 10);
    expect(p.record(16, 11)).toBe(0);
    expect(p.record(50, 14)).toBe(2); // frames 12 and 13 never reached the callback
    expect(p.record(66, 14 + MAX_COUNTED_GAP + 1)).toBe(0); // jump: a seek, not drops
    expect(p.record(66 + GAP_WINDOW_MS, 27)).toBe(0); // after a pause
    expect(p.record(GAP_WINDOW_MS + 80, 30)).toBe(2);
    expect(p.record(GAP_WINDOW_MS + 90)).toBe(0);
    expect(p.droppedFrames).toBe(4);
    p.reset();
    expect([p.droppedFrames, p.fps]).toEqual([0, 0]);
    expect(p.record(1000, 99)).toBe(0); // first callback after a reset has no reference
  });
});

describe('RateLimiter', () => {
  it('allows one event per interval (telemetry at <= 2 Hz)', () => {
    const r = new RateLimiter(500);
    expect(r.ready(0)).toBe(true);
    r.mark(0);
    expect(r.ready(499)).toBe(false);
    expect(r.delay(200)).toBe(300);
    expect(r.ready(500)).toBe(true);
    expect(r.delay(800)).toBe(0);
  });
});

// ---------------------------------------------------------------------------------------------
// Session start-up paths that run before any GPU object exists
// ---------------------------------------------------------------------------------------------

class FakeVideo {
  readonly listeners = new Map<string, Set<() => void>>();
  paused = true;
  readyState = 0;
  addEventListener(type: string, l: () => void): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(l);
  }
  removeEventListener(type: string, l: () => void): void {
    this.listeners.get(type)?.delete(l);
  }
  count(): number {
    return [...this.listeners.values()].reduce((n, s) => n + s.size, 0);
  }
}

function fakeManager(acquire: () => Promise<GpuContext>) {
  const state = { refs: 0, releases: 0, lost: 0, restored: 0 };
  const manager = {
    current: null,
    acquire: async () => {
      state.refs++;
      try {
        return await acquire();
      } catch (e) {
        state.refs--;
        throw e;
      }
    },
    release: () => {
      state.refs--;
      state.releases++;
    },
    onLost: () => {
      state.lost++;
      return () => state.lost--;
    },
    onRestored: () => {
      state.restored++;
      return () => state.restored--;
    },
    whenRecovered: async () => undefined,
  };
  return { manager: manager as unknown as GpuDeviceManager, state };
}

function fakeContext(video: FakeVideo) {
  const calls = { telemetry: 0, active: 0, errors: [] as string[], evaluations: 0 };
  const context: VideoSessionContext = {
    id: 'v1',
    video: video as unknown as HTMLVideoElement,
    siteRule: defaultSiteRule(),
    latches: new EligibilityLatches(),
    getSettings: () => defaultSettings(),
    getEligibility: () => {
      throw new Error('not used');
    },
    reportTelemetry: () => void calls.telemetry++,
    reportActive: () => void calls.active++,
    reportError: (m) => void calls.errors.push(m),
    requestEvaluation: () => void calls.evaluations++,
  };
  return { context, calls };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('VideoSRSession without a GPU', () => {
  it('no adapter: point 8 fails from then on and the gate is re-run; nothing is leaked', async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const { manager, state } = fakeManager(() => Promise.reject(new WebGpuUnavailableError('no WebGPU adapter')));
    const factory = new GpuSessionFactory({ deviceManager: manager });
    expect(factory.isGpuAvailable()).toBe(true);
    const video = new FakeVideo();
    const { context, calls } = fakeContext(video);
    const session = factory.create(context);
    session.attach();
    expect(video.count()).toBe(4); // encrypted, waitingforkey, seeked, loadeddata
    expect([state.lost, state.restored]).toEqual([1, 1]);
    await flush();
    expect(factory.isGpuAvailable()).toBe(false);
    expect(calls.evaluations).toBe(1);
    expect(calls.errors).toEqual([]);
    expect(session.isBudgetHealthy()).toBe(true);
    expect(state.refs).toBe(0);
    session.destroy();
    expect(video.count()).toBe(0);
    expect([state.lost, state.restored, state.releases]).toEqual([0, 0, 0]);
  });

  it('a failing device request is a session error (reported once), not a GPU verdict', async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const { manager } = fakeManager(() => Promise.reject(new Error('requestDevice failed')));
    const factory = new GpuSessionFactory({ deviceManager: manager });
    const video = new FakeVideo();
    const { context, calls } = fakeContext(video);
    const session = factory.create(context);
    session.attach();
    session.attach(); // idempotent
    await flush();
    expect(calls.errors).toEqual(['requestDevice failed']);
    expect(factory.isGpuAvailable()).toBe(true);
    session.destroy();
  });

  it('destroyed while the device is being acquired: the reference is returned', async () => {
    let resolve!: (ctx: GpuContext) => void;
    const { manager, state } = fakeManager(() => new Promise<GpuContext>((r) => (resolve = r)));
    const video = new FakeVideo();
    const { context, calls } = fakeContext(video);
    const session = new GpuSessionFactory({ deviceManager: manager }).create(context);
    session.attach();
    session.updateSettings({ ...defaultSettings(), qualityLevel: 2, sharpness: 0.9 }); // before the GPU: no-op on GPU state
    session.destroy();
    session.destroy();
    expect(state.refs).toBe(1);
    resolve({ hasF16: true } as GpuContext);
    await flush();
    expect(state.refs).toBe(0);
    expect(state.releases).toBe(1);
    expect([calls.errors, calls.active, calls.telemetry]).toEqual([[], 0, 0]);
  });

  it('reports no GPU when the frame has no WebGPU entry point', () => {
    vi.stubGlobal('navigator', {});
    expect(new GpuSessionFactory().isGpuAvailable()).toBe(false);
  });
});


describe('session eligibility reporting', () => {
  it('reports a session HDR signal despite a cached SDR probe and clears it on source change', () => {
    vi.useFakeTimers();
    const fx = new FakeEnv();
    const parent = new FakeNode();
    parent.connectedRoot = true;
    const video = Object.assign(new FakeNode(), { readyState: 4, currentSrc: 'https://example.com/sdr.webm', mediaKeys: null });
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    parent.appendChild(video);
    const doc = Object.assign(fx.doc, {
      documentElement: parent,
      querySelectorAll: (selector: string) => selector === 'video' ? [video] : [parent, video],
    });
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', {
      document: doc,
      location: { origin: 'https://example.com' },
      devicePixelRatio: 1,
      getComputedStyle: fx.env().getComputedStyle,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      setTimeout,
      clearTimeout,
      setInterval,
      clearInterval,
    });
    const probes = vi.fn();
    vi.stubGlobal('VideoFrame', class {
      readonly colorSpace = { transfer: 'bt709' };
      constructor() { probes(); }
      close(): void { /* no GPU resources */ }
    });
    const contexts: VideoSessionContext[] = [];
    const destroy = vi.fn();
    const controller = new ContentController({
      settings: defaultSettings(),
      siteRule: defaultSiteRule(),
      hostname: 'example.com',
      isTopFrame: true,
      getUrl: () => 'https://example.com/player',
      send: async () => undefined,
      sessionFactory: {
        isGpuAvailable: () => true,
        create: (context) => {
          contexts.push(context);
          return {
            attach: () => undefined,
            reattach: () => undefined,
            detach: () => undefined,
            refreshGeometry: () => undefined,
            isLayoutSupported: () => true,
            updateSettings: () => undefined,
            sourceChanged: () => undefined,
            isBudgetHealthy: () => true,
            destroy,
          };
        },
      },
    });
    try {
      controller.start();
      expect(controller.reports()[0]).toMatchObject({ state: 'eligible', reason: null });
      expect(probes).toHaveBeenCalledTimes(1);
      const context = contexts[0]!;
      context.latches.latchHdr(context.video);
      context.requestEvaluation();
      vi.advanceTimersByTime(30);
      expect(controller.reports()[0]).toMatchObject({ state: 'ineligible', reason: 'hdr', point: 6 });
      expect(destroy).toHaveBeenCalledTimes(1);
      expect(probes).toHaveBeenCalledTimes(1);
      video.currentSrc = 'https://example.com/new-sdr.webm';
      vi.advanceTimersByTime(2000);
      expect(controller.reports()[0]).toMatchObject({ state: 'eligible', reason: null });
      expect(probes).toHaveBeenCalledTimes(2);
      expect(contexts).toHaveLength(2);
    } finally {
      controller.stop();
    }
  });
});
