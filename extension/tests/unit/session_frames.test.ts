/**
 * VideoSRSession frame loop against a fake GPU device, a fake VideoFrame and the real
 * CanvasOverlay on the fake DOM (tests/unit/fake_dom.ts):
 * - the validated-first-frame gate: the native video is hidden only after the first frame's
 *   error scopes came back clean, and any GPU error fails the session with the video restored;
 * - every VideoFrame the loop creates is closed, on every exit path;
 * - the sharpness setting reaches the pipeline on initialisation and on a settings change.
 */
import { HIDDEN_OBJECT_POSITION } from '../../src/content/overlay';
import type { SessionTelemetry } from '../../src/shared/messages';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuSessionFactory, pipelineIdForTier, type RenderPipeline } from '../../src/content/session';
import { ContentController, type VideoSessionContext, type VideoSessionHandle } from '../../src/content/index';
import { EligibilityLatches } from '../../src/content/eligibility';
import { defaultSiteRule } from '../../src/content/site_rules';
import { GpuDeviceManager } from '../../src/pipeline/device';
import type { GpuContext, QualityTier } from '../../src/pipeline/types';
import { defaultSettings, type UserSettings } from '../../src/shared/settings';
import { FakeEnv, FakeNode } from './fake_dom';

// ---------------------------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------------------------

class FakeVideoFrame {
  static created = 0;
  static closed = 0;
  static throwNext: Error | null = null;
  static transfer = 'bt709';
  readonly colorSpace = { transfer: FakeVideoFrame.transfer };
  readonly displayWidth: number;
  readonly displayHeight: number;
  private open = true;

  constructor(source: { videoWidth: number; videoHeight: number }) {
    if (FakeVideoFrame.throwNext) {
      const e = FakeVideoFrame.throwNext;
      FakeVideoFrame.throwNext = null;
      throw e;
    }
    FakeVideoFrame.created++;
    this.displayWidth = source.videoWidth;
    this.displayHeight = source.videoHeight;
  }

  close(): void {
    if (!this.open) return;
    this.open = false;
    FakeVideoFrame.closed++;
  }

  static get live(): number {
    return FakeVideoFrame.created - FakeVideoFrame.closed;
  }
}

type ScopeResult = { message: string } | null;

class FakeDevice {
  readonly features = new Set(['shader-f16']);
  lose!: (info: GPUDeviceLostInfo) => void;
  readonly lost = new Promise<GPUDeviceLostInfo>((resolve) => { this.lose = resolve; });
  readonly destroy = vi.fn();
  /** Results of the next popErrorScope() calls in call order (default: no error). */
  readonly popResults: Array<Promise<ScopeResult>> = [];
  readonly openScopes: string[] = [];
  submits = 0;
  private readonly workDone: Array<() => void> = [];
  readonly queue = {
    submit: (): void => void this.submits++,
    onSubmittedWorkDone: (): Promise<void> => new Promise<void>((r) => this.workDone.push(r)),
    writeBuffer: (): void => undefined,
  };

  pushErrorScope(filter: string): void {
    this.openScopes.push(filter);
  }
  popErrorScope(): Promise<ScopeResult> {
    this.openScopes.pop();
    return this.popResults.shift() ?? Promise.resolve(null);
  }
  importExternalTexture(): object {
    return {};
  }
  createCommandEncoder(): object {
    return { finish: () => ({}), resolveQuerySet: () => undefined, copyBufferToBuffer: () => undefined };
  }
  createBuffer(d: { size: number }): object {
    return { size: d.size, destroy: () => undefined };
  }
  createQuerySet(d: { count: number }): object {
    return { count: d.count, destroy: () => undefined };
  }
  /** The GPU finishes everything submitted so far. */
  completeAll(): void {
    for (const r of this.workDone.splice(0)) r();
  }
  /** The GPU finishes only the oldest outstanding submission. */
  completeOne(): void {
    this.workDone.shift()?.();
  }
}

class FakePipeline implements RenderPipeline {
  readonly id: string;
  readonly scaleFactor = 2;
  readonly inits: number[][] = [];
  readonly sharpness: number[] = [];
  executed = 0;
  destroyed = false;
  failExecute = false;

  constructor(readonly tier: QualityTier) {
    this.id = pipelineIdForTier(tier);
  }
  async initialize(_device: GPUDevice, w: number, h: number, ow: number, oh: number): Promise<void> {
    this.inits.push([w, h, ow, oh]);
  }
  execute(): void {
    throw new Error('the session records frames with executeSlot');
  }
  executeSlot(): void {
    if (this.failExecute) throw new Error('encoder failure');
    this.executed++;
  }
  setSharpness(s: number): void {
    this.sharpness.push(s);
  }
  memoryBytes(): number {
    return 0;
  }
  destroy(): void {
    this.destroyed = true;
  }
}

class FakeVideo extends FakeNode {
  readyState = 4;
  paused = false;
  mediaKeys: unknown = null;
  playbackRate = 1;
  callback: ((now: number, meta: { presentedFrames: number; mediaTime: number }) => void) | null = null;
  private handles = 0;
  /** Every inline object-position the page or the extension wrote, in order. */
  readonly objectPositionWrites: string[] = [];

  constructor() {
    super();
    const set = this.style.setProperty.bind(this.style);
    this.style.setProperty = (name: string, value: string, priority = '') => {
      if (name === 'object-position') this.objectPositionWrites.push(value);
      set(name, value, priority);
    };
  }
  requestVideoFrameCallback(cb: FakeVideo['callback']): number {
    this.callback = cb;
    return ++this.handles;
  }
  cancelVideoFrameCallback(): void {
    this.callback = null;
  }
}

const flush = async (): Promise<void> => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
};

interface Rig {
  session: VideoSessionHandle;
  factory: GpuSessionFactory;
  fx: FakeEnv;
  parent: FakeNode;
  video: FakeVideo;
  device: FakeDevice;
  pipes: FakePipeline[];
  calls: { errors: string[]; active: number; evaluations: number; telemetry: SessionTelemetry[] };
  latches: EligibilityLatches;
  settings: { current: UserSettings };
  /** Delivers one requestVideoFrameCallback callback (if one is pending). */
  frame(): boolean;
  /** Advances the fake clock by `ms`. */
  tick(ms: number): void;
}

function rig(initial: Partial<UserSettings> = {}, deviceManager?: GpuDeviceManager): Rig {
  const fx = new FakeEnv();
  const canvasContext = { configure: vi.fn(), unconfigure: vi.fn(), getCurrentTexture: () => ({ createView: () => ({}) }) };
  fx.canvasContext = canvasContext;
  const body = new FakeNode();
  body.connectedRoot = true;
  const parent = new FakeNode();
  body.appendChild(parent);
  const video = new FakeVideo();
  parent.appendChild(video);
  parent.rect = { left: 0, top: 0, width: 800, height: 450 };
  video.rect = { left: 0, top: 0, width: 800, height: 450 };
  video.videoWidth = 640;
  video.videoHeight = 360;
  video.style.setProperty('object-position', '40% 60%');
  video.objectPositionWrites.length = 0;

  const device = new FakeDevice();
  const gpu = { adapter: {}, device, hasF16: true, hasTimestamps: false, storageFormat: 'bgra8unorm' } as unknown as GpuContext;
  const manager = {
    current: gpu,
    acquire: async () => gpu,
    release: () => undefined,
    onLost: () => () => undefined,
    onRestored: () => () => undefined,
    whenRecovered: async () => undefined,
  } as unknown as GpuDeviceManager;
  const pipes: FakePipeline[] = [];
  let clock = 1000;
  let presented = 0;
  const settings = { current: { ...defaultSettings(), ...initial } };
  const calls = { errors: [] as string[], active: 0, evaluations: 0, telemetry: [] as SessionTelemetry[] };
  const latches = new EligibilityLatches();
  const context: VideoSessionContext = {
    id: 'v1',
    video: video as unknown as HTMLVideoElement,
    siteRule: defaultSiteRule(),
    latches,
    getSettings: () => settings.current,
    getEligibility: () => {
      throw new Error('not used');
    },
    reportTelemetry: (telemetry) => void calls.telemetry.push(telemetry),
    reportActive: () => void calls.active++,
    reportError: (m) => void calls.errors.push(m),
    requestEvaluation: () => void calls.evaluations++,
  };
  const factory = new GpuSessionFactory({
    deviceManager: deviceManager ?? manager,
    createPipeline: (tier) => {
      const p = new FakePipeline(tier);
      pipes.push(p);
      return p;
    },
    now: () => clock,
    overlayEnvironment: fx.env(),
  });
  const session = factory.create(context);
  return {
    session,
    factory,
    fx,
    parent,
    video,
    device,
    pipes,
    calls,
    latches,
    settings,
    frame: () => {
      const cb = video.callback;
      if (!cb) return false;
      video.callback = null;
      presented++;
      cb(clock, { presentedFrames: presented, mediaTime: presented / 60 });
      return true;
    },
    tick: (ms) => {
      clock += ms;
    },
  };
}

/** Attach, let the first pipeline build and its first frame be submitted. */
async function startSession(r: Rig): Promise<void> {
  r.session.attach();
  await flush();
  expect(r.pipes).toHaveLength(1);
  expect(r.pipes[0]!.executed).toBe(1);
}

beforeEach(() => {
  vi.stubGlobal('chrome', { runtime: { id: 'test-extension', sendMessage: async () => undefined } });
  FakeVideoFrame.created = 0;
  FakeVideoFrame.closed = 0;
  FakeVideoFrame.throwNext = null;
  FakeVideoFrame.transfer = 'bt709';
  vi.stubGlobal('VideoFrame', FakeVideoFrame);
  vi.stubGlobal('GPUBufferUsage', { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128, QUERY_RESOLVE: 512 });
  vi.stubGlobal('GPUTextureUsage', { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 });
  vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------------------------
// Validated first frame and GPU errors
// ---------------------------------------------------------------------------------------------

describe('controller disable during detach grace', () => {
  it.each(['extension', 'frame site', 'top-level site', 'site rule'] as const)(
    'destroys the session and removes every extension node immediately when disabled by %s', async (disabledBy) => {
      const r = rig({ hudEnabled: true, siteOverrides: disabledBy === 'site rule' ? { 'example.com': true } : {} });
      const doc = Object.assign(r.fx.doc, {
        documentElement: r.parent,
        querySelectorAll: (selector: string) => selector === 'video' ? (r.video.isConnected ? [r.video] : []) : [r.parent, r.video],
      });
      let periodic!: () => void;
      vi.stubGlobal('document', doc);
      vi.stubGlobal('window', {
        document: doc, location: { origin: 'https://example.com' }, devicePixelRatio: 1,
        getComputedStyle: r.fx.env().getComputedStyle,
        addEventListener: () => undefined, removeEventListener: () => undefined, setTimeout, clearTimeout,
        setInterval: (cb: () => void) => { periodic = cb; return 1; }, clearInterval: () => undefined,
      });
      const create = vi.fn(() => r.session);
      const destroy = vi.spyOn(r.session, 'destroy');
      const detach = vi.spyOn(r.session, 'detach');
      const controller = new ContentController({
        settings: r.settings.current, siteRule: { ...defaultSiteRule(), enabled: disabledBy !== 'site rule' },
        hostname: 'example.com', topHostname: 'top.example', isTopFrame: false,
        getUrl: () => 'https://example.com/player', send: async () => undefined,
        sessionFactory: { isGpuAvailable: () => true, create },
      });
      try {
        controller.start();
        await flush();
        r.device.completeAll();
        await flush();
        expect(r.parent.children).toHaveLength(3); // video, canvas and HUD
        expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
        const id = controller.reports()[0]!.id;
        r.video.remove();
        periodic();
        expect(detach).toHaveBeenCalledTimes(1);
        expect(controller.reports()).toEqual([]);
        expect(r.parent.children).toHaveLength(2); // canvas and hidden HUD survive detach
        expect(destroy).not.toHaveBeenCalled();
        controller.applySettings({
          ...r.settings.current,
          enabled: disabledBy !== 'extension',
          siteOverrides: disabledBy === 'frame site' ? { 'example.com': false } :
            disabledBy === 'top-level site' ? { 'top.example': false } : {},
        });
        expect(r.parent.children).toEqual([]);
        expect(destroy).toHaveBeenCalledTimes(1);
        expect(r.pipes[0]!.destroyed).toBe(true);
        expect(r.video.callback).toBeNull();
        expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
        r.device.completeAll();
        await flush();
        expect(r.parent.children).toEqual([]);
        r.parent.appendChild(r.video);
        periodic();
        expect(controller.reports()[0]!.id).toBe(id);
        expect(create).toHaveBeenCalledTimes(1);
        expect(r.parent.children).toEqual([r.video]);
      } finally {
        controller.stop();
      }
    },
  );
});

describe('VideoSRSession: validated first frame', () => {
  it('recovers to healthy C when B is lost inside its restored callback', async () => {
    const first = new FakeDevice();
    const replacement = new FakeDevice();
    const healthy = new FakeDevice();
    const requestDevice = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(replacement).mockResolvedValueOnce(healthy);
    const sleep = vi.fn(async (_ms: number) => undefined);
    const manager = new GpuDeviceManager({
      gpu: { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU,
      sleep,
    });
    const restored = vi.fn((ctx: GpuContext) => {
      if (ctx.device === replacement as unknown as GPUDevice) {
        replacement.lose({ reason: 'unknown', message: 'B lost during notification' } as GPUDeviceLostInfo);
      }
    });
    const unsubscribe = manager.onRestored(restored);
    const r = rig({}, manager);
    const configured = r.fx.canvasContext as { configure: ReturnType<typeof vi.fn> };
    try {
      await startSession(r);
      first.completeAll();
      await flush();
      first.lose({ reason: 'unknown', message: 'A lost' } as GPUDeviceLostInfo);
      await Promise.resolve();
      await manager.whenRecovered();
      expect(manager.current?.device).toBe(healthy);
      expect(requestDevice).toHaveBeenCalledTimes(3);
      expect(sleep.mock.calls).toEqual([[1000], [2000]]);
      expect(restored.mock.calls.map(([ctx]) => ctx.device)).toEqual([replacement, healthy]);
      await flush();
      expect(configured.configure.mock.calls.at(-1)?.[0].device).toBe(healthy);
      expect(healthy.submits).toBe(1);
      healthy.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(r.calls.errors).toEqual([]);
    } finally {
      unsubscribe();
      r.session.destroy();
      await manager.whenRecovered();
    }
    expect(manager.referenceCount).toBe(0);
  });

  it('exhausts two replacement attempts lost during notification and restores native playback once', async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const first = new FakeDevice();
    const replacements = [new FakeDevice(), new FakeDevice()];
    const requestDevice = vi.fn().mockResolvedValueOnce(first)
      .mockResolvedValueOnce(replacements[0]).mockResolvedValueOnce(replacements[1])
      .mockRejectedValue(new Error('no more devices'));
    const sleep = vi.fn(async (_ms: number) => undefined);
    const manager = new GpuDeviceManager({
      gpu: { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU,
      sleep, maxAttempts: 2,
    });
    manager.onRestored((ctx) => {
      (ctx.device as unknown as FakeDevice).lose({ reason: 'unknown', message: 'replacement lost' } as GPUDeviceLostInfo);
    });
    const r = rig({}, manager);
    try {
      await startSession(r);
      first.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      first.lose({ reason: 'unknown', message: 'A lost' } as GPUDeviceLostInfo);
      await flush();
      await manager.whenRecovered();
      await flush();
      expect(requestDevice).toHaveBeenCalledTimes(3);
      expect(sleep.mock.calls).toEqual([[1000], [2000]]);
      expect(manager.current).toBeNull();
      expect(r.factory.isGpuAvailable()).toBe(false);
      expect(r.calls.evaluations).toBe(1);
      expect(r.video.callback).toBeNull();
      expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
      await manager.whenRecovered();
      await flush();
      expect(r.calls.evaluations).toBe(1);
    } finally {
      r.session.destroy();
    }
  });

  it('ignores an older recovery completion when B is lost in nested microtasks and binds healthy C', async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const first = new FakeDevice();
    const replacement = new FakeDevice();
    const healthy = new FakeDevice();
    const requestDevice = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(replacement).mockResolvedValueOnce(healthy);
    let resume!: () => void;
    const delay = new Promise<void>((resolve) => { resume = resolve; });
    const sleep = vi.fn((_ms: number) => sleep.mock.calls.length === 1 ? Promise.resolve() : delay);
    const manager = new GpuDeviceManager({
      gpu: { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU,
      sleep,
    });
    manager.onRestored((ctx) => {
      if (ctx.device === replacement as unknown as GPUDevice) {
        queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => {
          replacement.lose({ reason: 'unknown', message: 'B lost in nested microtasks' } as GPUDeviceLostInfo);
        }))));
      }
    });
    const r = rig({}, manager);
    const configured = r.fx.canvasContext as { configure: ReturnType<typeof vi.fn> };
    try {
      await startSession(r);
      first.lose({ reason: 'unknown', message: 'A lost' } as GPUDeviceLostInfo);
      await flush();
      expect(manager.current).toBeNull();
      expect(r.calls.evaluations).toBe(0);
      expect(r.factory.isGpuAvailable()).toBe(true);
      resume();
      await manager.whenRecovered();
      await flush();
      expect(manager.current?.device).toBe(healthy);
      expect(configured.configure.mock.calls.at(-1)?.[0].device).toBe(healthy);
      expect(sleep.mock.calls).toEqual([[1000], [2000]]);
      expect(r.calls.evaluations).toBe(0);
      expect(r.factory.isGpuAvailable()).toBe(true);
      expect(healthy.submits).toBe(1);
      healthy.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    } finally {
      resume();
      r.session.destroy();
    }
  });

  it('skips an already-lost replacement B and renders on healthy C after A is lost', async () => {
    const first = new FakeDevice();
    const lostReplacement = new FakeDevice();
    const healthy = new FakeDevice();
    const requestDevice = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(lostReplacement).mockResolvedValueOnce(healthy);
    const sleep = vi.fn(async (_ms: number) => undefined);
    const manager = new GpuDeviceManager({
      gpu: { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU,
      sleep,
    });
    const restored = vi.fn();
    const unsubscribe = manager.onRestored(restored);
    const r = rig({}, manager);
    const configured = r.fx.canvasContext as { configure: ReturnType<typeof vi.fn> };
    try {
      await startSession(r);
      first.completeAll();
      await flush();
      lostReplacement.lose({ reason: 'unknown', message: 'B already lost' } as GPUDeviceLostInfo);
      first.lose({ reason: 'unknown', message: 'A lost' } as GPUDeviceLostInfo);
      await flush();
      await manager.whenRecovered();
      expect(manager.current?.device).toBe(healthy);
      expect(sleep.mock.calls).toEqual([[1000], [2000]]);
      expect(restored).toHaveBeenCalledExactlyOnceWith(manager.current, manager.generation);
      expect(configured.configure.mock.calls.map(([config]) => config.device)).toEqual([first, healthy]);
      expect(lostReplacement.submits).toBe(0);
      expect(healthy.submits).toBe(1);
      healthy.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(r.calls.errors).toEqual([]);
    } finally {
      unsubscribe();
      r.session.destroy();
      await manager.whenRecovered();
    }
    expect(manager.referenceCount).toBe(0);
  });

  it('rebinds and renders after device loss before initial acquisition settles', async () => {
    const first = new FakeDevice();
    const healthy = new FakeDevice();
    let recover!: () => void;
    const recoveryDelay = new Promise<void>((resolve) => { recover = resolve; });
    const requestDevice = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(healthy);
    const manager = new GpuDeviceManager({
      gpu: { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU,
      sleep: () => recoveryDelay,
    });
    const r = rig({}, manager);
    const configured = r.fx.canvasContext as { configure: ReturnType<typeof vi.fn> };
    const acquired = vi.fn();
    const acquire = manager.acquire.bind(manager);
    vi.spyOn(manager, 'acquire').mockImplementation(async () => {
      const gpu = await acquire();
      acquired();
      return gpu;
    });
    const lost = vi.fn(() => expect(acquired).not.toHaveBeenCalled());
    const unsubscribe = manager.onLost(lost);
    try {
      first.lose({ reason: 'unknown', message: 'lost during initial acquisition' } as GPUDeviceLostInfo);
      r.session.attach();
      await flush();
      expect(lost).toHaveBeenCalledTimes(1);
      expect(acquired).toHaveBeenCalledTimes(1);
      expect(manager.current).toBeNull();
      expect(configured.configure).not.toHaveBeenCalled();
      recover();
      await manager.whenRecovered();
      await flush();
      expect(manager.current?.device).toBe(healthy);
      expect(configured.configure).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ device: healthy }));
      expect(r.pipes).toHaveLength(1);
      expect(r.pipes[0]!.executed).toBe(1);
      expect(first.submits).toBe(0);
      expect(healthy.submits).toBe(1);
      healthy.completeAll();
      await flush();
      expect(r.calls.active).toBe(1);
      expect(r.calls.errors).toEqual([]);
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    } finally {
      unsubscribe();
      r.session.destroy();
      recover();
      await manager.whenRecovered();
    }
    expect(manager.referenceCount).toBe(0);
    expect(healthy.destroy).toHaveBeenCalledTimes(1);
  });

  it('ignores a restored notification for a context that is no longer current', async () => {
    const first = new FakeDevice();
    const healthy = new FakeDevice();
    let recover!: () => void;
    const recoveryDelay = new Promise<void>((resolve) => { recover = resolve; });
    const requestDevice = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(healthy);
    const manager = new GpuDeviceManager({
      gpu: { requestAdapter: async () => ({ features: new Set(), limits: {}, requestDevice }) } as unknown as GPU,
      sleep: () => recoveryDelay,
    });
    let notify!: (gpu: GpuContext, generation: number) => void;
    const onRestored = manager.onRestored.bind(manager);
    vi.spyOn(manager, 'onRestored').mockImplementation((listener) => {
      notify = listener;
      return onRestored(listener);
    });
    const r = rig({}, manager);
    const configured = r.fx.canvasContext as { configure: ReturnType<typeof vi.fn> };
    try {
      await startSession(r);
      const stale = manager.current!;
      first.lose({ reason: 'unknown', message: 'A lost' } as GPUDeviceLostInfo);
      await flush();
      expect(manager.current).toBeNull();
      notify(stale, manager.generation);
      expect(configured.configure).toHaveBeenCalledTimes(1);
      expect(first.submits).toBe(1);
      recover();
      await manager.whenRecovered();
      await flush();
      expect(configured.configure.mock.calls.map(([config]) => config.device)).toEqual([first, healthy]);
      expect(healthy.submits).toBe(1);
    } finally {
      r.session.destroy();
      recover();
      await manager.whenRecovered();
    }
  });

  it('ignores a cancelled medium build rejection after returning to active low quality', async () => {
    const r = rig({ qualityLevel: 1 });
    let initialize: ReturnType<typeof vi.spyOn> | undefined;
    try {
      await startSession(r);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      let rejectMedium!: (error: Error) => void;
      initialize = vi.spyOn(FakePipeline.prototype, 'initialize').mockReturnValueOnce(
        new Promise<void>((_resolve, reject) => (rejectMedium = reject)),
      );
      r.settings.current = { ...r.settings.current, qualityLevel: 2 };
      r.session.updateSettings(r.settings.current);
      expect(r.frame()).toBe(true);
      await flush();
      expect(r.pipes).toHaveLength(2);
      expect(r.pipes[1]!.tier).toBe('medium');
      r.settings.current = { ...r.settings.current, qualityLevel: 1 };
      r.session.updateSettings(r.settings.current);
      rejectMedium(new Error('obsolete medium build failed'));
      await flush();
      expect(r.calls.errors).toEqual([]);
      expect(r.pipes).toHaveLength(2);
      expect(r.pipes[0]!.destroyed).toBe(false);
      expect(r.pipes[1]!.destroyed).toBe(true);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(r.calls.telemetry.at(-1)?.activeModel).toBe('rtx-vsr-low');
    } finally {
      initialize?.mockRestore();
      r.session.destroy();
    }
  });

  it('defers the first medium redraw when its pipeline becomes ready with a full ring', async () => {
    const r = rig({ qualityLevel: 1 });
    let initialize: ReturnType<typeof vi.spyOn> | undefined;
    try {
      await startSession(r);
      expect(r.pipes[0]!.tier).toBe('low');
      expect(r.frame()).toBe(true);
      expect(r.frame()).toBe(true);
      expect(r.device.submits).toBe(3);
      r.video.paused = true;
      let releaseMedium!: () => void;
      initialize = vi.spyOn(FakePipeline.prototype, 'initialize').mockReturnValueOnce(
        new Promise<void>((resolve) => (releaseMedium = resolve)),
      );
      r.settings.current = { ...r.settings.current, qualityLevel: 2 };
      r.session.updateSettings(r.settings.current);
      expect(r.pipes).toHaveLength(1);
      r.device.completeOne();
      await flush();
      expect(r.pipes).toHaveLength(2);
      expect(r.pipes[0]!.executed).toBe(4);
      expect(r.pipes[1]!.tier).toBe('medium');
      expect(r.pipes[1]!.executed).toBe(0);
      expect(r.device.submits).toBe(4);
      releaseMedium();
      await flush();
      expect(r.pipes[0]!.destroyed).toBe(true);
      expect(r.pipes[1]!.executed).toBe(0);
      expect(r.device.submits).toBe(4);
      r.device.completeAll();
      await flush();
      expect(r.pipes[1]!.executed).toBe(1);
      expect(r.device.submits).toBe(5);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(r.device.submits).toBe(5);
    } finally {
      initialize?.mockRestore();
      r.session.destroy();
    }
  });

  it('redraws a paused frame when fresh intersections recover unchanged placement', async () => {
    const r = rig();
    r.parent.computed.position = 'relative';
    try {
      await startSession(r);
      r.device.completeAll();
      await flush();
      const canvas = r.parent.children[0]!;
      const intersections = (): void => r.fx.fireIntersections([r.video, canvas].map((target) => ({
        target, intersectionRect: target.getBoundingClientRect(),
      })));
      intersections();
      r.video.paused = true;
      r.device.completeAll();
      await flush();
      const submits = r.device.submits;
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      r.video.rect.left += 100;
      r.parent.rect.left += 100;
      r.session.refreshGeometry();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
      expect(r.device.submits).toBe(submits);
      intersections();
      await flush();
      expect(r.device.submits).toBe(submits + 1);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(r.device.submits).toBe(submits + 1);
    } finally {
      r.session.destroy();
    }
  });

  it('redraws a paused presented frame after the canvas is removed and periodically remounted', async () => {
    const r = rig();
    try {
      await startSession(r);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      r.video.paused = true;
      r.device.completeAll();
      await flush();
      const canvas = r.parent.children[0]!;
      const submits = r.device.submits;
      canvas.remove();
      r.fx.fireChildList(r.parent);
      expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
      r.session.refreshGeometry();
      await flush();
      expect(r.parent.children[0]).toBe(canvas);
      expect(r.device.submits).toBe(submits + 1);
      expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    } finally {
      r.session.destroy();
    }
  });

  it.each(['scopes first', 'completion first'] as const)('telemetry counts each completed and validated frame once (%s)', async (order) => {
    const r = rig();
    try {
      await startSession(r);
      r.tick(500);
      r.device.completeAll();
      await flush();
      expect(r.calls.telemetry.at(-1)?.presentedFrames).toBe(1);
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      const furtherFrames = 4;
      for (let i = 0; i < furtherFrames; i++) {
        let release!: (v: ScopeResult) => void;
        if (order === 'completion first') {
          r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
        }
        r.tick(500);
        expect(r.frame()).toBe(true);
        await flush();
        expect(r.calls.telemetry.at(-1)?.presentedFrames).toBe(1 + i);
        r.device.completeAll();
        await flush();
        if (order === 'completion first') {
          expect(r.calls.telemetry.at(-1)?.presentedFrames).toBe(1 + i);
          r.tick(500);
          release(null);
          await flush();
        }
        expect(r.calls.telemetry.at(-1)?.presentedFrames).toBe(2 + i);
      }
      expect(r.calls.telemetry.at(-1)?.presentedFrames).toBe(1 + furtherFrames);
      expect(r.calls.active).toBe(1);
    } finally {
      r.session.destroy();
    }
  });

  it('a replacement completion presents after a stale first validation', async () => {
    const r = rig();
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    await startSession(r);
    r.video.paused = true;
    r.video.dispatch('seeking');
    r.video.dispatch('seeked');
    expect(r.pipes[0]!.executed).toBe(2);
    release(null);
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.session.destroy();
  });

  it('first validation keeps the native picture visible until GPU completion', async () => {
    const r = rig();
    await startSession(r);
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.session.destroy();
  });

  it('a paused resize retries after pending first validation and a full ring', async () => {
    const r = rig();
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    await startSession(r);
    r.frame();
    r.frame();
    r.video.paused = true;
    r.video.rect.width = 1000;
    r.video.rect.height = 600;
    r.fx.fireResize();
    await flush();
    release(null);
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(r.pipes).toHaveLength(2);
    expect(r.pipes[1]!.executed).toBe(1);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.session.destroy();
  });

  it.each(['sharpness', 'resize', 'seeked'] as const)('a full ring coalesces paused %s redraws until a slot is available', async (change) => {
    const r = rig();
    await startSession(r);
    r.frame();
    r.frame();
    r.video.paused = true;
    for (let i = 0; i < 3; i++) {
      if (change === 'sharpness') r.session.updateSettings({ ...r.settings.current, sharpness: 0.1 * i });
      if (change === 'resize') {
        r.video.rect.width = 900 + i;
        r.video.rect.height = 600;
        r.fx.fireResize();
      }
      if (change === 'seeked') r.video.dispatch('seeked');
    }
    await flush();
    expect(r.device.submits).toBe(3);
    r.device.completeAll();
    await flush();
    expect(r.device.submits).toBe(4);
    r.device.completeAll();
    await flush();
    expect(r.device.submits).toBe(4);
    r.session.destroy();
  });

  it.each(['missing runtime id', 'sendMessage invalidated'] as const)(
    'destroys a paused unchanged session after %s on periodic evaluation', async (invalidation) => {
      const r = rig({ hudEnabled: true });
      r.video.paused = true;
      const doc = Object.assign(r.fx.doc, {
        documentElement: r.parent,
        querySelectorAll: (selector: string) => selector === 'video' ? [r.video] : [r.parent, r.video],
      });
      let periodic!: () => void;
      vi.stubGlobal('document', doc);
      vi.stubGlobal('window', {
        document: doc, location: { origin: 'https://example.com' }, devicePixelRatio: 1,
        getComputedStyle: r.fx.env().getComputedStyle,
        addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout,
        setInterval: (cb: () => void) => { periodic = cb; return 1; }, clearInterval: vi.fn(),
      });
      const runtime = { id: 'test-extension' as string | undefined, sendMessage: vi.fn(async () => undefined) };
      vi.stubGlobal('chrome', { runtime });
      let context!: VideoSessionContext;
      const send = vi.fn(async () => undefined);
      const controller = new ContentController({
        settings: r.settings.current, siteRule: defaultSiteRule(), hostname: 'example.com', isTopFrame: true,
        getUrl: () => 'https://example.com/player', send,
        sessionFactory: { isGpuAvailable: () => true, create: (next) => { context = next; return r.session; } },
      });
      try {
        controller.start();
        await flush();
        r.device.completeAll();
        await flush();
        context.reportActive();
        expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
        expect(r.parent.children).toHaveLength(3);
        await new Promise((resolve) => setTimeout(resolve, 60));
        const reports = send.mock.calls.length;
        periodic();
        await new Promise((resolve) => setTimeout(resolve, 90));
        expect(send).toHaveBeenCalledTimes(reports);
        if (invalidation === 'missing runtime id') runtime.id = undefined;
        else runtime.sendMessage.mockImplementation(() => { throw new Error('Extension context invalidated.'); });
        for (let i = 0; i < 3; i++) {
          periodic();
          await new Promise((resolve) => setTimeout(resolve, 40));
        }
        expect(r.pipes[0]!.destroyed).toBe(true);
        expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
        expect(r.video.style.getPropertyValue('position')).toBe('');
        expect(r.parent.children).toEqual([r.video]);
        expect(controller.reports()).toEqual([]);
      } finally {
        controller.stop();
      }
    },
  );

  it('periodic evaluation refreshes paused object-position geometry and redraws the still frame', async () => {
    const r = rig();
    r.video.paused = true;
    r.video.rect.height = 600;
    r.video.style.removeProperty('object-position');
    const doc = Object.assign(r.fx.doc, {
      documentElement: r.parent,
      querySelectorAll: (selector: string) => selector === 'video' ? [r.video] : [r.parent, r.video],
    });
    let periodic!: () => void;
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', {
      document: doc,
      location: { origin: 'https://example.com' },
      devicePixelRatio: 1,
      getComputedStyle: r.fx.env().getComputedStyle,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      setTimeout,
      clearTimeout,
      setInterval: (cb: () => void) => { periodic = cb; return 1; },
      clearInterval: () => undefined,
    });
    const controller = new ContentController({
      settings: r.settings.current, siteRule: defaultSiteRule(), hostname: 'example.com', isTopFrame: true,
      getUrl: () => 'https://example.com/player', send: async () => undefined,
      sessionFactory: { isGpuAvailable: () => true, create: () => r.session },
    });
    try {
      controller.start();
      await flush();
      r.device.completeAll();
      await flush();
      const canvas = r.parent.children[0]!;
      expect(canvas.style.getPropertyValue('top')).toBe('75px');
      const submits = r.device.submits;
      r.video.computed.objectPosition = 'left top';
      periodic();
      await new Promise((resolve) => setTimeout(resolve, 40));
      expect(canvas.style.getPropertyValue('top')).toBe('0px');
      expect(r.device.submits).toBe(submits + 1);
    } finally {
      controller.stop();
    }
  });

  it('rejects an active session when the site object-position becomes an unsupported nested calc', async () => {
    const r = rig();
    r.video.paused = true;
    r.video.rect.height = 600;
    r.video.style.removeProperty('object-position');
    const doc = Object.assign(r.fx.doc, {
      documentElement: r.parent,
      querySelectorAll: (selector: string) => selector === 'video' ? [r.video] : [r.parent, r.video],
    });
    let periodic!: () => void;
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', {
      document: doc,
      location: { origin: 'https://example.com' },
      devicePixelRatio: 1,
      getComputedStyle: r.fx.env().getComputedStyle,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      setTimeout,
      clearTimeout,
      setInterval: (cb: () => void) => { periodic = cb; return 1; },
      clearInterval: () => undefined,
    });
    const controller = new ContentController({
      settings: r.settings.current, siteRule: defaultSiteRule(), hostname: 'example.com', isTopFrame: true,
      getUrl: () => 'https://example.com/player', send: async () => undefined,
      sessionFactory: { isGpuAvailable: () => true, create: () => r.session },
    });
    try {
      controller.start();
      await flush();
      r.device.completeAll();
      await flush();
      const canvas = r.parent.children[0]!;
      expect(canvas.style.getPropertyValue('top')).toBe('75px');
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      const submits = r.device.submits;
      r.video.computed.objectPosition = '50% calc(100% - min(10%, 40px))';
      periodic();
      await new Promise((resolve) => setTimeout(resolve, 40));
      expect(controller.reports()[0]?.reason).toBe('unsupported-layout');
      expect(r.video.style.getPropertyValue('object-position')).toBe('');
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      expect(r.device.submits).toBe(submits);
    } finally {
      controller.stop();
    }
  });

  it('signals unsupported-layout to the controller after the placement self-check fails', async () => {
    const r = rig();
    await startSession(r);
    r.device.completeAll();
    await flush();
    const canvas = r.parent.children[0]!;
    const rect = canvas.getBoundingClientRect();
    canvas.canvasRect = { ...rect, left: rect.left + 1 };
    const submits = r.device.submits;
    r.session.refreshGeometry();
    expect(r.session.isLayoutSupported()).toBe(false);
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(r.calls.evaluations).toBe(1);
    expect(r.video.callback).toBeNull();
    expect(r.device.submits).toBe(submits);
    r.session.destroy();
  });

  it('renders with a static parent without patching its position', async () => {
    const r = rig();
    r.parent.forceStaticPosition = true;
    await startSession(r);
    expect(r.session.isLayoutSupported()).toBe(true);
    expect(r.parent.style.getPropertyValue('position')).toBe('');
    expect(r.video.callback).not.toBeNull();
    expect(r.device.submits).toBe(1);
    expect(r.calls.evaluations).toBe(0);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.session.refreshGeometry();
    expect(r.session.isLayoutSupported()).toBe(true);
    expect(r.parent.style.getPropertyValue('position')).toBe('');
    r.session.destroy();
    expect(r.parent.style.getPropertyValue('position')).toBe('');
  });

  it('restores a paused picture before resizing and re-renders it before hiding again', async () => {
    const r = rig();
    await startSession(r);
    r.device.completeAll();
    await flush();
    r.video.paused = true;
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    r.video.rect = { ...r.video.rect, width: 1000, height: 600 };
    r.parent.rect = { ...r.video.rect };
    r.fx.fireResize();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(r.parent.children[0]!.style.getPropertyValue('visibility')).toBe('hidden');
    await flush();
    expect(r.pipes).toHaveLength(2);
    expect(r.pipes[1]!.executed).toBe(1);
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    release(null);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.session.destroy();
  });

  it('keeps the video visible until the first frame passed its error scopes, then hides it', async () => {
    const r = rig();
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    await startSession(r);
    // submitted, even completed on the GPU, but not validated yet: the video must stay visible
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(r.video.objectPositionWrites).toEqual([]);
    expect(r.calls.active).toBe(0);
    release(null);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    expect(r.calls.active).toBe(1);
    r.session.destroy();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
  });

  it('syncs site object-position changes after 500 ms without syncing every frame', async () => {
    const r = rig();
    r.video.rect.height = 600;
    r.video.style.removeProperty('object-position');
    await startSession(r);
    const canvas = r.parent.children[0]!;
    expect(canvas.style.getPropertyValue('top')).toBe('75px');
    r.video.computed.objectPosition = 'left top';
    r.device.completeAll();
    await flush();
    // First presentation flushes the hidden style with transitions suppressed. Count only the
    // following normal frame so this assertion still detects per-frame site-style suspension.
    const computed = vi.spyOn(r.video.style, 'getPropertyValue');
    r.tick(499);
    r.frame();
    expect(canvas.style.getPropertyValue('top')).toBe('75px');
    // A normal frame must not suspend the picture override to read site styles.
    expect(computed.mock.calls.filter(([name]) => name === 'object-position')).toHaveLength(0);
    r.device.completeAll();
    await flush();
    r.tick(1);
    r.frame();
    expect(canvas.style.getPropertyValue('top')).toBe('0px');
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.session.destroy();
  });

  it('keeps the picture visible when seeking invalidates a delayed validation', async () => {
    const r = rig();
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    await startSession(r);
    r.device.completeAll();
    await flush();
    r.video.dispatch('seeking');
    release(null);
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(r.calls.active).toBe(0);
    r.session.destroy();
  });

  it('a validation error on the first frame fails the session; the video is never hidden', async () => {
    const r = rig();
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    await startSession(r);
    // more frames arrive while the first one is being validated: none of them may hide the video
    r.tick(16);
    r.frame();
    r.tick(16);
    r.frame();
    expect(r.pipes[0]!.executed).toBe(3);
    expect(r.video.objectPositionWrites).toEqual([]);
    release({ message: 'Invalid BindGroup' });
    await flush();
    expect(r.calls.errors).toEqual(['GPU error in rtx-vsr-low: Invalid BindGroup']);
    expect(r.video.objectPositionWrites).toEqual([]);
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(r.calls.active).toBe(0);
    // the loop stopped: no further frame is rendered
    expect(r.video.callback).toBeNull();
    r.session.destroy();
  });

  it('an out-of-memory error on a later frame fails the session and restores the video', async () => {
    const r = rig();
    await startSession(r);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    r.device.popResults.push(Promise.resolve(null), Promise.resolve({ message: 'Out of memory' }));
    r.tick(16);
    expect(r.frame()).toBe(true);
    await flush();
    expect(r.calls.errors).toEqual(['GPU error in rtx-vsr-low: Out of memory']);
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    expect(r.video.callback).toBeNull();
    r.session.destroy();
  });

  it('a new pipeline (resolution change) is validated again before the video is hidden', async () => {
    const r = rig();
    await startSession(r);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    let release!: (v: ScopeResult) => void;
    r.device.popResults.push(new Promise((res) => (release = res)), Promise.resolve(null));
    r.video.videoWidth = 480;
    r.video.videoHeight = 270;
    r.tick(16);
    r.frame(); // size mismatch: video restored, new pipeline built and its first frame submitted
    await flush();
    expect(r.pipes).toHaveLength(2);
    expect(r.pipes[0]!.destroyed).toBe(true);
    expect(r.pipes[1]!.executed).toBe(1);
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    release(null);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    expect(r.calls.errors).toEqual([]);
    r.session.destroy();
  });
});

// ---------------------------------------------------------------------------------------------
// VideoFrame lifetime
// ---------------------------------------------------------------------------------------------

describe('VideoSRSession: every VideoFrame is closed', () => {
  it.each(['hdr', 'tainted'] as const)('restarts on a new SDR source before the deferred %s evaluation', async (reason) => {
    const r = rig();
    const doc = Object.assign(r.fx.doc, {
      documentElement: r.parent,
      querySelectorAll: (selector: string) => selector === 'video' ? [r.video] : [r.parent, r.video],
    });
    let periodic!: () => void;
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', {
      document: doc, location: { origin: 'https://example.com' }, devicePixelRatio: 1,
      getComputedStyle: r.fx.env().getComputedStyle,
      addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout,
      setInterval: (cb: () => void) => { periodic = cb; return 1; }, clearInterval: vi.fn(),
    });
    const create = vi.spyOn(r.factory, 'create');
    vi.spyOn(r.factory, 'isGpuAvailable').mockReturnValue(true);
    const controller = new ContentController({
      settings: r.settings.current, siteRule: defaultSiteRule(), hostname: 'example.com', isTopFrame: true,
      getUrl: () => 'https://example.com/player', send: async () => undefined, sessionFactory: r.factory,
    });
    let validate!: (result: ScopeResult) => void;
    try {
      controller.start();
      await flush();
      expect(create).toHaveBeenCalledTimes(1);
      expect(r.pipes[0]?.executed).toBe(1);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      if (reason === 'hdr') FakeVideoFrame.transfer = 'pq';
      else FakeVideoFrame.throwNext = new DOMException('tainted frame', 'SecurityError');
      r.tick(16);
      expect(r.frame()).toBe(true);
      expect(r.video.callback).toBeNull();
      expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
      // Discovery observes the new source synchronously, before the 30 ms gate debounce.
      FakeVideoFrame.transfer = 'bt709';
      r.video.setAttribute('src', 'https://example.com/new-sdr.mp4');
      r.device.popResults.push(new Promise((resolve) => { validate = resolve; }), Promise.resolve(null));
      periodic();
      await flush();
      expect(create).toHaveBeenCalledTimes(2);
      expect(r.pipes[0]!.destroyed).toBe(true);
      expect(r.pipes[1]!.executed).toBe(1);
      r.device.completeAll();
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
      validate(null);
      await flush();
      expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
      expect(controller.reports()[0]!.state).toBe('active');
      expect(FakeVideoFrame.live).toBe(0);
    } finally {
      validate?.(null);
      controller.stop();
    }
  });

  it('stops and requests an hdr gate failure when a processed frame switches to PQ', async () => {
    const r = rig();
    await startSession(r);
    r.device.completeAll();
    await flush();
    expect(r.video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    FakeVideoFrame.transfer = 'pq';
    r.tick(16);
    expect(r.frame()).toBe(true);
    expect(r.pipes[0]!.executed).toBe(1);
    expect(FakeVideoFrame.live).toBe(0);
    expect(r.latches.isHdrLatched(r.video)).toBe(true);
    expect(r.calls.evaluations).toBe(1);
    expect(r.calls.errors).toEqual([]);
    expect(r.video.callback).toBeNull();
    expect(r.video.style.getPropertyValue('object-position')).toBe('40% 60%');
    r.session.destroy();
  });

  it('on the pipeline-build path, every presented frame, a resize and a failing encode', async () => {
    const r = rig();
    await startSession(r);
    // first callback built the pipeline (frame 1, closed), the swap rendered frame 2
    expect(FakeVideoFrame.created).toBe(2);
    expect(FakeVideoFrame.live).toBe(0);
    for (let i = 0; i < 50; i++) {
      r.tick(16);
      r.frame();
      expect(FakeVideoFrame.live).toBe(0);
      if (i % 3 === 0) r.device.completeAll();
      await Promise.resolve();
    }
    r.device.completeAll();
    await flush();
    r.video.videoWidth = 320;
    r.video.videoHeight = 180;
    r.tick(16);
    r.frame();
    expect(FakeVideoFrame.live).toBe(0);
    await flush();
    // no decoded frame right now (InvalidStateError): nothing to close, the loop keeps going
    FakeVideoFrame.throwNext = Object.assign(new Error('no frame'), { name: 'InvalidStateError' });
    r.tick(16);
    r.frame();
    expect(r.video.callback).not.toBeNull();
    // an exception while encoding still closes the frame (and fails the session)
    r.pipes[r.pipes.length - 1]!.failExecute = true;
    r.tick(16);
    r.frame();
    expect(r.calls.errors).toEqual(['encoder failure']);
    expect(FakeVideoFrame.created).toBeGreaterThan(50);
    expect(FakeVideoFrame.live).toBe(0);
    r.session.destroy();
  });
});

// ---------------------------------------------------------------------------------------------
// Sharpness
// ---------------------------------------------------------------------------------------------

describe('VideoSRSession: sharpness setting', () => {
  it('re-renders exactly once when sharpness changes while paused', async () => {
    const r = rig();
    await startSession(r);
    r.video.paused = true;
    const executed = r.pipes[0]!.executed;
    r.settings.current = { ...r.settings.current, sharpness: 0.3 };
    r.session.updateSettings(r.settings.current);
    expect(r.pipes[0]!.executed).toBe(executed + 1);
    await flush();
    expect(r.pipes[0]!.executed).toBe(executed + 1);
    r.session.destroy();
  });

  it('is applied to every new pipeline and forwarded on a settings change', async () => {
    const r = rig({ sharpness: 0.8 });
    await startSession(r);
    expect(r.pipes[0]!.sharpness).toEqual([0.8]);
    r.settings.current = { ...r.settings.current, sharpness: 0.3 };
    r.session.updateSettings(r.settings.current);
    expect(r.pipes[0]!.sharpness).toEqual([0.8, 0.3]);
    // an unrelated change does not re-send it
    r.settings.current = { ...r.settings.current, hudEnabled: true };
    r.session.updateSettings(r.settings.current);
    expect(r.pipes[0]!.sharpness).toEqual([0.8, 0.3]);
    // a rebuilt pipeline (source resolution change) starts with the current value
    r.video.videoWidth = 480;
    r.video.videoHeight = 270;
    r.tick(16);
    r.frame();
    await flush();
    expect(r.pipes).toHaveLength(2);
    expect(r.pipes[1]!.sharpness).toEqual([0.3]);
    r.session.destroy();
  });
});
