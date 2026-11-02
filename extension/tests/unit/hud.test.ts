import { describe, expect, it } from 'vitest';
import { HUD_CLASS, HUD_LINE_COUNT, VideoHud, boxInContainer, formatHudLines, hudPosition } from '../../src/content/hud';
import type { SessionTelemetry, VideoReport } from '../../src/shared/messages';

const telemetry = (over: Partial<SessionTelemetry> = {}): SessionTelemetry => ({
  inputWidth: 960,
  inputHeight: 540,
  outputWidth: 1920,
  outputHeight: 1080,
  fps: 59.94,
  gpuTimeMeanMs: 4.2,
  gpuTimeP95Ms: 5.1,
  droppedFrames: 3,
  activeModel: 'rtx-vsr-low',
  timingSource: 'timestamp',
  presentedFrames: 1200,
  pipelineInits: 1,
  ...over,
});

const report = (over: Partial<VideoReport> = {}): VideoReport => ({
  id: 'v1',
  state: 'active',
  point: null,
  reason: null,
  reasonText: null,
  sourceWidth: 960,
  sourceHeight: 540,
  targetWidth: 1920,
  targetHeight: 1080,
  telemetry: telemetry(),
  error: null,
  ...over,
});

describe('formatHudLines', () => {
  it('active session: resolution, scale, model, GPU mean/p95, fps, dropped, timing source', () => {
    // 1920/960 = 2.00x; 4.2 -> "4.20"; 59.94 -> "59.9"
    expect(formatHudLines(report())).toEqual([
      'WebGPU VSR · ACTIVE',
      'Res   960×540 → 1920×1080 (2.00x)',
      'Model rtx-vsr-low',
      'GPU   4.20 ms mean · 5.10 ms p95 (GPU timestamps)',
      'FPS   59.9 · dropped 3',
    ]);
  });

  it('CPU cadence timing and non-integer scale', () => {
    // 1280/720 = 1.777... -> "1.78x"; 11.666 -> "11.67"; 23.976 -> "24.0"
    const lines = formatHudLines(
      report({ telemetry: telemetry({ inputWidth: 720, inputHeight: 404, outputWidth: 1280, outputHeight: 718, gpuTimeMeanMs: 11.666, gpuTimeP95Ms: 14, fps: 23.976, droppedFrames: 0, activeModel: 'rtx-vsr-medium', timingSource: 'cpu' }) }),
    );
    expect(lines[1]).toBe('Res   720×404 → 1280×718 (1.78x)');
    expect(lines[2]).toBe('Model rtx-vsr-medium');
    expect(lines[3]).toBe('GPU   11.67 ms mean · 14.00 ms p95 (CPU cadence)');
    expect(lines[4]).toBe('FPS   24.0 · dropped 0');
  });

  it('ineligible: shows the failing point and reason, no metrics', () => {
    const lines = formatHudLines(
      report({ state: 'ineligible', point: 5, reason: 'no-upscale-needed', reasonText: 'Display size is less than 1.15x the source resolution', sourceWidth: 1920, sourceHeight: 1080, telemetry: null }),
    );
    expect(lines).toEqual([
      'WebGPU VSR · OFF · [5] Display size is less than 1.15x the source resolution',
      'Res   1920×1080 → 1920×1080 target',
      'Model –',
      'GPU   –',
      'FPS   –',
    ]);
  });

  it('eligible without a session and error states', () => {
    expect(formatHudLines(report({ state: 'eligible', telemetry: null }))[0]).toBe('WebGPU VSR · ELIGIBLE · starting');
    expect(formatHudLines(report({ state: 'eligible', telemetry: null, targetWidth: 0, targetHeight: 0 }))[1]).toBe('Res   960×540 → – target');
    expect(formatHudLines(report({ state: 'error', telemetry: null, error: 'GPU device lost' }))[0]).toBe('WebGPU VSR · ERROR · GPU device lost');
  });

  it('always yields HUD_LINE_COUNT lines', () => {
    for (const r of [report(), report({ state: 'eligible', telemetry: null })]) expect(formatHudLines(r)).toHaveLength(HUD_LINE_COUNT);
  });
});

describe('HUD placement math', () => {
  it('boxInContainer subtracts the container origin and border, adds scroll', () => {
    // video at (130, 95) 640x360; container at (100, 60) with 2px border, scrolled 0/10
    expect(boxInContainer({ left: 130, top: 95, width: 640, height: 360 }, { left: 100, top: 60, width: 900, height: 600 }, { left: 2, top: 2 }, { left: 0, top: 10 })).toEqual({
      left: 28,
      top: 43,
      width: 640,
      height: 360,
    });
  });

  it('hudPosition sticks to the top-right corner with an 8px margin, never past the left edge', () => {
    // right edge 100+640 = 740; minus width 200 and margin 8 -> 532
    expect(hudPosition({ left: 100, top: 50, width: 640, height: 360 }, 200)).toEqual({ left: 532, top: 58 });
    // narrow box: 100+150-200-8 = 42 < 108 -> clamp to left + 8
    expect(hudPosition({ left: 100, top: 50, width: 150, height: 150 }, 200)).toEqual({ left: 108, top: 58 });
  });
});

// ---------------------------------------------------------------------------------------------
// Minimal DOM fakes for the VideoHud component
// ---------------------------------------------------------------------------------------------

class FakeStyle {
  props = new Map<string, { value: string; priority: string }>();
  setProperty(name: string, value: string, priority = ''): void {
    this.props.set(name, { value, priority });
  }
  getPropertyValue(name: string): string {
    return this.props.get(name)?.value ?? '';
  }
}

class FakeEl {
  style = new FakeStyle();
  children: FakeEl[] = [];
  parentElement: FakeEl | null = null;
  className = '';
  attrs: Record<string, string> = {};
  shadowRoot: FakeEl | null = null;
  attachedShadow: FakeEl | null = null;
  shadowMode: ShadowRootMode | null = null;
  textWrites = 0;
  private text = '';
  offsetWidth = 200;
  offsetParent: FakeEl | null = null;
  rect = { left: 0, top: 0, width: 0, height: 0 };
  clientLeft = 0;
  clientTop = 0;
  scrollLeft = 0;
  scrollTop = 0;
  isConnected = true;
  ownerDocument = { defaultView: { scrollX: 0, scrollY: 0 } };
  get textContent(): string {
    return this.text;
  }
  set textContent(v: string) {
    this.text = v;
    this.textWrites++;
  }
  setAttribute(n: string, v: string): void {
    this.attrs[n] = v;
  }
  attachShadow(options: ShadowRootInit): FakeEl {
    this.shadowMode = options.mode;
    this.attachedShadow = new FakeEl();
    if (options.mode === 'open') this.shadowRoot = this.attachedShadow;
    return this.attachedShadow;
  }
  appendChild(c: FakeEl): FakeEl {
    c.parentElement?.children.splice(c.parentElement.children.indexOf(c), 1);
    this.children.push(c);
    c.parentElement = this;
    return c;
  }
  remove(): void {
    if (!this.parentElement) return;
    this.parentElement.children.splice(this.parentElement.children.indexOf(this), 1);
    this.parentElement = null;
  }
  getBoundingClientRect(): { left: number; top: number; width: number; height: number } {
    return this.rect;
  }
}

class Clock {
  t = 0;
  timers: Array<{ at: number; fn: () => void; id: number }> = [];
  private seq = 0;
  now = (): number => this.t;
  set = (fn: () => void, ms: number): number => {
    const id = ++this.seq;
    this.timers.push({ at: this.t + ms, fn, id });
    return id;
  };
  clear = (h: unknown): void => {
    this.timers = this.timers.filter((x) => x.id !== h);
  };
  advance(ms: number): void {
    this.t += ms;
    const due = this.timers.filter((x) => x.at <= this.t);
    this.timers = this.timers.filter((x) => x.at > this.t);
    due.forEach((x) => x.fn());
  }
}

function makeHud(): { hud: VideoHud; container: FakeEl; clock: Clock; panel: FakeEl; lines: FakeEl[] } {
  const clock = new Clock();
  const container = new FakeEl();
  const doc = { createElement: () => new FakeEl() } as unknown as Document;
  const video = new FakeEl();
  const hud = new VideoHud(container as unknown as HTMLElement,
    { video: video as unknown as HTMLVideoElement, document: doc, now: clock.now, setTimeout: clock.set, clearTimeout: clock.clear,
      getComputedStyle: (element) => (element as unknown as FakeEl).style as unknown as CSSStyleDeclaration });
  const el = container.children[0] as FakeEl;
  const panel = el.shadowRoot!.children[1]!;
  return { hud, container, clock, panel, lines: panel.children };
}

describe('VideoHud', () => {
  it('builds a non-interactive panel with one element per line inside an open shadow root', () => {
    const { hud, container, lines } = makeHud();
    const el = container.children[0] as FakeEl;
    expect(el).toBe(hud.element as unknown as FakeEl);
    expect(el.className).toBe(HUD_CLASS);
    expect(el.shadowMode).toBe('open');
    expect(el.shadowRoot).toBe(el.attachedShadow);
    expect(el.children).toHaveLength(0);
    const css = el.shadowRoot!.children[0]!.textContent;
    expect(css).toContain('pointer-events: none !important');
    expect(css).toContain('position: absolute !important');
    expect(lines).toHaveLength(HUD_LINE_COUNT);
  });

  it('keeps host light DOM and attributes unchanged across content updates', () => {
    const { hud, container, clock, lines } = makeHud();
    const el = container.children[0]!;
    const attrs = { ...el.attrs };
    const styles = [...el.style.props];
    const video = new FakeEl();
    video.rect = { left: 20, top: 40, width: 800, height: 450 };
    hud.setAnchor(video as unknown as Element);
    for (let i = 0; i < 4; i++) {
      hud.update(report({ telemetry: telemetry({ fps: 30 + i }) }));
      clock.advance(300);
      expect(el.children).toHaveLength(0);
      expect(el.textWrites).toBe(0);
      expect(el.attrs).toEqual(attrs);
      expect([...el.style.props]).toEqual(styles);
    }
    expect(lines[4]!.textContent).toContain('33.0');
    expect(hud.renders).toBe(4);
  });

  it('renders at most once per 250 ms and always shows the latest report', () => {
    const { hud, clock, lines } = makeHud();
    hud.update(report());
    expect(hud.renders).toBe(1);
    expect(lines[4]?.textContent).toBe('FPS   59.9 · dropped 3');

    clock.advance(100);
    hud.update(report({ telemetry: telemetry({ droppedFrames: 4 }) }));
    clock.advance(20);
    hud.update(report({ telemetry: telemetry({ droppedFrames: 5 }) }));
    expect(hud.renders).toBe(1);
    expect(clock.timers).toHaveLength(1);
    expect(clock.timers[0]?.at).toBe(250);

    clock.advance(129); // t = 249
    expect(hud.renders).toBe(1);
    clock.advance(1); // t = 250
    expect(hud.renders).toBe(2);
    expect(lines[4]?.textContent).toBe('FPS   59.9 · dropped 5');

    clock.advance(1000);
    hud.update(report());
    expect(hud.renders).toBe(3); // idle long enough: immediate
  });

  it('writes only the lines whose text changed', () => {
    const { hud, clock, lines } = makeHud();
    hud.update(report());
    const before = lines.map((l) => l.textWrites);
    clock.advance(300);
    hud.update(report({ telemetry: telemetry({ fps: 30 }) }));
    const after = lines.map((l) => l.textWrites);
    expect(after.map((n, i) => n - (before[i] as number))).toEqual([0, 0, 0, 0, 1]);
  });

  it('positions itself at the anchor top-right in the container coordinate space', () => {
    const { hud, container, panel } = makeHud();
    const el = container.children[0] as FakeEl;
    container.rect = { left: 100, top: 60, width: 1000, height: 700 };
    el.offsetParent = container;
    panel.offsetWidth = 220;
    const video = new FakeEl();
    video.rect = { left: 120, top: 80, width: 800, height: 450 };
    hud.setAnchor(video as unknown as Element);
    expect(panel.style.getPropertyValue('visibility')).toBe('hidden'); // nothing rendered yet
    hud.update(report());
    // box in container: (20, 20) 800x450 -> left = 20 + 800 - 220 - 8 = 592, top = 28
    expect(panel.style.getPropertyValue('left')).toBe('592px');
    expect(panel.style.getPropertyValue('top')).toBe('28px');
    expect(panel.style.getPropertyValue('visibility')).toBe('visible');

    video.isConnected = false;
    hud.reposition();
    expect(panel.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('moveTo re-parents; destroy removes the element and cancels a pending render', () => {
    const { hud, container, clock } = makeHud();
    const other = new FakeEl();
    hud.moveTo(other as unknown as HTMLElement);
    expect(container.children).toHaveLength(0);
    expect(other.children).toHaveLength(1);

    hud.update(report());
    clock.advance(10);
    hud.update(report({ telemetry: telemetry({ fps: 1 }) }));
    expect(clock.timers).toHaveLength(1);
    hud.destroy();
    expect(other.children).toHaveLength(0);
    expect(clock.timers).toHaveLength(0);
    hud.update(report());
    expect(hud.renders).toBe(1);
  });
});
