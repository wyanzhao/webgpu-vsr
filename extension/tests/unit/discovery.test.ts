import { beforeEach, describe, expect, it } from 'vitest';
import {
  DETACH_GRACE_PERIOD_MS,
  POLL_INTERVAL_MS,
  VideoDiscovery,
  type DiscoveryEnvironment,
  type DiscoveryEventMap,
} from '../../src/content/discovery';

// ---------------------------------------------------------------------------------------------
// Fakes: a manual clock, a document with mutable video membership and a capturable observer.
// ---------------------------------------------------------------------------------------------

class FakeClock {
  now = 0;
  private seq = 0;
  private timers = new Map<number, { at: number; fn: () => void; every: number | null }>();

  setTimeout = (fn: () => void, ms: number): number => this.add(fn, ms, null);
  setInterval = (fn: () => void, ms: number): number => this.add(fn, ms, ms);
  clear = (h: unknown): void => {
    this.timers.delete(h as number);
  };

  private add(fn: () => void, ms: number, every: number | null): number {
    const id = ++this.seq;
    this.timers.set(id, { at: this.now + ms, fn, every });
    return id;
  }

  get pending(): number {
    return this.timers.size;
  }

  /** Advances time, firing due timers in order (intervals re-arm). */
  advance(ms: number): void {
    const end = this.now + ms;
    for (;;) {
      let nextId = -1;
      let nextAt = Infinity;
      for (const [id, t] of this.timers) {
        if (t.at <= end && t.at < nextAt) {
          nextAt = t.at;
          nextId = id;
        }
      }
      if (nextId < 0) break;
      const t = this.timers.get(nextId)!;
      this.now = t.at;
      if (t.every === null) this.timers.delete(nextId);
      else t.at += t.every;
      t.fn();
    }
    this.now = end;
  }
}

class FakeVideo {
  readonly nodeType = 1;
  readonly localName = 'video';
  isConnected = true;
  parentElement: object | null = {};
  currentSrc = '';
  attrs: Record<string, string> = {};
  /** Selectors this element matches (stands in for Element.matches). */
  matching = new Set<string>();
  constructor(src = '') {
    this.currentSrc = src;
  }
  matches(selector: string): boolean {
    if (selector.startsWith('!!')) throw new SyntaxError('bad selector');
    return this.matching.has(selector);
  }
  getAttribute(name: string): string | null {
    return this.attrs[name] ?? null;
  }
  getElementsByTagName(): unknown[] {
    return [];
  }
  readonly shadowRoot = null;
}

class FakeRoot {
  videos: FakeVideo[] = [];
  others: Array<{ shadowRoot: FakeRoot | null; host?: { isConnected: boolean } }> = [];
  host = { isConnected: true };
  querySelectorAll(selector: string): unknown[] {
    if (selector === 'video') return this.videos.filter((v) => v.isConnected);
    if (selector === '*') return [...this.others];
    return [];
  }
}

class FakeObserver {
  callback: (records: MutationRecord[]) => void;
  observed: unknown[] = [];
  disconnected = false;
  constructor(cb: (records: MutationRecord[]) => void) {
    this.callback = cb;
  }
  observe(target: unknown): void {
    this.observed.push(target);
  }
  disconnect(): void {
    this.disconnected = true;
  }
}

function removal(): MutationRecord {
  return { type: 'childList', addedNodes: [] as unknown as NodeList, removedNodes: [{}] as unknown as NodeList } as unknown as MutationRecord;
}
function addition(node: unknown): MutationRecord {
  return { type: 'childList', addedNodes: [node] as unknown as NodeList, removedNodes: [] as unknown as NodeList } as unknown as MutationRecord;
}
function srcAttr(target: unknown): MutationRecord {
  return { type: 'attributes', target, addedNodes: [], removedNodes: [] } as unknown as MutationRecord;
}

interface Harness {
  clock: FakeClock;
  root: FakeRoot;
  observer: () => FakeObserver;
  windowListeners: Map<string, Set<() => void>>;
  discovery: VideoDiscovery;
  events: Array<[keyof DiscoveryEventMap, string]>;
  fireWindow(type: string): void;
}

function harness(ignoreSelectors: string[] = [], spaEvents: string[] = []): Harness {
  const clock = new FakeClock();
  const root = new FakeRoot();
  let observer: FakeObserver | null = null;
  const windowListeners = new Map<string, Set<() => void>>();
  const env: DiscoveryEnvironment = {
    root: root as unknown as Document,
    observeTarget: {} as Node,
    createMutationObserver: (cb) => (observer = new FakeObserver(cb)),
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clear,
    setInterval: clock.setInterval,
    clearInterval: clock.clear,
    addWindowListener: (type, l) => {
      if (!windowListeners.has(type)) windowListeners.set(type, new Set());
      windowListeners.get(type)!.add(l);
    },
    removeWindowListener: (type, l) => windowListeners.get(type)?.delete(l),
  };
  const discovery = new VideoDiscovery({ rule: { ignoreSelectors, spaEvents }, env });
  const events: Array<[keyof DiscoveryEventMap, string]> = [];
  for (const e of ['attached', 'detached', 'reattached', 'destroyed', 'sourcechange'] as const) {
    discovery.on(e, (p) => events.push([e, (p as { id: string }).id]));
  }
  return {
    clock,
    root,
    observer: () => observer as FakeObserver,
    windowListeners,
    discovery,
    events,
    fireWindow: (type) => windowListeners.get(type)?.forEach((l) => l()),
  };
}

const asVideo = (v: FakeVideo): HTMLVideoElement => v as unknown as HTMLVideoElement;

describe('VideoDiscovery', () => {
  let h: Harness;
  let a: FakeVideo;
  let b: FakeVideo;

  beforeEach(() => {
    h = harness(['ytd-thumbnail video', '!!invalid']);
    a = new FakeVideo('https://example.com/a.webm');
    b = new FakeVideo('https://example.com/b.webm');
    h.root.videos.push(a, b);
  });

  it('attaches existing videos on start and observes the document', () => {
    h.discovery.start();
    expect(h.events).toEqual([
      ['attached', 'v1'],
      ['attached', 'v2'],
    ]);
    expect(h.discovery.stateOf(asVideo(a))).toBe('ATTACHED');
    expect(h.observer().observed).toHaveLength(1);
    expect(h.discovery.attachedVideos()).toHaveLength(2);
  });

  it('skips videos matching an ignore selector and videos without a parent', () => {
    b.matching.add('ytd-thumbnail video');
    const orphan = new FakeVideo();
    orphan.parentElement = null;
    h.root.videos.push(orphan);
    h.discovery.start();
    expect(h.events).toEqual([['attached', 'v1']]);
    expect(h.discovery.stateOf(asVideo(b))).toBe('DESTROYED');
  });

  it('removal enters the grace period; destroyed exactly at 5000 ms', () => {
    h.discovery.start();
    a.isConnected = false;
    h.observer().callback([removal()]);
    expect(h.discovery.stateOf(asVideo(a))).toBe('DETACHED_GRACE_PERIOD');
    expect(h.events.at(-1)).toEqual(['detached', 'v1']);

    h.clock.advance(DETACH_GRACE_PERIOD_MS - 1);
    expect(h.discovery.stateOf(asVideo(a))).toBe('DETACHED_GRACE_PERIOD');
    expect(h.events.some(([e]) => e === 'destroyed')).toBe(false);

    h.clock.advance(1);
    expect(h.discovery.stateOf(asVideo(a))).toBe('DESTROYED');
    expect(h.events.at(-1)).toEqual(['destroyed', 'v1']);
    expect(h.discovery.trackedCount).toBe(1);
  });

  it('re-attachment within the grace period emits reattached, keeps the id and cancels destruction', () => {
    h.discovery.start();
    a.isConnected = false;
    h.observer().callback([removal()]);
    h.clock.advance(120); // YouTube theater-mode re-parenting window
    a.isConnected = true;
    h.observer().callback([addition(a)]);
    expect(h.discovery.stateOf(asVideo(a))).toBe('ATTACHED');
    expect(h.events.slice(-2)).toEqual([
      ['detached', 'v1'],
      ['reattached', 'v1'],
    ]);
    h.clock.advance(DETACH_GRACE_PERIOD_MS * 2);
    expect(h.discovery.stateOf(asVideo(a))).toBe('ATTACHED');
    expect(h.events.filter(([e]) => e === 'destroyed')).toHaveLength(0);
    expect(h.discovery.idOf(asVideo(a))).toBe('v1');
  });

  it('a video that came back without any mutation is reattached when the grace timer fires', () => {
    h.discovery.start();
    a.isConnected = false;
    h.observer().callback([removal()]);
    a.isConnected = true; // re-inserted, but no mutation record delivered to us
    h.clock.advance(DETACH_GRACE_PERIOD_MS);
    expect(h.discovery.stateOf(asVideo(a))).toBe('ATTACHED');
    expect(h.events.at(-1)).toEqual(['reattached', 'v1']);
  });

  it('becoming excluded (e.g. moved into a miniplayer) counts as detachment', () => {
    h.discovery.start();
    a.matching.add('ytd-thumbnail video');
    h.clock.advance(POLL_INTERVAL_MS);
    expect(h.discovery.stateOf(asVideo(a))).toBe('DETACHED_GRACE_PERIOD');
  });

  it('polling finds videos added without a mutation record', () => {
    h.discovery.start();
    const c = new FakeVideo('https://example.com/c.webm');
    h.root.videos.push(c);
    h.clock.advance(POLL_INTERVAL_MS - 1);
    expect(h.discovery.stateOf(asVideo(c))).toBe('DESTROYED');
    h.clock.advance(1);
    expect(h.discovery.stateOf(asVideo(c))).toBe('ATTACHED');
    expect(h.events.at(-1)).toEqual(['attached', 'v3']);
  });

  it('polling also detaches videos removed without a mutation record', () => {
    h.discovery.start();
    b.isConnected = false;
    h.clock.advance(POLL_INTERVAL_MS);
    expect(h.discovery.stateOf(asVideo(b))).toBe('DETACHED_GRACE_PERIOD');
    // grace period is measured from detection (t = 2000) -> destroyed at t = 7000
    h.clock.advance(DETACH_GRACE_PERIOD_MS - 1);
    expect(h.discovery.stateOf(asVideo(b))).toBe('DETACHED_GRACE_PERIOD');
    h.clock.advance(1);
    expect(h.discovery.stateOf(asVideo(b))).toBe('DESTROYED');
  });

  it('a mutation adding a video triggers an immediate scan', () => {
    h.discovery.start();
    const c = new FakeVideo();
    h.root.videos.push(c);
    h.observer().callback([addition(c)]);
    expect(h.discovery.stateOf(asVideo(c))).toBe('ATTACHED');
  });

  it('src attribute mutations and currentSrc changes emit sourcechange once per change', () => {
    h.discovery.start();
    a.currentSrc = 'https://example.com/next.webm';
    h.observer().callback([srcAttr(a)]);
    h.observer().callback([srcAttr(a)]);
    expect(h.events.filter(([e]) => e === 'sourcechange')).toEqual([['sourcechange', 'v1']]);
    // currentSrc change noticed by a polling scan
    b.currentSrc = 'blob:https://example.com/123';
    h.clock.advance(POLL_INTERVAL_MS);
    expect(h.events.filter(([e]) => e === 'sourcechange')).toEqual([
      ['sourcechange', 'v1'],
      ['sourcechange', 'v2'],
    ]);
  });

  it('SPA navigation events (base + rule-specific) trigger a rescan', () => {
    h = harness([], ['yt-navigate-finish']);
    h.discovery.start();
    expect([...h.windowListeners.keys()].sort()).toEqual(['hashchange', 'pageshow', 'popstate', 'yt-navigate-finish']);
    const c = new FakeVideo();
    h.root.videos.push(c);
    h.fireWindow('yt-navigate-finish');
    expect(h.discovery.stateOf(asVideo(c))).toBe('ATTACHED');
  });

  it('emits sourcechange once for each srcObject identity change with empty URLs', () => {
    class FakeMediaStream {}
    const first = new FakeMediaStream() as MediaStream;
    const second = new FakeMediaStream() as MediaStream;
    const video = Object.assign(new FakeVideo(), { srcObject: first as MediaStream | null });
    h = harness();
    h.root.videos.push(video);
    const sources: string[] = [];
    h.discovery.on('sourcechange', ({ src }) => sources.push(src));
    h.discovery.start();
    h.clock.advance(POLL_INTERVAL_MS);
    expect(sources).toEqual([]);
    video.srcObject = second;
    h.clock.advance(POLL_INTERVAL_MS);
    h.clock.advance(POLL_INTERVAL_MS);
    expect(sources).toEqual(['']);
    video.srcObject = null;
    h.clock.advance(POLL_INTERVAL_MS);
    video.srcObject = first;
    h.clock.advance(POLL_INTERVAL_MS);
    expect(sources).toEqual(['', '', '']);
    expect(h.events.filter(([e]) => e === 'sourcechange')).toEqual([
      ['sourcechange', 'v1'], ['sourcechange', 'v1'], ['sourcechange', 'v1'],
    ]);
    h.discovery.stop();
  });

  it('discovers videos inside open shadow roots on a deep scan and observes the root', () => {
    const shadow = new FakeRoot();
    const inner = new FakeVideo();
    shadow.videos.push(inner);
    h.root.others.push({ shadowRoot: shadow });
    h.discovery.start();
    expect(h.discovery.stateOf(asVideo(inner))).toBe('ATTACHED');
    expect(h.observer().observed).toContain(shadow);
    // a later mutation-driven scan still sees the shadow video (known root)
    h.observer().callback([addition(new FakeVideo())]);
    expect(h.discovery.stateOf(asVideo(inner))).toBe('ATTACHED');
  });

  it('deep scans discover a new nested shadow root inside an already known root', () => {
    const outer = new FakeRoot();
    h.root.others.push({ shadowRoot: outer });
    h.discovery.start();
    expect(h.observer().observed.filter((root) => root === outer)).toHaveLength(1);
    const nested = new FakeRoot();
    const inner = new FakeVideo();
    nested.videos.push(inner);
    outer.others.push({ shadowRoot: nested });
    h.discovery.scan(true);
    expect(h.discovery.stateOf(asVideo(inner))).toBe('ATTACHED');
    h.discovery.scan(true);
    expect(h.observer().observed.filter((root) => root === outer)).toHaveLength(1);
    expect(h.observer().observed.filter((root) => root === nested)).toHaveLength(1);
    h.discovery.stop();
  });

  it('stop() destroys every tracked video and releases observers, timers and listeners', () => {
    h.discovery.start();
    a.isConnected = false;
    h.observer().callback([removal()]);
    h.discovery.stop();
    expect(h.events.filter(([e]) => e === 'destroyed').map(([, id]) => id).sort()).toEqual(['v1', 'v2']);
    expect(h.observer().disconnected).toBe(true);
    expect(h.clock.pending).toBe(0);
    expect([...h.windowListeners.values()].every((s) => s.size === 0)).toBe(true);
    const count = h.events.length;
    h.clock.advance(60_000);
    expect(h.events.length).toBe(count);
  });

  it('a listener that throws does not break discovery', () => {
    h.discovery.on('attached', () => {
      throw new Error('listener bug');
    });
    const errors: unknown[] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => errors.push(args);
    try {
      h.discovery.start();
    } finally {
      console.error = original;
    }
    expect(h.discovery.attachedVideos()).toHaveLength(2);
    expect(errors).toHaveLength(2);
  });
});
