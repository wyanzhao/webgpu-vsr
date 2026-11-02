/**
 * Hybrid <video> discovery with a per-video lifecycle state machine.
 *
 * Detection: a MutationObserver on documentElement (childList + subtree, plus `src` attributes)
 * for immediate reaction, a 2000 ms polling scan as the safety net (also discovers open shadow
 * roots, which are then observed too), and SPA navigation events (popstate, hashchange, pageshow
 * plus the site rule's events such as YouTube's `yt-navigate-finish`) that force a rescan.
 *
 * Lifecycle per video:
 *
 *   (new) --attached--> ATTACHED --(gone or excluded)--> DETACHED_GRACE_PERIOD --5000 ms--> DESTROYED
 *                          ^                                   |
 *                          +------------ reattached -----------+   (back within the grace period)
 *
 * `reattached` lets the session re-anchor its overlay without reallocating GPU resources (YouTube
 * theater mode re-parents the player for 16-120 ms). Timers and observers are injectable so the
 * state machine is testable with a fake clock.
 */

import { BASE_SPA_EVENTS, isExcludedByRule, type SiteRule } from './site_rules';

export type TrackedVideoState = 'ATTACHED' | 'DETACHED_GRACE_PERIOD' | 'DESTROYED';

export const DETACH_GRACE_PERIOD_MS = 5000;
export const POLL_INTERVAL_MS = 2000;

export interface DiscoveryEventMap {
  attached: { video: HTMLVideoElement; id: string };
  detached: { video: HTMLVideoElement; id: string };
  reattached: { video: HTMLVideoElement; id: string };
  destroyed: { video: HTMLVideoElement; id: string };
  sourcechange: { video: HTMLVideoElement; id: string; src: string };
  /** After every completed scan (cheap hook for periodic re-evaluation). */
  scan: { deep: boolean };
}

export type DiscoveryEvent = keyof DiscoveryEventMap;

export interface MutationObserverLike {
  observe(target: Node, options: MutationObserverInit): void;
  disconnect(): void;
}

export interface DiscoveryEnvironment {
  /** Document (or any root) to scan. */
  root: Pick<Document, 'querySelectorAll'>;
  /** Node the MutationObserver watches (documentElement); null disables the observer. */
  observeTarget: Node | null;
  createMutationObserver(callback: (records: MutationRecord[]) => void): MutationObserverLike | null;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
  setInterval(fn: () => void, ms: number): unknown;
  clearInterval(handle: unknown): void;
  addWindowListener(type: string, listener: () => void): void;
  removeWindowListener(type: string, listener: () => void): void;
}

export function browserDiscoveryEnvironment(): DiscoveryEnvironment {
  return {
    root: document,
    observeTarget: document.documentElement,
    createMutationObserver: (cb) => (typeof MutationObserver === 'undefined' ? null : new MutationObserver(cb)),
    setTimeout: (fn, ms) => window.setTimeout(fn, ms),
    clearTimeout: (h) => window.clearTimeout(h as number),
    setInterval: (fn, ms) => window.setInterval(fn, ms),
    clearInterval: (h) => window.clearInterval(h as number),
    addWindowListener: (type, l) => window.addEventListener(type, l),
    removeWindowListener: (type, l) => window.removeEventListener(type, l),
  };
}

export interface DiscoveryOptions {
  rule: Pick<SiteRule, 'ignoreSelectors' | 'spaEvents'>;
  env?: DiscoveryEnvironment;
  gracePeriodMs?: number;
  pollIntervalMs?: number;
  /** Walk the document for open shadow roots during polling scans (default true). */
  scanShadowRoots?: boolean;
}

interface Entry {
  id: string;
  video: HTMLVideoElement;
  state: TrackedVideoState;
  lastSrc: string;
  graceTimer: unknown;
}

const OBSERVE_OPTIONS: MutationObserverInit = { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] };

function isElement(node: unknown): node is Element {
  return typeof node === 'object' && node !== null && (node as Node).nodeType === 1;
}

export function isVideoElement(node: unknown): node is HTMLVideoElement {
  return isElement(node) && node.localName === 'video';
}

function sourceOf(video: HTMLVideoElement): string {
  return video.currentSrc || video.getAttribute('src') || '';
}

type Listener<E extends DiscoveryEvent> = (payload: DiscoveryEventMap[E]) => void;

export class VideoDiscovery {
  private readonly env: DiscoveryEnvironment;
  private rule: Pick<SiteRule, 'ignoreSelectors' | 'spaEvents'>;
  private readonly graceMs: number;
  private readonly pollMs: number;
  private readonly scanShadowRoots: boolean;
  private readonly entries = new Map<HTMLVideoElement, Entry>();
  private readonly lastSrcObjects = new WeakMap<HTMLVideoElement, HTMLVideoElement['srcObject']>();
  private readonly listeners: { [E in DiscoveryEvent]: Set<Listener<E>> } = {
    attached: new Set(),
    detached: new Set(),
    reattached: new Set(),
    destroyed: new Set(),
    sourcechange: new Set(),
    scan: new Set(),
  };
  private readonly shadowRoots = new Set<ShadowRoot>();
  private observer: MutationObserverLike | null = null;
  private pollTimer: unknown = null;
  private spaEvents: string[] = [];
  private nextId = 1;
  private running = false;

  private readonly onSpaNavigation = (): void => {
    this.scan(true);
  };

  constructor(options: DiscoveryOptions) {
    this.env = options.env ?? browserDiscoveryEnvironment();
    this.rule = options.rule;
    this.graceMs = options.gracePeriodMs ?? DETACH_GRACE_PERIOD_MS;
    this.pollMs = options.pollIntervalMs ?? POLL_INTERVAL_MS;
    this.scanShadowRoots = options.scanShadowRoots ?? true;
  }

  on<E extends DiscoveryEvent>(event: E, listener: Listener<E>): () => void {
    this.listeners[event].add(listener);
    return () => {
      this.listeners[event].delete(listener);
    };
  }

  get isRunning(): boolean {
    return this.running;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.observer = this.env.createMutationObserver((records) => this.onMutations(records));
    if (this.observer && this.env.observeTarget) this.observer.observe(this.env.observeTarget, OBSERVE_OPTIONS);
    this.pollTimer = this.env.setInterval(() => this.scan(true), this.pollMs);
    this.bindSpaEvents();
    this.scan(true);
  }

  /** Stops observing and destroys every tracked video (emits `destroyed` for each). */
  stop(): void {
    if (!this.running) return;
    this.running = false;
    this.observer?.disconnect();
    this.observer = null;
    if (this.pollTimer !== null) this.env.clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.unbindSpaEvents();
    this.shadowRoots.clear();
    for (const entry of [...this.entries.values()]) this.destroyEntry(entry);
  }

  /** Swaps the site rule (exclusions and SPA events) and rescans. */
  setRule(rule: Pick<SiteRule, 'ignoreSelectors' | 'spaEvents'>): void {
    this.rule = rule;
    if (this.running) {
      this.unbindSpaEvents();
      this.bindSpaEvents();
      this.scan(true);
    }
  }

  stateOf(video: HTMLVideoElement): TrackedVideoState {
    return this.entries.get(video)?.state ?? 'DESTROYED';
  }

  idOf(video: HTMLVideoElement): string | null {
    return this.entries.get(video)?.id ?? null;
  }

  /** Videos currently in the ATTACHED state. */
  attachedVideos(): HTMLVideoElement[] {
    return [...this.entries.values()].filter((e) => e.state === 'ATTACHED').map((e) => e.video);
  }

  get trackedCount(): number {
    return this.entries.size;
  }

  /**
   * Reconciles tracked videos with the DOM. `deep` additionally walks for open shadow roots.
   */
  scan(deep = false): void {
    if (!this.running) return;
    if (deep && this.scanShadowRoots) this.discoverShadowRoots();
    const present = new Set<HTMLVideoElement>();
    for (const video of this.collectVideos()) {
      if (!this.isTrackable(video)) continue;
      present.add(video);
      const entry = this.entries.get(video);
      if (!entry) {
        const created: Entry = { id: `v${this.nextId++}`, video, state: 'ATTACHED', lastSrc: sourceOf(video), graceTimer: null };
        this.entries.set(video, created);
        this.lastSrcObjects.set(video, video.srcObject);
        this.emit('attached', { video, id: created.id });
        continue;
      }
      if (entry.state === 'DETACHED_GRACE_PERIOD') this.reattach(entry);
      this.checkSource(entry);
    }
    for (const entry of [...this.entries.values()]) {
      if (entry.state === 'ATTACHED' && !present.has(entry.video)) this.detach(entry);
    }
    this.emit('scan', { deep });
  }

  private emit<E extends DiscoveryEvent>(event: E, payload: DiscoveryEventMap[E]): void {
    for (const listener of [...this.listeners[event]]) {
      try {
        listener(payload);
      } catch (err) {
        console.error('[webgpu-vsr] discovery listener failed', err);
      }
    }
  }

  private isTrackable(video: HTMLVideoElement): boolean {
    return video.isConnected && video.parentElement !== null && !isExcludedByRule(video, this.rule);
  }

  private collectVideos(): HTMLVideoElement[] {
    const out = Array.from(this.env.root.querySelectorAll('video')) as HTMLVideoElement[];
    for (const root of [...this.shadowRoots]) {
      if (!root.host.isConnected) {
        this.shadowRoots.delete(root);
        continue;
      }
      out.push(...(Array.from(root.querySelectorAll('video')) as HTMLVideoElement[]));
    }
    return out;
  }

  private discoverShadowRoots(): void {
    const visit = (root: Pick<Document, 'querySelectorAll'>): void => {
      for (const el of Array.from(root.querySelectorAll('*'))) {
        const sr = (el as Element).shadowRoot;
        if (sr && sr.host.isConnected) {
          if (!this.shadowRoots.has(sr)) {
            this.shadowRoots.add(sr);
            this.observer?.observe(sr, OBSERVE_OPTIONS);
          }
          visit(sr);
        }
      }
    };
    visit(this.env.root);
  }

  private onMutations(records: readonly MutationRecord[]): void {
    if (!this.running) return;
    let rescan = false;
    let removals = false;
    for (const record of records) {
      if (record.type === 'attributes') {
        const entry = isVideoElement(record.target) ? this.entries.get(record.target) : undefined;
        if (entry) this.checkSource(entry);
        else if (isVideoElement(record.target)) rescan = true;
        continue;
      }
      if (record.removedNodes.length > 0) removals = true;
      for (const node of Array.from(record.addedNodes)) {
        if (!isElement(node)) continue;
        if (node.localName === 'video' || node.getElementsByTagName('video').length > 0 || node.shadowRoot) {
          rescan = true;
          break;
        }
      }
    }
    if (rescan) {
      this.scan(false);
    } else if (removals) {
      for (const entry of [...this.entries.values()]) {
        if (entry.state === 'ATTACHED' && !this.isTrackable(entry.video)) this.detach(entry);
      }
    }
  }

  private checkSource(entry: Entry): void {
    const src = sourceOf(entry.video);
    const srcObject = entry.video.srcObject;
    if (src === entry.lastSrc && srcObject === this.lastSrcObjects.get(entry.video)) return;
    entry.lastSrc = src;
    this.lastSrcObjects.set(entry.video, srcObject);
    this.emit('sourcechange', { video: entry.video, id: entry.id, src });
  }

  private detach(entry: Entry): void {
    entry.state = 'DETACHED_GRACE_PERIOD';
    entry.graceTimer = this.env.setTimeout(() => this.onGraceExpired(entry), this.graceMs);
    this.emit('detached', { video: entry.video, id: entry.id });
  }

  private reattach(entry: Entry): void {
    if (entry.graceTimer !== null) this.env.clearTimeout(entry.graceTimer);
    entry.graceTimer = null;
    entry.state = 'ATTACHED';
    this.emit('reattached', { video: entry.video, id: entry.id });
  }

  private onGraceExpired(entry: Entry): void {
    entry.graceTimer = null;
    if (entry.state !== 'DETACHED_GRACE_PERIOD' || this.entries.get(entry.video) !== entry) return;
    // The video may have come back without a mutation or scan noticing it yet.
    if (this.running && this.isTrackable(entry.video)) {
      this.reattach(entry);
      this.checkSource(entry);
      return;
    }
    this.destroyEntry(entry);
  }

  private destroyEntry(entry: Entry): void {
    if (entry.graceTimer !== null) this.env.clearTimeout(entry.graceTimer);
    entry.graceTimer = null;
    entry.state = 'DESTROYED';
    this.entries.delete(entry.video);
    this.lastSrcObjects.delete(entry.video);
    this.emit('destroyed', { video: entry.video, id: entry.id });
  }

  private bindSpaEvents(): void {
    this.spaEvents = [...new Set([...BASE_SPA_EVENTS, ...this.rule.spaEvents])];
    for (const type of this.spaEvents) this.env.addWindowListener(type, this.onSpaNavigation);
  }

  private unbindSpaEvents(): void {
    for (const type of this.spaEvents) this.env.removeWindowListener(type, this.onSpaNavigation);
    this.spaEvents = [];
  }
}
