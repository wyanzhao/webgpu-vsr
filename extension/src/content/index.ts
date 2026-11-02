/**
 * Content script entry (runs in every frame, document_idle).
 *
 * Wires settings (chrome.storage.local), the site rule, {@link VideoDiscovery}
 * and the 9-point eligibility gate into a per-video registry, reports each video's verdict and
 * telemetry to the service worker (TELEMETRY_UPDATE -> per-tab state for the popup), and drives
 * the optional HUD.
 *
 * GPU processing is plugged in through {@link VideoSessionFactory}. The bootstrap installs the
 * WebGPU {@link GpuSessionFactory} (session.ts) unless another factory is injected
 * (bootstrapContentScript({ sessionFactory }) or controller.setSessionFactory()). Every video that
 * passes the gate gets a {@link VideoSessionHandle}; the controller forwards lifecycle events
 * (attach, reattach within the 5 s grace period, detach, source change, settings) and destroys
 * the handle as soon as the video becomes ineligible or is destroyed. Gate point 8 asks the
 * factory whether a GPU is available; point 9 asks the handle whether its budget controller has
 * throttled processing, and latches the verdict per video (until the quality, the enable switch
 * or the source changes). A video is `eligible` until its session presents the first frame.
 */

import { LOG_PREFIX, hasExtensionContext, httpHostname, isBenignMessagingError, extensionUrl } from '../shared/env';
import { defaultDiagnostics, loadDiagnostics, watchDiagnostics } from '../shared/diagnostics';
import {
  isExtensionMessage,
  sendRequest,
  type PongMessage,
  type SessionTelemetry,
  type TelemetryUpdateMessage,
  type VideoReport,
} from '../shared/messages';
import { loadSettings, sanitizeSettings, settingsEqual, siteOverrideFor, watchSettings, type UserSettings } from '../shared/settings';
import { VideoDiscovery } from './discovery';
import {
  EligibilityLatches,
  collectEligibilityInput,
  evaluateEligibility,
  probeVideoFrame,
  type EligibilityResult,
  type ProbedFrame,
} from './eligibility';
import { VideoHud } from './hud';
import { GpuSessionFactory } from './session';
import { defaultSiteRule, loadSiteRule, type SiteRule } from './site_rules';

// ---------------------------------------------------------------------------------------------
// Session injection contract (implemented by VideoSRSession / GpuSessionFactory in session.ts)
// ---------------------------------------------------------------------------------------------

/** What the controller gives a session. */
export interface VideoSessionContext {
  /** Stable id of the video within this frame (matches VideoReport.id). */
  readonly id: string;
  readonly video: HTMLVideoElement;
  readonly siteRule: SiteRule;
  /** Latch a SecurityError from `new VideoFrame(video)` with latches.latchTaint(video). */
  readonly latches: EligibilityLatches;
  getSettings(): UserSettings;
  /** The verdict that made the video eligible (source/target sizes). */
  getEligibility(): EligibilityResult;
  /** Live metrics; forwarded to the HUD and (throttled) to the service worker. */
  reportTelemetry(telemetry: SessionTelemetry): void;
  /** The first frame was presented on the overlay; the video is now `active`. */
  reportActive(): void;
  /**
   * Unrecoverable failure. The controller destroys the handle, keeps the native video visible and
   * does not recreate a session for this video until its source or the settings change.
   */
  reportError(message: string): void;
  /** Ask for the gate to be re-run soon (e.g. after latching taint or a budget change). */
  requestEvaluation(): void;
}

/** What a session gives the controller. Every method must be safe to call repeatedly. */
export interface VideoSessionHandle {
  /** The video passed the gate: build the overlay / GPU pipeline and start the frame loop. */
  attach(): void;
  /** The video was re-inserted within the grace period: re-anchor the overlay, keep GPU state. */
  reattach(): void;
  /** The video left the DOM (grace period running): pause and restore the native video. */
  detach(): void;
  /** Refresh geometry during periodic evaluation, including while paused. */
  refreshGeometry(): void;
  /** False when the overlay could not establish a positioning context. */
  isLayoutSupported(): boolean;
  /** Insertion changed the video, a sibling or a composed ancestor; sticky until source or parent changes. */
  hasInsertionLayoutFailure?(): boolean;
  /** Reads site values with the active overlay's picture override suspended. */
  readSiteStyle?<T>(read: () => T): T;
  updateSettings(settings: UserSettings): void;
  /** currentSrc, src or srcObject changed: reset per-source state; native video stays visible until the next frame. */
  sourceChanged(): void;
  /** Gate point 9: false once the session's budget controller has throttled processing. */
  isBudgetHealthy(): boolean;
  /** Release every GPU and DOM resource and restore the native video. */
  destroy(): void;
}

export interface VideoSessionFactory {
  /** Gate point 8: a healthy GPUDevice is available. */
  isGpuAvailable(): boolean;
  create(context: VideoSessionContext): VideoSessionHandle;
}

// ---------------------------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------------------------

interface VideoEntry {
  id: string;
  video: HTMLVideoElement;
  result: EligibilityResult | null;
  handle: VideoSessionHandle | null;
  active: boolean;
  telemetry: SessionTelemetry | null;
  error: string | null;
  budgetThrottled: boolean;
  probe: ProbedFrame | null;
  hud: VideoHud | null;
  cleanup: Array<() => void>;
}

export interface ContentControllerOptions {
  settings: UserSettings;
  siteRule: SiteRule;
  hostname: string;
  /** Hostname of the tab's top-level document (null if unknown); defaults to `hostname` in the top frame. */
  topHostname?: string | null;
  /** Current document URL (re-read on every report: SPA navigations change it in place). */
  getUrl: () => string;
  isTopFrame: boolean;
  sessionFactory: VideoSessionFactory;
  /** Delivers TELEMETRY_UPDATE to the service worker. */
  send: (message: TelemetryUpdateMessage) => Promise<void>;
  /** Called once when the controller stops (unsubscribe external listeners). */
  onStop?: () => void;
}

const MEDIA_EVENTS = [
  'loadedmetadata',
  'loadeddata',
  'resize',
  'playing',
  'emptied',
  'enterpictureinpicture',
  'leavepictureinpicture',
] as const;

const URGENT_REPORT_DELAY_MS = 50;
const TELEMETRY_REPORT_INTERVAL_MS = 1000;
const EVALUATION_DEBOUNCE_MS = 30;

export class ContentController {
  private settings: UserSettings;
  private readonly siteRule: SiteRule;
  private readonly hostname: string;
  private readonly topHostname: string | null;
  private readonly getUrl: () => string;
  private lastReportedUrl: string | null = null;
  private readonly isTopFrame: boolean;
  private factory: VideoSessionFactory;
  private readonly send: (message: TelemetryUpdateMessage) => Promise<void>;
  private readonly onStop: (() => void) | null;
  private readonly latches = new EligibilityLatches();
  private readonly unsupportedInsertions = new WeakMap<HTMLVideoElement, {
    src: string; currentSrc: string; parent: HTMLElement | null;
  }>();
  private readonly unsupportedHudInsertions = new WeakMap<HTMLVideoElement, {
    src: string; currentSrc: string; parent: HTMLElement | null;
  }>();
  private readonly entries = new Map<HTMLVideoElement, VideoEntry>();
  private readonly discovery: VideoDiscovery;
  private readonly resizeObserver: ResizeObserver | null;
  private evaluateTimer: number | null = null;
  private readonly dirty = new Set<HTMLVideoElement>();
  private reportTimer: number | null = null;
  private reportDue = Number.POSITIVE_INFINITY;
  private lastReportSignature: string | null = null;
  private everReported = false;
  private stopped = false;

  private readonly onViewportChange = (): void => this.scheduleEvaluation();

  constructor(options: ContentControllerOptions) {
    this.settings = options.settings;
    this.siteRule = options.siteRule;
    this.hostname = options.hostname;
    this.getUrl = options.getUrl;
    this.isTopFrame = options.isTopFrame;
    this.topHostname = options.topHostname !== undefined ? options.topHostname : options.isTopFrame ? options.hostname : null;
    this.factory = options.sessionFactory;
    this.send = options.send;
    this.onStop = options.onStop ?? null;
    this.discovery = new VideoDiscovery({ rule: options.siteRule });
    this.resizeObserver =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver((records) => {
            for (const r of records) this.scheduleEvaluation(r.target as HTMLVideoElement);
          });

    this.discovery.on('attached', ({ video, id }) => this.onAttached(video, id));
    this.discovery.on('detached', ({ video }) => this.onDetached(video));
    this.discovery.on('reattached', ({ video }) => this.onReattached(video));
    this.discovery.on('destroyed', ({ video }) => this.onDestroyed(video));
    this.discovery.on('sourcechange', ({ video }) => this.onSourceChange(video));
    this.discovery.on('scan', ({ deep }) => {
      if (!deep) return;
      void this.checkContext();
      if (this.stopped) return;
      this.scheduleEvaluation();
      // Same-document (SPA) navigations: the service worker may have dropped this tab's reports.
      if (this.lastReportedUrl !== null && this.lastReportedUrl !== this.getUrl()) this.resendReport();
    });
  }

  start(): void {
    document.addEventListener('fullscreenchange', this.onViewportChange);
    window.addEventListener('resize', this.onViewportChange, { passive: true });
    this.discovery.start();
    if (this.isTopFrame) this.scheduleReport(true);
  }

  /** Stops everything: destroys sessions and HUDs, restores videos, stops observers. */
  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    document.removeEventListener('fullscreenchange', this.onViewportChange);
    window.removeEventListener('resize', this.onViewportChange);
    this.discovery.stop();
    this.resizeObserver?.disconnect();
    if (this.evaluateTimer !== null) window.clearTimeout(this.evaluateTimer);
    if (this.reportTimer !== null) window.clearTimeout(this.reportTimer);
    this.evaluateTimer = null;
    this.reportTimer = null;
    if (this.onStop) this.safely(this.onStop);
  }

  get currentSettings(): UserSettings {
    return this.settings;
  }

  /** Installs (or replaces) the GPU session factory and re-runs the gate for every video. */
  setSessionFactory(factory: VideoSessionFactory): void {
    if (factory === this.factory) return;
    for (const entry of this.entries.values()) this.dropHandle(entry);
    this.factory = factory;
    this.scheduleEvaluation();
  }

  applySettings(next: UserSettings): void {
    if (settingsEqual(next, this.settings)) return;
    const previous = this.settings;
    this.settings = next;
    // Choosing another quality (or re-enabling) is the user's cue to retry throttled or failed
    // videos; cosmetic changes (HUD, sharpness) are not.
    const retry = next.qualityLevel !== previous.qualityLevel || (next.enabled && !previous.enabled);
    const override = siteOverrideFor(next, this.hostname);
    const topOverride = this.topHostname === null ? undefined : siteOverrideFor(next, this.topHostname);
    const disabled = !next.enabled || override === false || topOverride === false ||
      (override === undefined && !this.siteRule.enabled);
    for (const entry of this.entries.values()) {
      if (retry) {
        entry.budgetThrottled = false;
        entry.error = null;
      }
      if (disabled) {
        this.dropHandle(entry);
        entry.hud?.destroy();
        entry.hud = null;
        continue;
      }
      const handle = entry.handle;
      if (handle) this.safely(() => handle.updateSettings(next));
    }
    this.evaluateAll();
  }

  /** Current per-video reports (ATTACHED videos only). */
  reports(): VideoReport[] {
    const out: VideoReport[] = [];
    for (const entry of this.entries.values()) {
      if (this.discovery.stateOf(entry.video) !== 'ATTACHED' || !entry.result) continue;
      out.push(this.reportFor(entry));
    }
    return out.slice(0, 64);
  }

  /** Sends an empty report (frame unloading) so the service worker drops this frame. */
  flushEmpty(): void {
    if (!this.everReported) return;
    void this.deliver({ type: 'TELEMETRY_UPDATE', host: this.hostname, url: this.getUrl(), videos: [] });
    this.lastReportSignature = null;
  }

  /** Re-sends the current state (e.g. after a back/forward-cache restore). */
  resendReport(): void {
    this.lastReportSignature = null;
    this.scheduleReport(true);
  }

  // ---- discovery events -------------------------------------------------------------------

  private onAttached(video: HTMLVideoElement, id: string): void {
    const entry: VideoEntry = {
      id,
      video,
      result: null,
      handle: null,
      active: false,
      telemetry: null,
      error: null,
      budgetThrottled: false,
      probe: null,
      hud: null,
      cleanup: [],
    };
    const onMedia = (): void => this.scheduleEvaluation(video);
    for (const type of MEDIA_EVENTS) video.addEventListener(type, onMedia);
    entry.cleanup.push(() => {
      for (const type of MEDIA_EVENTS) video.removeEventListener(type, onMedia);
    });
    entry.cleanup.push(this.latches.watchDrmEvents(video, () => this.evaluate(entry)));
    if (this.resizeObserver) {
      this.resizeObserver.observe(video);
      entry.cleanup.push(() => this.resizeObserver?.unobserve(video));
    }
    this.entries.set(video, entry);
    this.evaluate(entry);
  }

  private onDetached(video: HTMLVideoElement): void {
    const entry = this.entries.get(video);
    if (!entry) return;
    this.safely(() => entry.handle?.detach());
    entry.hud?.setAnchor(null);
    this.scheduleReport(true);
  }

  private onReattached(video: HTMLVideoElement): void {
    const entry = this.entries.get(video);
    if (!entry) return;
    this.safely(() => entry.handle?.reattach());
    this.evaluate(entry);
  }

  private onDestroyed(video: HTMLVideoElement): void {
    const entry = this.entries.get(video);
    if (!entry) return;
    this.dropHandle(entry);
    entry.hud?.destroy();
    entry.hud = null;
    for (const fn of entry.cleanup) fn();
    this.latches.reset(video);
    this.entries.delete(video);
    this.dirty.delete(video);
    this.scheduleReport(true);
  }

  private onSourceChange(video: HTMLVideoElement): void {
    const entry = this.entries.get(video);
    if (!entry) return;
    this.latches.reset(video);
    entry.probe = null;
    entry.budgetThrottled = false;
    entry.error = null;
    entry.active = false;
    entry.telemetry = null;
    // Stops and pending frames belong to the old source; restart through the gate.
    this.dropHandle(entry);
    this.unsupportedInsertions.delete(video);
    this.evaluate(entry);
  }

  // ---- evaluation -------------------------------------------------------------------------

  private scheduleEvaluation(video?: HTMLVideoElement): void {
    if (this.stopped) return;
    if (video) this.dirty.add(video);
    else for (const v of this.entries.keys()) this.dirty.add(v);
    if (this.evaluateTimer !== null) return;
    this.evaluateTimer = window.setTimeout(() => {
      this.evaluateTimer = null;
      const batch = [...this.dirty];
      this.dirty.clear();
      for (const v of batch) {
        const entry = this.entries.get(v);
        if (entry) this.evaluate(entry);
      }
    }, EVALUATION_DEBOUNCE_MS);
  }

  private evaluateAll(): void {
    for (const entry of [...this.entries.values()]) this.evaluate(entry);
  }

  private frameProbe(entry: VideoEntry): ProbedFrame {
    if (this.latches.isHdrLatched(entry.video)) return { tainted: false, transfer: 'pq', inspected: true };
    if (entry.probe) return entry.probe;
    const probed = probeVideoFrame(entry.video);
    if (probed.tainted) this.latches.latchTaint(entry.video);
    if (probed.inspected) entry.probe = probed;
    return probed;
  }

  private evaluate(entry: VideoEntry): void {
    if (this.stopped || this.discovery.stateOf(entry.video) !== 'ATTACHED') return;
    const video = entry.video;
    const latched = this.unsupportedInsertions.get(video);
    if (latched && (latched.src !== video.src || latched.currentSrc !== video.currentSrc || latched.parent !== video.parentElement)) {
      if (entry.handle?.hasInsertionLayoutFailure?.()) this.dropHandle(entry);
      this.unsupportedInsertions.delete(video);
    }
    const hudLatched = this.unsupportedHudInsertions.get(video);
    if (hudLatched && (hudLatched.src !== video.src || hudLatched.currentSrc !== video.currentSrc || hudLatched.parent !== video.parentElement)) {
      this.unsupportedHudInsertions.delete(video);
    }
    if (entry.handle?.hasInsertionLayoutFailure?.()) {
      this.unsupportedInsertions.set(video, { src: video.src, currentSrc: video.currentSrc, parent: video.parentElement });
    }
    const factory = this.factory;
    const input = collectEligibilityInput(entry.video, {
      settings: this.settings,
      hostname: this.hostname,
      topHostname: this.topHostname,
      siteRuleEnabled: this.siteRule.enabled,
      layoutSupported: !this.unsupportedInsertions.has(video) && (entry.handle?.isLayoutSupported() ?? true),
      readSiteStyle: (read) => entry.handle?.readSiteStyle ? entry.handle.readSiteStyle(read) : read(),
      latches: this.latches,
      probeFrame: () => this.frameProbe(entry),
      gpuAvailable: () => factory.isGpuAvailable(),
      budgetHealthy: () => !entry.budgetThrottled && (entry.handle ? entry.handle.isBudgetHealthy() : true),
    });
    const result = evaluateEligibility(input);
    entry.result = result;
    if (result.reason === 'frame-budget') entry.budgetThrottled = true;

    if (!result.eligible) {
      this.dropHandle(entry);
    } else if (!entry.handle && entry.error === null) {
      this.createHandle(entry, factory);
    } else if (entry.handle) {
      this.safely(() => entry.handle?.refreshGeometry());
    }
    if (entry.handle?.hasInsertionLayoutFailure?.()) this.scheduleEvaluation(video);
    this.syncHud(entry);
    this.scheduleReport(true);
  }

  private createHandle(entry: VideoEntry, factory: VideoSessionFactory): void {
    let handle: VideoSessionHandle | null = null;
    const context: VideoSessionContext = {
      id: entry.id,
      video: entry.video,
      siteRule: this.siteRule,
      latches: this.latches,
      getSettings: () => this.settings,
      getEligibility: () => entry.result as EligibilityResult,
      reportTelemetry: (telemetry) => {
        if (entry.handle !== handle) return;
        entry.telemetry = telemetry;
        this.syncHud(entry);
        this.scheduleReport(false);
      },
      reportActive: () => {
        if (entry.handle !== handle || entry.active) return;
        entry.active = true;
        this.syncHud(entry);
        this.scheduleReport(true);
      },
      reportError: (message) => {
        if (entry.handle !== handle) return;
        entry.error = message.slice(0, 500) || 'session error';
        this.dropHandle(entry);
        this.syncHud(entry);
        this.scheduleReport(true);
      },
      requestEvaluation: () => {
        if (entry.handle === handle) this.scheduleEvaluation(entry.video);
      },
    };
    try {
      handle = factory.create(context);
      entry.handle = handle;
      entry.active = false;
      entry.telemetry = null;
      handle.attach();
    } catch (err) {
      entry.error = err instanceof Error ? err.message : String(err);
      console.error(`${LOG_PREFIX} session start failed`, err);
      this.dropHandle(entry);
    }
  }

  private dropHandle(entry: VideoEntry): void {
    const handle = entry.handle;
    if (handle?.hasInsertionLayoutFailure?.()) {
      const video = entry.video;
      this.unsupportedInsertions.set(video, { src: video.src, currentSrc: video.currentSrc, parent: video.parentElement });
    }
    entry.handle = null;
    entry.active = false;
    entry.telemetry = null;
    if (handle) this.safely(() => handle.destroy());
  }

  private safely(fn: () => void): void {
    try {
      fn();
    } catch (err) {
      console.error(`${LOG_PREFIX} session callback failed`, err);
    }
  }

  // ---- reporting --------------------------------------------------------------------------

  private reportFor(entry: VideoEntry): VideoReport {
    const r = entry.result as EligibilityResult;
    const state = entry.error !== null ? 'error' : !r.eligible ? 'ineligible' : entry.active ? 'active' : 'eligible';
    const ineligible = state === 'ineligible';
    const dim = (n: number): number => (Number.isFinite(n) && n > 0 ? Math.min(65536, Math.round(n)) : 0);
    return {
      id: entry.id,
      state,
      point: ineligible ? r.point : null,
      reason: ineligible ? r.reason : null,
      reasonText: ineligible ? r.message : null,
      sourceWidth: dim(r.sourceWidth),
      sourceHeight: dim(r.sourceHeight),
      targetWidth: dim(r.targetWidth),
      targetHeight: dim(r.targetHeight),
      telemetry: state === 'active' ? entry.telemetry : null,
      error: state === 'error' ? entry.error : null,
    };
  }

  private syncHud(entry: VideoEntry): void {
    // A detached video keeps its (hidden) HUD for the grace period; late session reports
    // (frames that were still in flight) must not tear it down.
    if (this.discovery.stateOf(entry.video) !== 'ATTACHED') return;
    const r = entry.result;
    // Only for videos that are actually on screen (passed the presentation and geometry points).
    const override = siteOverrideFor(this.settings, this.hostname);
    const topOverride = this.topHostname === null ? undefined : siteOverrideFor(this.settings, this.topHostname);
    const disabled = !this.settings.enabled || override === false || topOverride === false ||
      (override === undefined && !this.siteRule.enabled);
    const show = !disabled && this.settings.hudEnabled && !this.unsupportedHudInsertions.has(entry.video) &&
      r !== null && (r.point === null || r.point >= 4) && entry.video.parentElement !== null;
    if (!show) {
      entry.hud?.destroy();
      entry.hud = null;
      return;
    }
    if (!entry.hud) {
      entry.hud = new VideoHud(entry.video.parentElement as HTMLElement, { video: entry.video });
    } else {
      // Re-anchoring also repositions; it restores the anchor of a HUD whose video was detached.
      entry.hud.moveTo(entry.video.parentElement as HTMLElement);
    }
    if (entry.hud.hasInsertionLayoutFailure) {
      const video = entry.video;
      this.unsupportedHudInsertions.set(video, { src: video.src, currentSrc: video.currentSrc, parent: video.parentElement });
      entry.hud.destroy();
      entry.hud = null;
      return;
    }
    entry.hud.setAnchor(entry.video);
    entry.hud.update(this.reportFor(entry));
  }

  private scheduleReport(urgent: boolean): void {
    if (this.stopped) return;
    const delay = urgent ? URGENT_REPORT_DELAY_MS : TELEMETRY_REPORT_INTERVAL_MS;
    const due = performance.now() + delay;
    if (this.reportTimer !== null) {
      if (due >= this.reportDue) return;
      window.clearTimeout(this.reportTimer);
    }
    this.reportDue = due;
    this.reportTimer = window.setTimeout(() => {
      this.reportTimer = null;
      this.reportDue = Number.POSITIVE_INFINITY;
      this.flushReport();
    }, delay);
  }

  private flushReport(): void {
    const videos = this.reports();
    // Sub-frames without videos stay silent (ad iframes); the top frame always reports.
    if (!this.isTopFrame && videos.length === 0 && !this.everReported) return;
    const url = this.getUrl();
    const signature = JSON.stringify([url, videos]);
    if (signature === this.lastReportSignature) return;
    this.lastReportSignature = signature;
    this.lastReportedUrl = url;
    void this.deliver({ type: 'TELEMETRY_UPDATE', host: this.hostname, url, videos });
  }

  private async checkContext(): Promise<void> {
    if (!hasExtensionContext()) {
      this.stop();
      return;
    }
    try {
      // Reports may be unchanged for a paused video; probe before report deduplication.
      await chrome.runtime.sendMessage({ type: 'PING' });
    } catch (err) {
      if (!hasExtensionContext() || String(err).includes('Extension context invalidated')) this.stop();
    }
  }

  private async deliver(message: TelemetryUpdateMessage): Promise<void> {
    this.everReported = true;
    try {
      await this.send(message);
    } catch (err) {
      if (!hasExtensionContext() || String(err).includes('Extension context invalidated')) {
        // The extension was reloaded or removed: this script is orphaned. Restore the page.
        this.stop();
        return;
      }
      if (!isBenignMessagingError(err)) console.warn(`${LOG_PREFIX} telemetry delivery failed`, err);
      this.lastReportSignature = null;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------------------------

const CONTROLLER_KEY = Symbol.for('webgpu-vsr.content-controller');

type GlobalWithController = typeof globalThis & { [CONTROLLER_KEY]?: ContentController };

export interface BootstrapOptions {
  /** Replaces the default WebGPU session factory (GpuSessionFactory). */
  sessionFactory?: VideoSessionFactory;
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function sendTelemetry(message: TelemetryUpdateMessage): Promise<void> {
  await chrome.runtime.sendMessage(message);
}

/**
 * Hostname of the tab's top-level document. A sub-frame asks the service worker (it knows the
 * tab's URL); without an answer, the outermost ancestor origin (Chrome's location.ancestorOrigins)
 * is used, and null when neither is available.
 */
async function topLevelHostname(isTopFrame: boolean): Promise<string | null> {
  if (isTopFrame) return location.hostname;
  try {
    return (await sendRequest({ type: 'GET_FRAME_CONTEXT' })).topHost;
  } catch (err) {
    if (!isBenignMessagingError(err)) console.warn(`${LOG_PREFIX} frame context unavailable`, err);
    const ancestors = (location as Location & { ancestorOrigins?: DOMStringList }).ancestorOrigins;
    return ancestors && ancestors.length > 0 ? httpHostname(ancestors[ancestors.length - 1]) : null;
  }
}

/**
 * Starts the content controller once per frame. Returns the controller (or the already running
 * one), or null when there is no live extension context.
 */
export async function bootstrapContentScript(options: BootstrapOptions = {}): Promise<ContentController | null> {
  const g = globalThis as GlobalWithController;
  const existing = g[CONTROLLER_KEY];
  if (existing) {
    if (options.sessionFactory !== undefined) existing.setSessionFactory(options.sessionFactory);
    return existing;
  }
  if (!hasExtensionContext()) return null;

  const teardown: Array<() => void> = [];
  let bootstrappedController: ContentController | null = null;
  let bufferedSettings: UserSettings | null = null;
  const applySettings = (next: UserSettings): void => {
    if (bootstrappedController) bootstrappedController.applySettings(next);
    else bufferedSettings = next;
  };
  teardown.push(watchSettings(applySettings));
  const onMessage = (message: unknown, _sender: chrome.runtime.MessageSender, sendResponse: (r: unknown) => void): boolean => {
    if (!isExtensionMessage(message)) return false;
    if (message.type === 'PING') {
      const pong: PongMessage = { type: 'PONG', version: chrome.runtime.getManifest().version, context: 'content' };
      sendResponse(pong);
      return false;
    }
    return false;
  };
  chrome.runtime.onMessage.addListener(onMessage);
  teardown.push(() => chrome.runtime.onMessage.removeListener(onMessage));

  const hostname = location.hostname;
  const isTopFrame = window.top === window;
  const [settings, siteRuleResult, diagnostics, topHostname] = await Promise.all([
    loadSettings().catch((err: unknown) => {
      console.warn(`${LOG_PREFIX} settings unavailable, using defaults`, err);
      return sanitizeSettings(undefined);
    }),
    loadSiteRule(hostname, extensionUrl, fetchJson).catch((err: unknown) => ({
      rule: defaultSiteRule(),
      errors: [String(err)],
    })),
    loadDiagnostics().catch(() => defaultDiagnostics()),
    topLevelHostname(isTopFrame),
  ]);
  if (siteRuleResult.errors.length > 0) console.warn(`${LOG_PREFIX} site rule problems`, siteRuleResult.errors);

  // Re-check after the awaits: another injection may have won the race.
  if (g[CONTROLLER_KEY]) {
    for (const fn of teardown) fn();
    return g[CONTROLLER_KEY] as ContentController;
  }

  let sessionFactory = options.sessionFactory;
  if (!sessionFactory) {
    const gpuFactory = new GpuSessionFactory();
    gpuFactory.setDiagnostics(diagnostics);
    teardown.push(watchDiagnostics((next) => gpuFactory.setDiagnostics(next)));
    sessionFactory = gpuFactory;
  }
  const controller = new ContentController({
    settings: bufferedSettings ?? settings,
    siteRule: siteRuleResult.rule,
    hostname,
    topHostname,
    getUrl: () => location.href.slice(0, 2048),
    isTopFrame,
    sessionFactory,
    send: sendTelemetry,
    // Orphaned-script cleanup: periodic probes and delivery stop the controller when its context is gone.
    onStop: () => {
      for (const fn of teardown) {
        try {
          fn();
        } catch {
          // extension context already gone
        }
      }
    },
  });
  bootstrappedController = controller;
  g[CONTROLLER_KEY] = controller;

  const onPageHide = (): void => controller.flushEmpty();
  const onPageShow = (event: PageTransitionEvent): void => {
    if (event.persisted) controller.resendReport();
  };
  window.addEventListener('pagehide', onPageHide);
  window.addEventListener('pageshow', onPageShow);
  teardown.push(() => {
    window.removeEventListener('pagehide', onPageHide);
    window.removeEventListener('pageshow', onPageShow);
    delete g[CONTROLLER_KEY];
  });

  controller.start();
  return controller;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined' && hasExtensionContext()) {
  void bootstrapContentScript().catch((err: unknown) => console.error(`${LOG_PREFIX} content bootstrap failed`, err));
}
