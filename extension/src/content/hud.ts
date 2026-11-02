/**
 * Diagnostic HUD: a small text panel anchored to the top-right corner of the video's box inside
 * the player container. It shows input -> output resolution, active model, GPU mean/p95, fps,
 * dropped frames and the timing source (or the eligibility verdict when no session is active).
 *
 * Rendering cost is bounded: updates are coalesced to at most one DOM write pass every 250 ms,
 * and a pass only assigns `textContent` of pre-built line elements whose text changed.
 * The panel has `pointer-events: none`, so it never intercepts player input.
 */

import type { VideoReport } from '../shared/messages';
import type { Rect } from './overlay';
import { captureLayoutSnapshot, compareLayoutSnapshots, registerExtensionNode, type LayoutSnapshotEnvironment } from './invariance';

export const HUD_MIN_INTERVAL_MS = 250;
export const HUD_CLASS = 'webgpu-vsr-hud';
export const HUD_LINE_COUNT = 5;
const HUD_MARGIN_PX = 8;

function dims(w: number, h: number): string {
  return `${w}×${h}`;
}

/** Pure formatter: the HUD's text lines for a report (always {@link HUD_LINE_COUNT} lines). */
export function formatHudLines(report: VideoReport): string[] {
  const t = report.telemetry;
  const lines: string[] = [];
  switch (report.state) {
    case 'active':
      lines.push('WebGPU VSR · ACTIVE');
      break;
    case 'eligible':
      lines.push('WebGPU VSR · ELIGIBLE · starting');
      break;
    case 'error':
      lines.push(`WebGPU VSR · ERROR · ${report.error ?? 'unknown error'}`);
      break;
    case 'ineligible':
      lines.push(`WebGPU VSR · OFF · [${report.point ?? '?'}] ${report.reasonText ?? report.reason ?? ''}`.trimEnd());
      break;
  }
  if (t) {
    const scale = t.inputWidth > 0 ? t.outputWidth / t.inputWidth : 0;
    lines.push(`Res   ${dims(t.inputWidth, t.inputHeight)} → ${dims(t.outputWidth, t.outputHeight)} (${scale.toFixed(2)}x)`);
    lines.push(`Model ${t.activeModel}`);
    lines.push(`GPU   ${t.gpuTimeMeanMs.toFixed(2)} ms mean · ${t.gpuTimeP95Ms.toFixed(2)} ms p95 (${t.timingSource === 'timestamp' ? 'GPU timestamps' : 'CPU cadence'})`);
    lines.push(`FPS   ${t.fps.toFixed(1)} · dropped ${t.droppedFrames}`);
  } else {
    const target = report.targetWidth > 0 && report.targetHeight > 0 ? dims(report.targetWidth, report.targetHeight) : '–';
    lines.push(`Res   ${dims(report.sourceWidth, report.sourceHeight)} → ${target} target`);
    lines.push('Model –');
    lines.push('GPU   –');
    lines.push('FPS   –');
  }
  return lines;
}

/**
 * The anchor element's border box in the coordinate space of an absolutely positioned element
 * whose containing block is `offsetParent` (its padding box, scrolled content).
 */
export function boxInContainer(
  anchorRect: Rect,
  containerRect: Rect,
  containerBorder: { left: number; top: number },
  containerScroll: { left: number; top: number },
): Rect {
  return {
    left: anchorRect.left - containerRect.left - containerBorder.left + containerScroll.left,
    top: anchorRect.top - containerRect.top - containerBorder.top + containerScroll.top,
    width: anchorRect.width,
    height: anchorRect.height,
  };
}

/** HUD position (CSS px, in the container's coordinate space) for a video box. */
export function hudPosition(box: Rect, hudWidth: number): { left: number; top: number } {
  const left = Math.max(box.left + HUD_MARGIN_PX, box.left + box.width - hudWidth - HUD_MARGIN_PX);
  return { left, top: box.top + HUD_MARGIN_PX };
}

export interface HudOptions {
  video: HTMLVideoElement;
  document?: Document;
  now?: () => number;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
  minIntervalMs?: number;
  getComputedStyle?: (element: Element) => CSSStyleDeclaration;
}

export class VideoHud {
  readonly element: HTMLDivElement;
  private readonly panel: HTMLDivElement;
  private readonly lines: HTMLDivElement[] = [];
  private readonly lineText: string[] = [];
  private readonly now: () => number;
  private readonly setTimer: (fn: () => void, ms: number) => unknown;
  private readonly clearTimer: (handle: unknown) => void;
  private readonly minInterval: number;
  private pending: VideoReport | null = null;
  private timer: unknown = null;
  private lastRender = Number.NEGATIVE_INFINITY;
  private anchor: Element | null = null;
  private destroyed = false;
  private renderCount = 0;
  private readonly env: LayoutSnapshotEnvironment;
  private insertionLayoutFailure = false;
  private readonly video: HTMLVideoElement;

  constructor(container: HTMLElement, options: HudOptions) {
    const doc = options.document ?? document;
    this.video = options.video;
    this.env = { document: doc, getComputedStyle: options.getComputedStyle ?? ((element) => window.getComputedStyle(element)) };
    this.now = options.now ?? (() => performance.now());
    this.setTimer = options.setTimeout ?? ((fn, ms) => window.setTimeout(fn, ms));
    this.clearTimer = options.clearTimeout ?? ((h) => window.clearTimeout(h as number));
    this.minInterval = options.minIntervalMs ?? HUD_MIN_INTERVAL_MS;

    const el = doc.createElement('div');
    el.className = HUD_CLASS;
    el.setAttribute('aria-hidden', 'true');
    const shadow = el.attachShadow({ mode: 'open' });
    const style = doc.createElement('style');
    style.textContent = `
      :host {
        all: initial !important;
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
        z-index: 2147483646 !important;
      }
      .panel {
        position: absolute;
        width: max-content;
        margin: 0;
        padding: 6px 9px;
        border-radius: 8px;
        background: rgba(20, 20, 24, 0.82);
        color: #f2f2f7;
        font: 11px/1.45 ui-monospace, Menlo, Monaco, monospace;
        white-space: pre;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
        -webkit-backdrop-filter: blur(8px);
        backdrop-filter: blur(8px);
        left: 0;
        top: 0;
        visibility: hidden;
      }
      .panel > div:first-child { font-weight: 600; }
    `;
    shadow.appendChild(style);
    const panel = doc.createElement('div');
    panel.className = 'panel';
    shadow.appendChild(panel);
    this.panel = panel;
    for (let i = 0; i < HUD_LINE_COUNT; i++) {
      const line = doc.createElement('div');
      panel.appendChild(line);
      this.lines.push(line);
      this.lineText.push('');
    }
    this.element = el;
    registerExtensionNode(el);
    this.moveTo(container);
  }

  /** Number of DOM write passes so far (diagnostics and tests). */
  get renders(): number {
    return this.renderCount;
  }

  get hasInsertionLayoutFailure(): boolean {
    return this.insertionLayoutFailure;
  }

  /** Moves the HUD into another container (video re-parented). */
  moveTo(container: HTMLElement): void {
    if (this.destroyed || this.insertionLayoutFailure || this.element.parentElement === container) return;
    const before = captureLayoutSnapshot(this.video, this.env);
    container.appendChild(this.element);
    if (compareLayoutSnapshots(before, captureLayoutSnapshot(this.video, this.env))) {
      this.insertionLayoutFailure = true;
      this.destroy();
    }
  }

  /** Sets the element (the video) whose top-right corner the HUD sticks to; null hides it. */
  setAnchor(anchor: Element | null): void {
    this.anchor = anchor;
    this.applyPosition();
  }

  /** Re-reads layout and repositions (call on resize / re-parent). */
  reposition(): void {
    this.applyPosition();
  }

  /** Queues a report; rendered immediately if the last render is >= 250 ms old, else coalesced. */
  update(report: VideoReport): void {
    if (this.destroyed) return;
    this.pending = report;
    if (this.timer !== null) return;
    const wait = this.lastRender + this.minInterval - this.now();
    if (wait <= 0) {
      this.flush();
    } else {
      this.timer = this.setTimer(() => {
        this.timer = null;
        this.flush();
      }, wait);
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.timer !== null) this.clearTimer(this.timer);
    this.timer = null;
    this.pending = null;
    this.element.remove();
  }

  private flush(): void {
    if (this.destroyed || this.pending === null) return;
    const text = formatHudLines(this.pending);
    this.pending = null;
    this.lastRender = this.now();
    this.renderCount++;
    for (let i = 0; i < HUD_LINE_COUNT; i++) {
      const next = text[i] ?? '';
      if (this.lineText[i] !== next) {
        this.lineText[i] = next;
        (this.lines[i] as HTMLDivElement).textContent = next;
      }
    }
    this.applyPosition();
  }

  private applyPosition(): void {
    if (this.destroyed) return;
    const s = this.panel.style;
    const anchor = this.anchor;
    const container = this.element.offsetParent;
    const rect = anchor && anchor.isConnected ? anchor.getBoundingClientRect() : null;
    if (!rect || rect.width <= 0 || rect.height <= 0 || this.renderCount === 0) {
      s.setProperty('visibility', 'hidden', 'important');
      return;
    }
    let box: Rect = { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    if (container) {
      const cr = container.getBoundingClientRect();
      box = boxInContainer(
        box,
        { left: cr.left, top: cr.top, width: cr.width, height: cr.height },
        { left: container.clientLeft, top: container.clientTop },
        { left: container.scrollLeft, top: container.scrollTop },
      );
    } else {
      // No containing block (position: fixed ancestor-less root): rect is already in page space.
      const view = this.element.ownerDocument.defaultView;
      box = { ...box, left: box.left + (view?.scrollX ?? 0), top: box.top + (view?.scrollY ?? 0) };
    }
    const width = this.panel.offsetWidth || 0;
    const pos = hudPosition(box, width);
    s.setProperty('left', `${pos.left}px`, 'important');
    s.setProperty('top', `${pos.top}px`, 'important');
    s.setProperty('visibility', 'visible', 'important');
  }
}
