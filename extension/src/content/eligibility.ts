/**
 * 9-point ordered eligibility gate. Points are evaluated strictly in order and the first failing
 * point is reported with a reason code and human-readable text:
 *
 *   1 DRM / EME             5 resolution / upscale threshold
 *   2 presentation mode     6 dynamic range (HDR)
 *   3 DOM geometry          7 user / site settings
 *   4 CORS / taint          8 WebGPU device availability
 *                           9 frame-budget health
 *
 * {@link evaluateEligibility} is pure: every input is a plain value or an injected probe. Probes
 * (frame inspection, GPU, budget) are called lazily, only when the gate reaches their point, and at
 * most once per evaluation. {@link collectEligibilityInput} builds the input from a live video.
 */

import { type UserSettings, siteOverrideFor } from '../shared/settings';
import {
  computeLetterboxGeometry,
  deepFullscreenElement,
  videoEffectsSupported,
  positionPatchSupported,
  parseObjectFit,
  parseObjectPosition,
} from './overlay';

export type EligibilityPoint = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

export const INELIGIBLE_REASONS = [
  'drm',
  'picture-in-picture',
  'native-fullscreen',
  'not-connected',
  'not-rendered',
  'too-small',
  'no-video-dimensions',
  'unsupported-layout',
  'tainted',
  'cross-origin',
  'no-upscale-needed',
  'hdr',
  'extension-disabled',
  'site-disabled',
  'site-rule-disabled',
  'webgpu-unavailable',
  'frame-budget',
] as const;

export type IneligibleReason = (typeof INELIGIBLE_REASONS)[number];

export const REASON_POINT: Readonly<Record<IneligibleReason, EligibilityPoint>> = Object.freeze({
  drm: 1,
  'picture-in-picture': 2,
  'native-fullscreen': 2,
  'not-connected': 3,
  'not-rendered': 3,
  'too-small': 3,
  'no-video-dimensions': 3,
  'unsupported-layout': 3,
  tainted: 4,
  'cross-origin': 4,
  'no-upscale-needed': 5,
  hdr: 6,
  'extension-disabled': 7,
  'site-disabled': 7,
  'site-rule-disabled': 7,
  'webgpu-unavailable': 8,
  'frame-budget': 9,
});

export const REASON_TEXT: Readonly<Record<IneligibleReason, string>> = Object.freeze({
  drm: 'DRM-protected (Encrypted Media Extensions) video is left untouched',
  'picture-in-picture': 'Video is in Picture-in-Picture',
  'native-fullscreen': 'The video element itself is fullscreen; an overlay cannot be shown above it',
  'not-connected': 'Video is not attached to the document',
  'not-rendered': 'Video is not rendered (display: none, visibility: hidden, opacity: 0 or no layout box)',
  'too-small': 'Video is smaller than 128 x 128 CSS px',
  'no-video-dimensions': 'Video has no decoded dimensions yet',
  'unsupported-layout': 'Unsupported object-fit / object-position, transformed or zoomed layout',
  tainted: 'Video frames are cross-origin tainted and cannot be read by WebGPU',
  'cross-origin': 'Cross-origin video without a crossorigin attribute',
  'no-upscale-needed': 'Display size is less than 1.15x the source resolution',
  hdr: 'HDR video (PQ / HLG transfer) is left untouched',
  'extension-disabled': 'WebGPU VSR is turned off',
  'site-disabled': 'Turned off for this site',
  'site-rule-disabled': 'Disabled by the built-in rule for this site',
  'webgpu-unavailable': 'WebGPU device unavailable',
  'frame-budget': 'Frame budget exceeded; processing is throttled',
});

export const MIN_ELEMENT_CSS_SIZE = 128;
export const MIN_UPSCALE_FACTOR = 1.15;
export const HDR_TRANSFERS: readonly string[] = ['pq', 'hlg', 'smpte2084', 'arib-std-b67'];

export interface FrameProbeResult {
  /** `new VideoFrame(video)` threw a SecurityError. */
  tainted: boolean;
  /** VideoFrame.colorSpace.transfer, or null when unknown / no frame was available. */
  transfer: string | null;
}

export interface EligibilityInput {
  drm: { mediaKeys: boolean; encryptedEventLatched: boolean };
  presentation: { pictureInPicture: boolean; videoIsFullscreenElement: boolean };
  geometry: {
    connected: boolean;
    layoutSupported: boolean;
    display: string;
    visibility: string;
    opacity: number;
    /** The element generates a layout box (getClientRects().length > 0). */
    hasLayoutBox: boolean;
    clientWidth: number;
    clientHeight: number;
    videoWidth: number;
    videoHeight: number;
    objectFit: string;
    objectPosition: string;
    transform: string;
    scale: string;
    rotate: string;
    translate: string;
    currentCSSZoom: number;
    ancestors: Array<{ transform: string; scale: string; rotate: string; zoom: number; currentCSSZoom: number }>;
    renderedWidth: number;
    renderedHeight: number;
    offsetWidth: number;
    offsetHeight: number;
  };
  origin: {
    /** currentSrc (or src); null/'' for srcObject streams. */
    sourceUrl: string | null;
    documentOrigin: string;
    crossOriginAttribute: string | null;
    /** Set by the session when `new VideoFrame(video)` threw a SecurityError. */
    taintLatched: boolean;
  };
  /** Output target in device pixels after the 1080p ceiling (from the rendered content rect). */
  target: { width: number; height: number };
  settings: UserSettings;
  /** Hostname of the frame the video is in. */
  hostname: string;
  /**
   * Hostname of the tab's top-level document ("the site" of the popup's per-site switch), or null
   * when unknown. An explicit per-site "off" for it applies to every frame of the tab, including
   * cross-origin player iframes.
   */
  topHostname: string | null;
  siteRuleEnabled: boolean;
  probeFrame: () => FrameProbeResult;
  gpuAvailable: () => boolean;
  budgetHealthy: () => boolean;
}

export interface EligibilityResult {
  eligible: boolean;
  /** First failing point, or null when eligible. */
  point: EligibilityPoint | null;
  reason: IneligibleReason | null;
  message: string | null;
  sourceWidth: number;
  sourceHeight: number;
  targetWidth: number;
  targetHeight: number;
}

/** Origin of a media URL; blob:/data: and stream sources count as same-origin. */
export function mediaSourceIsSameOrigin(sourceUrl: string | null, documentOrigin: string): boolean {
  if (!sourceUrl) return true; // srcObject (MediaStream / MediaSource handle) or no source yet
  if (sourceUrl.startsWith('data:')) return true;
  try {
    const url = new URL(sourceUrl, documentOrigin === 'null' ? undefined : documentOrigin);
    // blob: URLs carry their creator's origin (`new URL('blob:https://a/x').origin === 'https://a'`).
    return url.origin === documentOrigin;
  } catch {
    return false;
  }
}

export function isHdrTransfer(transfer: string | null | undefined): boolean {
  return typeof transfer === 'string' && HDR_TRANSFERS.includes(transfer.toLowerCase());
}

/** Computed matrices that change only the 2D origin can be reproduced from viewport rects. */
function isTranslation(transform: string): boolean {
  if (transform === 'none') return true;
  const match = /^(matrix|matrix3d)\(([^)]+)\)$/.exec(transform);
  if (!match) return false;
  const values = match[2]!.split(',').map(Number);
  if (!values.every(Number.isFinite)) return false;
  if (match[1] === 'matrix') return values.length === 6 && values.slice(0, 4).every((v, i) => v === [1, 0, 0, 1][i]);
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  return values.length === 16 && values.every((v, i) => i === 12 || i === 13 || v === identity[i]);
}

/** Evaluates the 9 points in order; see the module comment. */
export function evaluateEligibility(input: EligibilityInput): EligibilityResult {
  const g = input.geometry;
  const base = {
    sourceWidth: g.videoWidth,
    sourceHeight: g.videoHeight,
    targetWidth: input.target.width,
    targetHeight: input.target.height,
  };
  const fail = (reason: IneligibleReason): EligibilityResult => ({
    ...base,
    eligible: false,
    point: REASON_POINT[reason],
    reason,
    message: REASON_TEXT[reason],
  });

  // 1. DRM / EME
  if (input.drm.mediaKeys || input.drm.encryptedEventLatched) return fail('drm');

  // 2. Presentation mode
  if (input.presentation.pictureInPicture) return fail('picture-in-picture');
  if (input.presentation.videoIsFullscreenElement) return fail('native-fullscreen');

  // 3. DOM geometry / visibility
  if (!g.connected) return fail('not-connected');
  if (g.display === 'none' || g.visibility === 'hidden' || g.visibility === 'collapse' || g.opacity === 0 || !g.hasLayoutBox) {
    return fail('not-rendered');
  }
  if (!(g.videoWidth > 0) || !(g.videoHeight > 0)) return fail('no-video-dimensions');
  if (g.clientWidth < MIN_ELEMENT_CSS_SIZE || g.clientHeight < MIN_ELEMENT_CSS_SIZE) return fail('too-small');
  if (
    !g.layoutSupported || parseObjectFit(g.objectFit) === null || parseObjectPosition(g.objectPosition) === null ||
    g.transform !== 'none' || g.scale !== 'none' || g.rotate !== 'none' || g.translate !== 'none' ||
    g.currentCSSZoom !== 1 ||
    g.ancestors.some((a) => !isTranslation(a.transform) || a.scale !== 'none' || a.rotate !== 'none' || a.zoom !== 1 || a.currentCSSZoom !== 1) ||
    Math.abs(g.renderedWidth - g.offsetWidth) > 0.5 ||
    Math.abs(g.renderedHeight - g.offsetHeight) > 0.5
  ) {
    return fail('unsupported-layout');
  }

  // 4. CORS / taint
  if (input.origin.taintLatched) return fail('tainted');
  if (
    input.origin.crossOriginAttribute === null &&
    !mediaSourceIsSameOrigin(input.origin.sourceUrl, input.origin.documentOrigin)
  ) {
    return fail('cross-origin');
  }
  const frame = input.probeFrame();
  if (frame.tainted) return fail('tainted');

  // 5. Resolution / upscale threshold (limiting axis) and strictly larger output
  const { width: tw, height: th } = input.target;
  const scale = Math.min(tw / g.videoWidth, th / g.videoHeight);
  if (!(scale >= MIN_UPSCALE_FACTOR) || !(tw * th > g.videoWidth * g.videoHeight)) return fail('no-upscale-needed');

  // 6. Dynamic range
  if (isHdrTransfer(frame.transfer)) return fail('hdr');

  // 7. User / site settings. An explicit per-site user choice wins over the built-in rule. The
  //    popup's switch names the tab's site: turning it off also stops embedded players (iframes
  //    of other origins), and a player's own host turned off stays off on every embedding page.
  if (!input.settings.enabled) return fail('extension-disabled');
  const override = siteOverrideFor(input.settings, input.hostname);
  const topOverride = input.topHostname !== null ? siteOverrideFor(input.settings, input.topHostname) : undefined;
  if (override === false || topOverride === false) return fail('site-disabled');
  if (override === undefined && !input.siteRuleEnabled) return fail('site-rule-disabled');

  // 8. WebGPU device availability
  if (!input.gpuAvailable()) return fail('webgpu-unavailable');

  // 9. Frame budget health
  if (!input.budgetHealthy()) return fail('frame-budget');

  return { ...base, eligible: true, point: null, reason: null, message: null };
}

// ---------------------------------------------------------------------------------------------
// Live-DOM adapters
// ---------------------------------------------------------------------------------------------

/**
 * Per-video latched ineligibility signals that outlive a single evaluation:
 * - DRM: an `encrypted` or `waitingforkey` event was observed;
 * - taint: the session saw `new VideoFrame(video)` throw a SecurityError;
 * - HDR: the session captured a frame with a PQ / HLG transfer.
 * All are cleared on a media source change.
 */
export class EligibilityLatches {
  private drm = new WeakSet<object>();
  private taint = new WeakSet<object>();
  private hdr = new WeakSet<object>();

  latchDrm(video: object): void {
    this.drm.add(video);
  }

  latchTaint(video: object): void {
    this.taint.add(video);
  }

  latchHdr(video: object): void {
    this.hdr.add(video);
  }

  isHdrLatched(video: object): boolean {
    return this.hdr.has(video);
  }

  isDrmLatched(video: object): boolean {
    return this.drm.has(video);
  }

  isTaintLatched(video: object): boolean {
    return this.taint.has(video);
  }

  reset(video: object): void {
    this.drm.delete(video);
    this.taint.delete(video);
    this.hdr.delete(video);
  }

  /** Latches DRM on `encrypted` / `waitingforkey`; returns a function that removes the listeners. */
  watchDrmEvents(
    video: Pick<EventTarget, 'addEventListener' | 'removeEventListener'>,
    onLatched?: () => void,
  ): () => void {
    const listener = (): void => {
      this.latchDrm(video);
      onLatched?.();
    };
    video.addEventListener('encrypted', listener);
    video.addEventListener('waitingforkey', listener);
    return () => {
      video.removeEventListener('encrypted', listener);
      video.removeEventListener('waitingforkey', listener);
    };
  }
}

function isSecurityError(err: unknown): boolean {
  if (typeof err !== 'object' || err === null) return false;
  const e = err as { name?: unknown; message?: unknown };
  return e.name === 'SecurityError' || (typeof e.message === 'string' && /tainted|cross-origin/i.test(e.message));
}

export interface ProbedFrame extends FrameProbeResult {
  /** A frame was actually inspected (result can be cached for the current source). */
  inspected: boolean;
}

/**
 * Wraps the current frame in a VideoFrame to learn its transfer function and whether it is
 * tainted; the frame is closed immediately. Needs readyState >= HAVE_CURRENT_DATA.
 */
export function probeVideoFrame(video: HTMLVideoElement): ProbedFrame {
  if (video.readyState < 2 || typeof VideoFrame === 'undefined') return { tainted: false, transfer: null, inspected: false };
  let frame: VideoFrame | null = null;
  try {
    frame = new VideoFrame(video);
    return { tainted: false, transfer: frame.colorSpace?.transfer ?? null, inspected: true };
  } catch (err) {
    if (isSecurityError(err)) return { tainted: true, transfer: null, inspected: true };
    return { tainted: false, transfer: null, inspected: false };
  } finally {
    frame?.close();
  }
}

export interface LiveInputDeps {
  settings: UserSettings;
  hostname: string;
  topHostname: string | null;
  siteRuleEnabled: boolean;
  latches: EligibilityLatches;
  probeFrame: () => FrameProbeResult;
  gpuAvailable: () => boolean;
  budgetHealthy: () => boolean;
  /** Whether an active overlay established a usable parent positioning context. */
  layoutSupported?: boolean;
  /** Active overlays suspend their picture-hiding override for these synchronous reads. */
  readSiteStyle?: <T>(read: () => T) => T;
  /** Defaults to the global window. */
  win?: Pick<Window, 'getComputedStyle' | 'devicePixelRatio' | 'location'> & { document: Document };
}

/** Reads a live video element into an {@link EligibilityInput}. */
export function collectEligibilityInput(video: HTMLVideoElement, deps: LiveInputDeps): EligibilityInput {
  const win = deps.win ?? window;
  const doc = win.document;
  const style = win.getComputedStyle(video);
  const readObjectStyle = () => {
    const site = win.getComputedStyle(video);
    return { objectFit: site.objectFit, objectPosition: site.objectPosition };
  };
  const { objectFit, objectPosition } = deps.readSiteStyle ? deps.readSiteStyle(readObjectStyle) : readObjectStyle();
  const connected = video.isConnected;
  const rects = connected ? video.getClientRects().length : 0;
  const dpr = win.devicePixelRatio || 1;

  const cssZoom = (el: Element): number => (el as Element & { currentCSSZoom?: number }).currentCSSZoom ?? 1;
  const composedParent = (el: Element): Element | null =>
    el.assignedSlot ?? el.parentElement ??
    (el.parentNode && 'host' in el.parentNode ? (el.parentNode as ShadowRoot).host : null);
  const ancestors: EligibilityInput['geometry']['ancestors'] = [];
  for (let parent = composedParent(video); parent; parent = composedParent(parent)) {
    const computed = win.getComputedStyle(parent);
    ancestors.push({
      transform: computed.transform || 'none',
      scale: computed.scale || 'none',
      rotate: computed.rotate || 'none',
      zoom: !computed.zoom || computed.zoom === 'normal' ? 1 : Number(computed.zoom),
      currentCSSZoom: cssZoom(parent),
    });
    if (parent === doc.documentElement) break;
  }

  // Target resolution from the rendered content rect inside the content box.
  const num = (v: string): number => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : 0;
  };
  const rect = video.getBoundingClientRect();
  const boxW = Math.max(0, rect.width - num(style.borderLeftWidth) - num(style.borderRightWidth) - num(style.paddingLeft) - num(style.paddingRight));
  const boxH = Math.max(0, rect.height - num(style.borderTopWidth) - num(style.borderBottomWidth) - num(style.paddingTop) - num(style.paddingBottom));
  const fit = parseObjectFit(objectFit);
  const position = parseObjectPosition(objectPosition);
  const geometry = fit && position ? computeLetterboxGeometry(
    video.videoWidth,
    video.videoHeight,
    boxW,
    boxH,
    fit,
    { objectPosition: position, devicePixelRatio: dpr },
  ) : null;

  return {
    drm: { mediaKeys: video.mediaKeys != null, encryptedEventLatched: deps.latches.isDrmLatched(video) },
    presentation: {
      pictureInPicture: (doc as Document & { pictureInPictureElement?: Element | null }).pictureInPictureElement === video,
      videoIsFullscreenElement: deepFullscreenElement(doc) === video,
    },
    geometry: {
      connected,
      layoutSupported: (deps.layoutSupported ?? true) && video.assignedSlot == null && videoEffectsSupported(style, video.parentElement ? win.getComputedStyle(video.parentElement).display : '') &&
        positionPatchSupported(style) && (!video.parentElement || positionPatchSupported(win.getComputedStyle(video.parentElement))),
      display: style.display,
      visibility: style.visibility,
      opacity: Number(style.opacity),
      hasLayoutBox: rects > 0,
      clientWidth: video.clientWidth,
      clientHeight: video.clientHeight,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      objectFit,
      objectPosition,
      transform: style.transform || 'none',
      scale: style.scale || 'none',
      rotate: style.rotate || 'none',
      translate: style.translate || 'none',
      currentCSSZoom: cssZoom(video),
      ancestors,
      renderedWidth: rect.width,
      renderedHeight: rect.height,
      offsetWidth: video.offsetWidth,
      offsetHeight: video.offsetHeight,
    },
    origin: {
      sourceUrl: video.currentSrc || video.getAttribute('src') || null,
      documentOrigin: win.location.origin,
      crossOriginAttribute: video.getAttribute('crossorigin'),
      taintLatched: deps.latches.isTaintLatched(video),
    },
    target: geometry ? { width: geometry.backingWidth, height: geometry.backingHeight } : { width: 0, height: 0 },
    settings: deps.settings,
    hostname: deps.hostname,
    topHostname: deps.topHostname,
    siteRuleEnabled: deps.siteRuleEnabled,
    probeFrame: deps.probeFrame,
    gpuAvailable: deps.gpuAvailable,
    budgetHealthy: deps.budgetHealthy,
  };
}
