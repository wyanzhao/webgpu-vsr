/**
 * Sibling-canvas overlay: geometry (object-fit letterbox/pillarbox/crop at sub-pixel precision,
 * 1080p output ceiling) and the DOM component that keeps a `<canvas class="webgpu-vsr-canvas">`
 * aligned under the video's rendered content and owns the fail-safe picture-hiding contract.
 *
 * Geometry is kept in CSS px floats end to end; only the canvas backing store is integral.
 */

import { captureLayoutSnapshot, compareLayoutSnapshots, registerExtensionNode } from './invariance';

// ---------------------------------------------------------------------------------------------
// Pure geometry
// ---------------------------------------------------------------------------------------------

export type ObjectFit = 'contain' | 'cover' | 'fill' | 'none' | 'scale-down';

/** One axis of object-position: offset = percent/100 * (box - content) + px. */
export interface PositionComponent {
  percent: number;
  px: number;
}

export interface ObjectPosition {
  x: PositionComponent;
  y: PositionComponent;
}

export const CENTER_POSITION: Readonly<ObjectPosition> = Object.freeze({
  x: Object.freeze({ percent: 50, px: 0 }),
  y: Object.freeze({ percent: 50, px: 0 }),
});

/**
 * - `exact`: content fills the box exactly (aspect ratios match or object-fit: fill with equal aspect)
 * - `letterbox`: full width, bars above and below
 * - `pillarbox`: full height, bars left and right
 * - `windowbox`: smaller than the box on both axes (object-fit none / scale-down)
 * - `crop`: content exceeds the box on at least one axis (cover / none); the canvas is clipped
 * - `stretch`: object-fit: fill with a different aspect ratio
 */
export type FitMode = 'exact' | 'letterbox' | 'pillarbox' | 'windowbox' | 'crop' | 'stretch';

export interface Size {
  width: number;
  height: number;
}

export interface LetterboxGeometry {
  mode: FitMode;
  /** Rendered video content rect relative to the video's content box (CSS px, may exceed the box). */
  contentX: number;
  contentY: number;
  contentWidth: number;
  contentHeight: number;
  /** Intersection of the content rect with the video content box (CSS px). */
  visibleX: number;
  visibleY: number;
  visibleWidth: number;
  visibleHeight: number;
  /** clip-path insets (CSS px, relative to the content rect) that crop the canvas to the box. */
  clipTop: number;
  clipRight: number;
  clipBottom: number;
  clipLeft: number;
  /** CSS px per source pixel. */
  scaleX: number;
  scaleY: number;
  /** Target output size in device pixels after the 1080p ceiling (integral). */
  backingWidth: number;
  backingHeight: number;
}

export interface GeometryOptions {
  objectPosition?: ObjectPosition;
  devicePixelRatio?: number;
}

export const MAX_OUTPUT_LONG_SIDE = 1920;
export const MAX_OUTPUT_SHORT_SIDE = 1080;

const EPS = 1e-9;

function nearlyEqual(a: number, b: number): boolean {
  return Math.abs(a - b) <= EPS * Math.max(1, Math.abs(a), Math.abs(b));
}

function positive(n: number): boolean {
  return Number.isFinite(n) && n > 0;
}

/**
 * Applies the 1080p ceiling to a device-pixel size: long side <= 1920, short side <= 1080,
 * aspect preserved, never upscaled, rounded to integers (>= 1).
 */
export function computeTargetResolution(deviceWidth: number, deviceHeight: number): Size {
  if (!positive(deviceWidth) || !positive(deviceHeight)) return { width: 0, height: 0 };
  const longSide = Math.max(deviceWidth, deviceHeight);
  const shortSide = Math.min(deviceWidth, deviceHeight);
  const scale = Math.min(1, MAX_OUTPUT_LONG_SIDE / longSide, MAX_OUTPUT_SHORT_SIDE / shortSide);
  return {
    width: Math.max(1, Math.round(deviceWidth * scale)),
    height: Math.max(1, Math.round(deviceHeight * scale)),
  };
}

export interface CanvasResolution extends Size {
  /** True when the canvas shows the network's internal `scale*W x scale*H` image unresampled. */
  presentsInternal: boolean;
}

/**
 * Canvas backing-store size (IMPLEMENTATION_NOTES 3.12). With an internal scale (2 for the VSR
 * models), when the target covers the internal image on both axes the internal image is presented
 * directly and the compositor scales the rest; otherwise the canvas is the target size and the
 * pipeline resamples. `internalScale = null` (Quality 0 bicubic) always renders at the target.
 */
export function computeCanvasResolution(
  target: Size,
  sourceWidth: number,
  sourceHeight: number,
  internalScale: number | null,
): CanvasResolution {
  if (internalScale !== null && positive(internalScale) && positive(sourceWidth) && positive(sourceHeight)) {
    const iw = Math.round(sourceWidth * internalScale);
    const ih = Math.round(sourceHeight * internalScale);
    if (target.width >= iw && target.height >= ih) {
      return { width: iw, height: ih, presentsInternal: true };
    }
  }
  return { width: target.width, height: target.height, presentsInternal: false };
}

/**
 * Computes where the video content is drawn inside its content box for a CSS object-fit, at
 * sub-pixel precision. The axis that limits the fit is set exactly equal to the box size (no
 * floating-point drift), the other is derived from the intrinsic aspect ratio.
 * Returns null when any dimension is not a positive finite number.
 */
export function computeLetterboxGeometry(
  videoWidth: number,
  videoHeight: number,
  boxWidth: number,
  boxHeight: number,
  objectFit: ObjectFit = 'contain',
  options: GeometryOptions = {},
): LetterboxGeometry | null {
  if (!positive(videoWidth) || !positive(videoHeight) || !positive(boxWidth) || !positive(boxHeight)) return null;
  const position = options.objectPosition ?? CENTER_POSITION;
  const dpr = positive(options.devicePixelRatio ?? 1) ? (options.devicePixelRatio ?? 1) : 1;

  // Cross-multiplied aspect comparison: box relatively taller (or equal) than the video.
  const boxTaller = boxWidth * videoHeight <= boxHeight * videoWidth;
  const containSize = (): Size =>
    boxTaller
      ? { width: boxWidth, height: (boxWidth * videoHeight) / videoWidth }
      : { width: (boxHeight * videoWidth) / videoHeight, height: boxHeight };

  let content: Size;
  switch (objectFit) {
    case 'fill':
      content = { width: boxWidth, height: boxHeight };
      break;
    case 'cover':
      content = boxTaller
        ? { width: (boxHeight * videoWidth) / videoHeight, height: boxHeight }
        : { width: boxWidth, height: (boxWidth * videoHeight) / videoWidth };
      break;
    case 'none':
      content = { width: videoWidth, height: videoHeight };
      break;
    case 'scale-down':
      content = videoWidth <= boxWidth && videoHeight <= boxHeight ? { width: videoWidth, height: videoHeight } : containSize();
      break;
    case 'contain':
    default:
      content = containSize();
      break;
  }

  const contentX = (position.x.percent / 100) * (boxWidth - content.width) + position.x.px;
  const contentY = (position.y.percent / 100) * (boxHeight - content.height) + position.y.px;

  const visLeft = Math.max(0, contentX);
  const visTop = Math.max(0, contentY);
  const visRight = Math.min(boxWidth, contentX + content.width);
  const visBottom = Math.min(boxHeight, contentY + content.height);
  const visibleWidth = Math.max(0, visRight - visLeft);
  const visibleHeight = Math.max(0, visBottom - visTop);

  const fullW = nearlyEqual(content.width, boxWidth);
  const fullH = nearlyEqual(content.height, boxHeight);
  const overflow = content.width > boxWidth + EPS * boxWidth || content.height > boxHeight + EPS * boxHeight;
  const sameAspect = nearlyEqual(boxWidth * videoHeight, boxHeight * videoWidth);

  let mode: FitMode;
  if (objectFit === 'fill') mode = sameAspect ? 'exact' : 'stretch';
  else if (overflow || contentX < -EPS || contentY < -EPS) mode = 'crop';
  else if (fullW && fullH) mode = 'exact';
  else if (fullW) mode = 'letterbox';
  else if (fullH) mode = 'pillarbox';
  else mode = 'windowbox';

  const target = computeTargetResolution(content.width * dpr, content.height * dpr);
  return {
    mode,
    contentX,
    contentY,
    contentWidth: content.width,
    contentHeight: content.height,
    visibleX: visLeft,
    visibleY: visTop,
    visibleWidth,
    visibleHeight,
    clipTop: visibleHeight > 0 ? visTop - contentY : 0,
    clipLeft: visibleWidth > 0 ? visLeft - contentX : 0,
    clipBottom: visibleHeight > 0 ? contentY + content.height - visBottom : content.height,
    clipRight: visibleWidth > 0 ? contentX + content.width - visRight : content.width,
    scaleX: content.width / videoWidth,
    scaleY: content.height / videoHeight,
    backingWidth: target.width,
    backingHeight: target.height,
  };
}

/** Normalises a computed `object-fit` value; unknown values -> null. */
export function parseObjectFit(value: string | null | undefined): ObjectFit | null {
  const v = (value ?? '').trim().toLowerCase();
  return v === 'contain' || v === 'cover' || v === 'fill' || v === 'none' || v === 'scale-down' ? v : null;
}

const X_KEYWORDS: Record<string, number> = { left: 0, center: 50, right: 100 };
const Y_KEYWORDS: Record<string, number> = { top: 0, center: 50, bottom: 100 };

function parseLengthOrPercent(token: string): PositionComponent | null {
  let m = /^(-?\d*\.?\d+(?:e-?\d+)?)%$/i.exec(token);
  if (m) return { percent: Number(m[1]), px: 0 };
  m = /^(-?\d*\.?\d+(?:e-?\d+)?)px$/i.exec(token);
  if (m) return { percent: 0, px: Number(m[1]) };
  if (/^-?0$/.test(token)) return { percent: 0, px: 0 };
  // calc(P% + Lpx) / calc(P% - Lpx) as serialised by Blink for mixed values.
  m = /^calc\(\s*(-?\d*\.?\d+)%\s*([+-])\s*(\d*\.?\d+)px\s*\)$/i.exec(token);
  if (m) return { percent: Number(m[1]), px: (m[2] === '-' ? -1 : 1) * Number(m[3]) };
  return null;
}

function splitTokens(value: string): string[] {
  const tokens: string[] = [];
  let depth = 0;
  let cur = '';
  for (const ch of value.trim()) {
    if (ch === '(') depth++;
    if (ch === ')') depth = Math.max(0, depth - 1);
    if (/\s/.test(ch) && depth === 0) {
      if (cur) tokens.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  if (cur) tokens.push(cur);
  return tokens;
}

/**
 * Parses a computed `object-position` (1, 2 or 4 value syntax with keywords, %, px and Blink's
 * `calc(P% ± Lpx)`). Returns null for anything it cannot represent exactly.
 */
export function parseObjectPosition(value: string | null | undefined): ObjectPosition | null {
  const raw = (value ?? '').trim().toLowerCase();
  const t = splitTokens(raw);
  const kw = (token: string, table: Record<string, number>): PositionComponent | null =>
    token in table ? { percent: table[token] as number, px: 0 } : null;

  if (t.length === 1) {
    const [a] = t as [string];
    if (a in Y_KEYWORDS && !(a in X_KEYWORDS)) return { x: { percent: 50, px: 0 }, y: kw(a, Y_KEYWORDS)! };
    const x = kw(a, X_KEYWORDS) ?? parseLengthOrPercent(a);
    return x ? { x, y: { percent: 50, px: 0 } } : null;
  }
  if (t.length === 2) {
    const [a, b] = t as [string, string];
    // Keyword pairs may be given in either order ("top left").
    if ((a === 'top' || a === 'bottom') && (b in X_KEYWORDS)) {
      return { x: kw(b, X_KEYWORDS)!, y: kw(a, Y_KEYWORDS)! };
    }
    const x = kw(a, X_KEYWORDS) ?? parseLengthOrPercent(a);
    const y = kw(b, Y_KEYWORDS) ?? parseLengthOrPercent(b);
    return x && y ? { x, y } : null;
  }
  if (t.length === 4) {
    const [k1, l1, k2, l2] = t as [string, string, string, string];
    const off1 = parseLengthOrPercent(l1);
    const off2 = parseLengthOrPercent(l2);
    if (!off1 || !off2) return null;
    const edge = (k: string, off: PositionComponent, horizontal: boolean): PositionComponent | null => {
      if (horizontal ? k === 'left' : k === 'top') return { percent: off.percent, px: off.px };
      if (horizontal ? k === 'right' : k === 'bottom') return { percent: 100 - off.percent, px: -off.px };
      return null;
    };
    const hFirst = k1 === 'left' || k1 === 'right';
    const x = hFirst ? edge(k1, off1, true) : edge(k2, off2, true);
    const y = hFirst ? edge(k2, off2, false) : edge(k1, off1, false);
    return x && y ? { x, y } : null;
  }
  return null;
}

/**
 * The canvas is earlier in tree order with the video's z-index, so the positioned video's
 * transparent picture area, native controls and captions paint above it.
 */
export function canvasZIndexFor(videoZIndex: string | null | undefined): string {
  const v = (videoZIndex ?? '').trim();
  return /^-?\d+$/.test(v) ? v : 'auto';
}

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface BoxEdges {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface PlacementInput {
  /** getBoundingClientRect() of the video (border box). */
  videoRect: Rect;
  /** Canvas viewport origin measured with left/top set to zero. */
  origin: { left: number; top: number };
  videoBorder: BoxEdges;
  videoPadding: BoxEdges;
}

export interface CanvasPlacement {
  /** CSS px relative to the measured containing-block origin. */
  left: number;
  top: number;
  width: number;
  height: number;
  clip: BoxEdges;
  /** The video's content box in the same coordinate space (HUD anchoring). */
  videoBox: Rect;
}

/** The object-fit box of a video: its border box minus borders and padding. */
export function videoContentBox(input: PlacementInput): Rect {
  const { videoRect, origin, videoBorder, videoPadding } = input;
  return {
    left: videoRect.left - origin.left + videoBorder.left + videoPadding.left,
    top: videoRect.top - origin.top + videoBorder.top + videoPadding.top,
    width: Math.max(0, videoRect.width - videoBorder.left - videoBorder.right - videoPadding.left - videoPadding.right),
    height: Math.max(0, videoRect.height - videoBorder.top - videoBorder.bottom - videoPadding.top - videoPadding.bottom),
  };
}

export function computeCanvasPlacement(input: PlacementInput, geometry: LetterboxGeometry): CanvasPlacement {
  const box = videoContentBox(input);
  return {
    left: box.left + geometry.contentX,
    top: box.top + geometry.contentY,
    width: geometry.contentWidth,
    height: geometry.contentHeight,
    clip: { top: geometry.clipTop, right: geometry.clipRight, bottom: geometry.clipBottom, left: geometry.clipLeft },
    videoBox: box,
  };
}

/** CSS px serialisation that keeps sub-pixel precision without float noise. */
export function cssPx(value: number): string {
  const rounded = Math.round(value * 10000) / 10000;
  return `${Object.is(rounded, -0) ? 0 : rounded}px`;
}

// ---------------------------------------------------------------------------------------------
// Fail-safe picture hiding
// ---------------------------------------------------------------------------------------------

/** The subset of CSSStyleDeclaration used to hide/restore a video. */
export interface InlineStyle {
  getPropertyValue(name: string): string;
  getPropertyPriority(name: string): string;
  setProperty(name: string, value: string, priority?: string): void;
  removeProperty(name: string): string;
}

/** Flushes managed writes before restoring the page's transition declaration. */
export function withTransitionsSuppressed<T>(
  el: { style: InlineStyle },
  fn: () => T,
  flush: () => void,
): T {
  const style = el.style;
  const value = style.getPropertyValue('transition-property');
  const priority = style.getPropertyPriority('transition-property');
  style.setProperty('transition-property', 'none', 'important');
  try {
    return fn();
  } finally {
    try {
      flush();
    } finally {
      if (style.getPropertyValue('transition-property') === 'none' &&
        style.getPropertyPriority('transition-property') === 'important') {
        if (value === '') style.removeProperty('transition-property');
        else style.setProperty('transition-property', value, priority);
      }
    }
  }
}

export const HIDDEN_OBJECT_POSITION = '0px 100000px';

const PICTURE_PROPERTIES = {
  'object-position': HIDDEN_OBJECT_POSITION,
  'background-color': 'transparent',
  'background-image': 'none',
} as const;

type PictureProperty = keyof typeof PICTURE_PROPERTIES;
const PICTURE_PROPERTY_NAMES = Object.keys(PICTURE_PROPERTIES) as PictureProperty[];

/**
 * Hides only the decoded picture and poster, leaving native controls and captions visible.
 * Restores exact inline values and priorities, adopting page rewrites while hidden.
 */
export class VideoPictureGuard {
  private saved: Record<PictureProperty, { value: string; priority: string }> | null = null;

  constructor(
    private readonly style: InlineStyle,
    private readonly flushStyle: () => void = () => {},
    private readonly beforeWrite: () => void = () => {},
  ) {}

  get hidden(): boolean {
    return this.saved !== null;
  }

  hide(): void {
    if (this.saved !== null) {
      this.reassert();
      return;
    }
    this.beforeWrite();
    this.saved = Object.fromEntries(PICTURE_PROPERTY_NAMES.map((name) => [name, {
      value: this.style.getPropertyValue(name),
      priority: this.style.getPropertyPriority(name),
    }])) as NonNullable<VideoPictureGuard['saved']>;
    this.apply();
  }

  /** Re-applies the hidden state if the page overwrote it; adopts the page's value for restore. */
  reassert(): void {
    if (this.saved === null) return;
    for (const name of PICTURE_PROPERTY_NAMES) {
      const value = this.style.getPropertyValue(name);
      const priority = this.style.getPropertyPriority(name);
      if (value === PICTURE_PROPERTIES[name] && priority === 'important') continue;
      this.saved[name] = { value, priority };
      withTransitionsSuppressed({ style: this.style }, () => {
        this.style.setProperty(name, PICTURE_PROPERTIES[name], 'important');
      }, this.flushStyle);
    }
  }

  restore(): void {
    if (this.saved === null) return;
    this.applySiteStyle();
    this.saved = null;
  }

  /** Reads site styles synchronously with our overrides suspended; no paint can intervene. */
  readSiteStyle<T>(read: () => T): T {
    if (this.saved === null) return read();
    this.applySiteStyle();
    try {
      return read();
    } finally {
      this.apply();
    }
  }

  private applySiteStyle(): void {
    if (this.saved === null) return;
    withTransitionsSuppressed({ style: this.style }, () => {
      for (const name of PICTURE_PROPERTY_NAMES) {
        if (this.style.getPropertyValue(name) !== PICTURE_PROPERTIES[name] || this.style.getPropertyPriority(name) !== 'important') continue;
        const { value, priority } = this.saved![name];
        if (value === '') this.style.removeProperty(name);
        else this.style.setProperty(name, value, priority);
      }
    }, this.flushStyle);
  }

  private apply(): void {
    withTransitionsSuppressed({ style: this.style }, () => {
      for (const name of PICTURE_PROPERTY_NAMES) {
        this.style.setProperty(name, PICTURE_PROPERTIES[name], 'important');
      }
    }, this.flushStyle);
  }
}

// ---------------------------------------------------------------------------------------------
// DOM overlay component
// ---------------------------------------------------------------------------------------------

export interface ResizeObserverLike {
  observe(target: Element): void;
  disconnect(): void;
}

export interface MutationObserverLike {
  observe(target: Node, options: MutationObserverInit): void;
  disconnect(): void;
}

export interface IntersectionObserverLike {
  observe(target: Element): void;
  unobserve(target: Element): void;
  disconnect(): void;
}

/** Browser services used by {@link CanvasOverlay}; injectable for tests. */
export interface OverlayEnvironment {
  document: Document;
  getComputedStyle(el: Element): CSSStyleDeclaration;
  elementsFromPoint(x: number, y: number): Element[];
  devicePixelRatio(): number;
  createResizeObserver(callback: () => void): ResizeObserverLike;
  /** Child-list observer of the video's parent (detects a page removing the canvas). */
  createMutationObserver(callback: () => void): MutationObserverLike;
  createIntersectionObserver(callback: (entries: IntersectionObserverEntry[]) => void): IntersectionObserverLike | null;
  addWindowListener(type: string, listener: () => void): void;
  removeWindowListener(type: string, listener: () => void): void;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export function browserOverlayEnvironment(): OverlayEnvironment {
  return {
    document,
    getComputedStyle: (el) => window.getComputedStyle(el),
    elementsFromPoint: (x, y) => document.elementsFromPoint(x, y),
    devicePixelRatio: () => window.devicePixelRatio || 1,
    createResizeObserver: (cb) => new ResizeObserver(() => cb()),
    createMutationObserver: (cb) => new MutationObserver(() => cb()),
    createIntersectionObserver: (cb) => typeof IntersectionObserver === 'undefined' ? null :
      new IntersectionObserver(cb, { threshold: [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1] }),
    addWindowListener: (type, l) => window.addEventListener(type, l, { passive: true }),
    removeWindowListener: (type, l) => window.removeEventListener(type, l),
    setTimeout: (fn, ms) => window.setTimeout(fn, ms),
    clearTimeout: (h) => window.clearTimeout(h as number),
  };
}

/** Resolves the fullscreen element through open shadow roots. */
export function deepFullscreenElement(doc: Document): Element | null {
  let element = doc.fullscreenElement;
  while (element?.shadowRoot?.fullscreenElement) element = element.shadowRoot.fullscreenElement;
  return element;
}

function composedContains(container: Element, element: Element): boolean {
  for (let current: Element | null = element; current; current = current.assignedSlot ?? current.parentElement ??
    (current.parentNode && 'host' in current.parentNode ? (current.parentNode as ShadowRoot).host : null)) {
    if (current === container) return true;
  }
  return false;
}

export type FullscreenState = 'none' | 'container' | 'video';

export interface OverlayLayout {
  geometry: LetterboxGeometry;
  placement: CanvasPlacement;
  target: Size;
  /** Desired canvas backing store (internal image or target). */
  canvas: CanvasResolution;
  /** Backing store actually applied (lags `canvas` while a fullscreen transition settles). */
  backing: Size;
  fullscreen: FullscreenState;
  sourceWidth: number;
  sourceHeight: number;
}

export interface OverlayOptions {
  /** Internal network scale (2 for RTX VSR); null renders at the target size (Quality 0). */
  internalScale?: number | null;
  /** Called after every sync whose layout or backing store changed. */
  onLayout?: (layout: OverlayLayout) => void;
  /** Debounce for backing-store resizes during fullscreen transitions. */
  fullscreenResizeDebounceMs?: number;
  env?: OverlayEnvironment;
}

export const OVERLAY_CANVAS_CLASS = 'webgpu-vsr-canvas';

const MEDIA_RESTORE_EVENTS = ['seeking', 'emptied', 'error', 'abort'] as const;
const MEDIA_SYNC_EVENTS = ['loadedmetadata', 'resize'] as const;

function px(value: string | null | undefined): number {
  const n = parseFloat(value ?? '');
  return Number.isFinite(n) ? n : 0;
}

/** Effects and positioning that cannot be reproduced by a sibling canvas. */
export function videoEffectsSupported(style: Pick<CSSStyleDeclaration, 'clip' | 'clipPath' | 'maskImage' | 'webkitMaskImage' | 'mixBlendMode' | 'position' | 'backdropFilter'> & { objectViewBox?: string; webkitBackdropFilter?: string; boxShadow?: string; outlineStyle?: string; outlineOffset?: string; order?: string }, parentDisplay = ''): boolean {
  return (!style.clip || style.clip === 'auto') &&
    (!style.clipPath || style.clipPath === 'none') &&
    (!style.maskImage || style.maskImage === 'none') &&
    (!style.webkitMaskImage || style.webkitMaskImage === 'none') &&
    (!style.mixBlendMode || style.mixBlendMode === 'normal') &&
    (!style.objectViewBox || style.objectViewBox === 'none') &&
    (!style.backdropFilter || style.backdropFilter === 'none') &&
    (!style.webkitBackdropFilter || style.webkitBackdropFilter === 'none') &&
    !/\binset\b/i.test(style.boxShadow ?? '') &&
    (!style.outlineStyle || style.outlineStyle === 'none' || px(style.outlineOffset) >= 0) &&
    (!['flex', 'inline-flex', 'grid', 'inline-grid'].includes(parentDisplay) || !style.order || style.order === '0') &&
    style.position !== 'fixed';
}

/** A static box with offsets or a z-index would change layout or stacking if made relative. */
export function positionPatchSupported(style: Pick<CSSStyleDeclaration, 'position' | 'top' | 'right' | 'bottom' | 'left' | 'inset' | 'zIndex'>): boolean {
  return style.position !== 'static' || [style.top, style.right, style.bottom, style.left, style.inset, style.zIndex]
    .every((value) => !value || value === 'auto');
}

function sameLayout(a: OverlayLayout | null, b: OverlayLayout): boolean {
  if (!a) return false;
  const pa = a.placement;
  const pb = b.placement;
  return (
    pa.left === pb.left &&
    pa.top === pb.top &&
    pa.width === pb.width &&
    pa.height === pb.height &&
    pa.clip.top === pb.clip.top &&
    pa.clip.right === pb.clip.right &&
    pa.clip.bottom === pb.clip.bottom &&
    pa.clip.left === pb.clip.left &&
    a.canvas.width === b.canvas.width &&
    a.canvas.height === b.canvas.height &&
    a.backing.width === b.backing.width &&
    a.backing.height === b.backing.height &&
    a.fullscreen === b.fullscreen &&
    a.sourceWidth === b.sourceWidth &&
    a.sourceHeight === b.sourceHeight
  );
}

/**
 * Injects and maintains `<canvas class="webgpu-vsr-canvas">` as the previous sibling of the video.
 *
 * Picture-hiding contract: the native picture stays visible until {@link markFramePresented} is called by
 * the session after a frame was presented on the canvas, and only while the canvas is mounted
 * right before the video ({@link isMounted}); {@link restoreVideo} (also triggered automatically on
 * seeking/emptied/error/abort, detachment, native video fullscreen, destroy, and when the page
 * removes or moves the canvas or the video out of their sibling position, e.g. a player that
 * re-renders its container but keeps the same <video>) restores the original inline picture styles and
 * hides the canvas until the next presented frame.
 */
export class CanvasOverlay {
  readonly video: HTMLVideoElement;
  readonly canvas: HTMLCanvasElement;
  presentationEpoch = 0;
  private readonly env: OverlayEnvironment;
  private readonly picture: VideoPictureGuard;
  private readonly onLayout?: (layout: OverlayLayout) => void;
  private readonly debounceMs: number;
  private internalScale: number | null;
  private parent: HTMLElement | null = null;
  private videoPositionPatched: { previous: string; priority: string } | null = null;
  private videoHadStyleAttribute: boolean | null = null;
  private videoStacking: Array<{ index: number; above: Element | null } | null> | null = null;
  private observer: ResizeObserverLike | null = null;
  private childObserver: MutationObserverLike | null = null;
  private intersectionObserver: IntersectionObserverLike | null = null;
  private readonly intersections = new Map<Element, IntersectionObserverEntry>();
  private intersectionsPending = false;
  /** Visible content-fit rect relative to the video's viewport border box. */
  private visibleCanvasRect: Rect | null = null;
  private currentLayout: OverlayLayout | null = null;
  private framePresented = false;
  private hasPresented = false;
  private remountNeedsRedraw = false;
  private unsupportedLayout = false;
  private insertionLayoutFailure = false;
  private destroyed = false;
  private fullscreenTransition = false;
  private transitionTimer: unknown = null;

  private readonly onViewport = (): void => {
    this.sync();
  };
  private readonly onFullscreenChange = (): void => {
    this.fullscreenTransition = true;
    this.armTransitionTimer();
    this.sync();
  };
  private readonly onRestoreEvent = (): void => {
    this.restoreVideo();
  };
  /** The parent's children changed: a hidden video whose canvas is gone must become visible. */
  private readonly onChildList = (): void => {
    if (!this.destroyed && !this.isMounted) {
      if (this.framePresented) this.restoreVideo();
      this.disconnectIntersections();
      this.unpatchVideo();
    }
  };

  constructor(video: HTMLVideoElement, options: OverlayOptions = {}) {
    this.video = video;
    this.env = options.env ?? browserOverlayEnvironment();
    this.onLayout = options.onLayout;
    this.internalScale = options.internalScale ?? null;
    this.debounceMs = options.fullscreenResizeDebounceMs ?? 150;
    this.picture = new VideoPictureGuard(video.style, () => { void this.env.getComputedStyle(video).objectPosition; },
      () => this.recordVideoStyleAttribute());
    this.canvas = this.env.document.createElement('canvas');
    registerExtensionNode(this.canvas);
    this.canvas.className = OVERLAY_CANVAS_CLASS;
    this.canvas.setAttribute('aria-hidden', 'true');
    this.applyStaticStyle();
    this.setCanvasVisible(false);

    for (const type of MEDIA_RESTORE_EVENTS) video.addEventListener(type, this.onRestoreEvent);
    for (const type of MEDIA_SYNC_EVENTS) video.addEventListener(type, this.onViewport);
    this.env.addWindowListener('resize', this.onViewport);
    this.env.document.addEventListener('fullscreenchange', this.onFullscreenChange);

    this.mount();
    this.sync();
  }

  get layout(): OverlayLayout | null {
    return this.currentLayout;
  }

  /** True once a frame has been presented and the native video picture is hidden. */
  get videoHidden(): boolean {
    return this.picture.hidden;
  }

  get layoutSupported(): boolean {
    return !this.unsupportedLayout;
  }

  get hasInsertionLayoutFailure(): boolean {
    return this.insertionLayoutFailure;
  }

  /** Reads the site's computed values without the picture-hiding overrides. */
  readSiteStyle<T>(read: () => T): T {
    this.picture.reassert();
    return this.picture.readSiteStyle(read);
  }

  get isDestroyed(): boolean {
    return this.destroyed;
  }

  get containerElement(): HTMLElement | null {
    return this.parent;
  }

  /** The canvas sits right before the video in the parent the overlay is anchored to. */
  get isMounted(): boolean {
    const parent = this.parent;
    return (
      parent !== null &&
      this.video.assignedSlot == null &&
      this.video.parentElement === parent &&
      this.canvas.parentElement === parent &&
      this.canvas.nextSibling === this.video &&
      this.video.isConnected
    );
  }

  setInternalScale(scale: number | null): void {
    if (scale === this.internalScale) return;
    this.internalScale = scale;
    this.sync();
  }

  /**
   * Called by the session after a frame completed and passed validation on the canvas. Shows the
   * canvas and hides only the video picture (atomically within the same task).
   */
  markFramePresented(): void {
    if (this.destroyed) return;
    // A canvas the page removed (or separated from the video) shows nothing: keep the video.
    if (!this.canPresent()) return;
    const wasHidden = this.picture.hidden;
    this.framePresented = true;
    this.hasPresented = true;
    this.setCanvasVisible(true);
    this.picture.hide();
    if (!wasHidden && this.intersections.size > 0) this.sync();
  }

  /** Hides the picture only if a frame has already been presented; returns whether it is hidden. */
  hideVideo(): boolean {
    if (this.destroyed || !this.framePresented || !this.canPresent()) return false;
    const wasHidden = this.picture.hidden;
    this.setCanvasVisible(true);
    this.picture.hide();
    if (!wasHidden && this.intersections.size > 0) this.sync();
    return this.picture.hidden;
  }

  private canPresent(): boolean {
    const layout = this.currentLayout;
    return !this.unsupportedLayout && !this.intersectionsPending && this.isMounted && layout !== null && layout.fullscreen !== 'video' &&
      layout.geometry.visibleWidth > 0 && layout.geometry.visibleHeight > 0;
  }

  /** Restores the original inline picture styles and hides the canvas until the next presented frame. */
  restoreVideo(): void {
    this.presentationEpoch++;
    this.framePresented = false;
    this.picture.restore();
    this.removeEmptyVideoStyleAttribute();
    this.setCanvasVisible(false);
  }

  /**
   * Re-anchors after the video was re-parented (theater mode, SPA re-mounts): moves the canvas to
   * the video's current parent and re-observes. GPU resources owned by the session are untouched.
   */
  reattach(): void {
    if (this.destroyed) return;
    this.mount();
    this.sync();
  }

  /** Recomputes geometry, applies CSS and (possibly debounced) backing-store size. */
  sync(): OverlayLayout | null {
    if (this.destroyed || this.insertionLayoutFailure) return null;
    if (!this.video.isConnected || !this.video.parentElement) {
      this.restoreVideo();
      this.disconnectIntersections();
      this.unpatchVideo();
      return this.currentLayout;
    }
    if (this.video.parentElement !== this.parent || this.canvas.nextSibling !== this.video) this.mount();
    if (this.insertionLayoutFailure) return null;
    if (!this.isMounted) this.disconnectIntersections();
    this.picture.reassert();

    const doc = this.env.document;
    const fsEl = deepFullscreenElement(doc);
    const fullscreen: FullscreenState = fsEl === this.video ? 'video' : fsEl && composedContains(fsEl, this.video) ? 'container' : 'none';
    if (fullscreen === 'video') {
      // A fullscreen <video> is composited alone; a sibling canvas cannot appear above it.
      this.restoreVideo();
    }

    const vStyle = this.picture.readSiteStyle(() => {
      const style = this.env.getComputedStyle(this.video);
      // Copy values while the override is suspended: computed styles can be live.
      return {
        objectFit: style.objectFit,
        objectPosition: style.objectPosition,
        zIndex: style.zIndex,
        opacity: style.opacity,
        filter: style.filter || 'none',
        effectsSupported: videoEffectsSupported(style, this.video.parentElement ? this.env.getComputedStyle(this.video.parentElement).display : ''),
        borderTopLeftRadius: style.borderTopLeftRadius,
        borderTopRightRadius: style.borderTopRightRadius,
        borderBottomRightRadius: style.borderBottomRightRadius,
        borderBottomLeftRadius: style.borderBottomLeftRadius,
        borderTopWidth: style.borderTopWidth,
        borderRightWidth: style.borderRightWidth,
        borderBottomWidth: style.borderBottomWidth,
        borderLeftWidth: style.borderLeftWidth,
        paddingTop: style.paddingTop,
        paddingRight: style.paddingRight,
        paddingBottom: style.paddingBottom,
        paddingLeft: style.paddingLeft,
      };
    });
    const fit = parseObjectFit(vStyle.objectFit);
    const position = parseObjectPosition(vStyle.objectPosition);
    const positionSupported = this.ensureVideoPosition();
    if (this.insertionLayoutFailure) return null;
    this.unsupportedLayout = this.video.assignedSlot != null || !vStyle.effectsSupported || !fit || !position ||
      !positionSupported;
    if (this.unsupportedLayout) {
      this.restoreVideo();
      this.currentLayout = null;
      return null;
    }
    const sourceWidth = this.video.videoWidth;
    const sourceHeight = this.video.videoHeight;
    const vr = this.video.getBoundingClientRect();
    this.applyStaticStyle();
    this.canvas.style.setProperty('left', '0px', 'important');
    this.canvas.style.setProperty('top', '0px', 'important');
    const origin = this.canvas.getBoundingClientRect();
    const input: PlacementInput = {
      videoRect: { left: vr.left, top: vr.top, width: vr.width, height: vr.height },
      origin: { left: origin.left, top: origin.top },
      videoBorder: {
        top: px(vStyle.borderTopWidth),
        right: px(vStyle.borderRightWidth),
        bottom: px(vStyle.borderBottomWidth),
        left: px(vStyle.borderLeftWidth),
      },
      videoPadding: {
        top: px(vStyle.paddingTop),
        right: px(vStyle.paddingRight),
        bottom: px(vStyle.paddingBottom),
        left: px(vStyle.paddingLeft),
      },
    };
    const box = videoContentBox(input);
    const geometry = computeLetterboxGeometry(sourceWidth, sourceHeight, box.width, box.height, fit!, {
      objectPosition: position!,
      devicePixelRatio: this.env.devicePixelRatio(),
    });
    if (!geometry) {
      this.restoreVideo();
      return this.currentLayout;
    }
    const placement = computeCanvasPlacement(input, geometry);
    const target = { width: geometry.backingWidth, height: geometry.backingHeight };
    const canvasRes = computeCanvasResolution(target, sourceWidth, sourceHeight, this.internalScale);

    // Normalise overlapping radii on the outer border box before deriving inner radii.
    const radius = (value: string, axis: number): number | null => {
      const part = value.includes('(') ? null : parseLengthOrPercent(value);
      const resolved = part ? part.percent / 100 * axis + part.px : NaN;
      return Number.isFinite(resolved) && resolved >= 0 ? resolved : null;
    };
    const corners = [
      [vStyle.borderTopLeftRadius, 'left', 'top'],
      [vStyle.borderTopRightRadius, 'right', 'top'],
      [vStyle.borderBottomRightRadius, 'right', 'bottom'],
      [vStyle.borderBottomLeftRadius, 'left', 'bottom'],
    ] as const;
    const outerRadii: Array<[number, number]> = [];
    for (const [value] of corners) {
      const parts = splitTokens(value);
      const x = parts.length >= 1 && parts.length <= 2 ? radius(parts[0]!, vr.width) : null;
      const y = parts.length >= 1 && parts.length <= 2 ? radius(parts[1] ?? parts[0]!, vr.height) : null;
      if (x === null || y === null) {
        this.unsupportedLayout = true;
        this.restoreVideo();
        this.currentLayout = null;
        return null;
      }
      outerRadii.push([x, y]);
    }
    const [tl, tr, br, bl] = outerRadii as [[number, number], [number, number], [number, number], [number, number]];
    const f = Math.min(1, vr.width / (tl[0] + tr[0]), vr.width / (bl[0] + br[0]),
      vr.height / (tl[1] + bl[1]), vr.height / (tr[1] + br[1]));
    const radii = corners.map(([, horizontal, vertical], i) => [
      Math.max(0, outerRadii[i]![0] * f - input.videoBorder[horizontal] - input.videoPadding[horizontal]),
      Math.max(0, outerRadii[i]![1] * f - input.videoBorder[vertical] - input.videoPadding[vertical]),
    ]);
    this.applyStaticStyle();
    this.setCanvasVisible(this.framePresented && fullscreen !== 'video');
    const s = this.canvas.style;
    s.setProperty('z-index', canvasZIndexFor(vStyle.zIndex), 'important');
    s.setProperty('opacity', vStyle.opacity, 'important');
    s.setProperty('filter', vStyle.filter, 'important');
    s.setProperty('left', cssPx(placement.left), 'important');
    s.setProperty('top', cssPx(placement.top), 'important');
    s.setProperty('width', cssPx(placement.width), 'important');
    s.setProperty('height', cssPx(placement.height), 'important');
    const c = placement.clip;
    if (radii.some(([x, y]) => x! > 0 && y! > 0)) {
      const xs = radii.map(([x]) => cssPx(x!));
      const ys = radii.map(([, y]) => cssPx(y!));
      const compact = (values: string[]): string => values.every((v) => v === values[0]) ? values[0]! : values.join(' ');
      const round = xs.every((x, i) => x === ys[i]) ? compact(xs) : `${compact(xs)} / ${compact(ys)}`;
      const merge = (edge: number, crop: number): number => crop > 0 ? Math.max(edge, crop) : edge;
      const top = merge(-geometry.contentY, c.top);
      const right = merge(geometry.contentX + geometry.contentWidth - box.width, c.right);
      const bottom = merge(geometry.contentY + geometry.contentHeight - box.height, c.bottom);
      const left = merge(-geometry.contentX, c.left);
      s.setProperty('clip-path', `inset(${cssPx(top)} ${cssPx(right)} ${cssPx(bottom)} ${cssPx(left)} round ${round})`, 'important');
    } else if (c.top > 0 || c.right > 0 || c.bottom > 0 || c.left > 0) {
      s.setProperty('clip-path', `inset(${cssPx(c.top)} ${cssPx(c.right)} ${cssPx(c.bottom)} ${cssPx(c.left)})`, 'important');
    } else {
      s.setProperty('clip-path', 'none', 'important');
    }
    const canvasStyle = this.env.getComputedStyle(this.canvas);
    const expectedStyles = {
      display: 'block',
      transform: 'none',
      rotate: 'none',
      scale: 'none',
      translate: 'none',
      clip: 'auto',
      clipPath: s.getPropertyValue('clip-path'),
      maskImage: 'none',
      mixBlendMode: 'normal',
      opacity: vStyle.opacity,
      filter: vStyle.filter,
      objectFit: 'fill',
    };
    if (Object.entries(expectedStyles).some(([name, value]) => canvasStyle[name as keyof typeof expectedStyles] !== value) ||
      (canvasStyle.zoom !== undefined && canvasStyle.zoom !== '' && canvasStyle.zoom !== '1') ||
      (this.framePresented && fullscreen !== 'video' && canvasStyle.visibility !== 'visible')) {
      this.unsupportedLayout = true;
      this.restoreVideo();
      this.currentLayout = null;
      return null;
    }
    // Check the actual viewport placement, independent of the assumed containing block.
    const canvasRect = this.canvas.getBoundingClientRect();
    const expectedLeft = vr.left + input.videoBorder.left + input.videoPadding.left + geometry.contentX;
    const expectedTop = vr.top + input.videoBorder.top + input.videoPadding.top + geometry.contentY;
    const edgeErrors = [
      canvasRect.left - expectedLeft,
      canvasRect.top - expectedTop,
      canvasRect.left + canvasRect.width - (expectedLeft + geometry.contentWidth),
      canvasRect.top + canvasRect.height - (expectedTop + geometry.contentHeight),
    ];
    if (edgeErrors.some((error) => !Number.isFinite(error) || Math.abs(error) > 0.5)) {
      this.unsupportedLayout = true;
      this.restoreVideo();
      this.currentLayout = null;
      return null;
    }
    this.visibleCanvasRect = {
      left: input.videoBorder.left + input.videoPadding.left + geometry.visibleX,
      top: input.videoBorder.top + input.videoPadding.top + geometry.visibleY,
      width: geometry.visibleWidth,
      height: geometry.visibleHeight,
    };
    const visibilityOptions = { checkOpacity: true, checkVisibilityCSS: true, contentVisibilityAuto: true };
    const wasPending = this.intersectionsPending;
    const intersectionsSupported = this.intersectionsSupported(vr, canvasRect);
    if (intersectionsSupported === null) this.restoreVideo();
    if (intersectionsSupported === false ||
      (this.framePresented && fullscreen !== 'video' && typeof this.canvas.checkVisibility === 'function' &&
        typeof this.video.checkVisibility === 'function' && !this.canvas.checkVisibility(visibilityOptions) &&
        this.video.checkVisibility(visibilityOptions))) {
      this.unsupportedLayout = true;
      this.restoreVideo();
      this.currentLayout = null;
      return null;
    }
    this.applyBackingStore(canvasRes);
    const layout: OverlayLayout = {
      geometry,
      placement,
      target,
      canvas: canvasRes,
      backing: { width: this.canvas.width, height: this.canvas.height },
      fullscreen,
      sourceWidth,
      sourceHeight,
    };
    if (geometry.visibleWidth <= 0 || geometry.visibleHeight <= 0) this.restoreVideo();
    else if (this.framePresented && fullscreen !== 'video') this.setCanvasVisible(true);

    const changed = !sameLayout(this.currentLayout, layout) || this.remountNeedsRedraw ||
      (wasPending && intersectionsSupported === true);
    this.remountNeedsRedraw = false;
    this.currentLayout = layout;
    if (changed) this.onLayout?.(layout);
    return layout;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.restoreVideo();
    this.observer?.disconnect();
    this.observer = null;
    this.childObserver?.disconnect();
    this.childObserver = null;
    this.disconnectIntersections();
    if (this.transitionTimer !== null) this.env.clearTimeout(this.transitionTimer);
    this.transitionTimer = null;
    for (const type of MEDIA_RESTORE_EVENTS) this.video.removeEventListener(type, this.onRestoreEvent);
    for (const type of MEDIA_SYNC_EVENTS) this.video.removeEventListener(type, this.onViewport);
    this.env.removeWindowListener('resize', this.onViewport);
    this.env.document.removeEventListener('fullscreenchange', this.onFullscreenChange);
    this.unpatchVideo();
    this.canvas.remove();
    this.parent = null;
  }

  private mount(): void {
    const parent = this.video.parentElement;
    if (!parent || this.insertionLayoutFailure) return;
    if (!this.isMounted) this.disconnectIntersections();
    if (parent !== this.parent) {
      this.unpatchVideo();
      this.parent = parent;
      this.observer?.disconnect();
      this.observer = this.env.createResizeObserver(this.onViewport);
      this.observer.observe(this.video);
      this.observer.observe(parent);
      this.childObserver?.disconnect();
      this.childObserver = this.env.createMutationObserver(this.onChildList);
      this.childObserver.observe(parent, { childList: true });
    }
    if (this.canvas.parentElement !== parent || this.canvas.nextSibling !== this.video) {
      if (this.hasPresented) {
        this.restoreVideo();
        this.remountNeedsRedraw = true;
      }
      const before = captureLayoutSnapshot(this.video, this.env);
      parent.insertBefore(this.canvas, this.video);
      if (compareLayoutSnapshots(before, captureLayoutSnapshot(this.video, this.env))) {
        this.rejectInsertion();
        return;
      }
    }
    if (!this.intersectionObserver && this.isMounted) {
      const observer = this.env.createIntersectionObserver((entries) => {
        if (this.destroyed || this.intersectionObserver !== observer || !this.isMounted) return;
        for (const entry of entries) this.intersections.set(entry.target, entry);
        if (this.picture.hidden || this.intersectionsPending) this.sync();
      });
      this.intersectionObserver = observer;
      observer?.observe(this.video);
      observer?.observe(this.canvas);
    }
  }

  private rejectInsertion(): void {
    this.insertionLayoutFailure = true;
    this.unsupportedLayout = true;
    this.restoreVideo();
    this.unpatchVideo();
    this.canvas.remove();
    this.disconnectIntersections();
    this.currentLayout = null;
  }

  private disconnectIntersections(): void {
    this.intersectionObserver?.disconnect();
    this.intersectionObserver = null;
    this.intersections.clear();
    this.intersectionsPending = false;
    this.visibleCanvasRect = null;
  }

  private intersectionsSupported(videoRect: Rect, canvasRect: Rect): boolean | null {
    if (((!this.picture.hidden || !this.framePresented) && !this.intersectionsPending) || !this.visibleCanvasRect) return true;
    const video = this.intersections.get(this.video);
    const canvas = this.intersections.get(this.canvas);
    const stale = (entry: IntersectionObserverEntry, rect: Rect): boolean => {
      const bounds = entry.boundingClientRect;
      return [bounds.left - rect.left, bounds.top - rect.top,
        bounds.right - rect.left - rect.width, bounds.bottom - rect.top - rect.height]
        .some((error) => !Number.isFinite(error) || Math.abs(error) > 1);
    };
    // Called once per sync: each stale target gets one new initial observation, not a polling loop.
    for (const [target, entry, rect] of [[this.video, video, videoRect], [this.canvas, canvas, canvasRect]] as const) {
      if (entry && stale(entry, rect)) {
        this.intersectionsPending = true;
        this.intersections.delete(target);
        this.intersectionObserver?.unobserve(target);
        this.intersectionObserver?.observe(target);
      }
    }
    if (!this.intersections.has(this.video) || !this.intersections.has(this.canvas)) return this.intersectionsPending ? null : true;
    this.intersectionsPending = false;
    if (!video || !canvas) return true;
    const visible = this.visibleCanvasRect;
    const expectedLeft = Math.max(video.intersectionRect.left, videoRect.left + visible.left);
    const expectedTop = Math.max(video.intersectionRect.top, videoRect.top + visible.top);
    const expectedRight = Math.min(video.intersectionRect.right, videoRect.left + visible.left + visible.width);
    const expectedBottom = Math.min(video.intersectionRect.bottom, videoRect.top + visible.top + visible.height);
    const expectedArea = Math.max(0, expectedRight - expectedLeft) * Math.max(0, expectedBottom - expectedTop);
    const actual = canvas.intersectionRect;
    const hasArea = actual.width > 0 && actual.height > 0;
    const shortfall = expectedArea >= 1 && (!hasArea || actual.left > expectedLeft + 1 ||
      actual.top > expectedTop + 1 || actual.right < expectedRight - 1 || actual.bottom < expectedBottom - 1);
    const excess = hasArea && (actual.left < expectedLeft - 1 || actual.top < expectedTop - 1 ||
      actual.right > expectedRight + 1 || actual.bottom > expectedBottom + 1);
    const supported = !shortfall && !excess;
    this.intersectionsPending = !supported;
    return supported;
  }

  /** Samples paint order at the centre and quarter points of the video's content rect. */
  private videoStackingAtPoints(): Array<{ index: number; above: Element | null } | null> {
    return this.picture.readSiteStyle(() => {
      const style = this.env.getComputedStyle(this.video);
      const rect = this.video.getBoundingClientRect();
      const left = rect.left + px(style.borderLeftWidth) + px(style.paddingLeft);
      const top = rect.top + px(style.borderTopWidth) + px(style.paddingTop);
      const width = rect.width - px(style.borderLeftWidth) - px(style.borderRightWidth) - px(style.paddingLeft) - px(style.paddingRight);
      const height = rect.height - px(style.borderTopWidth) - px(style.borderBottomWidth) - px(style.paddingTop) - px(style.paddingBottom);
      const fit = parseObjectFit(style.objectFit);
      const position = parseObjectPosition(style.objectPosition);
      const geometry = fit && position ? computeLetterboxGeometry(this.video.videoWidth, this.video.videoHeight,
        width, height, fit, { objectPosition: position }) : null;
      if (!geometry) return [];
      const view = this.env.document.defaultView;
      return [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]].map(([x, y]) => {
        const vx = left + geometry.contentX + geometry.contentWidth * x!;
        const vy = top + geometry.contentY + geometry.contentHeight * y!;
        if (vx < 0 || vy < 0 || vx >= (view?.innerWidth ?? 0) || vy >= (view?.innerHeight ?? 0)) return null;
        const elements = this.env.elementsFromPoint(vx, vy).filter((element) => element !== this.canvas);
        const index = elements.indexOf(this.video);
        return { index, above: index > 0 ? elements[index - 1]! : null };
      });
    });
  }

  /** Reads offset semantics with our owned position patch synchronously suspended. */
  private videoPositionOffsetsSupported(): boolean {
    const style = this.video.style;
    const saved = this.videoPositionPatched;
    const owned = saved && style.getPropertyValue('position') === 'relative' &&
      style.getPropertyPriority('position') === 'important';
    if (owned) {
      this.withVideoTransitionsSuppressed(() => {
        if (saved.previous === '') style.removeProperty('position');
        else style.setProperty('position', saved.previous, saved.priority);
      });
    }
    try {
      const computed = this.env.getComputedStyle(this.video);
      return [computed.top, computed.right, computed.bottom, computed.left, computed.zIndex]
        .every((value) => !value || value === 'auto');
    } finally {
      if (owned) this.withVideoTransitionsSuppressed(() => style.setProperty('position', 'relative', 'important'));
    }
  }

  /** Re-validates the minimal video patch, adopting page position rewrites for restore. */
  private ensureVideoPosition(): boolean {
    const computed = this.env.getComputedStyle(this.video);
    if (this.videoPositionPatched && !this.videoPositionOffsetsSupported()) {
      this.unpatchVideo();
      return false;
    }
    if (computed.position === 'static') {
      if (!positionPatchSupported(computed)) {
        this.unpatchVideo();
        return false;
      }
      const style = this.video.style;
      const previous = style.getPropertyValue('position');
      const priority = style.getPropertyPriority('position');
      const owned = previous === 'relative' && priority === 'important';
      const saved = owned && this.videoPositionPatched ? this.videoPositionPatched : { previous, priority };
      const snapshotBefore = captureLayoutSnapshot(this.video, this.env);
      const before = this.video.getBoundingClientRect();
      this.videoStacking = this.videoStackingAtPoints();
      this.withVideoTransitionsSuppressed(() => style.setProperty('position', 'relative', 'important'));
      this.videoPositionPatched = saved;
      const snapshotAfter = captureLayoutSnapshot(this.video, this.env);
      const ignore = new Set(['position']);
      const beforeProperties = snapshotBefore.elements.get(this.video)!.properties;
      const afterProperties = snapshotAfter.elements.get(this.video)!.properties;
      // Offsets are validated with the patch suspended, so ignore their derived auto-to-zero
      // resolution on this video; shorthands may enumerate several equivalent components.
      for (const name of ['top', 'right', 'bottom', 'left', 'inset-block-start', 'inset-block-end',
        'inset-inline-start', 'inset-inline-end', 'inset', 'inset-block', 'inset-inline']) {
        if (beforeProperties.get(name)?.split(/\s+/).every((value) => value === 'auto') &&
          afterProperties.get(name)?.split(/\s+/).every((value) => value === '0px')) ignore.add(name);
      }
      if (compareLayoutSnapshots(snapshotBefore, snapshotAfter, ignore)) {
        this.rejectInsertion();
        return false;
      }
      const after = this.video.getBoundingClientRect();
      const errors = [after.left - before.left, after.top - before.top,
        after.left + after.width - before.left - before.width, after.top + after.height - before.top - before.height];
      const patched = this.env.getComputedStyle(this.video);
      if (patched.position === 'static' || !this.videoPositionOffsetsSupported() ||
        errors.some((error) => !Number.isFinite(error) || Math.abs(error) > 0.5)) {
        this.unpatchVideo();
        return false;
      }
    }
    if (this.videoPositionPatched && this.videoStacking) {
      const current = this.videoStackingAtPoints();
      if (current.some((point, i) => {
        const before = this.videoStacking![i];
        return point && before && (point.index !== before.index || point.above !== before.above);
      })) {
        this.unpatchVideo();
        return false;
      }
    }
    return true;
  }

  private withVideoTransitionsSuppressed<T>(fn: () => T): T {
    this.recordVideoStyleAttribute();
    return withTransitionsSuppressed(this.video, fn, () => { void this.env.getComputedStyle(this.video).objectPosition; });
  }

  private recordVideoStyleAttribute(): void {
    if (this.videoHadStyleAttribute === null) this.videoHadStyleAttribute = this.video.hasAttribute('style');
  }

  private removeEmptyVideoStyleAttribute(): void {
    if (this.videoHadStyleAttribute === false && this.video.getAttribute('style')?.trim() === '') {
      this.video.removeAttribute('style');
    }
  }

  private unpatchVideo(): void {
    if (this.videoPositionPatched && this.video.style.getPropertyValue('position') === 'relative' && this.video.style.getPropertyPriority('position') === 'important') {
      const { previous, priority } = this.videoPositionPatched;
      this.withVideoTransitionsSuppressed(() => {
        if (previous === '') this.video.style.removeProperty('position');
        else this.video.style.setProperty('position', previous, priority);
      });
    }
    this.videoPositionPatched = null;
    this.videoStacking = null;
    this.removeEmptyVideoStyleAttribute();
  }

  private applyStaticStyle(): void {
    const s = this.canvas.style;
    const set = (name: string, value: string): void => s.setProperty(name, value, 'important');
    set('all', 'initial');
    set('position', 'absolute');
    set('pointer-events', 'none');
    set('display', 'block');
    set('margin', '0');
    set('padding', '0');
    set('border', 'none');
    set('background', 'transparent');
    set('transform', 'none');
    set('right', 'auto');
    set('bottom', 'auto');
    set('min-width', '0');
    set('min-height', '0');
    set('max-width', 'none');
    set('max-height', 'none');
    set('box-sizing', 'content-box');
    set('clip-path', 'none');
    set('mask', 'none');
    set('-webkit-mask', 'none');
    set('mask-image', 'none');
    set('-webkit-mask-image', 'none');
    set('mix-blend-mode', 'normal');
    set('backdrop-filter', 'none');
    set('isolation', 'auto');
    set('outline', 'none');
    set('box-shadow', 'none');
    set('object-fit', 'fill');
    set('object-position', '50% 50%');
    set('zoom', '1');
    set('content-visibility', 'visible');
  }

  private setCanvasVisible(visible: boolean): void {
    this.canvas.style.setProperty('visibility', visible ? 'visible' : 'hidden', 'important');
  }

  /** (Re)starts the window after which a fullscreen transition counts as settled. */
  private armTransitionTimer(): void {
    if (this.transitionTimer !== null) this.env.clearTimeout(this.transitionTimer);
    this.transitionTimer = this.env.setTimeout(() => {
      this.transitionTimer = null;
      this.fullscreenTransition = false;
      this.sync();
    }, this.debounceMs);
  }

  private applyBackingStore(size: Size): void {
    if (size.width < 1 || size.height < 1) return;
    if (this.canvas.width === size.width && this.canvas.height === size.height) return;
    if (this.fullscreenTransition) {
      // Resizing the backing store recreates the WebGPU swapchain; while the fullscreen zoom
      // animation fires ResizeObserver every frame, apply the size only once resizing settles.
      this.armTransitionTimer();
      return;
    }
    this.restoreVideo();
    this.canvas.width = size.width;
    this.canvas.height = size.height;
  }
}
