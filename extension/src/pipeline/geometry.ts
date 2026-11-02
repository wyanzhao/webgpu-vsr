/**
 * Frame geometry for the folded low/medium RTX VSR network
 * (docs/IMPLEMENTATION_NOTES.md section 3.1, 3.11 and 3.12).
 */

/** Low/medium models align the source to 32 pixels (RUNTIME-CONTRACT.md, alignment A32). */
export const MODEL_ALIGNMENT = 32;
/** Number of encoder resolutions (levels 0..4). */
export const LEVEL_COUNT = 5;
/** Presentation ceiling: long side <= 1920, short side <= 1080. */
export const CEILING_LONG = 1920;
export const CEILING_SHORT = 1080;

export interface Size {
  width: number;
  height: number;
}

function requirePositiveInt(name: string, v: number): void {
  if (!Number.isInteger(v) || v <= 0) throw new RangeError(`${name} must be a positive integer, got ${v}`);
}

/** `A*ceil(x/A)` with A = 32. */
export function alignUp(x: number, alignment = MODEL_ALIGNMENT): number {
  requirePositiveInt('x', x);
  return alignment * Math.ceil(x / alignment);
}

/** Aligned inference size TW x TH (area beyond the source is edge-replicated). */
export function alignedSize(width: number, height: number): Size {
  return { width: alignUp(width), height: alignUp(height) };
}

/**
 * Feature-map size of every level: level 0 is the folded grid (TW/2 x TH/2),
 * level l is level 0 / 2^l. Because TW, TH are multiples of 32, levels 0..3 have even
 * sizes (the 2x2 pool never straddles an edge) and level 4 is an integer.
 */
export function levelSizes(width: number, height: number): Size[] {
  const a = alignedSize(width, height);
  const out: Size[] = [];
  for (let l = 0; l < LEVEL_COUNT; l++) {
    out.push({ width: a.width / 2 / (1 << l), height: a.height / 2 / (1 << l) });
  }
  return out;
}

/** The network's internal output image: 2W x 2H (crop of 2TW x 2TH). */
export function internalSize(width: number, height: number): Size {
  requirePositiveInt('width', width);
  requirePositiveInt('height', height);
  return { width: 2 * width, height: 2 * height };
}

/**
 * Clamp a display rectangle (device pixels) to the 1920x1080 presentation ceiling while
 * preserving its aspect ratio. Sizes below the ceiling are only rounded.
 */
export function clampToCeiling(displayWidth: number, displayHeight: number): Size {
  if (!(displayWidth > 0) || !(displayHeight > 0)) {
    throw new RangeError(`display size must be positive, got ${displayWidth}x${displayHeight}`);
  }
  const long = Math.max(displayWidth, displayHeight);
  const short = Math.min(displayWidth, displayHeight);
  const s = Math.min(1, CEILING_LONG / long, CEILING_SHORT / short);
  const landscape = displayWidth >= displayHeight;
  const maxW = landscape ? CEILING_LONG : CEILING_SHORT;
  const maxH = landscape ? CEILING_SHORT : CEILING_LONG;
  return {
    width: Math.max(1, Math.min(maxW, Math.round(displayWidth * s))),
    height: Math.max(1, Math.min(maxH, Math.round(displayHeight * s))),
  };
}

export type OutputMode = 'direct' | 'resample';

export interface OutputDecision extends Size {
  /** 'direct': the SR head writes the canvas; 'resample': Catmull-Rom pass writes the canvas. */
  mode: OutputMode;
}

/**
 * Output-size decision (notes 3.12). For the VSR models: if the clamped target covers the
 * internal 2W x 2H image in both axes, present the internal image directly (the compositor
 * scales the rest); otherwise resample internal -> target. Quality 0 always resamples the
 * source to the target.
 */
export function decideOutput(
  inWidth: number,
  inHeight: number,
  displayWidth: number,
  displayHeight: number,
  quality0: boolean,
): OutputDecision {
  const target = clampToCeiling(displayWidth, displayHeight);
  if (quality0) return { ...target, mode: 'resample' };
  const internal = internalSize(inWidth, inHeight);
  if (target.width >= internal.width && target.height >= internal.height) {
    return { ...internal, mode: 'direct' };
  }
  return { ...target, mode: 'resample' };
}

/** True when the VSR pipeline for this in/out pair writes the canvas from the head directly. */
export function isDirectOutput(inWidth: number, inHeight: number, outWidth: number, outHeight: number): boolean {
  return outWidth === 2 * inWidth && outHeight === 2 * inHeight;
}

/**
 * Resample scale factors `float(srcW)/float(outW)` (RUNTIME-CONTRACT.md: integer
 * numerators are formed first, then the ratio is evaluated as a float32 division).
 */
export function resampleScale(srcWidth: number, srcHeight: number, outWidth: number, outHeight: number): { x: number; y: number } {
  requirePositiveInt('srcWidth', srcWidth);
  requirePositiveInt('srcHeight', srcHeight);
  requirePositiveInt('outWidth', outWidth);
  requirePositiveInt('outHeight', outHeight);
  return {
    x: Math.fround(Math.fround(srcWidth) / Math.fround(outWidth)),
    y: Math.fround(Math.fround(srcHeight) / Math.fround(outHeight)),
  };
}
