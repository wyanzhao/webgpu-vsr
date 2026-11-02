import { describe, expect, it } from 'vitest';
import {
  alignUp,
  alignedSize,
  clampToCeiling,
  decideOutput,
  internalSize,
  isDirectOutput,
  levelSizes,
  resampleScale,
} from '../../src/pipeline/geometry';

describe('alignment and levels', () => {
  it('aligns to 32 (low/medium A32)', () => {
    expect(alignUp(1)).toBe(32);
    expect(alignUp(32)).toBe(32);
    expect(alignUp(33)).toBe(64);
    expect(alignedSize(960, 540)).toEqual({ width: 960, height: 544 });
    expect(alignedSize(1280, 720)).toEqual({ width: 1280, height: 736 });
    expect(alignedSize(100, 60)).toEqual({ width: 128, height: 64 });
    expect(() => alignUp(0)).toThrow(RangeError);
    expect(() => alignUp(10.5)).toThrow(RangeError);
  });

  it('level 0 is the folded grid and each level halves', () => {
    expect(levelSizes(960, 540)).toEqual([
      { width: 480, height: 272 },
      { width: 240, height: 136 },
      { width: 120, height: 68 },
      { width: 60, height: 34 },
      { width: 30, height: 17 },
    ]);
    for (const [w, h] of [
      [1, 1],
      [37, 19],
      [1919, 1081],
    ]) {
      const L = levelSizes(w, h);
      for (let l = 0; l < 4; l++) {
        expect(L[l].width % 2).toBe(0);
        expect(L[l].height % 2).toBe(0);
        expect(L[l + 1].width * 2).toBe(L[l].width);
      }
      expect(L[0].width * 2).toBe(alignUp(w));
    }
  });

  it('internal image is 2W x 2H (crop of 2TW x 2TH)', () => {
    expect(internalSize(100, 60)).toEqual({ width: 200, height: 120 });
  });
});

describe('presentation ceiling and output decision', () => {
  it('clamps to 1920x1080 preserving aspect', () => {
    expect(clampToCeiling(3840, 2160)).toEqual({ width: 1920, height: 1080 });
    expect(clampToCeiling(2560, 1440)).toEqual({ width: 1920, height: 1080 });
    expect(clampToCeiling(3000, 1000)).toEqual({ width: 1920, height: 640 });
    expect(clampToCeiling(1080, 1920)).toEqual({ width: 1080, height: 1920 });
    expect(clampToCeiling(1200, 2400)).toEqual({ width: 960, height: 1920 });
    expect(clampToCeiling(1280.4, 719.6)).toEqual({ width: 1280, height: 720 });
    expect(() => clampToCeiling(0, 10)).toThrow(RangeError);
  });

  it('presents the internal image directly when the target covers 2W x 2H', () => {
    expect(decideOutput(960, 540, 1920, 1080, false)).toEqual({ width: 1920, height: 1080, mode: 'direct' });
    // target larger than the internal image: compositor scales the 2x image
    expect(decideOutput(640, 360, 1920, 1080, false)).toEqual({ width: 1280, height: 720, mode: 'direct' });
  });

  it('resamples when the internal image exceeds the clamped target', () => {
    expect(decideOutput(1280, 720, 1920, 1080, false)).toEqual({ width: 1920, height: 1080, mode: 'resample' });
    expect(decideOutput(960, 540, 1600, 900, false)).toEqual({ width: 1600, height: 900, mode: 'resample' });
    // one axis short is enough to require resampling
    expect(decideOutput(960, 540, 1920, 1000, false).mode).toBe('resample');
  });

  it('quality 0 always resamples the source to the target', () => {
    expect(decideOutput(960, 540, 1920, 1080, true)).toEqual({ width: 1920, height: 1080, mode: 'resample' });
    expect(decideOutput(1920, 1080, 3840, 2160, true)).toEqual({ width: 1920, height: 1080, mode: 'resample' });
  });

  it('identifies the direct path from in/out sizes', () => {
    expect(isDirectOutput(960, 540, 1920, 1080)).toBe(true);
    expect(isDirectOutput(1280, 720, 1920, 1080)).toBe(false);
  });
});

describe('resample scale', () => {
  it('is float32(src)/float32(out)', () => {
    const s = resampleScale(2560, 1440, 1920, 1080);
    expect(s.x).toBe(Math.fround(2560 / 1920));
    expect(s.y).toBe(Math.fround(1440 / 1080));
    expect(s.x).not.toBe(2560 / 1920); // rounded to float32
    expect(resampleScale(960, 540, 1920, 1080)).toEqual({ x: 0.5, y: 0.5 });
    expect(() => resampleScale(0, 1, 1, 1)).toThrow(RangeError);
  });
});
