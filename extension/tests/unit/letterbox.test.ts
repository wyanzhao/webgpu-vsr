import { describe, expect, it } from 'vitest';
import {
  computeCanvasResolution,
  computeLetterboxGeometry,
  computeTargetResolution,
  parseObjectFit,
  parseObjectPosition,
  type LetterboxGeometry,
} from '../../src/content/overlay';

/** Asserts the content rect (x, y, w, h) with tight float tolerance. */
function expectRect(g: LetterboxGeometry | null, x: number, y: number, w: number, h: number): void {
  expect(g).not.toBeNull();
  const geo = g as LetterboxGeometry;
  expect(geo.contentX).toBeCloseTo(x, 9);
  expect(geo.contentY).toBeCloseTo(y, 9);
  expect(geo.contentWidth).toBeCloseTo(w, 9);
  expect(geo.contentHeight).toBeCloseTo(h, 9);
}

describe('object-fit: contain', () => {
  it('16:9 video in a 4:3 box is letterboxed (bars top and bottom)', () => {
    // 1920x1080 in 800x600: width limits -> 800 x (800*1080/1920 = 450); y = (600-450)/2 = 75
    const g = computeLetterboxGeometry(1920, 1080, 800, 600, 'contain');
    expectRect(g, 0, 75, 800, 450);
    expect(g?.mode).toBe('letterbox');
    expect(g?.scaleX).toBeCloseTo(800 / 1920, 12);
    expect(g?.scaleY).toBeCloseTo(450 / 1080, 12);
    expect([g?.clipTop, g?.clipRight, g?.clipBottom, g?.clipLeft]).toEqual([0, 0, 0, 0]);
  });

  it('4:3 video in a 16:9 box is pillarboxed (bars left and right)', () => {
    // 640x480 in 1280x720: height limits -> (720*640/480 = 960) x 720; x = (1280-960)/2 = 160
    const g = computeLetterboxGeometry(640, 480, 1280, 720);
    expectRect(g, 160, 0, 960, 720);
    expect(g?.mode).toBe('pillarbox');
    expect(g?.visibleWidth).toBeCloseTo(960, 9);
  });

  it('21:9 video in a 16:9 box is letterboxed', () => {
    // 2560x1080 in 1280x720: 1280 x (1280*1080/2560 = 540); y = (720-540)/2 = 90
    const g = computeLetterboxGeometry(2560, 1080, 1280, 720);
    expectRect(g, 0, 90, 1280, 540);
    expect(g?.mode).toBe('letterbox');
  });

  it('matching aspect ratio fills the box exactly', () => {
    const g = computeLetterboxGeometry(1280, 720, 640, 360);
    expectRect(g, 0, 0, 640, 360);
    expect(g?.mode).toBe('exact');
    expect(g?.scaleX).toBe(0.5);
  });

  it('keeps sub-pixel letterbox offsets (no integer rounding)', () => {
    // 1920x1080 in 333x250: h = 333*1080/1920 = 187.3125, y = (250-187.3125)/2 = 31.34375
    const g = computeLetterboxGeometry(1920, 1080, 333, 250);
    expect(g?.contentWidth).toBe(333);
    expect(g?.contentHeight).toBe(187.3125);
    expect(g?.contentY).toBe(31.34375);
    expect(g?.contentX).toBe(0);
  });

  it('keeps sub-pixel pillarbox offsets on fractional boxes', () => {
    // 640x480 in 301.5x200: w = 200*640/480 = 266.666..., x = (301.5 - 266.666...)/2 = 17.41666...
    const g = computeLetterboxGeometry(640, 480, 301.5, 200);
    expectRect(g, (301.5 - 800 / 3) / 2, 0, 800 / 3, 200);
    expect(g?.contentHeight).toBe(200); // limiting axis is exact, not derived
  });

  it('honours object-position on the free axis', () => {
    // pillarbox 960x720 in 1280x720 with object-position 0% 0% -> x = 0; with 100% -> x = 320
    expectRect(computeLetterboxGeometry(640, 480, 1280, 720, 'contain', { objectPosition: parseObjectPosition('0% 0%')! }), 0, 0, 960, 720);
    expectRect(computeLetterboxGeometry(640, 480, 1280, 720, 'contain', { objectPosition: parseObjectPosition('right top')! }), 320, 0, 960, 720);
    expectRect(computeLetterboxGeometry(640, 480, 1280, 720, 'contain', { objectPosition: parseObjectPosition('10px 0px')! }), 10, 0, 960, 720);
  });
});

describe('object-fit: cover / fill / none / scale-down', () => {
  it('cover crops and reports clip insets', () => {
    // 1920x1080 in 800x600: height limits -> (600*1920/1080 = 1066.666...) x 600; x = (800 - 1066.666...)/2 = -133.333...
    const g = computeLetterboxGeometry(1920, 1080, 800, 600, 'cover');
    const w = (600 * 1920) / 1080;
    expectRect(g, (800 - w) / 2, 0, w, 600);
    expect(g?.mode).toBe('crop');
    expect(g?.clipLeft).toBeCloseTo((w - 800) / 2, 9);
    expect(g?.clipRight).toBeCloseTo((w - 800) / 2, 9);
    expect(g?.clipTop).toBe(0);
    expect(g?.visibleX).toBe(0);
    expect(g?.visibleWidth).toBeCloseTo(800, 9);
  });

  it('cover with matching aspect is exact', () => {
    expect(computeLetterboxGeometry(1920, 1080, 1280, 720, 'cover')?.mode).toBe('exact');
  });

  it('fill stretches to the box with independent scales', () => {
    const g = computeLetterboxGeometry(640, 480, 1280, 720, 'fill');
    expectRect(g, 0, 0, 1280, 720);
    expect(g?.mode).toBe('stretch');
    expect(g?.scaleX).toBe(2);
    expect(g?.scaleY).toBe(1.5);
    expect(computeLetterboxGeometry(640, 360, 1280, 720, 'fill')?.mode).toBe('exact');
  });

  it('none draws at intrinsic size, centred (windowbox when smaller)', () => {
    // 640x360 in 1280x720 -> x = 320, y = 180
    const g = computeLetterboxGeometry(640, 360, 1280, 720, 'none');
    expectRect(g, 320, 180, 640, 360);
    expect(g?.mode).toBe('windowbox');
  });

  it('none larger than the box crops on both axes', () => {
    // 1920x1080 in 1280x720 -> x = -320, y = -180, clip 320 / 180 each side
    const g = computeLetterboxGeometry(1920, 1080, 1280, 720, 'none');
    expectRect(g, -320, -180, 1920, 1080);
    expect(g?.mode).toBe('crop');
    expect([g?.clipTop, g?.clipRight, g?.clipBottom, g?.clipLeft]).toEqual([180, 320, 180, 320]);
    expect([g?.visibleX, g?.visibleY, g?.visibleWidth, g?.visibleHeight]).toEqual([0, 0, 1280, 720]);
  });

  it('scale-down behaves like none when the video fits, contain otherwise', () => {
    expectRect(computeLetterboxGeometry(640, 360, 1280, 720, 'scale-down'), 320, 180, 640, 360);
    expectRect(computeLetterboxGeometry(1920, 1080, 800, 600, 'scale-down'), 0, 75, 800, 450);
  });
});

describe('invalid input', () => {
  it.each([
    [0, 1080, 800, 600],
    [1920, 0, 800, 600],
    [1920, 1080, 0, 600],
    [1920, 1080, 800, Number.NaN],
    [-1, 1080, 800, 600],
  ])('returns null for %d x %d in %d x %d', (vw, vh, bw, bh) => {
    expect(computeLetterboxGeometry(vw, vh, bw, bh)).toBeNull();
  });
});

describe('1080p ceiling and backing store', () => {
  it('computeTargetResolution never upscales and caps long 1920 / short 1080', () => {
    expect(computeTargetResolution(1600, 900)).toEqual({ width: 1600, height: 900 });
    // 2560x1440: scale = min(1920/2560, 1080/1440) = 0.75 -> 1920x1080
    expect(computeTargetResolution(2560, 1440)).toEqual({ width: 1920, height: 1080 });
    // 3840x1600 (21:9 at 4K width): long side limits, 0.5 -> 1920x800
    expect(computeTargetResolution(3840, 1600)).toEqual({ width: 1920, height: 800 });
    // portrait 1440x2560 -> 1080x1920
    expect(computeTargetResolution(1440, 2560)).toEqual({ width: 1080, height: 1920 });
    // 4:3 at 2880x2160: short side limits, 0.5 -> 1440x1080
    expect(computeTargetResolution(2880, 2160)).toEqual({ width: 1440, height: 1080 });
    expect(computeTargetResolution(0, 100)).toEqual({ width: 0, height: 0 });
  });

  it('geometry backing size uses the content rect times DPR under the ceiling', () => {
    // 1920x1080 contain in 800x600 @2x: content 800x450 -> 1600x900 device px
    const g = computeLetterboxGeometry(1920, 1080, 800, 600, 'contain', { devicePixelRatio: 2 });
    expect([g?.backingWidth, g?.backingHeight]).toEqual([1600, 900]);
    // 1280x720 box @2x = 2560x1440 -> capped to 1920x1080
    const h = computeLetterboxGeometry(960, 540, 1280, 720, 'contain', { devicePixelRatio: 2 });
    expect([h?.backingWidth, h?.backingHeight]).toEqual([1920, 1080]);
    // sub-pixel content height 187.3125 @1x rounds only in the backing store: 333x187
    const s = computeLetterboxGeometry(1920, 1080, 333, 250);
    expect([s?.backingWidth, s?.backingHeight]).toEqual([333, 187]);
  });

  it('computeCanvasResolution presents the internal 2x image when the target covers it', () => {
    expect(computeCanvasResolution({ width: 1920, height: 1080 }, 960, 540, 2)).toEqual({ width: 1920, height: 1080, presentsInternal: true });
    expect(computeCanvasResolution({ width: 1920, height: 1080 }, 640, 360, 2)).toEqual({ width: 1280, height: 720, presentsInternal: true });
    // target smaller than 2W x 2H -> resample to the target
    expect(computeCanvasResolution({ width: 1600, height: 900 }, 960, 540, 2)).toEqual({ width: 1600, height: 900, presentsInternal: false });
    // Quality 0 (no internal image) always renders at the target
    expect(computeCanvasResolution({ width: 1920, height: 1080 }, 640, 360, null)).toEqual({ width: 1920, height: 1080, presentsInternal: false });
  });
});

describe('computed-style parsing', () => {
  it('parseObjectFit accepts the five keywords', () => {
    expect(parseObjectFit('contain')).toBe('contain');
    expect(parseObjectFit('Scale-Down')).toBe('scale-down');
    expect(parseObjectFit('stretch')).toBeNull();
  });

  it('parseObjectPosition handles keywords, %, px, calc and edge offsets', () => {
    expect(parseObjectPosition('50% 50%')).toEqual({ x: { percent: 50, px: 0 }, y: { percent: 50, px: 0 } });
    expect(parseObjectPosition('left top')).toEqual({ x: { percent: 0, px: 0 }, y: { percent: 0, px: 0 } });
    expect(parseObjectPosition('top right')).toEqual({ x: { percent: 100, px: 0 }, y: { percent: 0, px: 0 } });
    expect(parseObjectPosition('bottom')).toEqual({ x: { percent: 50, px: 0 }, y: { percent: 100, px: 0 } });
    expect(parseObjectPosition('12.5px 0px')).toEqual({ x: { percent: 0, px: 12.5 }, y: { percent: 0, px: 0 } });
    expect(parseObjectPosition('calc(100% - 10px) 25%')).toEqual({ x: { percent: 100, px: -10 }, y: { percent: 25, px: 0 } });
    expect(parseObjectPosition('right 10px bottom 20px')).toEqual({ x: { percent: 100, px: -10 }, y: { percent: 100, px: -20 } });
    expect(parseObjectPosition('top 5% left 3px')).toEqual({ x: { percent: 0, px: 3 }, y: { percent: 5, px: 0 } });
    expect(parseObjectPosition('1em 2vw')).toBeNull();
    expect(parseObjectPosition('left center top')).toBeNull();
  });

  it('a right-edge offset positions the content as CSS does', () => {
    // pillarbox 960 wide in 1280: free space 320; "right 10px" -> 100% of 320 - 10 = 310
    const g = computeLetterboxGeometry(640, 480, 1280, 720, 'contain', { objectPosition: parseObjectPosition('right 10px top 0px')! });
    expectRect(g, 310, 0, 960, 720);
  });
});
