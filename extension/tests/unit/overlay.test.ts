import { beforeEach, describe, expect, it } from 'vitest';
import {
  CanvasOverlay,
  HIDDEN_OBJECT_POSITION,
  OVERLAY_CANVAS_CLASS,
  VideoPictureGuard,
  canvasZIndexFor,
  computeCanvasPlacement,
  computeLetterboxGeometry,
  cssPx,
  type LetterboxGeometry,
  type OverlayLayout,
} from '../../src/content/overlay';
import { FakeEnv, FakeNode, FakeStyle } from './fake_dom';

// ---------------------------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------------------------

describe('VideoPictureGuard', () => {
  const properties = ['object-position', 'background-color', 'background-image'];
  const hiddenValues = [HIDDEN_OBJECT_POSITION, 'transparent', 'none'];
  const values = (style: FakeStyle) => properties.map((name) => [style.getPropertyValue(name), style.getPropertyPriority(name)]);

  it.each([['', ''], ['all', ''], ['object-position, background-color', 'important']])(
    'suppresses transitions for every picture write and restores %s with priority %s', (value, priority) => {
      const style = new FakeStyle();
      if (value) style.setProperty('transition-property', value, priority);
      const guard = new VideoPictureGuard(style);
      guard.hide();
      guard.readSiteStyle(() => undefined);
      guard.restore();
      guard.hide();
      style.setProperty('object-position', 'left top');
      style.managedWrites.pop(); // The page's own write is not managed by the guard.
      guard.reassert();
      guard.restore();
      expect(style.managedWrites.length).toBeGreaterThan(12);
      expect(style.managedWrites.every((write) => write.transition === 'none' && write.priority === 'important')).toBe(true);
      expect([style.getPropertyValue('transition-property'), style.getPropertyPriority('transition-property')]).toEqual([value, priority]);
    },
  );

  it('restores transition ownership after a page rewrite during the style flush', () => {
    const style = new FakeStyle();
    style.setProperty('transition-property', 'all');
    const guard = new VideoPictureGuard(style, () => style.setProperty('transition-property', 'opacity', 'important'));
    guard.hide();
    guard.restore();
    expect([style.getPropertyValue('transition-property'), style.getPropertyPriority('transition-property')]).toEqual(['opacity', 'important']);
  });

  it('saves and restores all inline values and priorities', () => {
    const style = new FakeStyle();
    style.setProperty('object-position', '40% 60%', 'important');
    style.setProperty('background-color', '#123456');
    style.setProperty('background-image', 'url("poster.png")', 'important');
    const original = values(style);
    const guard = new VideoPictureGuard(style);
    guard.hide();
    expect(values(style)).toEqual(hiddenValues.map((v) => [v, 'important']));
    expect(guard.hidden).toBe(true);
    guard.restore();
    expect(values(style)).toEqual(original);
    expect(guard.hidden).toBe(false);
  });

  it('removes all managed properties when there were none', () => {
    const style = new FakeStyle();
    const guard = new VideoPictureGuard(style);
    guard.hide();
    guard.restore();
    expect(properties.every((name) => !style.props.has(name))).toBe(true);
  });

  it('repeated hide() does not overwrite saved originals or rewrite the style', () => {
    const style = new FakeStyle();
    style.setProperty('object-position', '40% 60%');
    const guard = new VideoPictureGuard(style);
    guard.hide();
    const writes = style.writes;
    guard.hide();
    guard.hide();
    expect(style.writes).toBe(writes);
    guard.restore();
    expect(style.getPropertyValue('object-position')).toBe('40% 60%');
  });

  it('restore preserves page writes made since the last reassert', () => {
    const style = new FakeStyle();
    const guard = new VideoPictureGuard(style);
    guard.hide();
    style.setProperty('object-position', 'left top');
    style.setProperty('background-color', 'transparent');
    style.removeProperty('background-image');
    guard.restore();
    expect(values(style)).toEqual([['left top', ''], ['transparent', ''], ['', '']]);
  });

  it('reassert() adopts page rewrites of values, priorities and absent properties', () => {
    const style = new FakeStyle();
    const guard = new VideoPictureGuard(style);
    guard.hide();
    style.setProperty('object-position', '20% 80%', 'important');
    style.setProperty('background-color', 'transparent'); // same value, changed priority
    style.removeProperty('background-image');
    guard.reassert();
    expect(values(style)).toEqual(hiddenValues.map((v) => [v, 'important']));
    guard.restore();
    expect(values(style)).toEqual([['20% 80%', 'important'], ['transparent', ''], ['', '']]);
    expect(style.props.has('background-image')).toBe(false);
  });

  it('readSiteStyle exposes the page values and re-applies the overrides afterwards', () => {
    const style = new FakeStyle();
    style.setProperty('object-position', '40% 60%', 'important');
    style.setProperty('background-color', '#123456');
    const original = values(style);
    const guard = new VideoPictureGuard(style);
    guard.hide();
    expect(guard.readSiteStyle(() => values(style))).toEqual(original);
    expect(values(style)).toEqual(hiddenValues.map((v) => [v, 'important']));
    expect(guard.hidden).toBe(true);
    expect(() => guard.readSiteStyle(() => { throw new Error('read failed'); })).toThrow('read failed');
    expect(values(style)).toEqual(hiddenValues.map((v) => [v, 'important']));
    guard.restore();
    expect(values(style)).toEqual(original);
  });

  it('restore(), reassert() and readSiteStyle are no-ops when not hidden', () => {
    const style = new FakeStyle();
    style.setProperty('object-position', '40% 60%');
    const writes = style.writes;
    const guard = new VideoPictureGuard(style);
    guard.restore();
    guard.reassert();
    expect(guard.readSiteStyle(() => style.getPropertyValue('object-position'))).toBe('40% 60%');
    expect(style.writes).toBe(writes);
  });
});

describe('placement helpers', () => {
  it('canvasZIndexFor: n -> n, auto stays auto', () => {
    expect(canvasZIndexFor('5')).toBe('5');
    expect(canvasZIndexFor('-1')).toBe('-1');
    expect(canvasZIndexFor('auto')).toBe('auto');
    expect(canvasZIndexFor('')).toBe('auto');
  });

  it('computeCanvasPlacement maps the content rect relative to the measured origin', () => {
    // video border box at (110, 70) 806x456 with 2px border + 1px padding -> content box 800x450
    // measured origin includes a containing block at (100, 50), 4px border and 30px scroll
    const geometry = computeLetterboxGeometry(640, 480, 800, 450) as LetterboxGeometry; // pillarbox: 600x450 at x=100
    const p = computeCanvasPlacement(
      {
        videoRect: { left: 110, top: 70, width: 806, height: 456 },
        origin: { left: 104, top: 24 },
        videoBorder: { top: 2, right: 2, bottom: 2, left: 2 },
        videoPadding: { top: 1, right: 1, bottom: 1, left: 1 },
      },
      geometry,
    );
    // content box origin: x = 110-100-4+0+2+1 = 9, y = 70-50-4+30+2+1 = 49
    expect(p.videoBox).toEqual({ left: 9, top: 49, width: 800, height: 450 });
    expect([p.left, p.top, p.width, p.height]).toEqual([109, 49, 600, 450]);
  });

  it('cssPx keeps 4 decimals and normalises -0', () => {
    expect(cssPx(31.34375)).toBe('31.3438px');
    expect(cssPx(17.416666666)).toBe('17.4167px');
    expect(cssPx(-0.00001)).toBe('0px');
  });
});

// ---------------------------------------------------------------------------------------------
// CanvasOverlay against the fake DOM
// ---------------------------------------------------------------------------------------------

describe('CanvasOverlay', () => {
  let fx: FakeEnv;
  let body: FakeNode;
  let parent: FakeNode;
  let video: FakeNode;
  let controls: FakeNode;
  let layouts: OverlayLayout[];

  beforeEach(() => {
    fx = new FakeEnv();
    body = new FakeNode();
    body.connectedRoot = true;
    parent = new FakeNode();
    body.appendChild(parent);
    video = new FakeNode();
    controls = new FakeNode();
    parent.appendChild(video);
    parent.appendChild(controls);
    parent.rect = { left: 90, top: 40, width: 1000, height: 700 };
    parent.computed = { borderLeftWidth: '2px', borderTopWidth: '2px' };
    video.rect = { left: 100, top: 50, width: 800, height: 600 };
    video.videoWidth = 1920;
    video.videoHeight = 1080;
    video.style.setProperty('object-position', '50% 50%');
    layouts = [];
  });

  const make = (internalScale: number | null = null): CanvasOverlay =>
    new CanvasOverlay(video as unknown as HTMLVideoElement, { env: fx.env(), internalScale, onLayout: (l) => layouts.push(l) });
  const canvasOf = (o: CanvasOverlay): FakeNode => o.canvas as unknown as FakeNode;

  describe('style attribute ownership', () => {
    it.each(['restore', 'destroy'])('removes an originally absent style attribute after %s', (action) => {
      video.removeAttribute('style');
      // Picture restoration can remove the last declaration without a position patch.
      if (action === 'restore') video.computed.position = 'absolute';
      const o = make();
      o.markFramePresented();
      expect(o.videoHidden).toBe(true);
      expect(video.hasAttribute('style')).toBe(true);
      if (action === 'restore') o.restoreVideo();
      else o.destroy();
      expect(video.hasAttribute('style')).toBe(false);
      o.destroy();
      expect(video.hasAttribute('style')).toBe(false);
    });

    it('removes an originally absent style attribute after a rejected position patch rollback', () => {
      video.removeAttribute('style');
      controls.computedRule = () => ({ visibility: video.style.getPropertyValue('position') === 'relative' ? 'hidden' : 'visible' });
      const o = make();
      expect(o.hasInsertionLayoutFailure).toBe(true);
      expect(video.hasAttribute('style')).toBe(false);
      o.destroy();
      expect(video.hasAttribute('style')).toBe(false);
    });

    it.each(['restore', 'rollback', 'destroy'])('keeps an original empty style attribute after %s', (action) => {
      video.removeAttribute('style');
      video.setAttribute('style', '');
      if (action === 'restore') video.computed.position = 'absolute';
      if (action === 'rollback') video.positionShift.left = 1;
      const o = make();
      if (action !== 'rollback') o.markFramePresented();
      if (action === 'restore') o.restoreVideo();
      else if (action === 'destroy') o.destroy();
      else expect(o.layoutSupported).toBe(false);
      expect(video.getAttribute('style')).toBe('');
      o.destroy();
      expect(video.getAttribute('style')).toBe('');
    });

    it.each(['restore', 'destroy'])('keeps a page declaration written while hidden after %s', (action) => {
      video.removeAttribute('style');
      if (action === 'restore') video.computed.position = 'absolute';
      const o = make();
      o.markFramePresented();
      video.style.setProperty('color', 'red');
      if (action === 'restore') o.restoreVideo();
      else o.destroy();
      expect(video.hasAttribute('style')).toBe(true);
      expect(video.style.getPropertyValue('color')).toBe('red');
      o.destroy();
      expect(video.hasAttribute('style')).toBe(true);
    });
  });

  describe('visibility fail-safe', () => {
    it.each(['left', 'top', 'right', 'bottom', 'empty'])('restores native picture when the canvas intersection is clipped at %s', (edge) => {
      const o = make();
      const canvas = canvasOf(o);
      o.markFramePresented();
      const actual = { ...canvas.getBoundingClientRect() };
      if (edge === 'left') actual.left += actual.width / 2;
      if (edge === 'top') actual.top += actual.height / 2;
      if (edge === 'left' || edge === 'right') actual.width /= 2;
      if (edge === 'top' || edge === 'bottom') actual.height /= 2;
      if (edge === 'empty') actual.width = actual.height = 0;
      // Separate deliveries retain the latest entry for each target.
      fx.fireIntersections([{ target: video, intersectionRect: video.getBoundingClientRect() }]);
      expect(o.videoHidden).toBe(true);
      fx.fireIntersections([{ target: canvas, intersectionRect: actual }]);
      expect(o.videoHidden).toBe(false);
      expect(o.layoutSupported).toBe(false);
      expect(o.layout).toBeNull();
      expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      o.markFramePresented();
      expect(o.videoHidden).toBe(false);
    });

    it('keeps equally viewport-clipped video and canvas supported', () => {
      video.rect.top = -300; // lower half of the video remains in the viewport
      const o = make();
      const canvas = canvasOf(o);
      o.markFramePresented();
      fx.fireIntersections([
        { target: video, intersectionRect: { ...video.rect, top: 0, height: 300 } },
        { target: canvas, intersectionRect: { ...canvas.getBoundingClientRect(), top: 0, height: 225 } },
      ]);
      o.sync();
      expect(o.layoutSupported).toBe(true);
      expect(o.videoHidden).toBe(true);
    });

    it.each(['video', 'canvas'])('restores native picture for stale %s bounds until fresh failing entries reject the layout', (target) => {
      const o = make();
      const canvas = canvasOf(o);
      const oldVideoRect = video.getBoundingClientRect();
      const oldCanvasRect = canvas.getBoundingClientRect();
      video.rect.left += 10;
      o.sync();
      o.markFramePresented();
      const entries = [
        { target: video, boundingClientRect: target === 'video' ? oldVideoRect : video.getBoundingClientRect(),
          intersectionRect: video.getBoundingClientRect() },
        { target: canvas, boundingClientRect: target === 'canvas' ? oldCanvasRect : canvas.getBoundingClientRect(),
          intersectionRect: { ...canvas.getBoundingClientRect(), width: 0, height: 0 } },
      ];
      fx.fireIntersections(entries);
      o.sync();
      expect(o.layoutSupported).toBe(true);
      expect(o.videoHidden).toBe(false);
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      expect(fx.intersectionObservers[0]!.unobserved).toEqual([target === 'video' ? video : canvas]);
      o.markFramePresented();
      expect(o.videoHidden).toBe(false);
      fx.fireIntersections(entries.map((entry) => ({ ...entry, boundingClientRect: entry.target.getBoundingClientRect() })));
      expect(o.layoutSupported).toBe(false);
      expect(o.videoHidden).toBe(false);
      o.sync();
      o.markFramePresented();
      expect(o.layoutSupported).toBe(false);
      expect(o.videoHidden).toBe(false);
    });

    it('reobserves both stale targets once and waits for fresh passing entries for both after a geometry change', () => {
      const o = make();
      const canvas = canvasOf(o);
      const passing = (target: FakeNode) => ({ target, intersectionRect: target.getBoundingClientRect() });
      fx.fireIntersections([passing(video), passing(canvas)]);
      o.markFramePresented();
      expect(o.videoHidden).toBe(true);
      parent.rect.left += 10;
      video.rect.left += 10;
      o.sync();
      expect(o.videoHidden).toBe(false);
      expect(o.layoutSupported).toBe(true);
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      const observer = fx.intersectionObservers[0]!;
      expect(observer.unobserved).toEqual([video, canvas]);
      o.sync();
      o.markFramePresented();
      expect(observer.unobserved).toEqual([video, canvas]);
      expect(o.videoHidden).toBe(false);
      fx.fireIntersections([passing(canvas)]);
      o.markFramePresented();
      expect(o.videoHidden).toBe(false);
      fx.fireIntersections([passing(video)]);
      expect(o.layoutSupported).toBe(true);
      o.markFramePresented();
      expect(o.videoHidden).toBe(true);
      expect(observer.unobserved).toEqual([video, canvas]);
      o.destroy();
    });

    it('refreshes stale cached bounds before hiding the native picture for the first time', () => {
      const o = make();
      const canvas = canvasOf(o);
      const oldRect = video.getBoundingClientRect();
      video.rect.left += 10;
      o.sync();
      fx.fireIntersections([
        { target: video, boundingClientRect: oldRect, intersectionRect: video.getBoundingClientRect() },
        { target: canvas, intersectionRect: canvas.getBoundingClientRect() },
      ]);
      o.markFramePresented();
      expect(o.videoHidden).toBe(false);
      expect(o.layoutSupported).toBe(true);
      expect(fx.intersectionObservers[0]!.unobserved).toEqual([video]);
      fx.fireIntersections([{ target: video, intersectionRect: video.getBoundingClientRect() }]);
      o.markFramePresented();
      expect(o.videoHidden).toBe(true);
      o.destroy();
    });

    it('checks the latest intersections during sync after the canvas is shown', () => {
      const o = make();
      const canvas = canvasOf(o);
      fx.fireIntersections([
        { target: video, intersectionRect: video.getBoundingClientRect() },
        { target: canvas, intersectionRect: { ...canvas.getBoundingClientRect(), width: 400 } },
      ]);
      expect(o.layoutSupported).toBe(true); // native picture is still visible
      o.markFramePresented();
      o.sync();
      expect(o.layoutSupported).toBe(false);
      expect(o.videoHidden).toBe(false);
    });

    it('allows one CSS pixel of intersection edge error and ignores expected areas below one square pixel', () => {
      const o = make();
      const canvas = canvasOf(o);
      o.markFramePresented();
      const rect = canvas.getBoundingClientRect();
      fx.fireIntersections([
        { target: video, intersectionRect: video.getBoundingClientRect() },
        { target: canvas, intersectionRect: { left: rect.left + 1, top: rect.top + 1, width: rect.width - 2, height: rect.height - 2 } },
      ]);
      expect(o.videoHidden).toBe(true);
      fx.fireIntersections([
        { target: video, intersectionRect: { ...rect, width: 0.5, height: 1 } },
        { target: canvas, intersectionRect: { ...rect, width: 0, height: 0 } },
      ]);
      expect(o.layoutSupported).toBe(true);
      expect(o.videoHidden).toBe(true);
      fx.fireIntersections([{ target: video, intersectionRect: { ...rect, width: 1, height: 1 } }]);
      expect(o.layoutSupported).toBe(false);
      expect(o.videoHidden).toBe(false);
    });

    it('restores native picture when checkVisibility is false on the canvas only', () => {
      const o = make();
      const calls: Array<CheckVisibilityOptions | undefined> = [];
      canvasOf(o).checkVisibility = (options) => { calls.push(options); return false; };
      video.checkVisibility = (options) => { calls.push(options); return true; };
      o.markFramePresented();
      o.sync();
      expect(o.layoutSupported).toBe(false);
      expect(o.videoHidden).toBe(false);
      expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
      expect(calls).toEqual(Array(2).fill({ checkOpacity: true, checkVisibilityCSS: true, contentVisibilityAuto: true }));
    });

    it('keeps mirrored invisibility supported when checkVisibility is false on both targets', () => {
      const o = make();
      canvasOf(o).checkVisibility = () => false;
      video.checkVisibility = () => false;
      o.markFramePresented();
      o.sync();
      expect(o.layoutSupported).toBe(true);
      expect(o.videoHidden).toBe(true);
    });

    it('preserves behaviour when IntersectionObserver and checkVisibility are unavailable', () => {
      fx.intersectionObserverAvailable = false;
      const o = make();
      o.markFramePresented();
      o.sync();
      expect(fx.intersectionObservers).toHaveLength(0);
      expect(o.layoutSupported).toBe(true);
      expect(o.videoHidden).toBe(true);
    });

    it('observes both targets and disconnects on unmount, re-parent and destroy', () => {
      const o = make();
      const canvas = canvasOf(o);
      expect(fx.intersectionObservers[0]?.targets).toEqual([video, canvas]);
      canvas.remove();
      fx.fireChildList(parent);
      expect(fx.intersectionObservers[0]?.disconnected).toBe(true);
      o.reattach();
      expect(fx.intersectionObservers[1]?.targets).toEqual([video, canvas]);
      const theater = new FakeNode();
      body.appendChild(theater);
      theater.appendChild(video);
      o.reattach();
      expect(fx.intersectionObservers[1]?.disconnected).toBe(true);
      expect(fx.intersectionObservers[2]?.targets).toEqual([video, canvas]);
      video.remove();
      o.sync();
      expect(fx.intersectionObservers[2]?.disconnected).toBe(true);
      theater.appendChild(video);
      o.reattach();
      o.destroy();
      expect(fx.intersectionObservers).toHaveLength(4);
      expect(fx.intersectionObservers.every((observer) => observer.disconnected)).toBe(true);
    });
  });

  it('preserves the static player and panel containing block while measuring the canvas origin', () => {
    body.computed.position = 'relative'; // <main>
    body.rect = { left: 20.25, top: 10.5, width: 1200, height: 900 };
    body.computed.borderLeftWidth = '3px';
    body.computed.borderTopWidth = '4px';
    body.scrollLeft = 7;
    body.scrollTop = 9;
    parent.attrs.id = 'player';
    controls.attrs.id = 'panel';
    controls.computed.position = 'absolute';
    const before = controls.containingBlock;
    const o = make();
    expect(parent.style.getPropertyValue('position')).toBe('');
    expect(fx.env().getComputedStyle(parent as unknown as Element).position).toBe('static');
    expect(controls.containingBlock).toBe(before);
    expect(controls.containingBlock).toBe(body);
    expect(o.layoutSupported).toBe(true);
    expect(canvasOf(o).style.getPropertyValue('left')).toBe('83.75px');
    expect(canvasOf(o).style.getPropertyValue('top')).toBe('119.5px');
    expect(canvasOf(o).getBoundingClientRect()).toEqual({ left: 100, top: 125, width: 800, height: 450 });
    o.destroy();
  });

  it('keeps the video patch when Chrome resolves relative auto offsets to zero', () => {
    video.style.setProperty('position', 'static', 'important');
    const computed = () => fx.env().getComputedStyle(video as unknown as Element);
    expect([computed().top, computed().right, computed().bottom, computed().left]).toEqual(['auto', 'auto', 'auto', 'auto']);
    const o = make();
    expect(video.style.props.get('position')).toEqual({ value: 'relative', priority: 'important' });
    expect([computed().top, computed().right, computed().bottom, computed().left]).toEqual(['0px', '0px', '0px', '0px']);
    expect(o.layoutSupported).toBe(true);
    o.markFramePresented();
    for (let i = 0; i < 2; i++) {
      o.sync();
      expect(video.style.props.get('position')).toEqual({ value: 'relative', priority: 'important' });
      expect(o.layoutSupported).toBe(true);
      expect(o.videoHidden).toBe(true);
    }
    o.destroy();
    expect(video.style.props.get('position')).toEqual({ value: 'static', priority: 'important' });
  });

  it('rejects a page top offset read with the relative position patch suspended', () => {
    const o = make();
    expect(o.layoutSupported).toBe(true);
    o.markFramePresented();
    video.style.setProperty('top', '5px');
    o.sync();
    const computed = fx.env().getComputedStyle(video as unknown as Element);
    expect(computed.position).toBe('static');
    expect(computed.top).toBe('5px');
    expect(video.style.getPropertyValue('position')).toBe('');
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.destroy();
  });

  it.each([['z-index', '-1'], ['top', '5px']])('removes the video patch after a page %s write', (property, value) => {
    const o = make();
    o.markFramePresented();
    video.style.setProperty(property!, value!);
    o.sync();
    expect(video.style.getPropertyValue('position')).toBe('');
    expect(video.style.getPropertyValue(property!)).toBe(value);
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it.each(['install', 'sync'])('rolls back when a following sibling moves below the patched video on %s', (phase) => {
    let changed = phase === 'install';
    const points: Array<[number, number]> = [];
    fx.elementsFromPoint = (x, y) => {
      points.push([x, y]);
      return changed && video.style.getPropertyValue('position') === 'relative' ?
        [video, controls] : [controls, video];
    };
    const o = make();
    if (phase === 'sync') {
      expect(o.layoutSupported).toBe(true);
      o.markFramePresented();
      changed = true;
      o.sync();
    }
    expect(video.style.getPropertyValue('position')).toBe('');
    expect(o.layoutSupported).toBe(false);
    expect(o.videoHidden).toBe(false);
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    expect(points.slice(0, 5)).toEqual([[500, 350], [300, 237.5], [700, 237.5], [300, 462.5], [700, 462.5]]);
    o.destroy();
  });

  it('excludes the canvas and skips stacking points outside the viewport', () => {
    video.rect.left = -400;
    const calls: Array<[number, number]> = [];
    fx.elementsFromPoint = (x, y) => {
      calls.push([x, y]);
      return [...parent.children.filter((node) => node.isCanvas), controls, video];
    };
    const o = make();
    expect(calls).toHaveLength(6);
    o.sync();
    expect(calls).toHaveLength(9);
    expect(o.layoutSupported).toBe(true);
    o.destroy();
  });

  it.each(['left', 'top', 'right', 'bottom', 'empty'])('rejects a canvas escaping an ancestor overflow clip at %s', (edge) => {
    const o = make();
    o.markFramePresented();
    const canvas = canvasOf(o);
    const expected = { ...canvas.getBoundingClientRect() };
    if (edge === 'left') { expected.left += 5; expected.width -= 5; }
    if (edge === 'top') { expected.top += 5; expected.height -= 5; }
    if (edge === 'right') expected.width -= 5;
    if (edge === 'bottom') expected.height -= 5;
    if (edge === 'empty') { expected.width = 0; expected.height = 0; }
    fx.fireIntersections([
      { target: video, intersectionRect: expected },
      { target: canvas, intersectionRect: canvas.getBoundingClientRect() },
    ]);
    expect(o.layoutSupported).toBe(false);
    expect(o.videoHidden).toBe(false);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    o.destroy();
  });

  it('measures placement through a translateZ(0) ancestor containing block', () => {
    body.computed.transform = 'translateZ(0)';
    body.rect = { left: 24.25, top: 18.5, width: 1200, height: 900 };
    const o = make();
    const canvas = canvasOf(o);
    expect(canvas.containingBlock).toBe(body);
    expect(parent.style.getPropertyValue('position')).toBe('');
    expect(o.layoutSupported).toBe(true);
    expect(canvas.style.getPropertyValue('left')).toBe('75.75px');
    expect(canvas.style.getPropertyValue('top')).toBe('106.5px');
    expect(canvas.getBoundingClientRect()).toEqual({ left: 100, top: 125, width: 800, height: 450 });
    body.rect.left += 10.125;
    body.scrollTop = 12;
    o.sync();
    expect(canvas.getBoundingClientRect()).toEqual({ left: 100, top: 125, width: 800, height: 450 });
    o.destroy();
  });

  it('reapplies a removed video position patch on sync and adopts the page value for restore', () => {
    video.style.setProperty('position', 'static', 'important');
    const o = make();
    o.markFramePresented();
    video.style.removeProperty('position');
    o.sync();
    expect(video.style.props.get('position')).toEqual({ value: 'relative', priority: 'important' });
    expect(o.layoutSupported).toBe(true);
    expect(o.videoHidden).toBe(true);
    o.destroy();
    expect(video.style.getPropertyValue('position')).toBe('');
  });

  it('restores the native picture when a video position patch remains computed static on sync', () => {
    const o = make();
    o.markFramePresented();
    video.style.removeProperty('position');
    video.forceStaticPosition = true;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it.each(['video'] as const)('reapplies a removed %s position patch without losing page ownership', (target) => {
    const node = target === 'video' ? video : parent;
    const o = make();
    node.style.setProperty('position', 'static');
    o.sync();
    expect(o.layoutSupported).toBe(true);
    expect(node.style.props.get('position')).toEqual({ value: 'relative', priority: 'important' });
    o.destroy();
    expect(node.style.props.get('position')).toEqual({ value: 'static', priority: '' });
  });

  it.each(['video'] as const)('rejects static offsets on the %s before patching', (target) => {
    const node = target === 'video' ? video : parent;
    for (const property of ['top', 'right', 'bottom', 'left', 'inset'] as const) {
      node.computed[property] = '0px';
      const o = make();
      expect(o.layoutSupported, property).toBe(false);
      expect(node.style.getPropertyValue('position'), property).toBe('');
      o.markFramePresented();
      expect(o.videoHidden).toBe(false);
      o.destroy();
      delete node.computed[property];
    }
  });

  it.each(['video'] as const)('rolls back a %s position patch that moves the video by more than 0.5px', (target) => {
    const node = target === 'video' ? video : parent;
    node.positionShift.left = 0.51;
    const before = video.getBoundingClientRect();
    const o = make();
    expect(o.layoutSupported).toBe(false);
    expect(node.style.getPropertyValue('position')).toBe('');
    expect(video.getBoundingClientRect()).toEqual(before);
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it('rejects a static negative-z-index video inside a positioned black parent without patching', () => {
    parent.computed.position = 'relative';
    parent.style.setProperty('background-color', 'black');
    video.computed.zIndex = '-1';
    const o = make();
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(video.style.getPropertyValue('position')).toBe('');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.destroy();
  });

  it.each(['video'] as const)('rejects a static %s with a non-auto z-index before patching', (target) => {
    const node = target === 'video' ? video : parent;
    for (const zIndex of ['-1', '0', '7']) {
      node.computed.zIndex = zIndex;
      const o = make();
      expect(o.layoutSupported, zIndex).toBe(false);
      expect(node.style.getPropertyValue('position'), zIndex).toBe('');
      o.markFramePresented();
      expect(o.videoHidden).toBe(false);
      o.destroy();
    }
  });

  it.each([
    ['backdropFilter', 'blur(4px)'],
    ['webkitBackdropFilter', 'blur(4px)'],
    ['position', 'fixed'],
  ] as const)('rejects video %s during sync', (property, value) => {
    const o = make();
    o.markFramePresented();
    if (property === 'position') video.style.setProperty('position', value, 'important');
    else video.computed[property] = value;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.videoHidden).toBe(false);
    expect(o.layout).toBeNull();
    o.destroy();
  });

  it('overrides a canvas stylesheet clip-path and reasserts hardened styles on sync', () => {
    const o = make();
    const canvas = canvasOf(o);
    canvas.computed.clipPath = 'inset(50%)';
    const hardened = {
      'clip-path': 'none', mask: 'none', '-webkit-mask': 'none', 'mask-image': 'none',
      '-webkit-mask-image': 'none', 'mix-blend-mode': 'normal', 'backdrop-filter': 'none',
      isolation: 'auto', outline: 'none', 'box-shadow': 'none', 'object-fit': 'fill',
      'object-position': '50% 50%', zoom: '1', 'content-visibility': 'visible',
    };
    for (const name of Object.keys(hardened)) canvas.style.removeProperty(name);
    canvas.style.setProperty('visibility', 'hidden');
    o.markFramePresented();
    o.sync();
    for (const [name, value] of Object.entries(hardened)) {
      expect(canvas.style.props.get(name), name).toEqual({ value, priority: 'important' });
    }
    expect(fx.env().getComputedStyle(canvas as unknown as Element).clipPath).toBe('none');
    expect(o.layoutSupported).toBe(true);
    expect(o.videoHidden).toBe(true);
    o.destroy();
  });

  it('isolates canvas rotate from page rules with an inline important all reset', () => {
    const o = make();
    const canvas = canvasOf(o);
    canvas.stylesheetImportant.rotate = '180deg';
    o.markFramePresented();
    o.sync();
    expect(fx.env().getComputedStyle(canvas as unknown as Element).rotate).toBe('none');
    expect(canvas.style.props.get('all')).toEqual({ value: 'initial', priority: 'important' });
    expect(o.layoutSupported).toBe(true);
    expect(o.videoHidden).toBe(true);
    o.destroy();
  });

  it.each([
    ['display', 'none'], ['visibility', 'hidden'], ['clipPath', 'inset(50%)'],
    ['transform', 'matrix(1, 0, 0, 1, 10, 0)'], ['rotate', '180deg'],
    ['scale', '2'], ['translate', '10px'], ['clip', 'rect(0px, 10px, 10px, 0px)'], ['zoom', '2'],
    ['maskImage', 'url(mask.png)'], ['mixBlendMode', 'multiply'], ['opacity', '0'],
    ['filter', 'blur(4px)'], ['objectFit', 'contain'],
  ] as const)('restores the native picture on an unoverridable canvas %s mismatch', (property, value) => {
    const o = make();
    o.markFramePresented();
    canvasOf(o).computedOverrides[property] = value;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it.each([
    ['clip', 'rect(0px, 400px, 300px, 0px)'],
    ['clipPath', 'circle(40%)'],
    ['maskImage', 'linear-gradient(black, transparent)'],
    ['webkitMaskImage', 'linear-gradient(black, transparent)'],
    ['mixBlendMode', 'multiply'],
    ['objectViewBox', 'inset(10%)'],
  ] as const)('restores native picture when video effect %s appears', (property, value) => {
    const o = make();
    o.markFramePresented();
    video.computed[property] = value;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
  });

  it.each([
    ['objectFit', 'bogus'],
    ['objectFit', ''],
    ['objectPosition', '50% calc(100% - min(10%, 40px))'],
    ['objectPosition', ''],
  ] as const)('fails closed when site %s cannot be parsed: %s', (property, value) => {
    const o = make();
    o.markFramePresented();
    if (property === 'objectPosition') video.style.setProperty('object-position', value);
    video.computed[property] = value;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it('mirrors video filter and updates it on later syncs', () => {
    video.computed.filter = 'brightness(0.5) blur(2px)';
    const o = make();
    const canvas = canvasOf(o);
    expect(canvas.style.props.get('filter')).toEqual({ value: 'brightness(0.5) blur(2px)', priority: 'important' });
    o.markFramePresented();
    video.computed.filter = 'grayscale(1)';
    o.sync();
    expect(canvas.style.getPropertyValue('filter')).toBe('grayscale(1)');
    video.computed.filter = 'none';
    o.sync();
    expect(canvas.style.getPropertyValue('filter')).toBe('none');
  });

  it.each([
    ['exact', 450, 'inset(0px 0px 0px 0px round 12px)'],
    ['letterboxed', 600, 'inset(-75px 0px -75px 0px round 12px)'],
  ] as const)('clips %s video to rounded content box', (_name, height, clip) => {
    video.rect.height = height;
    Object.assign(video.computed, { borderTopLeftRadius: '12px', borderTopRightRadius: '12px',
      borderBottomRightRadius: '12px', borderBottomLeftRadius: '12px' });
    const o = make();
    expect(canvasOf(o).style.getPropertyValue('clip-path')).toBe(clip);
    o.markFramePresented();
    Object.assign(video.computed, { borderTopLeftRadius: '0px', borderTopRightRadius: '0px',
      borderBottomRightRadius: '0px', borderBottomLeftRadius: '0px' });
    o.sync();
    expect(canvasOf(o).style.getPropertyValue('clip-path')).toBe('none');
  });

  it('subtracts adjacent borders and padding from each elliptical corner radius and merges crop', () => {
    video.computed.objectFit = 'cover';
    video.rect = { left: 0, top: 0, width: 800, height: 600 };
    Object.assign(video.computed, { borderTopLeftRadius: '20px 30px', borderTopRightRadius: '2px 3px',
      borderBottomRightRadius: '40px 50px', borderBottomLeftRadius: '60px 70px',
      borderLeftWidth: '2px', paddingLeft: '3px', borderRightWidth: '4px', paddingRight: '5px',
      borderTopWidth: '6px', paddingTop: '7px', borderBottomWidth: '8px', paddingBottom: '9px' });
    const o = make();
    const g = o.layout!.geometry;
    expect(canvasOf(o).style.getPropertyValue('clip-path')).toBe(
      `inset(0px ${cssPx(g.clipRight)} 0px ${cssPx(g.clipLeft)} round 15px 0px 31px 55px / 17px 0px 33px 53px)`,
    );
  });

  it.each(['calc(10% + 20px)', '20px calc(10% + 20px)'])('rejects a corner radius with an unsupported component: %s', (radius) => {
    const o = make();
    o.markFramePresented();
    video.computed.borderTopLeftRadius = radius;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it('normalizes overlapping outer radii before subtracting the left border', () => {
    video.videoWidth = 90;
    video.videoHeight = 100;
    video.rect = { left: 100, top: 50, width: 200, height: 200 };
    Object.assign(video.computed, { objectFit: 'fill', borderLeftWidth: '20px',
      borderTopLeftRadius: '200px', borderTopRightRadius: '200px',
      borderBottomRightRadius: '0px', borderBottomLeftRadius: '0px' });
    const o = make();
    expect(o.layoutSupported).toBe(true);
    expect(canvasOf(o).style.getPropertyValue('clip-path')).toBe(
      'inset(0px 0px 0px 0px round 80px 100px 0px 0px / 100px 100px 0px 0px)',
    );
    o.destroy();
  });

  it('keeps a square exact-fit video free of clip-path', () => {
    video.rect.height = 450;
    const o = make();
    expect(canvasOf(o).style.getPropertyValue('clip-path')).toBe('none');
  });

  it('mirrors video opacity at sync cadence while the picture is hidden', () => {
    video.computed.opacity = '0.5';
    const o = make();
    const canvas = canvasOf(o);
    expect(canvas.style.props.get('opacity')).toEqual({ value: '0.5', priority: 'important' });
    o.markFramePresented();
    video.computed.opacity = '0.8';
    expect(canvas.style.getPropertyValue('opacity')).toBe('0.5');
    o.sync();
    expect(canvas.style.props.get('opacity')).toEqual({ value: '0.8', priority: 'important' });
    expect(o.videoHidden).toBe(true);
    o.destroy();
  });

  it('a backing resize before first presentation invalidates the presentation epoch', () => {
    const o = make();
    const epoch = o.presentationEpoch;
    video.rect.width = 900;
    o.sync();
    expect(o.presentationEpoch).toBeGreaterThan(epoch);
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.destroy();
  });

  it('a fully displaced object-position keeps clipping and the native picture visible', () => {
    const o = make();
    o.markFramePresented();
    video.style.setProperty('object-position', '2000px 2000px');
    o.sync();
    expect(o.layout?.geometry.visibleWidth).toBe(0);
    expect(o.layout?.geometry.visibleHeight).toBe(0);
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.hideVideo()).toBe(false);
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('clip-path')).not.toBe('');
    o.destroy();
  });

  it('leaves a stylesheet-important static parent untouched', () => {
    parent.stylesheetImportant.position = 'static';
    const o = make();
    expect(fx.env().getComputedStyle(parent as unknown as Element).position).toBe('static');
    expect(parent.style.getPropertyValue('position')).toBe('');
    o.markFramePresented();
    expect(o.videoHidden).toBe(true);
    parent.forceStaticPosition = true;
    o.sync();
    expect(o.layoutSupported).toBe(true);
    expect(o.videoHidden).toBe(true);
    o.destroy();
    expect(parent.style.getPropertyValue('position')).toBe('');
  });

  it('supports placement under a display: contents parent via the measured origin', () => {
    const o = make();
    o.markFramePresented();
    parent.computed.display = 'contents';
    o.sync();
    expect(o.layoutSupported).toBe(true);
    expect(o.layout).not.toBeNull();
    expect(o.videoHidden).toBe(true);
    expect(canvasOf(o).getBoundingClientRect()).toEqual({ left: 100, top: 125, width: 800, height: 450 });
    o.destroy();
  });

  it.each([
    ['left', -0.51, 0, 0, 0],
    ['top', 0, -0.51, 0, 0],
    ['right', 0, 0, 0.51, 0],
    ['bottom', 0, 0, 0, 0.51],
  ] as const)('rejects a canvas whose %s edge misses the expected viewport rect by more than 0.5px', (_edge, dx, dy, dw, dh) => {
    const o = make();
    o.markFramePresented();
    const canvas = canvasOf(o);
    const rect = canvas.getBoundingClientRect();
    canvas.canvasRect = { left: rect.left + dx, top: rect.top + dy, width: rect.width + dw, height: rect.height + dh };
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.layout).toBeNull();
    expect(o.videoHidden).toBe(false);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it('supports a canvas whose viewport edges match the content-fit rect within 0.5px', () => {
    parent.scrollLeft = 12;
    parent.scrollTop = 15;
    Object.assign(video.computed, { borderLeftWidth: '3px', paddingLeft: '4px',
      borderTopWidth: '5px', paddingTop: '6px' });
    const o = make();
    const canvas = canvasOf(o);
    const g = o.layout!.geometry;
    canvas.canvasRect = { left: video.rect.left + 7 + g.contentX + 0.5,
      top: video.rect.top + 11 + g.contentY - 0.5, width: g.contentWidth, height: g.contentHeight };
    expect(o.sync()).not.toBeNull();
    expect(o.layoutSupported).toBe(true);
    o.markFramePresented();
    expect(o.videoHidden).toBe(true);
    o.destroy();
  });

  it.each(['display', 'visibility', 'position', 'opacity', 'zIndex', 'order'] as const)(
    'rolls back an insertion-induced sibling %s change', (property) => {
      controls.computedRule = () => parent.children.length > 2 ? { [property]: 'changed' } : {};
      const o = make();
      expect(o.hasInsertionLayoutFailure).toBe(true);
      expect(parent.children).toEqual([video, controls]);
      expect(o.videoHidden).toBe(false);
      o.destroy();
    },
  );

  it.each([0.51, -0.51])('rolls back an insertion-induced sibling rect shift of %s px', (shift) => {
    const original = controls.getBoundingClientRect.bind(controls);
    controls.getBoundingClientRect = () => {
      const rect = original();
      return { ...rect, left: rect.left + (parent.children.length > 2 ? shift : 0) };
    };
    const o = make();
    expect(o.hasInsertionLayoutFailure).toBe(true);
    expect(parent.children).toEqual([video, controls]);
    o.destroy();
  });

  it.each(['video opacity', 'sibling visibility', 'sibling rect'])('rolls back a video patch that changes %s', (effect) => {
    const patched = () => video.style.getPropertyValue('position') === 'relative';
    if (effect === 'video opacity') video.computedRule = () => ({ opacity: patched() ? '0.5' : '1' });
    else if (effect === 'sibling visibility') controls.computedRule = () => ({ visibility: patched() ? 'hidden' : 'visible' });
    else {
      const original = controls.getBoundingClientRect.bind(controls);
      controls.getBoundingClientRect = () => ({ ...original(), left: patched() ? 1 : 0 });
    }
    const o = make();
    expect(o.hasInsertionLayoutFailure).toBe(true);
    expect(o.layoutSupported).toBe(false);
    expect(parent.children).toEqual([video, controls]);
    expect(video.style.getPropertyValue('position')).toBe('');
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it.each(['first-child video', 'nth-child sibling'])('rolls back insertion that changes %s', (rule) => {
    if (rule === 'first-child video') {
      video.computedRule = () => ({ display: parent.children[0] === video ? 'block' : 'none' });
    } else {
      controls.computedRule = () => ({ visibility: parent.children.indexOf(controls) === 1 ? 'visible' : 'hidden' });
    }
    const o = make();
    expect(o.layoutSupported).toBe(false);
    expect(o.hasInsertionLayoutFailure).toBe(true);
    expect(parent.children).toEqual([video, controls]);
    expect(video.style.getPropertyValue('position')).toBe('');
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    o.sync();
    o.reattach();
    o.markFramePresented();
    expect(parent.children).toEqual([video, controls]);
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it.each([
    ['inset box-shadow', { boxShadow: 'rgb(0, 0, 0) 0px 0px 8px inset' }, 'block'],
    ['negative outline-offset', { outlineStyle: 'solid', outlineOffset: '-1px' }, 'block'],
    ...['flex', 'inline-flex', 'grid', 'inline-grid'].map((display) => [`${display} order`, { order: '1' }, display] as const),
  ] as const)('rejects %s and restores an active native picture', (_name, effect, display) => {
    const o = make();
    o.markFramePresented();
    expect(o.videoHidden).toBe(true);
    Object.assign(video.computed, effect);
    parent.computed.display = display;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.videoHidden).toBe(false);
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
    o.destroy();
  });

  it('injects a pass-through canvas as the previous sibling of the video', () => {
    const o = make();
    const c = canvasOf(o);
    expect(parent.children).toEqual([c, video, controls]);
    expect(c.className).toBe(OVERLAY_CANVAS_CLASS);
    expect(c.style.props.get('pointer-events')).toEqual({ value: 'none', priority: 'important' });
    expect(c.style.props.get('position')).toEqual({ value: 'absolute', priority: 'important' });
    expect(c.attrs['aria-hidden']).toBe('true');
    expect(fx.observers[0]?.targets).toEqual([video, parent]);
  });

  it('leaves a static parent untouched on mount and destroy', () => {
    const o = make();
    expect(parent.style.getPropertyValue('position')).toBe('');
    o.destroy();
    expect(parent.style.props.has('position')).toBe(false);
  });

  it('leaves a shared parent untouched while either overlay is mounted', () => {
    parent.style.setProperty('position', 'static', 'important');
    const first = make();
    const secondVideo = new FakeNode();
    secondVideo.videoWidth = video.videoWidth;
    secondVideo.videoHeight = video.videoHeight;
    secondVideo.rect = { ...video.rect };
    parent.appendChild(secondVideo);
    const second = new CanvasOverlay(secondVideo as unknown as HTMLVideoElement, { env: fx.env() });
    first.destroy();
    expect(parent.style.props.get('position')).toEqual({ value: 'static', priority: 'important' });
    second.destroy();
    expect(parent.style.props.get('position')).toEqual({ value: 'static', priority: 'important' });
  });

  it('leaves an already positioned parent alone', () => {
    parent.computed.position = 'absolute';
    make();
    expect(parent.style.props.has('position')).toBe(false);
  });

  it('suppresses and flushes transitions for video patch install, suspend, reapply and restore', () => {
    video.style.setProperty('transition-property', 'all', 'important');
    video.style.managedWrites = [];
    const original = fx.env().getComputedStyle;
    const env = fx.env();
    let flushes = 0;
    env.getComputedStyle = (el) => {
      if (el === video as unknown as Element && video.style.getPropertyValue('transition-property') === 'none') flushes++;
      return original(el);
    };
    const o = new CanvasOverlay(video as unknown as HTMLVideoElement, { env });
    o.sync();
    o.destroy();
    expect(video.style.managedWrites.filter((write) => write.name === 'position').length).toBeGreaterThanOrEqual(6);
    expect(video.style.managedWrites.every((write) => write.transition === 'none' && write.priority === 'important')).toBe(true);
    expect(flushes).toBeGreaterThanOrEqual(video.style.managedWrites.length);
    expect(video.style.props.get('transition-property')).toEqual({ value: 'all', priority: 'important' });
  });

  it('positions a static video while mounted and restores its exact inline position', () => {
    video.style.setProperty('position', 'static', 'important');
    const o = make();
    expect(video.style.props.get('position')).toEqual({ value: 'relative', priority: 'important' });
    o.destroy();
    expect(video.style.props.get('position')).toEqual({ value: 'static', priority: 'important' });
  });

  it.each([['absolute', 'important'], ['relative', '']])('preserves a page video position write of %s with priority %s', (value, priority) => {
    const o = make();
    video.style.setProperty('position', value, priority);
    o.destroy();
    expect(video.style.props.get('position')).toEqual({ value, priority });
  });

  it.each([['absolute', ''], ['relative', '']])('preserves a page parent position write of %s with priority %s', (value, priority) => {
    const o = make();
    parent.style.setProperty('position', value, priority);
    o.destroy();
    expect(parent.style.props.get('position')).toEqual({ value, priority });
  });

  it('restores an absent video position on canvas removal and patches it again on reattach', () => {
    const o = make();
    canvasOf(o).remove();
    fx.fireChildList(parent);
    expect(video.style.props.has('position')).toBe(false);
    o.reattach();
    expect(video.style.props.get('position')).toEqual({ value: 'relative', priority: 'important' });
    o.destroy();
    expect(video.style.props.has('position')).toBe(false);
  });

  it('leaves an absolutely positioned video alone', () => {
    video.style.setProperty('position', 'absolute');
    const o = make();
    expect(video.style.props.get('position')).toEqual({ value: 'absolute', priority: '' });
    o.destroy();
    expect(video.style.props.get('position')).toEqual({ value: 'absolute', priority: '' });
  });

  it('restores the video position before checking its new parent styling', () => {
    const o = make();
    const theater = new FakeNode();
    body.appendChild(theater);
    theater.appendChild(video);
    video.computed.position = 'absolute';
    o.reattach();
    expect(video.style.props.has('position')).toBe(false);
    expect(o.isMounted).toBe(true);
    o.destroy();
    expect(video.style.props.has('position')).toBe(false);
  });

  it('isMounted is false when the canvas is moved after the video or removed', () => {
    const o = make();
    expect(o.isMounted).toBe(true);
    parent.insertBefore(canvasOf(o), controls);
    expect(o.isMounted).toBe(false);
    o.sync();
    expect(o.isMounted).toBe(true);
    canvasOf(o).remove();
    expect(o.isMounted).toBe(false);
  });

  it('uses the site object-position for geometry while the picture is hidden', () => {
    const o = make();
    o.markFramePresented();
    o.sync();
    expect(o.layout?.geometry.contentY).toBe(75); // 50% of the 150px letterbox space
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    video.style.setProperty('object-position', '40% 20%', 'important');
    o.sync();
    expect(o.layout?.geometry.contentY).toBe(30);
    o.restoreVideo();
    expect(video.style.props.get('object-position')).toEqual({ value: '40% 20%', priority: 'important' });
  });

  it('z-index equals the video z-index (auto stays auto)', () => {
    video.computed.position = 'relative';
    video.computed.zIndex = '7';
    expect(canvasOf(make()).style.getPropertyValue('z-index')).toBe('7');
    video.computed.zIndex = 'auto';
    expect(canvasOf(make()).style.getPropertyValue('z-index')).toBe('auto');
  });

  it('aligns to the letterboxed content rect at sub-pixel precision and sizes the backing store', () => {
    // 16:9 in 800x600: 800x450 at y=75; initial containing-block origin is (0, 0)
    fx.dpr = 2;
    const o = make();
    const s = canvasOf(o).style;
    expect([s.getPropertyValue('left'), s.getPropertyValue('top'), s.getPropertyValue('width'), s.getPropertyValue('height')]).toEqual(['100px', '125px', '800px', '450px']);
    expect(s.props.get('clip-path')).toEqual({ value: 'none', priority: 'important' });
    // 800x450 CSS @2x = 1600x900 device px (under the 1080p ceiling)
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([1600, 900]);
    expect(o.layout?.geometry.mode).toBe('letterbox');
    expect(layouts).toHaveLength(1);
  });

  it('uses the internal 2x image as backing store when the target covers it', () => {
    video.videoWidth = 640;
    video.videoHeight = 360;
    fx.dpr = 2; // target 1600x900 >= internal 1280x720
    const o = make(2);
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([1280, 720]);
    expect(o.layout?.canvas.presentsInternal).toBe(true);
    o.setInternalScale(null);
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([1600, 900]);
  });

  it('clips the canvas to the box for object-fit: cover', () => {
    video.computed.objectFit = 'cover';
    const o = make();
    const s = canvasOf(o).style;
    // cover 1920x1080 in 800x600 -> 1066.6667 wide at x = -133.3333 (+100 box origin)
    expect(s.getPropertyValue('left')).toBe('-33.3333px');
    expect(s.getPropertyValue('width')).toBe('1066.6667px');
    expect(s.getPropertyValue('clip-path')).toBe('inset(0px 133.3333px 0px 133.3333px)');
  });

  it('keeps the video visible until the first presented frame', () => {
    const o = make();
    expect(o.hideVideo()).toBe(false);
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');

    o.markFramePresented();
    expect([video.style.getPropertyValue('object-position'), video.style.getPropertyPriority('object-position')]).toEqual([HIDDEN_OBJECT_POSITION, 'important']);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('visible');
    expect(o.videoHidden).toBe(true);
    expect(o.hideVideo()).toBe(true);
  });

  it.each(['seeking', 'emptied', 'error', 'abort'])('restores the original picture styles on %s', (event) => {
    const o = make();
    o.markFramePresented();
    video.dispatch(event);
    expect([video.style.getPropertyValue('object-position'), video.style.getPropertyPriority('object-position')]).toEqual(['50% 50%', '']);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    expect(o.hideVideo()).toBe(false); // needs a new presented frame
    o.markFramePresented();
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
  });

  it('restores the video when it is detached (unmount) and on destroy', () => {
    const o = make();
    o.markFramePresented();
    video.remove();
    o.sync();
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(video.style.props.has('position')).toBe(false);

    parent.insertBefore(video, controls);
    o.reattach();
    o.markFramePresented();
    o.destroy();
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(parent.children.includes(canvasOf(o))).toBe(false);
    expect(video.listenerCount()).toBe(0);
    expect(fx.observers.every((ob) => ob.disconnected)).toBe(true);
    expect([...fx.windowListeners.values()].every((s) => s.size === 0)).toBe(true);
    expect(fx.doc.listenerCount()).toBe(0);
  });

  it('restores the video when the page removes the canvas, and never hides it without one', () => {
    const o = make();
    o.markFramePresented();
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    canvasOf(o).remove(); // e.g. a framework that strips unknown children
    fx.fireChildList(parent);
    expect([video.style.getPropertyValue('object-position'), video.style.getPropertyPriority('object-position')]).toEqual(['50% 50%', '']);
    expect(o.isMounted).toBe(false);
    o.markFramePresented(); // a frame rendered into the detached canvas shows nothing
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(o.hideVideo()).toBe(false);
    o.reattach();
    expect(parent.children).toEqual([canvasOf(o), video, controls]);
    expect(o.isMounted).toBe(true);
    o.markFramePresented();
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
  });

  it('a container re-render that keeps the same <video> restores it until the canvas is back', () => {
    const o = make();
    o.markFramePresented();
    for (const child of [...parent.children]) child.remove(); // container.textContent = ''
    parent.appendChild(video); // ... and the player re-appends its video element
    fx.fireChildList(parent);
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    o.reattach();
    expect(parent.children).toEqual([canvasOf(o), video]);
    o.markFramePresented();
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    // the canvas moved apart from the video (something inserted in between) is not mounted either
    parent.insertBefore(controls, video);
    fx.fireChildList(parent);
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    // its own re-mount (a child-list change while mounted) does not restore anything
    o.reattach();
    o.markFramePresented();
    fx.fireChildList(parent);
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);
    o.destroy();
    expect(fx.mutationObservers.every((m) => m.disconnected)).toBe(true);
  });

  it('removes the object-position property on restore when the video had none', () => {
    video.style.removeProperty('object-position');
    const o = make();
    o.markFramePresented();
    o.restoreVideo();
    expect(video.style.props.has('object-position')).toBe(false);
  });

  it('re-anchors into a new parent after re-parenting and restores the old parent', () => {
    const o = make();
    const theater = new FakeNode();
    body.appendChild(theater);
    theater.rect = { left: 0, top: 0, width: 1280, height: 720 };
    theater.appendChild(video);
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    o.reattach();
    expect(theater.children).toEqual([canvasOf(o), video]);
    expect(parent.style.props.has('position')).toBe(false);
    expect(theater.style.getPropertyValue('position')).toBe('');
    expect(canvasOf(o).style.getPropertyValue('width')).toBe('1280px');
    expect(o.layout?.geometry.mode).toBe('exact');
  });

  it('a fullscreen <video> restores the native video; a fullscreen container keeps the overlay', () => {
    const o = make();
    o.markFramePresented();
    fx.doc.fullscreenElement = parent;
    fx.doc.dispatch('fullscreenchange');
    expect(o.layout?.fullscreen).toBe('container');
    expect(video.style.getPropertyValue('object-position')).toBe(HIDDEN_OBJECT_POSITION);

    fx.doc.fullscreenElement = video;
    fx.doc.dispatch('fullscreenchange');
    expect(o.layout?.fullscreen).toBe('video');
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
    o.markFramePresented(); // refused while the video itself is fullscreen
    expect(video.style.getPropertyValue('object-position')).toBe('50% 50%');
  });

  it('restores native picture when the video is assigned to a named slot', () => {
    const o = make();
    o.markFramePresented();
    const slot = new FakeNode();
    slot.setAttribute('name', 'player');
    video.setAttribute('slot', 'player');
    video.assignedSlot = slot;
    o.sync();
    expect(o.layoutSupported).toBe(false);
    expect(o.isMounted).toBe(false);
    expect(o.videoHidden).toBe(false);
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
  });

  it('restores native picture for video fullscreen through open shadow roots', () => {
    const host = new FakeNode();
    body.appendChild(host);
    Object.assign(parent, { parentNode: { host } });
    parent.remove();
    parent.connectedRoot = true;
    host.shadowRoot = { fullscreenElement: null };
    const o = make();
    o.markFramePresented();
    fx.doc.fullscreenElement = host;
    fx.doc.dispatch('fullscreenchange');
    expect(o.layout?.fullscreen).toBe('container');
    expect(o.videoHidden).toBe(true);
    host.shadowRoot.fullscreenElement = video;
    fx.doc.dispatch('fullscreenchange');
    expect(o.layout?.fullscreen).toBe('video');
    expect(o.videoHidden).toBe(false);
    expect(canvasOf(o).style.getPropertyValue('visibility')).toBe('hidden');
    o.markFramePresented();
    expect(o.videoHidden).toBe(false);
  });

  it('debounces backing-store resizes during a fullscreen transition while CSS follows immediately', () => {
    const o = make();
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([800, 450]);
    fx.doc.fullscreenElement = parent;
    video.rect = { left: 0, top: 0, width: 1600, height: 1200 };
    fx.doc.dispatch('fullscreenchange');
    fx.fireResize();
    expect(canvasOf(o).style.getPropertyValue('width')).toBe('1600px');
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([800, 450]); // not yet
    expect(o.layout?.backing).toEqual({ width: 800, height: 450 });
    expect(o.layout?.canvas).toMatchObject({ width: 1600, height: 900 });
    const before = layouts.length;
    fx.runTimers();
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([1600, 900]);
    expect(layouts.length).toBe(before + 1); // the applied size is reported
    // outside a transition, resizes apply immediately
    video.rect = { left: 0, top: 0, width: 800, height: 600 };
    fx.fireResize();
    expect([canvasOf(o).width, canvasOf(o).height]).toEqual([800, 450]);
  });

  it('reports layout changes only when something changed', () => {
    const o = make();
    const n = layouts.length;
    o.sync();
    fx.fireResize();
    expect(layouts.length).toBe(n);
    video.rect = { ...video.rect, width: 640 };
    fx.fireResize();
    expect(layouts.length).toBe(n + 1);
  });
});
