import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ContentController, type VideoSessionContext, type VideoSessionHandle } from '../../src/content/index';
import { CanvasOverlay } from '../../src/content/overlay';
import { defaultSettings } from '../../src/shared/settings';
import { defaultSiteRule } from '../../src/content/site_rules';
import { FakeEnv, FakeNode } from './fake_dom';

beforeEach(() => {
  vi.stubGlobal('chrome', { runtime: { id: 'test-extension', sendMessage: async () => undefined } });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('insertion layout latch', () => {
  it.each(['first-child video', 'nth-child sibling', 'first-child video filter', 'has ancestor opacity',
    'positioned sibling opacity', 'positioned sibling filter', 'positioned ancestor opacity', 'positioned video filter'])(
    'keeps %s unsupported across later evaluations until source or parent changes', (rule) => {
    vi.useFakeTimers();
    const fx = new FakeEnv();
    const parent = new FakeNode();
    parent.connectedRoot = true;
    const video = Object.assign(new FakeNode(), { readyState: 1, mediaKeys: null, src: 'one.mp4', currentSrc: 'one.mp4' });
    const sibling = new FakeNode();
    parent.appendChild(video);
    parent.appendChild(sibling);
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    if (rule === 'first-child video') {
      video.computedRule = () => ({ display: parent.children[0] === video ? 'block' : 'none' });
    } else if (rule === 'nth-child sibling') {
      sibling.computedRule = () => ({ opacity: parent.children.indexOf(sibling) === 1 ? '1' : '0.5' });
    } else if (rule.startsWith('positioned ')) {
      const positioned = (): boolean => video.style.getPropertyValue('position') === 'relative';
      if (rule === 'positioned sibling opacity') {
        sibling.computedRule = () => ({ opacity: positioned() ? '0' : '1' });
      } else if (rule === 'positioned sibling filter') {
        sibling.computedRule = () => ({ filter: positioned() ? 'opacity(0)' : 'none' });
      } else if (rule === 'positioned ancestor opacity') {
        parent.computedRule = () => ({ opacity: positioned() ? '0' : '1' });
      } else {
        video.computedRule = () => ({ filter: positioned() ? 'opacity(0)' : 'none' });
      }
    } else {
      parent.afterInsertion = (node) => {
        if (!node.isCanvas) return;
        if (rule === 'first-child video filter') {
          video.computedRule = () => ({ filter: parent.children[0] === video ? 'none' : 'opacity(0)' });
        } else {
          parent.computedRule = () => ({ opacity: parent.children[0] === video ? '1' : '0' });
        }
      };
    }
    const doc = Object.assign(fx.doc, {
      documentElement: parent,
      querySelectorAll: (selector: string) => selector === 'video' ? [video] : [parent, video, sibling],
    });
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', {
      document: doc, devicePixelRatio: 1, location: { origin: 'https://example.com' },
      getComputedStyle: fx.env().getComputedStyle,
      addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout, setInterval, clearInterval,
    });
    const overlays: CanvasOverlay[] = [];
    const create = vi.fn((context: VideoSessionContext): VideoSessionHandle => {
      let overlay: CanvasOverlay;
      return {
        attach: () => {
          overlay = new CanvasOverlay(context.video, { env: fx.env() });
          overlays.push(overlay);
          if (!overlay.layoutSupported) context.requestEvaluation();
        },
        reattach: () => overlay.reattach(), detach: () => overlay.restoreVideo(),
        refreshGeometry: () => { overlay.sync(); },
        isLayoutSupported: () => overlay.layoutSupported,
        hasInsertionLayoutFailure: () => overlay.hasInsertionLayoutFailure,
        readSiteStyle: (read) => overlay.readSiteStyle(read),
        updateSettings: () => undefined, sourceChanged: () => undefined,
        isBudgetHealthy: () => true, destroy: () => overlay.destroy(),
      };
    });
    const settings = defaultSettings();
    const controller = new ContentController({
      settings, siteRule: defaultSiteRule(), hostname: 'example.com', getUrl: () => 'https://example.com/player',
      isTopFrame: true, sessionFactory: { isGpuAvailable: () => true, create }, send: async () => undefined,
    });
    try {
      controller.start();
      vi.advanceTimersByTime(100);
      expect(controller.reports().map((report) => report.reason)).toEqual(['unsupported-layout']);
      expect(parent.children).toEqual([video, sibling]);
      expect(video.style.getPropertyValue('position')).toBe('');
      expect(overlays[0]?.videoHidden).toBe(false);
      expect(overlays[0]?.hasInsertionLayoutFailure).toBe(true);
      expect(overlays[0]?.isDestroyed).toBe(true);
      vi.advanceTimersByTime(6000);
      controller.applySettings({ ...settings, qualityLevel: 0 });
      vi.advanceTimersByTime(100);
      expect(create).toHaveBeenCalledTimes(1);
      expect(controller.reports().map((report) => report.reason)).toEqual(['unsupported-layout']);
      video.src = 'two.mp4'; // currentSrc can still name the old resource while loading.
      video.dispatch('loadedmetadata');
      vi.advanceTimersByTime(100);
      expect(create).toHaveBeenCalledTimes(2);
      expect(controller.reports().map((report) => report.reason)).toEqual(['unsupported-layout']);
      video.currentSrc = 'two.mp4';
      video.dispatch('loadedmetadata');
      vi.advanceTimersByTime(100);
      expect(create).toHaveBeenCalledTimes(3);
      expect(controller.reports().map((report) => report.reason)).toEqual(['unsupported-layout']);
      video.computedRule = null;
      sibling.computedRule = null;
      parent.computedRule = null;
      parent.afterInsertion = null;
      const newParent = new FakeNode();
      parent.appendChild(newParent);
      newParent.appendChild(video);
      video.dispatch('loadedmetadata');
      vi.advanceTimersByTime(100);
      expect(create).toHaveBeenCalledTimes(4);
      expect(controller.reports().map((report) => report.reason)).toEqual([null]);
      expect(overlays[3]?.isMounted).toBe(true);
    } finally {
      controller.stop();
    }
  });

  it('removes and latches a selector-changing HUD with no GPU across later evaluations', () => {
    vi.useFakeTimers();
    const fx = new FakeEnv();
    const parent = new FakeNode();
    parent.connectedRoot = true;
    const video = Object.assign(new FakeNode(), { readyState: 1, mediaKeys: null, src: 'one.mp4', currentSrc: 'one.mp4' });
    parent.appendChild(video);
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    const nativeStyle = [...video.style.props];
    const nativeRect = video.getBoundingClientRect();
    let insertions = 0;
    const afterInsertion = (node: FakeNode): void => {
      if (node.className !== 'webgpu-vsr-hud') return;
      insertions++;
      video.computedRule = () => ({ opacity: video.nextSibling ? '0' : '1' });
    };
    parent.afterInsertion = afterInsertion;
    const doc = Object.assign(fx.doc, {
      documentElement: parent,
      querySelectorAll: (selector: string) => selector === 'video' ? [video] : [parent, video],
    });
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', {
      document: doc, devicePixelRatio: 1, location: { origin: 'https://example.com' },
      getComputedStyle: fx.env().getComputedStyle,
      addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout, setInterval, clearInterval,
    });
    const create = vi.fn();
    const settings = { ...defaultSettings(), hudEnabled: true };
    const controller = new ContentController({
      settings, siteRule: defaultSiteRule(), hostname: 'example.com', getUrl: () => 'https://example.com/player',
      isTopFrame: true, sessionFactory: { isGpuAvailable: () => false, create }, send: async () => undefined,
    });
    try {
      controller.start();
      expect(insertions).toBe(1);
      expect(parent.children).toEqual([video]);
      expect(controller.reports().map((report) => report.reason)).toEqual(['webgpu-unavailable']);
      vi.advanceTimersByTime(6000);
      controller.applySettings({ ...settings, qualityLevel: 0 });
      controller.applySettings({ ...settings, hudEnabled: false });
      controller.applySettings(settings);
      vi.advanceTimersByTime(100);
      expect(insertions).toBe(1);
      expect([...video.style.props]).toEqual(nativeStyle);
      expect(video.style.writes).toBe(0);
      expect(video.getBoundingClientRect()).toEqual(nativeRect);
      expect(fx.env().getComputedStyle(video as unknown as Element).opacity).toBe('1');
      expect(create).not.toHaveBeenCalled();
      video.src = 'two.mp4';
      video.dispatch('loadedmetadata');
      vi.advanceTimersByTime(100);
      expect(insertions).toBe(2);
      video.currentSrc = 'two.mp4';
      video.dispatch('loadedmetadata');
      vi.advanceTimersByTime(100);
      expect(insertions).toBe(3);
      const newParent = new FakeNode();
      parent.appendChild(newParent);
      newParent.appendChild(video);
      newParent.afterInsertion = afterInsertion;
      video.dispatch('loadedmetadata');
      vi.advanceTimersByTime(100);
      expect(insertions).toBe(4);
      expect(newParent.children).toEqual([video]);
      vi.advanceTimersByTime(6000);
      expect(insertions).toBe(4);
      expect([...video.style.props]).toEqual(nativeStyle);
    } finally {
      controller.stop();
    }
  });
});
