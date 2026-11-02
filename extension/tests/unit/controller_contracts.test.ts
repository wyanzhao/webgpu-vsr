import { afterEach, describe, expect, it, vi } from 'vitest';
import { ContentController, type VideoSessionContext, type VideoSessionHandle } from '../../src/content/index';
import { POLL_INTERVAL_MS } from '../../src/content/discovery';
import { defaultSettings, type UserSettings } from '../../src/shared/settings';
import { defaultSiteRule } from '../../src/content/site_rules';
import { FakeEnv, FakeNode } from './fake_dom';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('controller srcObject identity', () => {
  it.each(['frame-budget', 'tainted', 'hdr', 'drm', 'error'] as const)(
    'resets source state and starts a fresh session after a %s stop and MediaStream swap', (reason) => {
      vi.useFakeTimers();
      class FakeMediaStream {}
      const first = new FakeMediaStream();
      const second = new FakeMediaStream();
      const probe = vi.fn();
      vi.stubGlobal('VideoFrame', class {
        readonly colorSpace = { transfer: 'bt709' };
        constructor() { probe(); }
        close(): void {}
      });
      const fx = new FakeEnv();
      const parent = new FakeNode();
      parent.connectedRoot = true;
      const video = Object.assign(new FakeNode(), {
        readyState: 4, mediaKeys: null, src: '', currentSrc: '', srcObject: first,
      });
      parent.appendChild(video);
      video.videoWidth = 640;
      video.videoHeight = 360;
      video.rect = { left: 0, top: 0, width: 1280, height: 720 };
      const doc = Object.assign(fx.doc, {
        documentElement: parent,
        querySelectorAll: (selector: string) => selector === 'video' ? [video] : [parent, video],
      });
      vi.stubGlobal('document', doc);
      vi.stubGlobal('window', {
        document: doc, location: { origin: 'https://example.com' }, devicePixelRatio: 1,
        getComputedStyle: fx.env().getComputedStyle,
        addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout, setInterval, clearInterval,
      });
      vi.stubGlobal('chrome', { runtime: { id: 'test-extension', sendMessage: async () => undefined } });
      const contexts: VideoSessionContext[] = [];
      const destroy = vi.fn();
      let budgetHealthy = true;
      const create = vi.fn((context: VideoSessionContext): VideoSessionHandle => {
        contexts.push(context);
        budgetHealthy = true;
        return {
          attach: vi.fn(), reattach: vi.fn(), detach: vi.fn(), refreshGeometry: vi.fn(),
          isLayoutSupported: () => true, updateSettings: vi.fn(), sourceChanged: vi.fn(),
          isBudgetHealthy: () => budgetHealthy, destroy,
        };
      });
      const controller = new ContentController({
        settings: defaultSettings(), siteRule: defaultSiteRule(), hostname: 'example.com', isTopFrame: true,
        getUrl: () => 'https://example.com/player', send: async () => undefined,
        sessionFactory: { isGpuAvailable: () => true, create },
      });
      try {
        controller.start();
        expect(create).toHaveBeenCalledTimes(1);
        expect(probe).toHaveBeenCalledTimes(1);
        const context = contexts[0]!;
        if (reason === 'frame-budget') budgetHealthy = false;
        else if (reason === 'tainted') context.latches.latchTaint(video);
        else if (reason === 'hdr') context.latches.latchHdr(video);
        else if (reason === 'drm') video.dispatch('encrypted');
        else context.reportError('old source failed');
        context.requestEvaluation();
        vi.advanceTimersByTime(POLL_INTERVAL_MS);
        expect(destroy).toHaveBeenCalledTimes(1);
        expect(create).toHaveBeenCalledTimes(1);
        expect(controller.reports()[0]).toMatchObject(reason === 'error' ?
          { state: 'error', error: 'old source failed' } : { state: 'ineligible', reason });
        expect(probe).toHaveBeenCalledTimes(1);

        video.srcObject = second;
        vi.advanceTimersByTime(POLL_INTERVAL_MS);
        expect(create).toHaveBeenCalledTimes(2);
        expect(contexts[1]).not.toBe(context);
        expect(contexts[1]!.video.srcObject).toBe(second);
        expect(context.latches.isDrmLatched(video)).toBe(false);
        expect(context.latches.isTaintLatched(video)).toBe(false);
        expect(context.latches.isHdrLatched(video)).toBe(false);
        expect(probe).toHaveBeenCalledTimes(2);
        expect(controller.reports()[0]).toMatchObject({ state: 'eligible', reason: null, error: null, telemetry: null });
        vi.advanceTimersByTime(POLL_INTERVAL_MS);
        expect(create).toHaveBeenCalledTimes(2);
        expect(probe).toHaveBeenCalledTimes(2);
      } finally {
        controller.stop();
      }
    },
  );
});

describe('controller HUD disable contract', () => {
  it.each(['extension', 'frame site', 'top-level site', 'site rule', 'site before gate 7'] as const)(
    'removes every extension node when disabled by %s', (disabledBy) => {
      vi.useFakeTimers();
      const fx = new FakeEnv();
      const parent = new FakeNode();
      parent.connectedRoot = true;
      const video = Object.assign(new FakeNode(), { readyState: 1, mediaKeys: null, currentSrc: '' });
      parent.appendChild(video);
      video.videoWidth = 640;
      video.videoHeight = 360;
      video.rect = { left: 0, top: 0, width: 1280, height: 720 };
      const doc = Object.assign(fx.doc, {
        documentElement: parent,
        querySelectorAll: (selector: string) => selector === 'video' ? [video] : [parent, video],
      });
      vi.stubGlobal('document', doc);
      vi.stubGlobal('window', {
        document: doc, location: { origin: 'https://example.com' }, devicePixelRatio: 1,
        getComputedStyle: fx.env().getComputedStyle,
        addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout, setInterval, clearInterval,
      });
      vi.stubGlobal('chrome', { runtime: { id: 'test-extension', sendMessage: async () => undefined } });
      const settings = { ...defaultSettings(), hudEnabled: true };
      const disabled: UserSettings = {
        ...settings,
        enabled: disabledBy !== 'extension',
        siteOverrides: disabledBy === 'top-level site' ? { 'top.example': false } :
          disabledBy === 'site before gate 7' || disabledBy === 'frame site' ? { 'example.com': false } : {},
      };
      const siteRule = { ...defaultSiteRule(), enabled: disabledBy !== 'site rule' };
      const controller = new ContentController({
        settings, siteRule, hostname: 'example.com', topHostname: 'top.example', isTopFrame: false,
        getUrl: () => 'https://example.com/player', send: async () => undefined,
        sessionFactory: { isGpuAvailable: () => false, create: vi.fn() },
      });
      try {
        controller.start();
        if (disabledBy !== 'site rule') expect(parent.children).toHaveLength(2);
        if (disabledBy === 'site before gate 7') video.rect = { left: 0, top: 0, width: 640, height: 360 };
        controller.applySettings(disabled);
        expect(parent.children).toEqual([video]);
        vi.advanceTimersByTime(6000);
        expect(parent.children).toEqual([video]);
      } finally {
        controller.stop();
      }
    },
  );
});
