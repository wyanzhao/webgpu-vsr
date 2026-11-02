import { afterEach, describe, expect, it, vi } from 'vitest';
import { bootstrapContentScript, type ContentController, type VideoSessionHandle } from '../../src/content/index';
import { defaultSettings } from '../../src/shared/settings';
import { defaultSiteRule } from '../../src/content/site_rules';
import { FakeEnv, FakeNode } from './fake_dom';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('content bootstrap', () => {
  it.each(['storage', 'out-of-order broadcasts'] as const)('applies %s disable during a delayed site-rule fetch before discovery', async (channel) => {
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
    const location = { hostname: 'example.com', origin: 'https://example.com', href: 'https://example.com/player' };
    const win = {
      document: doc, location, devicePixelRatio: 1, getComputedStyle: fx.env().getComputedStyle,
      addEventListener: vi.fn(), removeEventListener: vi.fn(), setTimeout, clearTimeout, setInterval, clearInterval,
    };
    Object.assign(win, { top: win });
    vi.stubGlobal('document', doc);
    vi.stubGlobal('window', win);
    vi.stubGlobal('location', location);
    type StorageListener = (changes: Record<string, { newValue: unknown }>, area: string) => void;
    type MessageListener = (message: unknown, sender: object, respond: (reply: unknown) => void) => boolean;
    const storageListeners = new Set<StorageListener>();
    const messageListeners = new Set<MessageListener>();
    let listenersBeforeRead = false;
    let storedSettings = defaultSettings();
    vi.stubGlobal('chrome', {
      storage: {
        local: { get: async () => {
          listenersBeforeRead = storageListeners.size === 1 && messageListeners.size === 1;
          return { settings: storedSettings };
        } },
        onChanged: {
          addListener: (listener: StorageListener) => storageListeners.add(listener),
          removeListener: (listener: StorageListener) => storageListeners.delete(listener),
        },
      },
      runtime: {
        id: 'test-extension', getURL: (path: string) => `chrome-extension://test/${path}`,
        sendMessage: async () => undefined, getManifest: () => ({ version: '1.0.0' }),
        onMessage: {
          addListener: (listener: MessageListener) => messageListeners.add(listener),
          removeListener: (listener: MessageListener) => messageListeners.delete(listener),
        },
      },
    });
    let release!: (response: unknown) => void;
    vi.stubGlobal('fetch', vi.fn(() => new Promise((resolve) => { release = resolve; })));
    const create = vi.fn(() => ({ attach: vi.fn(), destroy: vi.fn() }) as unknown as VideoSessionHandle);
    const pending = bootstrapContentScript({ sessionFactory: { isGpuAvailable: () => true, create } });
    let controller: ContentController | null = null;
    try {
      for (let i = 0; i < 5; i++) await Promise.resolve();
      const disabled = { ...defaultSettings(), enabled: false };
      storedSettings = disabled;
      for (const listener of storageListeners) listener({ settings: { newValue: disabled } }, 'local');
      if (channel === 'out-of-order broadcasts') {
        // Storage ends disabled; independently awaited broadcasts finish false, then stale true.
        for (const enabled of [false, true]) {
          for (const listener of messageListeners) {
            listener({ type: 'SETTINGS_CHANGED', settings: { ...disabled, enabled } }, {}, () => undefined);
          }
        }
      }
      release({ ok: true, json: async () => defaultSiteRule() });
      controller = await pending;
      expect(controller?.currentSettings.enabled).toBe(false);
      expect(listenersBeforeRead).toBe(true);
      expect(controller?.reports().map((report) => report.reason)).toEqual(['extension-disabled']);
      expect(create).not.toHaveBeenCalled();
    } finally {
      controller?.stop();
    }
    expect(storageListeners.size).toBe(0);
    expect(messageListeners.size).toBe(0);
  });
});
