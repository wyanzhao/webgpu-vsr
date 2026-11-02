import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SETTINGS,
  MAX_SITE_OVERRIDES,
  SETTINGS_STORAGE_KEY,
  SerialQueue,
  applySettingsPatch,
  coerceQualityLevel,
  coerceSharpness,
  defaultSettings,
  initializeSettings,
  loadSettings,
  normalizeHostname,
  sanitizeSettings,
  sanitizeSettingsPatch,
  saveSettings,
  settingsEqual,
  siteOverrideFor,
  watchSettings,
  type SettingsStorageArea,
  type StorageChangeLike,
  type StorageChangedEvent,
  type UserSettings,
} from '../../src/shared/settings';

/** In-memory chrome.storage.local stand-in that deep-copies like structured clone and counts writes. */
class MemoryArea implements SettingsStorageArea {
  data: Record<string, unknown> = {};
  writes = 0;
  async get(key: string): Promise<Record<string, unknown>> {
    await Promise.resolve();
    return key in this.data ? { [key]: structuredClone(this.data[key]) } : {};
  }
  async set(items: Record<string, unknown>): Promise<void> {
    await Promise.resolve();
    this.writes++;
    for (const [k, v] of Object.entries(items)) this.data[k] = structuredClone(v);
  }
}

describe('defaults', () => {
  it('match IMPLEMENTATION_NOTES section 5 exactly', () => {
    expect({ ...DEFAULT_SETTINGS, siteOverrides: { ...DEFAULT_SETTINGS.siteOverrides } }).toEqual({
      enabled: true,
      qualityLevel: 1,
      sharpness: 0.5,
      hudEnabled: false,
      siteOverrides: {},
    });
    expect(Object.isFrozen(DEFAULT_SETTINGS)).toBe(true);
  });

  it('defaultSettings() returns an independent mutable copy', () => {
    const a = defaultSettings();
    a.siteOverrides['x.com'] = false;
    expect(defaultSettings().siteOverrides).toEqual({});
  });
});

describe('field coercion', () => {
  it('quality: rounds, clamps to 0..2, rejects non-numbers', () => {
    expect(coerceQualityLevel(0)).toBe(0);
    expect(coerceQualityLevel(2)).toBe(2);
    expect(coerceQualityLevel(1.4)).toBe(1);
    expect(coerceQualityLevel(1.6)).toBe(2);
    expect(coerceQualityLevel(7)).toBe(2);
    expect(coerceQualityLevel(-3)).toBe(0);
    expect(coerceQualityLevel('2')).toBeNull();
    expect(coerceQualityLevel(Number.NaN)).toBeNull();
    expect(coerceQualityLevel(Infinity)).toBeNull();
  });

  it('sharpness: clamps to [0,1], rejects non-finite', () => {
    expect(coerceSharpness(0.25)).toBe(0.25);
    expect(coerceSharpness(-0.5)).toBe(0);
    expect(coerceSharpness(3)).toBe(1);
    expect(coerceSharpness(Number.NaN)).toBeNull();
    expect(coerceSharpness('0.5')).toBeNull();
  });

  it('hostnames are lower-cased, stripped of a trailing dot and validated', () => {
    expect(normalizeHostname('WWW.YouTube.com.')).toBe('www.youtube.com');
    expect(normalizeHostname('[::1]')).toBe('[::1]');
    expect(normalizeHostname('example.com:8080')).toBeNull();
    expect(normalizeHostname('a b.com')).toBeNull();
    expect(normalizeHostname('')).toBeNull();
    expect(normalizeHostname('x'.repeat(254))).toBeNull();
    expect(normalizeHostname(42)).toBeNull();
  });
});

describe('sanitizeSettings', () => {
  it('returns defaults for non-objects', () => {
    for (const raw of [undefined, null, 5, 'x', [1, 2]]) expect(sanitizeSettings(raw)).toEqual(defaultSettings());
  });

  it('keeps valid fields and replaces invalid ones field by field', () => {
    const s = sanitizeSettings({
      enabled: false,
      qualityLevel: 'high',
      sharpness: 9,
      hudEnabled: 'yes',
      siteOverrides: { 'Example.COM': false, 'bad host': true, 'ok.org': 'true', 'b.net': true },
      contentMode: 'anime',
    });
    expect(s).toEqual({
      enabled: false,
      qualityLevel: 1,
      sharpness: 1,
      hudEnabled: false,
      siteOverrides: { 'example.com': false, 'b.net': true },
    });
    expect('contentMode' in s).toBe(false);
  });

  it('caps the number of stored overrides', () => {
    const many: Record<string, boolean> = {};
    for (let i = 0; i < MAX_SITE_OVERRIDES + 50; i++) many[`h${i}.com`] = i % 2 === 0;
    expect(Object.keys(sanitizeSettings({ siteOverrides: many }).siteOverrides)).toHaveLength(MAX_SITE_OVERRIDES);
  });
});

describe('patches', () => {
  it('sanitizeSettingsPatch drops unknown and invalid keys', () => {
    expect(sanitizeSettingsPatch(null)).toBeNull();
    expect(sanitizeSettingsPatch({ enabled: 'no', foo: 1, sharpness: 0.75, siteOverrides: { 'A.com': null, 'b c': true } })).toEqual({
      sharpness: 0.75,
      siteOverrides: { 'a.com': null },
    });
  });

  it('applySettingsPatch merges per field and per host; null removes an override', () => {
    const base: UserSettings = { ...defaultSettings(), siteOverrides: { 'a.com': false, 'b.com': true } };
    const next = applySettingsPatch(base, { qualityLevel: 2, siteOverrides: { 'a.com': null, 'c.com': false } });
    expect(next).toEqual({ ...defaultSettings(), qualityLevel: 2, siteOverrides: { 'b.com': true, 'c.com': false } });
    // input untouched
    expect(base.siteOverrides).toEqual({ 'a.com': false, 'b.com': true });
  });

  it('applySettingsPatch ignores a junk patch', () => {
    const base = { ...defaultSettings(), enabled: false };
    expect(applySettingsPatch(base, 'garbage')).toEqual(base);
  });

  it('settingsEqual compares every field including overrides', () => {
    const a = defaultSettings();
    expect(settingsEqual(a, defaultSettings())).toBe(true);
    expect(settingsEqual(a, { ...a, sharpness: 0.55 })).toBe(false);
    expect(settingsEqual(a, { ...a, siteOverrides: { 'x.com': true } })).toBe(false);
    expect(settingsEqual({ ...a, siteOverrides: { 'x.com': true } }, { ...a, siteOverrides: { 'x.com': false } })).toBe(false);
  });

  it('siteOverrideFor normalises the host and distinguishes absent from false', () => {
    const s = { ...defaultSettings(), siteOverrides: { 'www.youtube.com': false } };
    expect(siteOverrideFor(s, 'WWW.YOUTUBE.COM')).toBe(false);
    expect(siteOverrideFor(s, 'example.com')).toBeUndefined();
    expect(siteOverrideFor(s, 'hasOwnProperty')).toBeUndefined();
  });
});

describe('storage helpers', () => {
  it('loadSettings sanitizes whatever is stored', async () => {
    const area = new MemoryArea();
    expect(await loadSettings(area)).toEqual(defaultSettings());
    area.data[SETTINGS_STORAGE_KEY] = { enabled: false, qualityLevel: 5, sharpness: -1 };
    expect(await loadSettings(area)).toEqual({ ...defaultSettings(), enabled: false, qualityLevel: 2, sharpness: 0 });
  });

  it('saveSettings performs read-modify-write and returns what was written', async () => {
    const area = new MemoryArea();
    area.data[SETTINGS_STORAGE_KEY] = { ...defaultSettings(), siteOverrides: { 'a.com': false } };
    const written = await saveSettings({ hudEnabled: true }, area);
    expect(written).toEqual({ ...defaultSettings(), hudEnabled: true, siteOverrides: { 'a.com': false } });
    expect(area.data[SETTINGS_STORAGE_KEY]).toEqual(written);
  });

  it('initializeSettings fills missing fields without clobbering valid stored ones', async () => {
    const area = new MemoryArea();
    area.data[SETTINGS_STORAGE_KEY] = { enabled: false, qualityLevel: 2 };
    const { settings, wrote } = await initializeSettings(area);
    expect(wrote).toBe(true);
    expect(settings).toEqual({ ...defaultSettings(), enabled: false, qualityLevel: 2 });
    expect(area.data[SETTINGS_STORAGE_KEY]).toEqual(settings);

    // Already canonical: no second write.
    const again = await initializeSettings(area);
    expect(again.wrote).toBe(false);
    expect(area.writes).toBe(1);
  });

  it('initializeSettings writes defaults into empty storage', async () => {
    const area = new MemoryArea();
    const { settings, wrote } = await initializeSettings(area);
    expect(wrote).toBe(true);
    expect(settings).toEqual(defaultSettings());
    expect(area.data[SETTINGS_STORAGE_KEY]).toEqual(defaultSettings());
  });

  it('SerialQueue keeps concurrent read-modify-writes from losing updates', async () => {
    const area = new MemoryArea();
    const queue = new SerialQueue();
    await Promise.all([
      queue.run(() => saveSettings({ enabled: false }, area)),
      queue.run(() => saveSettings({ qualityLevel: 0 }, area)),
      queue.run(() => saveSettings({ siteOverrides: { 'x.com': true } }, area)),
    ]);
    expect(await loadSettings(area)).toEqual({ ...defaultSettings(), enabled: false, qualityLevel: 0, siteOverrides: { 'x.com': true } });
  });

  it('without serialisation the same concurrent writes do lose updates (negative control)', async () => {
    const area = new MemoryArea();
    await Promise.all([saveSettings({ enabled: false }, area), saveSettings({ qualityLevel: 0 }, area)]);
    const s = await loadSettings(area);
    expect(s.enabled === false && s.qualityLevel === 0).toBe(false);
  });

  it('SerialQueue continues after a failing task', async () => {
    const queue = new SerialQueue();
    const order: number[] = [];
    const failing = queue.run(async () => {
      order.push(1);
      throw new Error('boom');
    });
    const ok = queue.run(async () => {
      order.push(2);
      return 'done';
    });
    await expect(failing).rejects.toThrow('boom');
    await expect(ok).resolves.toBe('done');
    expect(order).toEqual([1, 2]);
  });

  it('watchSettings reacts only to local-area settings changes and unsubscribes', () => {
    const listeners = new Set<(c: Record<string, StorageChangeLike>, a: string) => void>();
    const events: StorageChangedEvent = {
      addListener: (cb) => listeners.add(cb),
      removeListener: (cb) => listeners.delete(cb),
    };
    const seen: UserSettings[] = [];
    const stop = watchSettings((s) => seen.push(s), events);
    const fire = (c: Record<string, StorageChangeLike>, a: string): void => listeners.forEach((l) => l(c, a));

    fire({ [SETTINGS_STORAGE_KEY]: { newValue: { enabled: false } } }, 'session');
    fire({ other: { newValue: 1 } }, 'local');
    fire({ [SETTINGS_STORAGE_KEY]: { newValue: { enabled: false, sharpness: 2 } } }, 'local');
    expect(seen).toEqual([{ ...defaultSettings(), enabled: false, sharpness: 1 }]);

    stop();
    expect(listeners.size).toBe(0);
  });
});
