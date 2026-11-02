/**
 * User settings: schema, defaults, validation and chrome.storage.local persistence.
 *
 * Settings live only in `chrome.storage.local` under one key ({@link SETTINGS_STORAGE_KEY}).
 * Nothing here caches state in memory, so every reader (service worker after a restart,
 * popup, content scripts) sees the persisted value. Stored values are treated as untrusted:
 * every read goes through {@link sanitizeSettings}.
 */

/** 0 = Quality 0 (Catmull-Rom bicubic), 1 = RTX VSR low (folded), 2 = RTX VSR medium (folded). */
export type QualityLevel = 0 | 1 | 2;

export interface UserSettings {
  enabled: boolean;
  qualityLevel: QualityLevel;
  /** Residual gain control in [0, 1]; 0.5 is the network's native output (gain 2*sharpness). */
  sharpness: number;
  hudEnabled: boolean;
  /** Per-hostname explicit enable (true) / disable (false). Absent = follow the site rule. */
  siteOverrides: Record<string, boolean>;
}

/**
 * A validated partial update. `siteOverrides` entries are merged per host; a `null` value removes
 * the override for that host (the site then follows its rule again).
 */
export interface SettingsPatch {
  enabled?: boolean;
  qualityLevel?: QualityLevel;
  sharpness?: number;
  hudEnabled?: boolean;
  siteOverrides?: Record<string, boolean | null>;
}

export const SETTINGS_STORAGE_KEY = 'settings';

export const DEFAULT_SETTINGS: Readonly<UserSettings> = Object.freeze({
  enabled: true,
  qualityLevel: 1,
  sharpness: 0.5,
  hudEnabled: false,
  siteOverrides: Object.freeze({}) as Record<string, boolean>,
});

/** Upper bound on stored overrides so a corrupted or hostile value cannot grow without limit. */
export const MAX_SITE_OVERRIDES = 2000;

export const QUALITY_LABELS: Readonly<Record<QualityLevel, string>> = Object.freeze({
  0: 'Basic · Catmull-Rom bicubic',
  1: 'RTX VSR Low',
  2: 'RTX VSR Medium',
});

/** Model / pipeline identifier used by the GPU pipelines for each quality level. */
export const QUALITY_PIPELINE_IDS: Readonly<Record<QualityLevel, string>> = Object.freeze({
  0: 'bicubic-q0',
  1: 'rtx-vsr-low',
  2: 'rtx-vsr-medium',
});

export function defaultSettings(): UserSettings {
  return { ...DEFAULT_SETTINGS, siteOverrides: {} };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Canonical hostname key: lower-case, no trailing dot, no port, max 253 chars and only characters
 * valid in a DNS name or an IPv6 literal. Returns null for anything else.
 */
export function normalizeHostname(host: unknown): string | null {
  if (typeof host !== 'string') return null;
  let h = host.trim().toLowerCase();
  if (h.endsWith('.')) h = h.slice(0, -1);
  if (h.length === 0 || h.length > 253) return null;
  if (/^\[[0-9a-f:.]+\]$/.test(h)) return h;
  if (!/^[a-z0-9_-]+(\.[a-z0-9_-]+)*$/.test(h)) return null;
  return h;
}

/** Coerces a quality value; non-integers are rounded, out-of-range values clamped, junk -> null. */
export function coerceQualityLevel(value: unknown): QualityLevel | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const q = Math.min(2, Math.max(0, Math.round(value)));
  return q as QualityLevel;
}

/** Clamps sharpness into [0, 1]; non-finite or non-number -> null. */
export function coerceSharpness(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return Math.min(1, Math.max(0, value));
}

function sanitizeOverrides(raw: unknown): Record<string, boolean> {
  const out: Record<string, boolean> = {};
  if (!isRecord(raw)) return out;
  let count = 0;
  for (const [key, value] of Object.entries(raw)) {
    if (count >= MAX_SITE_OVERRIDES) break;
    const host = normalizeHostname(key);
    if (host === null || typeof value !== 'boolean') continue;
    out[host] = value;
    count++;
  }
  return out;
}

/** Turns any stored value into valid settings, falling back to the default per field. */
export function sanitizeSettings(raw: unknown): UserSettings {
  const base = defaultSettings();
  if (!isRecord(raw)) return base;
  return {
    enabled: typeof raw.enabled === 'boolean' ? raw.enabled : base.enabled,
    qualityLevel: coerceQualityLevel(raw.qualityLevel) ?? base.qualityLevel,
    sharpness: coerceSharpness(raw.sharpness) ?? base.sharpness,
    hudEnabled: typeof raw.hudEnabled === 'boolean' ? raw.hudEnabled : base.hudEnabled,
    siteOverrides: sanitizeOverrides(raw.siteOverrides),
  };
}

/**
 * Validates an untrusted patch. Unknown keys and invalid values are dropped; returns null when the
 * input is not an object at all.
 */
export function sanitizeSettingsPatch(raw: unknown): SettingsPatch | null {
  if (!isRecord(raw)) return null;
  const patch: SettingsPatch = {};
  if (typeof raw.enabled === 'boolean') patch.enabled = raw.enabled;
  const q = coerceQualityLevel(raw.qualityLevel);
  if (q !== null) patch.qualityLevel = q;
  const s = coerceSharpness(raw.sharpness);
  if (s !== null) patch.sharpness = s;
  if (typeof raw.hudEnabled === 'boolean') patch.hudEnabled = raw.hudEnabled;
  if (isRecord(raw.siteOverrides)) {
    const overrides: Record<string, boolean | null> = {};
    let any = false;
    for (const [key, value] of Object.entries(raw.siteOverrides)) {
      const host = normalizeHostname(key);
      if (host === null) continue;
      if (typeof value === 'boolean' || value === null) {
        overrides[host] = value;
        any = true;
      }
    }
    if (any) patch.siteOverrides = overrides;
  }
  return patch;
}

/** Applies a (validated or untrusted) patch to settings and returns new sanitized settings. */
export function applySettingsPatch(current: UserSettings, rawPatch: unknown): UserSettings {
  const base = sanitizeSettings(current);
  const patch = sanitizeSettingsPatch(rawPatch);
  if (!patch) return base;
  const siteOverrides = { ...base.siteOverrides };
  if (patch.siteOverrides) {
    for (const [host, value] of Object.entries(patch.siteOverrides)) {
      if (value === null) delete siteOverrides[host];
      else siteOverrides[host] = value;
    }
  }
  return sanitizeSettings({
    enabled: patch.enabled ?? base.enabled,
    qualityLevel: patch.qualityLevel ?? base.qualityLevel,
    sharpness: patch.sharpness ?? base.sharpness,
    hudEnabled: patch.hudEnabled ?? base.hudEnabled,
    siteOverrides,
  });
}

export function settingsEqual(a: UserSettings, b: UserSettings): boolean {
  if (
    a.enabled !== b.enabled ||
    a.qualityLevel !== b.qualityLevel ||
    a.sharpness !== b.sharpness ||
    a.hudEnabled !== b.hudEnabled
  ) {
    return false;
  }
  const ak = Object.keys(a.siteOverrides);
  const bk = Object.keys(b.siteOverrides);
  if (ak.length !== bk.length) return false;
  return ak.every((k) => b.siteOverrides[k] === a.siteOverrides[k]);
}

/** Explicit user override for `host`, or undefined when the site follows its rule. */
export function siteOverrideFor(settings: UserSettings, host: string): boolean | undefined {
  const key = normalizeHostname(host);
  if (key === null) return undefined;
  return Object.prototype.hasOwnProperty.call(settings.siteOverrides, key) ? settings.siteOverrides[key] : undefined;
}

/** Minimal storage surface (chrome.storage.local satisfies it); injectable for tests. */
export interface SettingsStorageArea {
  get(keys: string): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
}

export interface StorageChangeLike {
  oldValue?: unknown;
  newValue?: unknown;
}

/** Minimal chrome.storage.onChanged surface. */
export interface StorageChangedEvent {
  addListener(cb: (changes: Record<string, StorageChangeLike>, areaName: string) => void): void;
  removeListener(cb: (changes: Record<string, StorageChangeLike>, areaName: string) => void): void;
}

function localArea(): SettingsStorageArea {
  return chrome.storage.local as unknown as SettingsStorageArea;
}

export async function loadSettings(area: SettingsStorageArea = localArea()): Promise<UserSettings> {
  const stored = await area.get(SETTINGS_STORAGE_KEY);
  return sanitizeSettings(stored[SETTINGS_STORAGE_KEY]);
}

/**
 * Read-modify-write of a patch. Returns the settings that were written. Callers that may race
 * (the service worker handling several UPDATE_SETTINGS) serialise calls with {@link SerialQueue}.
 */
export async function saveSettings(patch: unknown, area: SettingsStorageArea = localArea()): Promise<UserSettings> {
  const current = await loadSettings(area);
  const next = applySettingsPatch(current, patch);
  await area.set({ [SETTINGS_STORAGE_KEY]: next });
  return next;
}

/**
 * Install-time initialisation: keeps every valid stored field, fills missing/invalid ones with
 * defaults, and writes only when the stored value is missing or differs from its sanitized form.
 * Returns the effective settings and whether a write happened.
 */
export async function initializeSettings(
  area: SettingsStorageArea = localArea(),
): Promise<{ settings: UserSettings; wrote: boolean }> {
  const stored = await area.get(SETTINGS_STORAGE_KEY);
  const raw = stored[SETTINGS_STORAGE_KEY];
  const settings = sanitizeSettings(raw);
  const needsWrite = !isRecord(raw) || JSON.stringify(raw) !== JSON.stringify(settings);
  if (needsWrite) await area.set({ [SETTINGS_STORAGE_KEY]: settings });
  return { settings, wrote: needsWrite };
}

/**
 * Calls `onChange` with sanitized settings whenever the stored settings change in the local area.
 * Returns an unsubscribe function.
 */
export function watchSettings(
  onChange: (settings: UserSettings) => void,
  events: StorageChangedEvent = chrome.storage.onChanged as unknown as StorageChangedEvent,
): () => void {
  const listener = (changes: Record<string, StorageChangeLike>, areaName: string): void => {
    if (areaName !== 'local') return;
    const change = changes[SETTINGS_STORAGE_KEY];
    if (!change) return;
    onChange(sanitizeSettings(change.newValue));
  };
  events.addListener(listener);
  return () => events.removeListener(listener);
}

/** Runs async tasks strictly one after another (used for read-modify-write of settings). */
export class SerialQueue {
  private tail: Promise<unknown> = Promise.resolve();

  run<T>(task: () => Promise<T>): Promise<T> {
    const result = this.tail.then(task, task);
    this.tail = result.catch(() => undefined);
    return result;
  }
}
