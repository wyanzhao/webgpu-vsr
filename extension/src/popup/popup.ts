/**
 * Action popup: settings controls and the active tab's live session readout.
 *
 * Writes go through UPDATE_SETTINGS (the service worker serialises read-modify-write of
 * chrome.storage.local); content scripts pick changes up via storage.onChanged and the
 * SETTINGS_CHANGED broadcast. The tab readout polls GET_TAB_STATE once per second.
 *
 * The target tab is the active tab of the current window, or `?tabId=<id>` when the popup page is
 * opened directly (e.g. in its own tab for inspection).
 */

import { LOG_PREFIX, httpHostname } from '../shared/env';
import { sendRequest, type TabState, type VideoReport } from '../shared/messages';
import {
  QUALITY_LABELS,
  coerceQualityLevel,
  loadSettings,
  siteOverrideFor,
  watchSettings,
  type SettingsPatch,
  type UserSettings,
} from '../shared/settings';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`popup element #${id} missing`);
  return el as T;
};

const ui = {
  status: $<HTMLSpanElement>('status'),
  enabled: $<HTMLInputElement>('enabled'),
  site: $<HTMLInputElement>('site-enabled'),
  siteHost: $<HTMLSpanElement>('site-host'),
  hud: $<HTMLInputElement>('hud'),
  quality: $<HTMLSelectElement>('quality'),
  qualitySub: $<HTMLSpanElement>('quality-sub'),
  sharpness: $<HTMLInputElement>('sharpness'),
  sharpnessValue: $<HTMLOutputElement>('sharpness-value'),
  videos: $<HTMLUListElement>('videos'),
  error: $<HTMLParagraphElement>('error'),
};

const QUALITY_SUBTITLES: Record<0 | 1 | 2, string> = {
  0: 'Bicubic resampling, no network',
  1: 'Folded U-Net (16-64 ch), 2x internal',
  2: 'Folded U-Net (32-64 ch), 2x internal',
};

let settings: UserSettings | null = null;
let tabState: TabState | null = null;
let tabId: number | null = null;
let tabHost: string | null = null;
let pendingWrites = 0;
let pollTimer: number | null = null;

function host(): string | null {
  return tabHost ?? tabState?.host ?? null;
}

function showError(message: string | null): void {
  ui.error.hidden = message === null;
  ui.error.textContent = message ?? '';
}

function dims(w: number, h: number): string {
  return `${w}×${h}`;
}

function describeVideo(v: VideoReport): { title: string; detail: string } {
  const t = v.telemetry;
  switch (v.state) {
    case 'active': {
      const detail = t
        ? `${dims(t.inputWidth, t.inputHeight)} → ${dims(t.outputWidth, t.outputHeight)} · ${t.activeModel}\n` +
          `GPU ${t.gpuTimeMeanMs.toFixed(2)} ms (p95 ${t.gpuTimeP95Ms.toFixed(2)}) · ${t.fps.toFixed(1)} fps · ${t.droppedFrames} dropped · ${t.timingSource === 'timestamp' ? 'GPU timestamps' : 'CPU cadence'}`
        : `${dims(v.sourceWidth, v.sourceHeight)} → ${dims(v.targetWidth, v.targetHeight)}`;
      return { title: 'Super-resolving', detail };
    }
    case 'eligible':
      return {
        title: 'Eligible',
        detail: `${dims(v.sourceWidth, v.sourceHeight)} → ${dims(v.targetWidth, v.targetHeight)} · waiting for the GPU session`,
      };
    case 'error':
      return { title: 'GPU session failed', detail: v.error ?? 'unknown error' };
    case 'ineligible':
    default:
      return {
        title: `Not applied (check ${v.point ?? '?'} of 9)`,
        detail: `${v.reasonText ?? v.reason ?? ''}${v.sourceWidth > 0 ? ` · ${dims(v.sourceWidth, v.sourceHeight)}` : ''}`,
      };
  }
}

function renderStatus(videos: VideoReport[]): void {
  let state: string;
  let text: string;
  const h = host();
  if (!settings) {
    state = 'pending';
    text = '…';
  } else if (!settings.enabled) {
    state = 'off';
    text = 'Off';
  } else if (h && siteOverrideFor(settings, h) === false) {
    state = 'off';
    text = 'Off for site';
  } else if (videos.some((v) => v.state === 'active')) {
    state = 'active';
    text = 'Active';
  } else if (videos.some((v) => v.state === 'error')) {
    state = 'error';
    text = 'Error';
  } else if (videos.some((v) => v.state === 'eligible')) {
    state = 'ready';
    text = 'Ready';
  } else if (videos.length > 0) {
    state = 'skipped';
    text = 'Not applied';
  } else {
    state = 'off';
    text = 'No video';
  }
  ui.status.dataset.state = state;
  ui.status.textContent = text;
}

function render(): void {
  const videos = tabState ? tabState.frames.flatMap((f) => f.videos) : [];
  renderStatus(videos);
  if (settings) {
    const h = host();
    ui.enabled.checked = settings.enabled;
    ui.hud.checked = settings.hudEnabled;
    ui.site.disabled = h === null;
    ui.site.checked = h === null ? false : siteOverrideFor(settings, h) !== false;
    ui.siteHost.textContent = h ?? 'No website in this tab';
    ui.quality.value = String(settings.qualityLevel);
    ui.quality.title = QUALITY_LABELS[settings.qualityLevel];
    ui.qualitySub.textContent = QUALITY_SUBTITLES[settings.qualityLevel];
    // Do not fight the user while the slider is being dragged.
    if (document.activeElement !== ui.sharpness) ui.sharpness.value = String(settings.sharpness);
    ui.sharpnessValue.value = Number(ui.sharpness.value).toFixed(2);
  }

  const items: HTMLLIElement[] = [];
  if (tabId === null) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No active tab';
    items.push(li);
  } else if (videos.length === 0) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = tabState?.host ? 'No video found on this page' : 'This page cannot be enhanced (no content script)';
    items.push(li);
  } else {
    for (const v of videos) {
      const { title, detail } = describeVideo(v);
      const li = document.createElement('li');
      li.dataset.videoId = v.id;
      const t = document.createElement('div');
      t.className = 'video-title';
      t.dataset.state = v.state;
      t.textContent = title;
      const d = document.createElement('div');
      d.className = 'video-detail';
      d.textContent = detail;
      li.append(t, d);
      items.push(li);
    }
  }
  ui.videos.replaceChildren(...items);
}

async function update(patch: SettingsPatch): Promise<void> {
  pendingWrites++;
  try {
    const response = await sendRequest({ type: 'UPDATE_SETTINGS', patch });
    settings = response.settings;
    showError(null);
  } catch (err) {
    showError(`Could not save settings: ${err instanceof Error ? err.message : String(err)}`);
    settings = await loadSettings().catch(() => settings);
  } finally {
    pendingWrites--;
    render();
  }
}

async function refreshTabState(): Promise<void> {
  if (tabId === null) return;
  try {
    const response = await sendRequest({ type: 'GET_TAB_STATE', tabId });
    tabState = response.state;
    if (pendingWrites === 0) settings = response.settings;
    render();
  } catch (err) {
    console.warn(`${LOG_PREFIX} tab state unavailable`, err);
  }
}

async function resolveTab(): Promise<void> {
  const param = new URLSearchParams(location.search).get('tabId');
  if (param !== null && /^\d+$/.test(param)) {
    tabId = Number(param);
    try {
      const tab = await chrome.tabs.get(tabId);
      tabHost = httpHostname(tab.url);
    } catch {
      tabHost = null;
    }
    return;
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && typeof tab.id === 'number' && !(tab.url ?? '').startsWith(chrome.runtime.getURL(''))) {
    tabId = tab.id;
    tabHost = httpHostname(tab.url);
  }
}

function bind(): void {
  ui.enabled.addEventListener('change', () => void update({ enabled: ui.enabled.checked }));
  ui.hud.addEventListener('change', () => void update({ hudEnabled: ui.hud.checked }));
  ui.site.addEventListener('change', () => {
    const h = host();
    if (h) void update({ siteOverrides: { [h]: ui.site.checked } });
  });
  ui.quality.addEventListener('change', () => {
    const q = coerceQualityLevel(Number(ui.quality.value));
    if (q !== null) void update({ qualityLevel: q });
  });
  ui.sharpness.addEventListener('input', () => {
    ui.sharpnessValue.value = Number(ui.sharpness.value).toFixed(2);
  });
  ui.sharpness.addEventListener('change', () => void update({ sharpness: Number(ui.sharpness.value) }));
  watchSettings((next) => {
    if (pendingWrites > 0) return;
    settings = next;
    render();
  });
  window.addEventListener('pagehide', () => {
    if (pollTimer !== null) window.clearInterval(pollTimer);
  });
}

async function init(): Promise<void> {
  bind();
  try {
    settings = (await sendRequest({ type: 'GET_SETTINGS' })).settings;
  } catch (err) {
    console.warn(`${LOG_PREFIX} service worker unavailable, reading storage`, err);
    settings = await loadSettings();
  }
  try {
    await resolveTab();
  } catch (err) {
    console.warn(`${LOG_PREFIX} active tab unavailable`, err);
  }
  render();
  await refreshTabState();
  pollTimer = window.setInterval(() => void refreshTabState(), 1000);
  document.body.dataset.ready = 'true';
}

void init().catch((err: unknown) => {
  showError(`Popup failed to start: ${err instanceof Error ? err.message : String(err)}`);
});
