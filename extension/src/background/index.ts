/**
 * MV3 service worker.
 *
 * Stateless by design: Chrome terminates the worker after ~30 s idle, so nothing lives in globals.
 * - settings: chrome.storage.local (initialised on install without clobbering existing values);
 * - per-tab session telemetry: chrome.storage.session, one key per frame
 *   (`tab:<tabId>:frame:<frameId>`), so concurrent frame reports never race and the data
 *   survives worker restarts (it is cleared with the browser session, on navigation and on tab
 *   removal);
 * - toolbar badge per tab (see badgeFor in shared/messages.ts).
 */

import { LOG_PREFIX, httpHostname, isBenignMessagingError } from '../shared/env';
import {
  badgeFor,
  isFrameReport,
  isRequestMessage,
  tabStateFromFrames,
  type ErrorMessage,
  type FrameReport,
  type RequestMessage,
  type ResponseMessage,
  type SettingsChangedMessage,
  type TabState,
} from '../shared/messages';
import {
  SETTINGS_STORAGE_KEY,
  SerialQueue,
  initializeSettings,
  loadSettings,
  sanitizeSettings,
  saveSettings,
  type UserSettings,
} from '../shared/settings';

const settingsWrites = new SerialQueue();

const frameKey = (tabId: number, frameId: number): string => `tab:${tabId}:frame:${frameId}`;
const tabPrefix = (tabId: number): string => `tab:${tabId}:frame:`;
const FRAME_KEY_RE = /^tab:(\d+):frame:(\d+)$/;

async function allSessionEntries(): Promise<Record<string, unknown>> {
  return (await chrome.storage.session.get(null)) as Record<string, unknown>;
}

async function readTabState(tabId: number): Promise<TabState> {
  const all = await allSessionEntries();
  const prefix = tabPrefix(tabId);
  const frames = Object.entries(all)
    .filter(([key]) => key.startsWith(prefix))
    .map(([, value]) => value);
  return tabStateFromFrames(tabId, frames);
}

async function clearTab(tabId: number): Promise<void> {
  const all = await allSessionEntries();
  const keys = Object.keys(all).filter((k) => k.startsWith(tabPrefix(tabId)));
  if (keys.length > 0) await chrome.storage.session.remove(keys);
}

async function applyBadge(tabId: number, settings: UserSettings, state: TabState | null): Promise<void> {
  const badge = badgeFor(settings, state);
  try {
    await chrome.action.setBadgeText({ tabId, text: badge.text });
    if (badge.text) await chrome.action.setBadgeBackgroundColor({ tabId, color: badge.color });
  } catch (err) {
    if (!isBenignMessagingError(err)) console.warn(`${LOG_PREFIX} badge update failed`, err);
  }
}

/** Recomputes the default badge and every open tab's badge from storage. */
async function refreshAllBadges(settings?: UserSettings): Promise<void> {
  const effective = settings ?? (await loadSettings());
  await chrome.action.setBadgeText({ text: effective.enabled ? '' : 'OFF' });
  await chrome.action.setBadgeBackgroundColor({ color: badgeFor(effective, null).color });
  const [tabs, all] = await Promise.all([chrome.tabs.query({}), allSessionEntries()]);
  const byTab = new Map<number, unknown[]>();
  for (const [key, value] of Object.entries(all)) {
    const m = FRAME_KEY_RE.exec(key);
    if (!m) continue;
    const tabId = Number(m[1]);
    const list = byTab.get(tabId) ?? [];
    list.push(value);
    byTab.set(tabId, list);
  }
  await Promise.all(
    tabs
      .filter((t): t is chrome.tabs.Tab & { id: number } => typeof t.id === 'number' && t.id >= 0)
      .map((t) => applyBadge(t.id, effective, tabStateFromFrames(t.id, byTab.get(t.id) ?? []))),
  );
}

async function broadcastSettings(settings: UserSettings): Promise<void> {
  const message: SettingsChangedMessage = { type: 'SETTINGS_CHANGED', settings };
  const tabs = await chrome.tabs.query({});
  await Promise.all(
    tabs.map(async (tab) => {
      if (typeof tab.id !== 'number' || tab.id < 0) return;
      try {
        await chrome.tabs.sendMessage(tab.id, message);
      } catch (err) {
        // Tabs without our content script (chrome://, the Web Store, PDF viewer, discarded tabs).
        if (!isBenignMessagingError(err)) console.warn(`${LOG_PREFIX} settings broadcast failed`, err);
      }
    }),
  );
}

function isExtensionPage(sender: chrome.runtime.MessageSender): boolean {
  const origin = chrome.runtime.getURL('');
  return typeof sender.url === 'string' && sender.url.startsWith(origin);
}

async function handleRequest(message: RequestMessage, sender: chrome.runtime.MessageSender): Promise<ResponseMessage> {
  switch (message.type) {
    case 'PING':
      return { type: 'PONG', version: chrome.runtime.getManifest().version, context: 'service-worker' };

    case 'GET_SETTINGS':
      return { type: 'GET_SETTINGS_RESPONSE', settings: await loadSettings() };

    case 'UPDATE_SETTINGS': {
      if (!isExtensionPage(sender)) return { type: 'ERROR', error: 'UPDATE_SETTINGS is only accepted from extension pages' };
      const settings = await settingsWrites.run(() => saveSettings(message.patch));
      return { type: 'GET_SETTINGS_RESPONSE', settings };
    }

    case 'GET_TAB_STATE': {
      if (!isExtensionPage(sender)) return { type: 'ERROR', error: 'GET_TAB_STATE is only accepted from extension pages' };
      const [state, settings] = await Promise.all([readTabState(message.tabId), loadSettings()]);
      return { type: 'TAB_STATE', state, settings };
    }

    case 'GET_FRAME_CONTEXT': {
      // A sub-frame's content script cannot read its tab's URL (cross-origin): the per-site
      // switch names the tab's top-level site, which the gate applies to every frame.
      if (!sender.tab) return { type: 'ERROR', error: 'GET_FRAME_CONTEXT must come from a content script in a tab' };
      return { type: 'FRAME_CONTEXT', topHost: httpHostname(sender.tab.url) };
    }

    case 'TELEMETRY_UPDATE': {
      const tabId = sender.tab?.id;
      const frameId = sender.frameId;
      if (typeof tabId !== 'number' || tabId < 0 || typeof frameId !== 'number' || frameId < 0) {
        return { type: 'ERROR', error: 'TELEMETRY_UPDATE must come from a content script in a tab' };
      }
      const key = frameKey(tabId, frameId);
      const report: FrameReport = { frameId, host: message.host, url: message.url, videos: message.videos, updatedAt: Date.now() };
      if (!isFrameReport(report)) return { type: 'ERROR', error: 'invalid frame report' };
      if (report.videos.length === 0 && frameId !== 0) await chrome.storage.session.remove(key);
      else await chrome.storage.session.set({ [key]: report });
      const [state, settings] = await Promise.all([readTabState(tabId), loadSettings()]);
      await applyBadge(tabId, settings, state);
      return { type: 'ACK' };
    }
  }
}

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) return false;
  if (!isRequestMessage(message)) {
    // Broadcasts (SETTINGS_CHANGED) and responses are not for us; reject malformed requests.
    const type = typeof message === 'object' && message !== null ? (message as { type?: unknown }).type : undefined;
    if (type === 'SETTINGS_CHANGED') return false;
    const error: ErrorMessage = { type: 'ERROR', error: `invalid message${typeof type === 'string' ? ` ${type}` : ''}` };
    sendResponse(error);
    return false;
  }
  handleRequest(message, sender).then(
    (response) => sendResponse(response),
    (err: unknown) => {
      console.error(`${LOG_PREFIX} ${message.type} failed`, err);
      const error: ErrorMessage = { type: 'ERROR', error: err instanceof Error ? err.message : String(err) };
      sendResponse(error);
    },
  );
  return true; // async response
});

chrome.runtime.onInstalled.addListener(() => {
  void initializeSettings()
    .then(({ settings }) => refreshAllBadges(settings))
    .catch((err: unknown) => console.error(`${LOG_PREFIX} install initialisation failed`, err));
});

chrome.runtime.onStartup.addListener(() => {
  void refreshAllBadges().catch((err: unknown) => console.error(`${LOG_PREFIX} startup badge refresh failed`, err));
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== 'local' || !changes[SETTINGS_STORAGE_KEY]) return;
  const settings = sanitizeSettings(changes[SETTINGS_STORAGE_KEY].newValue);
  void Promise.all([broadcastSettings(settings), refreshAllBadges(settings)]).catch((err: unknown) =>
    console.error(`${LOG_PREFIX} settings propagation failed`, err),
  );
});

chrome.tabs.onRemoved.addListener((tabId) => {
  void clearTab(tabId).catch((err: unknown) => console.error(`${LOG_PREFIX} tab cleanup failed`, err));
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  // A new document is loading in the tab: drop every frame report of the previous document.
  // The new document's content scripts report afresh once they start.
  if (changeInfo.status !== 'loading' || changeInfo.url === undefined) return;
  void clearTab(tabId)
    .then(async () => applyBadge(tabId, await loadSettings(), null))
    .catch((err: unknown) => console.error(`${LOG_PREFIX} navigation cleanup failed`, err));
});
