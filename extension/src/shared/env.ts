/**
 * Runtime environment helpers shared by the service worker, popup and content scripts.
 *
 * The extension targets Chrome MV3 only, so `chrome.*` promise APIs are used directly; these
 * helpers exist to make the "am I running inside a live extension context?" question explicit
 * (an orphaned content script loses `chrome.runtime.id` after the extension is reloaded) and to
 * keep extension-context error handling in one place.
 */

export const LOG_PREFIX = '[webgpu-vsr]';

/** True when `chrome.runtime` is reachable and the extension context has not been invalidated. */
export function hasExtensionContext(): boolean {
  try {
    return typeof chrome !== 'undefined' && typeof chrome.runtime?.id === 'string' && chrome.runtime.id.length > 0;
  } catch {
    return false;
  }
}

/** Errors Chrome raises when the other end of a message is missing; these are expected, not bugs. */
export function isBenignMessagingError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return (
    message.includes('Receiving end does not exist') ||
    message.includes('Could not establish connection') ||
    message.includes('message port closed') ||
    message.includes('Extension context invalidated') ||
    message.includes('No tab with id') ||
    message.includes('The page keeping the extension port is moved into back/forward cache')
  );
}

/** Resolves an extension-packaged resource path (e.g. `sites/youtube.json`) to its URL. */
export function extensionUrl(path: string): string {
  return chrome.runtime.getURL(path);
}

/** WebGPU entry point presence (secure contexts only). Does not request an adapter. */
export function hasWebGpuApi(nav: { gpu?: unknown } | undefined = globalThis.navigator as { gpu?: unknown } | undefined): boolean {
  return typeof nav === 'object' && nav !== null && typeof nav.gpu === 'object' && nav.gpu !== null;
}

/** Hostname of an http(s) URL, or null for other schemes and malformed URLs. */
export function httpHostname(url: string | undefined | null): string | null {
  if (!url) return null;
  try {
    const u = new URL(url);
    return u.protocol === 'http:' || u.protocol === 'https:' ? u.hostname : null;
  } catch {
    return null;
  }
}
