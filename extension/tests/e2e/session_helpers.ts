/**
 * Shared helpers for the GPU session e2e specs (video_overlay, downgrade): test pages around the
 * VP9 clips in tests/media, settings / diagnostics writes through the service worker, and reads of
 * the per-tab reports the service worker stores in chrome.storage.session.
 */
import type { BrowserContext, Frame, Page, Worker } from '@playwright/test';
import { TEST_ORIGIN, expect } from './fixtures';
import { HIDDEN_OBJECT_POSITION } from '../../src/content/overlay';
import type { SessionTelemetry, VideoReport } from '../../src/shared/messages';

export type Clip = 'clip_540p30.webm' | 'clip_720p60.webm' | 'clip_scope_24.webm';

export interface PlayerSpec {
  id: string;
  /** CSS box of the player container (the video fills it with object-fit: contain). */
  width: number;
  height: number;
  /** Clip to play inside this player; none for an empty container. */
  clip?: Clip;
  /** Inline style of the video element (e.g. an original object-position). */
  videoStyle?: string;
  loop?: boolean;
  controls?: boolean;
  /** /media path of a default English subtitle track. */
  captions?: string;
}

/** A page with player containers laid out left to right; videos are muted and autoplay. */
export function playerPage(players: PlayerSpec[]): string {
  const blocks = players
    .map((p) => {
      const video = p.clip
        ? `<video id="${p.id}-video" src="/media/${p.clip}" muted autoplay playsinline ${p.loop ? 'loop' : ''} ${p.controls ? 'controls' : ''} style="display:block;width:100%;height:100%;${p.videoStyle ?? ''}">${p.captions ? `<track default kind="subtitles" srclang="en" src="${p.captions}">` : ''}</video>`
        : '';
      return `<div class="player" id="${p.id}" style="width:${p.width}px;height:${p.height}px">${video}</div>`;
    })
    .join('');
  return `<!doctype html><html><head><meta charset="utf-8"><title>webgpu-vsr session</title>
<style>body{margin:0;background:#000;display:flex;align-items:flex-start} .player{position:relative;background:#202020;flex:none}</style></head>
<body>${blocks}</body></html>`;
}

/** Whether the video picture is hidden while its native controls and captions remain visible. */
export async function pictureHidden(page: Page | Frame): Promise<boolean> {
  return page.evaluate((hiddenPosition) => {
    const style = document.querySelector('video')!.style;
    return style.getPropertyValue('object-position') === hiddenPosition && style.getPropertyPriority('object-position') === 'important';
  }, HIDDEN_OBJECT_POSITION);
}

/** Serves `html` at TEST_ORIGIN/session/<name>.html and returns the URL. */
export async function servePage(context: BrowserContext, name: string, html: string): Promise<string> {
  const path = `/session/${name}.html`;
  await context.route(`${TEST_ORIGIN}${path}`, (route) => route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html }));
  return `${TEST_ORIGIN}${path}`;
}

export async function serviceWorker(context: BrowserContext): Promise<Worker> {
  return context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
}

/** Writes a settings patch straight to chrome.storage.local (as the popup's service worker does). */
export async function patchSettings(worker: Worker, patch: Record<string, unknown>): Promise<void> {
  await expect.poll(() => worker.evaluate(async () => (await chrome.storage.local.get('settings')).settings !== undefined)).toBe(true);
  await worker.evaluate(async (p) => {
    const { settings } = await chrome.storage.local.get('settings');
    await chrome.storage.local.set({ settings: { ...(settings as Record<string, unknown>), ...p } });
  }, patch);
}

/** Writes the developer diagnostics key (src/shared/diagnostics.ts). */
export async function setDiagnostics(worker: Worker, value: unknown): Promise<void> {
  await worker.evaluate(async (v) => chrome.storage.local.set({ diagnostics: v }), value);
}

export async function tabIdFor(worker: Worker, url: string): Promise<number> {
  return worker.evaluate(async (u) => {
    const [tab] = await chrome.tabs.query({ url: u });
    if (!tab || typeof tab.id !== 'number') throw new Error(`no tab for ${u}`);
    return tab.id;
  }, url);
}

/** The top frame's video reports as stored by the service worker. */
export async function topFrameVideos(worker: Worker, tabId: number): Promise<VideoReport[]> {
  return worker.evaluate(async (id) => {
    const key = `tab:${id}:frame:0`;
    const stored = (await chrome.storage.session.get(key)) as Record<string, { videos: VideoReport[] } | undefined>;
    return stored[key]?.videos ?? [];
  }, tabId);
}

/** The single top-frame report (asserts there is exactly one video). */
export async function onlyVideo(worker: Worker, tabId: number): Promise<VideoReport | null> {
  const videos = await topFrameVideos(worker, tabId);
  return videos.length === 1 ? (videos[0] as VideoReport) : null;
}

/** Waits until the video is active with telemetry matching `accept`, and returns that telemetry. */
export async function waitForTelemetry(
  worker: Worker,
  tabId: number,
  accept: (t: SessionTelemetry) => boolean,
  timeout = 20000,
): Promise<SessionTelemetry> {
  let found: SessionTelemetry | null = null;
  await expect
    .poll(
      async () => {
        const v = await onlyVideo(worker, tabId);
        if (v?.state === 'active' && v.telemetry && accept(v.telemetry)) found = v.telemetry;
        return found !== null;
      },
      { timeout, intervals: [100] },
    )
    .toBe(true);
  return found as unknown as SessionTelemetry;
}

/** Opens a page with a fixed viewport and collects page errors. */
export async function openPage(context: BrowserContext, url: string, viewport = { width: 1600, height: 1000 }): Promise<{ page: Page; errors: string[] }> {
  const page = await context.newPage();
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  await page.setViewportSize(viewport);
  await page.goto(url);
  return { page, errors };
}

/** Name of the content script's isolated world (the extension name in the manifest). */
export const EXTENSION_WORLD = 'WebGPU VSR - Video Super Resolution';

/**
 * Opens a page and returns, besides the page, an evaluator for the content script's isolated
 * world (reachable through CDP only): test-only access to the extension's globals.
 */
export async function openPageWithExtensionWorld(
  context: BrowserContext,
  url: string,
  viewport = { width: 1600, height: 1000 },
): Promise<{ page: Page; errors: string[]; inExtension: <T>(expression: string) => Promise<T> }> {
  const page = await context.newPage();
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  const client = await context.newCDPSession(page);
  const worlds: { id: number; name: string }[] = [];
  client.on('Runtime.executionContextCreated', (e) => worlds.push(e.context));
  await client.send('Runtime.enable');
  await page.setViewportSize(viewport);
  await page.goto(url);
  const inExtension = async <T>(expression: string): Promise<T> => {
    await expect.poll(() => worlds.some((w) => w.name === EXTENSION_WORLD)).toBe(true);
    const world = worlds.filter((w) => w.name === EXTENSION_WORLD).pop()!;
    const r = await client.send('Runtime.evaluate', { contextId: world.id, returnByValue: true, expression });
    if (r.exceptionDetails) throw new Error(`extension world: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
    return r.result.value as T;
  };
  return { page, errors, inExtension };
}

/**
 * Isolated-world instrumentation of VideoFrame: counts frames constructed through the global
 * constructor (which the session uses for every requestVideoFrameCallback) and their first
 * close(). `globalThis.__webgpuVsrFrames` = { created, closed, maxOpen } where maxOpen is the
 * largest number of unclosed frames seen when a new one was created.
 */
export const COUNT_VIDEO_FRAMES = `(() => {
  if (globalThis.__webgpuVsrFrames) return true;
  const Native = globalThis.VideoFrame;
  const stats = { created: 0, closed: 0, maxOpen: 0 };
  const open = new WeakSet();
  const close = Native.prototype.close;
  Native.prototype.close = function () {
    if (open.has(this)) { open.delete(this); stats.closed++; }
    return close.call(this);
  };
  globalThis.VideoFrame = class extends Native {
    constructor(...args) {
      super(...args);
      stats.maxOpen = Math.max(stats.maxOpen, stats.created - stats.closed);
      stats.created++;
      open.add(this);
    }
  };
  globalThis.__webgpuVsrFrames = stats;
  return true;
})()`;

/**
 * Page-world statistics helper (installed with addInitScript): `__webgpuVsrLuma(source, w, h, gx, gy)`
 * draws an image source (video or the overlay canvas) into a 2D canvas of w x h and returns the
 * mean BT.601 luma of each cell of a gx x gy grid, plus the global mean and standard deviation.
 */
export const LUMA_HELPER = `
window.__webgpuVsrLuma = (source, w, h, gx, gy) => {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(source, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h).data;
  const cells = new Array(gx * gy).fill(0);
  const counts = new Array(gx * gy).fill(0);
  let sum = 0, sq = 0;
  for (let y = 0; y < h; y++) {
    const cy = Math.min(gy - 1, Math.floor((y * gy) / h));
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      const cell = cy * gx + Math.min(gx - 1, Math.floor((x * gx) / w));
      cells[cell] += l; counts[cell]++;
      sum += l; sq += l * l;
    }
  }
  const n = w * h;
  const mean = sum / n;
  return { cells: cells.map((v, i) => v / counts[i]), mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)) };
};
`;

/** Pearson correlation of two equally long vectors. */
export function correlation(a: number[], b: number[]): number {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = (a[i] as number) - ma;
    const db = (b[i] as number) - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  return cov / Math.sqrt(va * vb);
}
