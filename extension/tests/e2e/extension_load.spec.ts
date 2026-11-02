import type { BrowserContext, ConsoleMessage, Worker } from '@playwright/test';
import { TEST_ORIGIN, expect, extensionTest as test } from './fixtures';

/** Second origin for a cross-origin (out-of-process) iframe. */
const CROSS_ORIGIN = 'http://127.0.0.1:4173';

/**
 * A playing 640x360 MediaStream video (canvas.captureStream, no codec or network needed) shown in
 * a 960x540 box: target 960x540 = 1.5x, so it passes all nine eligibility points. A second video
 * from the same stream is only 100x80 CSS px (fails point 3, too-small). Two iframes (same-origin
 * and cross-origin) each contain a video without a source (fails point 3, no-video-dimensions).
 */
const STREAM_SCRIPT = `
  const c = document.createElement('canvas');
  c.width = 640; c.height = 360;
  const g = c.getContext('2d');
  let n = 0;
  (function draw() {
    g.fillStyle = 'hsl(' + (n++ % 360) + ', 70%, 50%)';
    g.fillRect(0, 0, 640, 360);
    g.fillStyle = '#fff';
    g.fillRect((n * 4) % 600, 150, 40, 60);
    requestAnimationFrame(draw);
  })();
  const stream = c.captureStream(30);
  for (const v of document.querySelectorAll('video[data-stream]')) {
    v.muted = true;
    v.srcObject = stream;
    v.play();
  }
`;

const PAGE = `<!doctype html><html><head><meta charset="utf-8"><title>webgpu-vsr e2e</title>
<style>body{margin:0;background:#111} .player{position:relative;width:960px;height:540px} video{display:block}</style></head>
<body>
  <div class="player" id="main-player"><video id="main" data-stream style="width:960px;height:540px"></video></div>
  <div><video id="tiny" data-stream style="width:100px;height:80px"></video></div>
  <iframe id="same" src="/e2e/frame.html" width="400" height="300"></iframe>
  <iframe id="cross" src="${CROSS_ORIGIN}/e2e/frame.html" width="400" height="300"></iframe>
  <script>${STREAM_SCRIPT}</script>
</body></html>`;

const FRAME = `<!doctype html><html><body style="margin:0"><video style="width:320px;height:180px"></video></body></html>`;

async function serveE2ePages(context: BrowserContext): Promise<void> {
  for (const origin of [TEST_ORIGIN, CROSS_ORIGIN]) {
    await context.route(`${origin}/e2e/**`, async (route) => {
      const path = new URL(route.request().url()).pathname;
      const body = path === '/e2e/page.html' ? PAGE : path === '/e2e/frame.html' ? FRAME : null;
      if (body === null) await route.fulfill({ status: 404, body: 'not found' });
      else await route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body });
    });
  }
}

async function serviceWorker(context: BrowserContext): Promise<Worker> {
  return context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
}

interface StoredVideo {
  id: string;
  state: string;
  point: number | null;
  reason: string | null;
  sourceWidth: number;
  sourceHeight: number;
  targetWidth: number;
  targetHeight: number;
}
interface StoredFrame {
  frameId: number;
  host: string;
  url: string;
  videos: StoredVideo[];
}

/** Frame reports the service worker stored in chrome.storage.session for a tab. */
async function frameReports(worker: Worker, tabId: number): Promise<StoredFrame[]> {
  return worker.evaluate(async (id) => {
    const all = (await chrome.storage.session.get(null)) as Record<string, unknown>;
    return Object.entries(all)
      .filter(([k]) => k.startsWith(`tab:${id}:frame:`))
      .map(([, v]) => v as StoredFrame)
      .sort((a, b) => a.frameId - b.frameId);
  }, tabId);
}

/** Writes settings straight to chrome.storage.local, as another extension page would. */
async function patchStoredSettings(worker: Worker, patch: Record<string, unknown>): Promise<void> {
  await worker.evaluate(async (p) => {
    const { settings } = await chrome.storage.local.get('settings');
    await chrome.storage.local.set({ settings: { ...(settings as Record<string, unknown>), ...p } });
  }, patch);
}

async function tabIdFor(worker: Worker, url: string): Promise<number> {
  return worker.evaluate(async (u) => {
    const [tab] = await chrome.tabs.query({ url: u });
    if (!tab || typeof tab.id !== 'number') throw new Error(`no tab for ${u}`);
    return tab.id;
  }, url);
}

test.describe('extension load', () => {
  test('service worker starts, initialises settings and logs no errors', async ({ context, extensionId }) => {
    const worker = await serviceWorker(context);
    const errors: string[] = [];
    worker.on('console', (m: ConsoleMessage) => {
      if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text());
    });
    expect(worker.url()).toBe(`chrome-extension://${extensionId}/background.js`);

    const manifest = await worker.evaluate(() => chrome.runtime.getManifest());
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.content_scripts?.[0]?.all_frames).toBe(true);

    // onInstalled wrote the defaults (IMPLEMENTATION_NOTES section 5).
    await expect
      .poll(() => worker.evaluate(async () => (await chrome.storage.local.get('settings')).settings))
      .toEqual({ enabled: true, qualityLevel: 1, sharpness: 0.5, hudEnabled: false, siteOverrides: {} });

    // Round-trip a request through the runtime message router from an extension page.
    const page = await context.newPage();
    await page.goto(`chrome-extension://${extensionId}/popup.html`);
    const pong = await page.evaluate(() => chrome.runtime.sendMessage({ type: 'PING' }));
    expect(pong).toEqual({ type: 'PONG', version: '1.0.0', context: 'service-worker' });
    const invalid = await page.evaluate(() => chrome.runtime.sendMessage({ type: 'UPDATE_SETTINGS', patch: { sharpness: 7 } }));
    expect(invalid).toEqual({ type: 'ERROR', error: 'invalid message UPDATE_SETTINGS' });
    expect(errors).toEqual([]);
  });

  test('content script runs in every frame and reports eligibility verdicts', async ({ context, extensionId }) => {
    await serveE2ePages(context);
    const worker = await serviceWorker(context);
    const swErrors: string[] = [];
    worker.on('console', (m: ConsoleMessage) => {
      if (m.type() === 'error') swErrors.push(m.text());
    });
    const page = await context.newPage();
    const pageErrors: string[] = [];
    page.on('pageerror', (e) => pageErrors.push(e.message));
    await page.goto(`${TEST_ORIGIN}/e2e/page.html`);
    await expect.poll(() => page.evaluate(() => (document.getElementById('main') as HTMLVideoElement).videoWidth)).toBe(640);
    const tabId = await tabIdFor(worker, `${TEST_ORIGIN}/e2e/page.html`);

    // Top frame + same-origin iframe + cross-origin iframe all report.
    await expect.poll(async () => (await frameReports(worker, tabId)).length, { timeout: 15000 }).toBe(3);
    await expect
      .poll(async () => {
        const top = (await frameReports(worker, tabId)).find((f) => f.frameId === 0);
        return top?.videos.map((v) => [v.state === 'active' ? 'eligible' : v.state, v.reason]).sort();
      })
      .toEqual([
        ['eligible', null],
        ['ineligible', 'too-small'],
      ]);

    const frames = await frameReports(worker, tabId);
    const top = frames.find((f) => f.frameId === 0)!;
    expect(top.host).toBe('localhost');
    const main = top.videos.find((v) => v.reason === null)!;
    expect([main.sourceWidth, main.sourceHeight, main.targetWidth, main.targetHeight]).toEqual([640, 360, 960, 540]);
    const tiny = top.videos.find((v) => v.reason === 'too-small')!;
    expect(tiny.point).toBe(3);

    const subframes = frames.filter((f) => f.frameId !== 0);
    expect(subframes.map((f) => f.host).sort()).toEqual(['127.0.0.1', 'localhost']);
    for (const f of subframes) {
      expect(f.videos).toHaveLength(1);
      expect([f.videos[0]?.state, f.videos[0]?.point, f.videos[0]?.reason]).toEqual(['ineligible', 3, 'no-video-dimensions']);
    }

    // Disabling the extension re-runs the gate in place: point 7 now fails; badge shows OFF.
    await patchStoredSettings(worker, { enabled: false });
    await expect
      .poll(async () => (await frameReports(worker, tabId)).find((f) => f.frameId === 0)?.videos.find((v) => v.id === main.id)?.reason)
      .toBe('extension-disabled');
    await expect.poll(() => worker.evaluate((id) => chrome.action.getBadgeText({ tabId: id }), tabId)).toBe('OFF');

    // Re-enable: eligible again; the badge clears (no GPU session is active).
    await patchStoredSettings(worker, { enabled: true });
    await expect
      .poll(async () => (await frameReports(worker, tabId)).find((f) => f.frameId === 0)?.videos.find((v) => v.id === main.id)?.point)
      .toBeNull();
    await expect.poll(() => worker.evaluate((id) => chrome.action.getBadgeText({ tabId: id }), tabId)).not.toBe('OFF');

    // HUD: anchored inside the player box, text-only, non-interactive.
    await patchStoredSettings(worker, { hudEnabled: true });
    const hud = page.locator('#main-player .webgpu-vsr-hud .panel');
    await expect(hud).toBeVisible();
    await expect(hud).toContainText('WebGPU VSR');
    await expect(hud).toContainText('640×360');
    expect(await hud.evaluate((el) => getComputedStyle(el).pointerEvents)).toBe('none');
    const [hudBox, videoBox] = await Promise.all([hud.boundingBox(), page.locator('#main').boundingBox()]);
    expect(hudBox && videoBox).toBeTruthy();
    // top-right corner of the video, 8px margin
    expect(Math.abs(hudBox!.x + hudBox!.width - (videoBox!.x + videoBox!.width - 8))).toBeLessThan(1);
    expect(Math.abs(hudBox!.y - (videoBox!.y + 8))).toBeLessThan(1);
    await expect(page.locator('#tiny + .webgpu-vsr-hud')).toHaveCount(0); // too small: no HUD

    // The popup shows the tab's live verdicts via GET_TAB_STATE.
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup.html?tabId=${tabId}`);
    await expect(popup.locator('#site-host')).toHaveText('localhost');
    await expect(popup.locator('#videos li')).toHaveCount(4);
    await expect(popup.locator('#videos')).toContainText('Not applied (check 3 of 9)');
    await expect(popup.locator('#videos')).toContainText('Video is smaller than 128 x 128 CSS px');

    // Removing the main video: it leaves the tab state at once (detached videos are not
    // reported) and its HUD is hidden, but the session is only destroyed after the 5 s grace period.
    const removedAt = Date.now();
    await page.evaluate(() => document.getElementById('main')!.remove());
    await expect
      .poll(async () => (await frameReports(worker, tabId)).find((f) => f.frameId === 0)?.videos.some((v) => v.id === main.id))
      .toBe(false);
    await expect(hud).toBeHidden();
    await expect(hud).toHaveCount(1);
    await expect(hud).toHaveCount(0, { timeout: 15000 });
    expect(Date.now() - removedAt).toBeGreaterThanOrEqual(4500);

    expect(pageErrors).toEqual([]);
    expect(swErrors).toEqual([]);
  });

  test('popup page renders with every control', async ({ context, extensionId }) => {
    const page = await context.newPage();
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(m.text());
    });
    await page.goto(`chrome-extension://${extensionId}/popup.html`);
    await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
    await expect(page).toHaveTitle('WebGPU VSR');
    await expect(page.locator('#enabled')).toBeChecked();
    await expect(page.locator('#hud')).not.toBeChecked();
    await expect(page.locator('#quality')).toHaveValue('1');
    await expect(page.locator('#quality option')).toHaveText(['Medium', 'Low', 'Quality 0 · Catmull-Rom Bicubic']);
    await expect(page.locator('#sharpness')).toHaveValue('0.5');
    await expect(page.locator('#status')).not.toHaveText('…');
    const logo = await page.locator('.logo').evaluate((img: HTMLImageElement) => img.naturalWidth);
    expect(logo).toBe(48);
    expect(errors).toEqual([]);
  });
});
