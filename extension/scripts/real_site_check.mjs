// Real-site playback check (docs/VALIDATION.md, "Method"): loads the built extension unpacked
// into installed, branded Google Chrome, opens each URL, optionally enters a player layout, plays
// for a while and prints the extension's own per-video reports from the service worker.
//
//   npm run build
//   node scripts/real_site_check.mjs [--seconds N] [--quality 0|1|2] [--headless] [--dpr N]
//                                    [--yt-quality hd720|hd1080|...] URL[#layout] ...
//
// --dpr sets the device scale factor (with a 1600x900 viewport; default: the real window).
// --yt-quality pins the YouTube player's source quality through its player API.
// layout: `theater` or `fullscreen` (YouTube keyboard shortcuts t / f), default the page's own.
// Needs a desktop session and network access; branded Chrome ignores --load-extension, so the
// extension is loaded through the CDP command Extensions.loadUnpacked.
import { chromium } from '@playwright/test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const DIST = resolve(import.meta.dirname, '../dist');
const args = process.argv.slice(2);
let seconds = 20;
let quality = null;
let headless = false;
let dpr = null;
let ytQuality = null;
const urls = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--seconds') seconds = Number(args[++i]);
  else if (args[i] === '--quality') quality = Number(args[++i]);
  else if (args[i] === '--headless') headless = true;
  else if (args[i] === '--dpr') dpr = Number(args[++i]);
  else if (args[i] === '--yt-quality') ytQuality = args[++i];
  else urls.push(args[i]);
}
if (urls.length === 0 || !(seconds > 0)) {
  console.error('usage: node scripts/real_site_check.mjs [--seconds N] [--quality 0|1|2] [--headless] URL[#theater|#fullscreen] ...');
  process.exit(2);
}

const profile = mkdtempSync(join(tmpdir(), 'webgpu-vsr-site-'));
const context = await chromium.launchPersistentContext(profile, {
  channel: 'chrome',
  headless,
  ignoreDefaultArgs: ['--disable-extensions'],
  ...(dpr ? { viewport: { width: 1600, height: 900 }, deviceScaleFactor: dpr } : { viewport: null }),
  args: ['--enable-unsafe-extension-debugging', '--autoplay-policy=no-user-gesture-required', '--window-size=1600,1000'],
});
try {
  const cdp = await context.browser()?.newBrowserCDPSession();
  if (!cdp) throw new Error('no browser CDP session');
  const { id } = await cdp.send('Extensions.loadUnpacked', { path: DIST });
  // The worker may start before or after loadUnpacked returns; poll instead of awaiting the event.
  let worker;
  for (let i = 0; i < 300 && !worker; i++) {
    worker = context.serviceWorkers().find((w) => w.url().includes(id));
    if (!worker) await new Promise((r) => setTimeout(r, 100));
  }
  if (!worker) throw new Error(`service worker of extension ${id} did not start`);
  if (quality !== null) {
    await worker.evaluate(async (q) => {
      const { settings } = await chrome.storage.local.get('settings');
      await chrome.storage.local.set({ settings: { ...settings, qualityLevel: q } });
    }, quality);
  }
  for (const entry of urls) {
    const [url, layout] = entry.split('#');
    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('video', { timeout: 30000 });
    await page.waitForTimeout(3000);
    // Consent dialogs and paused autoplay differ per site; start playback through the element.
    await page.evaluate(() => document.querySelector('video')?.play().catch(() => {}));
    if (ytQuality) {
      await page.evaluate((q) => {
        const player = document.getElementById('movie_player');
        player?.setPlaybackQualityRange?.(q, q);
      }, ytQuality);
    }
    if (layout === 'theater' || layout === 'fullscreen') {
      await page.locator('video').first().hover();
      await page.keyboard.press(layout === 'theater' ? 't' : 'f');
    }
    await page.waitForTimeout(seconds * 1000);
    const dpr = await page.evaluate(() => window.devicePixelRatio);
    const reports = await worker.evaluate(async () => Object.values(await chrome.storage.session.get(null)));
    console.log(`\n${entry} (devicePixelRatio ${dpr})`);
    for (const frame of reports.filter((r) => r && Array.isArray(r.videos))) {
      for (const v of frame.videos) {
        const t = v.telemetry;
        const line = [
          `  ${frame.host} ${v.id}: ${v.state}${v.reason ? ` (${v.reason})` : ''}`,
          `source ${v.sourceWidth}x${v.sourceHeight}`,
          `target ${v.targetWidth}x${v.targetHeight}`,
        ];
        if (t) {
          line.push(`${t.activeModel} ${t.inputWidth}x${t.inputHeight} -> ${t.outputWidth}x${t.outputHeight}`,
            `${t.fps.toFixed(1)} fps`, `GPU ${t.gpuTimeMeanMs.toFixed(1)} ms mean (${t.timingSource})`,
            `${t.droppedFrames} dropped`, `${t.presentedFrames} presented`);
        }
        if (v.error) line.push(`error ${v.error}`);
        console.log(line.join(', '));
      }
    }
    await page.close();
  }
} finally {
  await context.close();
  rmSync(profile, { recursive: true, force: true });
}
