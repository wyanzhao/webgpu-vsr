/**
 * Automatic downgrade: synthetic GPU latency, injected through the developer diagnostics key
 * (src/shared/diagnostics.ts), makes the budget controller see late frames. The session must
 * step Medium -> Low -> Quality 0 without a gap in presentation, throttle (gate point 9) when even
 * Quality 0 is over budget, keep its tier under paced playback and through main-thread stalls, and
 * keep a 60 fps clip playing with every VideoFrame closed.
 */
import { HIDDEN_OBJECT_POSITION } from '../../src/content/overlay';
import { extensionTest as test, expect } from './fixtures';
import type { SessionTelemetry } from '../../src/shared/messages';
import {
  COUNT_VIDEO_FRAMES,
  onlyVideo,
  openPage,
  openPageWithExtensionWorld,
  patchSettings,
  playerPage,
  pictureHidden,
  servePage,
  serviceWorker,
  setDiagnostics,
  tabIdFor,
  waitForTelemetry,
} from './session_helpers';

/**
 * Page-world recorder:
 * - `models`: the HUD's model line in order of appearance;
 * - `restores` / `seeks`: times the video's picture left its hidden state, and `seeking` events (loop
 *   restarts restore the video by design);
 * - `frames`: one sample per video frame of what the overlay canvas shows. The sampler's
 *   requestVideoFrameCallback is registered when the session first hides the video, i.e. after the
 *   session's own callback, so it runs in the same rendering step right after the session encoded
 *   the frame and reads that frame (32x18 thumbnail). `lit` = not black, `changed` = differs from
 *   the previous sample (the test pattern moves every frame).
 */
const RECORDER = `
window.__rec = { models: [], restores: [], seeks: [], frames: [] };
let currentModel = null;
const hudRoots = new WeakSet();
function recordHudModel() {
  const hud = document.querySelector('.webgpu-vsr-hud');
  const root = hud?.shadowRoot;
  if (!root) return;
  if (!hudRoots.has(root)) {
    hudRoots.add(root);
    new MutationObserver(recordHudModel).observe(root, { subtree: true, childList: true, characterData: true });
  }
  const line = Array.from(root.querySelectorAll('.panel > div')).map((c) => c.textContent || '').find((t) => t.startsWith('Model '));
  if (line && line !== 'Model –') {
    currentModel = line.slice(6);
    const models = window.__rec.models;
    if (models[models.length - 1] !== currentModel) models.push(currentModel);
  }
}
new MutationObserver(recordHudModel).observe(document, { subtree: true, childList: true, characterData: true });
let sampling = false;
function startSampling(video, canvas) {
  if (sampling) return;
  sampling = true;
  const small = document.createElement('canvas');
  small.width = 32; small.height = 18;
  const g = small.getContext('2d', { willReadFrequently: true });
  let prev = null;
  const step = () => video.requestVideoFrameCallback(() => {
    if (canvas.isConnected) {
      g.clearRect(0, 0, 32, 18);
      g.drawImage(canvas, 0, 0, 32, 18);
      const d = g.getImageData(0, 0, 32, 18).data;
      let lum = 0, sig = 0;
      for (let i = 0; i < d.length; i += 4) {
        const l = d[i] * 3 + d[i + 1] * 6 + d[i + 2];
        lum += l; sig = (sig * 31 + l) % 1000000007;
      }
      window.__rec.frames.push({ t: performance.now(), model: currentModel, lit: lum > 0, changed: sig !== prev });
      prev = sig;
    }
    step();
  });
  step();
}
let wasHidden = false;
new MutationObserver((records) => {
  for (const r of records) {
    const v = r.target;
    if (v.localName !== 'video') continue;
    const hidden = v.style.objectPosition === '${HIDDEN_OBJECT_POSITION}' && v.style.getPropertyPriority('object-position') === 'important';
    if (wasHidden && !hidden) window.__rec.restores.push(performance.now());
    if (hidden && !wasHidden && v.previousElementSibling) startSampling(v, v.previousElementSibling);
    wasHidden = hidden;
  }
}).observe(document, { subtree: true, attributes: true, attributeFilter: ['style'] });
document.addEventListener('seeking', () => window.__rec.seeks.push(performance.now()), true);
`;

interface FrameSample {
  t: number;
  model: string | null;
  lit: boolean;
  changed: boolean;
}

interface Recording {
  models: string[];
  restores: number[];
  seeks: number[];
  frames: FrameSample[];
}

const recording = (page: import('@playwright/test').Page): Promise<Recording> => page.evaluate(() => (window as unknown as { __rec: Recording }).__rec);

test.describe('automatic downgrade', () => {
  test('synthetic latency walks Medium -> Low -> Quality 0 while frames keep being presented', async ({ context }) => {
    const worker = await serviceWorker(context);
    await patchSettings(worker, { qualityLevel: 2, hudEnabled: true });
    await setDiagnostics(worker, { syntheticGpuLatencyMs: { 'rtx-vsr-medium': 40, 'rtx-vsr-low': 40 } });
    await context.addInitScript(RECORDER);
    const url = await servePage(context, 'downgrade', playerPage([{ id: 'a', width: 1600, height: 900, clip: 'clip_720p60.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);

    // Sample the service worker's telemetry (refreshed about once per second) until Quality 0.
    const samples: SessionTelemetry[] = [];
    await expect
      .poll(
        async () => {
          const v = await onlyVideo(worker, tabId);
          if (v?.state === 'active' && v.telemetry) samples.push(v.telemetry);
          return v?.telemetry?.activeModel ?? null;
        },
        { timeout: 30000, intervals: [100] },
      )
      .toBe('bicubic-q0');
    const atQ0 = samples[samples.length - 1] as SessionTelemetry;

    // Quality 0 carries no synthetic latency: it stays, keeps presenting, nothing is throttled.
    await page.waitForTimeout(3000);
    const end = await onlyVideo(worker, tabId);
    expect(end?.state).toBe('active');
    expect(end?.telemetry?.activeModel).toBe('bicubic-q0');
    expect(end!.telemetry!.presentedFrames).toBeGreaterThan(atQ0.presentedFrames + 100);
    // Each switch built exactly one new pipeline.
    expect(end!.telemetry!.pipelineInits).toBe(3);

    // Tier order, as shown by the HUD and as reported to the service worker.
    const rec = await recording(page);
    expect(rec.models).toEqual(['rtx-vsr-medium', 'rtx-vsr-low', 'bicubic-q0']);
    expect(samples.map((s) => s.activeModel).filter((m, i, all) => i === 0 || all[i - 1] !== m)).toEqual(['rtx-vsr-medium', 'rtx-vsr-low', 'bicubic-q0']);
    for (let i = 1; i < samples.length; i++) {
      expect((samples[i] as SessionTelemetry).presentedFrames).toBeGreaterThanOrEqual((samples[i - 1] as SessionTelemetry).presentedFrames);
    }

    // Frames kept arriving on the canvas under every tier, without a freeze.
    for (const model of ['rtx-vsr-medium', 'rtx-vsr-low', 'bicubic-q0']) {
      const fresh = rec.frames.filter((f) => f.model === model && f.lit && f.changed).length;
      expect(fresh, `fresh canvas frames under ${model}`).toBeGreaterThan(30);
    }
    const lit = rec.frames.filter((f) => f.lit).length;
    expect(lit / rec.frames.length).toBeGreaterThan(0.95);
    let maxGap = 0;
    for (let i = 1; i < rec.frames.length; i++) maxGap = Math.max(maxGap, (rec.frames[i] as FrameSample).t - (rec.frames[i - 1] as FrameSample).t);
    expect(maxGap).toBeLessThan(500);
    // Seamless swaps: the native video was never shown again except around loop seeks.
    for (const t of rec.restores) expect(rec.seeks.some((s) => t >= s && t - s < 250)).toBe(true);
    expect(errors).toEqual([]);
  });

  test('Quality 0 over budget throttles: video restored, gate point 9; a new quality retries', async ({ context }) => {
    const worker = await serviceWorker(context);
    await patchSettings(worker, { qualityLevel: 0, hudEnabled: true });
    await setDiagnostics(worker, { syntheticGpuLatencyMs: { 'bicubic-q0': 40 } });
    await context.addInitScript(RECORDER);
    const url = await servePage(context, 'throttle', playerPage([{ id: 'a', width: 1200, height: 900, clip: 'clip_540p30.webm', loop: true, videoStyle: 'object-position:40% 60%;background:#123456' }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);

    // 40 ms more per frame at 30 fps: every frame misses its 33.3 ms slot with GPU time over the
    // 23.3 ms threshold, so it is throttled after the warm-up plus 60 late frames.
    await expect.poll(async () => (await onlyVideo(worker, tabId))?.reason, { timeout: 20000 }).toBe('frame-budget');
    const v = await onlyVideo(worker, tabId);
    expect([v?.state, v?.point]).toEqual(['ineligible', 9]);
    const rec = await recording(page);
    expect(rec.models).toEqual(['bicubic-q0']);
    expect(rec.frames.filter((f) => f.lit && f.changed).length).toBeGreaterThan(30); // it did run first
    expect(await page.evaluate(() => [document.querySelector('video')!.style.objectPosition, document.querySelector('video')!.style.background, document.querySelectorAll('canvas.webgpu-vsr-canvas').length])).toEqual(['40% 60%', 'rgb(18, 52, 86)', 0]);
    // Latched: still throttled a moment later (no session was re-created behind the gate's back).
    await page.waitForTimeout(1500);
    expect((await onlyVideo(worker, tabId))?.reason).toBe('frame-budget');
    expect(await page.evaluate(() => document.querySelectorAll('canvas.webgpu-vsr-canvas').length)).toBe(0);

    // Choosing another quality retries the video.
    await setDiagnostics(worker, {});
    await patchSettings(worker, { qualityLevel: 1 });
    const t = await waitForTelemetry(worker, tabId, (tel) => tel.activeModel === 'rtx-vsr-low' && tel.presentedFrames > 0);
    expect(t.pipelineInits).toBe(1);
    await expect.poll(() => pictureHidden(page)).toBe(true);
    expect(errors).toEqual([]);
  });

  test('paced 720p60 playback keeps Low and Medium: GPU frames on time are not over budget', async ({ context }) => {
    test.setTimeout(120000);
    // Under paced playback the GPU clock drops until the work fills most of each 16.7 ms frame
    // (low here: about 3.4 ms back to back, 8-13 ms paced). The former mean-GPU-time rule
    // (0.70 x budget = 11.7 ms) stepped Low to Quality 0 and Medium to Low within seconds.
    const worker = await serviceWorker(context);
    for (const [level, model] of [
      [1, 'rtx-vsr-low'],
      [2, 'rtx-vsr-medium'],
    ] as const) {
      await patchSettings(worker, { qualityLevel: level });
      const url = await servePage(context, `paced-${level}`, playerPage([{ id: 'a', width: 1600, height: 900, clip: 'clip_720p60.webm', loop: true }]));
      const { page, errors } = await openPage(context, url);
      const warnings: string[] = [];
      page.on('console', (m) => {
        if (m.text().includes('[webgpu-vsr]')) warnings.push(m.text());
      });
      const tabId = await tabIdFor(worker, url);
      const start = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 30);
      expect([start.activeModel, start.outputWidth, start.outputHeight]).toEqual([model, 1600, 900]);
      const t0 = Date.now();
      await page.waitForTimeout(12000);
      const end = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > start.presentedFrames + 300);
      const seconds = (Date.now() - t0) / 1000;
      console.log(`[paced] ${model} 1280x720@60 -> 1600x900: GPU mean ${end.gpuTimeMeanMs} ms p95 ${end.gpuTimeP95Ms} ms, ${end.fps} fps`);
      expect([end.activeModel, end.pipelineInits]).toEqual([model, 1]);
      // Frames kept being presented throughout (headless Chromium's callback cadence itself varies
      // between about 46 and 60 fps on this clip, with or without GPU work).
      expect(end.presentedFrames - start.presentedFrames).toBeGreaterThan(40 * (seconds - 2));
      expect(warnings.filter((w) => w.includes('->'))).toEqual([]);
      expect(errors).toEqual([]);
      await page.close();
    }
  });

  test('main-thread stalls of 400 ms are not GPU timeouts: Medium stays', async ({ context }) => {
    const worker = await serviceWorker(context);
    await patchSettings(worker, { qualityLevel: 2 });
    const url = await servePage(context, 'stalls', playerPage([{ id: 'a', width: 1280, height: 720, clip: 'clip_540p30.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const warnings: string[] = [];
    page.on('console', (m) => {
      if (m.text().includes('[webgpu-vsr]')) warnings.push(m.text());
    });
    const tabId = await tabIdFor(worker, url);
    await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 30 && t.activeModel === 'rtx-vsr-medium');
    for (let i = 0; i < 4; i++) {
      // page jank: the main thread (shared with the content script) is blocked while frames are in flight
      await page.evaluate(() => {
        const t = performance.now();
        while (performance.now() - t < 400) {
          // busy
        }
      });
      await page.waitForTimeout(1500);
    }
    const end = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 60);
    expect([end.activeModel, end.pipelineInits]).toEqual(['rtx-vsr-medium', 1]);
    expect(warnings.filter((w) => w.includes('timed out'))).toEqual([]);
    expect(errors).toEqual([]);
  });

  test('a 60 fps clip keeps playing for more than 5 s with every VideoFrame closed', async ({ context }) => {
    const worker = await serviceWorker(context);
    // Quality 0 keeps the GPU far inside the 60 fps budget, so the run measures the frame loop alone.
    await patchSettings(worker, { qualityLevel: 0 });
    const url = await servePage(context, 'leak', playerPage([{ id: 'a', width: 1600, height: 900, clip: 'clip_720p60.webm', loop: true }]));
    const { page, errors, inExtension } = await openPageWithExtensionWorld(context, url);
    const tabId = await tabIdFor(worker, url);
    const decoded = () => page.evaluate(() => document.querySelector('video')!.getVideoPlaybackQuality().totalVideoFrames);
    const start = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 30);
    // Count the session's VideoFrames from here on (the content script's own VideoFrame global).
    expect(await inExtension<boolean>(COUNT_VIDEO_FRAMES)).toBe(true);
    const decodedStart = await decoded();
    const t0 = Date.now();
    await page.waitForTimeout(6000);
    const end = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > start.presentedFrames);
    const seconds = (Date.now() - t0) / 1000;
    const decodedEnd = await decoded();
    // Every frame the loop created was closed before its callback returned. (A missing close()
    // does not reliably stall this Chromium build's playback; it is detected here directly.)
    type FrameStats = { created: number; closed: number; maxOpen: number };
    const frames = await inExtension<FrameStats>('({ ...globalThis.__webgpuVsrFrames })');
    expect(frames.created).toBeGreaterThan(50 * (seconds - 1));
    expect(frames.created - frames.closed).toBe(0);
    expect(frames.maxOpen).toBe(0);
    // Decoding and presentation keep the clip's 60 fps cadence throughout.
    expect(decodedEnd - decodedStart).toBeGreaterThan(50 * seconds);
    // Service-worker telemetry is up to ~1 s old at either end.
    expect(end.presentedFrames - start.presentedFrames).toBeGreaterThan(50 * (seconds - 2));
    expect(end.fps).toBeGreaterThan(50);
    expect([end.activeModel, end.pipelineInits]).toEqual(['bicubic-q0', 1]);
    expect((await onlyVideo(worker, tabId))?.state).toBe('active');
    expect(errors).toEqual([]);
  });
});
