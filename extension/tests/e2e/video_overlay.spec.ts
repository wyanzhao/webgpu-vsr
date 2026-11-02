/**
 * End-to-end GPU session tests with the extension loaded: the sibling canvas
 * overlay over real VP9 playback, letterbox geometry, pointer pass-through, the atomic picture
 * swap with a pixel check of the upscaled picture, the DRM / off-switch rollback, every quality
 * tier, GPU device loss and recovery, and theater-mode re-parenting within (and beyond) the 5 s
 * grace period.
 */
import type { Page } from '@playwright/test';
import { HIDDEN_OBJECT_POSITION } from '../../src/content/overlay';
import { SECOND_ORIGIN, TEST_ORIGIN, extensionTest as test, expect, serveStatic } from './fixtures';
import {
  LUMA_HELPER,
  correlation,
  onlyVideo,
  openPage,
  patchSettings,
  playerPage,
  pictureHidden,
  servePage,
  serviceWorker,
  tabIdFor,
  topFrameVideos,
  waitForTelemetry,
} from './session_helpers';

interface LumaStats {
  cells: number[];
  mean: number;
  std: number;
}

/**
 * Records the moment the session first hides the video: the canvas's visibility and a copy of its
 * pixels (toDataURL) taken synchronously in the MutationObserver callback for that style change.
 */
const HIDE_RECORDER = `
window.__webgpuVsrHide = null;
new MutationObserver((records) => {
  for (const r of records) {
    const v = r.target;
    if (window.__webgpuVsrHide || v.localName !== 'video' || (v.style.objectPosition !== '${HIDDEN_OBJECT_POSITION}' || v.style.getPropertyPriority('object-position') !== 'important')) continue;
    const c = v.previousElementSibling;
    const isCanvas = !!c && c.classList.contains('webgpu-vsr-canvas');
    window.__webgpuVsrHide = {
      canvasNextToVideo: isCanvas,
      canvasVisibility: isCanvas ? getComputedStyle(c).visibility : null,
      dataUrl: isCanvas ? c.toDataURL('image/png') : null,
    };
  }
}).observe(document, { subtree: true, attributes: true, attributeFilter: ['style'] });
`;

type LumaWindow = Window & {
  __webgpuVsrLuma: (source: CanvasImageSource, w: number, h: number, gx: number, gy: number) => LumaStats;
  __webgpuVsrHide: { canvasNextToVideo: boolean; canvasVisibility: string | null; dataUrl: string | null } | null;
};

/** Luma grid of a PNG data URL (decoded in the page). */
async function lumaOfDataUrl(page: Page, dataUrl: string, w: number, h: number, gx: number, gy: number): Promise<LumaStats> {
  return page.evaluate(
    async ({ url, w, h, gx, gy }) => {
      const img = new Image();
      img.src = url;
      await img.decode();
      return (window as unknown as LumaWindow).__webgpuVsrLuma(img, w, h, gx, gy);
    },
    { url: dataUrl, w, h, gx, gy },
  );
}

/** The canvas's and the video's rects (CSS px, viewport space). */
async function rects(page: Page, videoSelector: string) {
  return page.evaluate((sel) => {
    const v = document.querySelector(sel) as HTMLVideoElement;
    const c = v.parentElement?.querySelector('canvas.webgpu-vsr-canvas') as HTMLCanvasElement | null;
    const r = (e: Element) => {
      const b = e.getBoundingClientRect();
      return { left: b.left, top: b.top, width: b.width, height: b.height };
    };
    return { video: r(v), canvas: c ? r(c) : null, backing: c ? [c.width, c.height] : null };
  }, videoSelector);
}

/** Luma statistics of what is on screen inside `selector` (a screenshot, i.e. the composited page). */
async function screenLuma(page: Page, selector: string): Promise<LumaStats> {
  const box = await page.evaluate((sel) => {
    const r = document.querySelector(sel)!.getBoundingClientRect();
    return { x: r.left, y: r.top, width: r.width, height: r.height };
  }, selector);
  const png = await page.screenshot({ clip: box });
  return lumaOfDataUrl(page, `data:image/png;base64,${png.toString('base64')}`, 320, 180, 8, 6);
}

/** Counts screenshot pixels differing by more than 40 in any channel inside/outside a band. */
async function screenshotDifference(page: Page, a: Buffer, b: Buffer, top: number, bottom: number): Promise<{ inside: number; outside: number }> {
  return page.evaluate(async ({ a, b, top, bottom }) => {
    const decode = async (base64: string) => {
      const img = new Image();
      img.src = `data:image/png;base64,${base64}`;
      await img.decode();
      const canvas = new OffscreenCanvas(img.width, img.height);
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      return ctx.getImageData(0, 0, img.width, img.height);
    };
    const [first, second] = await Promise.all([decode(a), decode(b)]);
    let inside = 0, outside = 0;
    for (let y = 0; y < first.height; y++) {
      for (let x = 0; x < first.width; x++) {
        const i = (y * first.width + x) * 4;
        if ([0, 1, 2, 3].some((c) => Math.abs(first.data[i + c]! - second.data[i + c]!) > 40)) {
          if (y >= top && y < bottom) inside++;
          else outside++;
        }
      }
    }
    return { inside, outside };
  }, { a: a.toString('base64'), b: b.toString('base64'), top, bottom });
}

function expectClose(actual: number, expected: number, tolerance = 0.01): void {
  expect(Math.abs(actual - expected), `${actual} vs ${expected}`).toBeLessThanOrEqual(tolerance);
}

test.describe('GPU session overlay', () => {
  test('16:9 clip in a 4:3 box: letterbox geometry, pass-through clicks, atomic hide and the upscaled picture', async ({ context }) => {
    const worker = await serviceWorker(context);
    await context.addInitScript(LUMA_HELPER + HIDE_RECORDER);
    const url = await servePage(context, 'letterbox', playerPage([{ id: 'a', width: 1200, height: 900, clip: 'clip_540p30.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);

    const t = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > 0);
    expect([t.activeModel, t.inputWidth, t.inputHeight]).toEqual(['rtx-vsr-low', 960, 540]);
    // 960x540 contained in 1200x900 -> 1200x675 content; target 1200x675 < internal 1920x1080 -> resample.
    expect([t.outputWidth, t.outputHeight]).toEqual([1200, 675]);

    // (c) The video was hidden only once the canvas was visible and already held a picture.
    const hide = await page.evaluate(() => (window as unknown as LumaWindow).__webgpuVsrHide);
    expect(hide).not.toBeNull();
    expect(hide!.canvasNextToVideo).toBe(true);
    expect(hide!.canvasVisibility).toBe('visible');
    const atHide = await lumaOfDataUrl(page, hide!.dataUrl!, 240, 135, 8, 6);
    expect(atHide.std).toBeGreaterThan(20);
    expect(await pictureHidden(page)).toBe(true);

    // (a) Sibling canvas right before the video, non-interactive, positioned over the content rect.
    const dom = await page.evaluate(() => {
      const v = document.querySelector('video')!;
      const c = v.previousElementSibling as HTMLCanvasElement;
      const cs = getComputedStyle(c);
      return {
        cls: c.className,
        parent: c.parentElement?.id,
        pointerEvents: cs.pointerEvents,
        position: cs.position,
        zIndex: c.style.getPropertyValue('z-index'),
        zPriority: c.style.getPropertyPriority('z-index'),
      };
    });
    expect(dom).toEqual({ cls: 'webgpu-vsr-canvas', parent: 'a', pointerEvents: 'none', position: 'absolute', zIndex: 'auto', zPriority: 'important' });
    const r = await rects(page, '#a-video');
    expect(r.canvas).not.toBeNull();
    expectClose(r.canvas!.left - r.video.left, 0);
    expectClose(r.canvas!.top - r.video.top, 112.5);
    expectClose(r.canvas!.width, 1200);
    expectClose(r.canvas!.height, 675);
    expect(r.backing).toEqual([1200, 675]);

    // (b) Clicks at the video's position reach the video, not the canvas.
    await page.evaluate(() => {
      const v = document.querySelector('video')!;
      (window as unknown as { clicks: number }).clicks = 0;
      v.addEventListener('click', () => (window as unknown as { clicks: number }).clicks++);
    });
    const hit = await page.evaluate(() => document.elementFromPoint(600, 450)?.id);
    expect(hit).toBe('a-video');
    await page.mouse.click(600, 450);
    await page.mouse.click(300, 200);
    expect(await page.evaluate(() => (window as unknown as { clicks: number }).clicks)).toBe(2);

    // Seeking restores the native video during the seek; the paused frame is then rendered once
    // on 'seeked' (paused-MSE workaround) and hidden again.
    await page.evaluate(() => {
      const v = document.querySelector('video')!;
      const w = window as unknown as { pictureAtSeeking: string | null };
      w.pictureAtSeeking = null;
      v.addEventListener('seeking', () => (w.pictureAtSeeking = v.style.objectPosition), { once: true });
      v.pause();
    });
    await page.evaluate(
      () =>
        new Promise<void>((resolve) => {
          const v = document.querySelector('video')!;
          v.addEventListener('seeked', () => resolve(), { once: true });
          v.currentTime = 2.0;
        }),
    );
    expect(await page.evaluate(() => (window as unknown as { pictureAtSeeking: string | null }).pictureAtSeeking)).toBe('');
    await expect.poll(() => pictureHidden(page)).toBe(true);

    // The canvas holds the upscaled current frame: coarse structure matches the source frame.
    const [canvasUrl, video] = await page.evaluate(() => {
      const c = document.querySelector('canvas.webgpu-vsr-canvas') as HTMLCanvasElement;
      const v = document.querySelector('video')!;
      return [c.toDataURL('image/png'), (window as unknown as LumaWindow).__webgpuVsrLuma(v, 1200, 675, 32, 18)] as const;
    });
    const canvas = await lumaOfDataUrl(page, canvasUrl, 1200, 675, 32, 18);
    expect(canvas.std).toBeGreaterThan(20);
    expect(Math.abs(canvas.mean - video.mean)).toBeLessThan(4);
    // Measured: 0.9996 / 2.2 for the shown frame vs 0.979 / 5.3 against another frame of the clip.
    expect(correlation(canvas.cells, video.cells)).toBeGreaterThan(0.995);
    const meanAbs = canvas.cells.reduce((s, v, i) => s + Math.abs(v - (video.cells[i] as number)), 0) / canvas.cells.length;
    expect(meanAbs).toBeLessThan(4);
    // A different frame of the same clip correlates less: the check is not satisfied by any frame.
    const other = await page.evaluate(
      () =>
        new Promise<LumaStats>((resolve) => {
          const v = document.querySelector('video')!;
          v.addEventListener('seeked', () => resolve((window as unknown as LumaWindow).__webgpuVsrLuma(v, 1200, 675, 32, 18)), { once: true });
          v.currentTime = 4.5;
        }),
    );
    expect(meanAbs).toBeLessThan(canvas.cells.reduce((s, v, i) => s + Math.abs(v - (other.cells[i] as number)), 0) / canvas.cells.length);

    expect(errors).toEqual([]);
  });

  test('native controls and WebVTT captions stay visible above the upscaled picture', async ({ context }) => {
    const worker = await serviceWorker(context);
    await context.addInitScript(LUMA_HELPER);
    const url = await servePage(context, 'native-controls-captions', playerPage([{
      id: 'a', width: 1280, height: 720, clip: 'clip_540p30.webm', loop: true,
      controls: true, captions: '/media/captions_en.vtt',
    }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    await expect.poll(() => pictureHidden(page)).toBe(true);
    await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 0);
    await expect.poll(() => page.evaluate(() => {
      const track = document.querySelector('video')!.textTracks[0];
      return track?.mode === 'showing' && (track.activeCues?.[0] as VTTCue | undefined)?.text === 'WEBGPU VSR CAPTION';
    })).toBe(true);
    await page.evaluate(() => document.querySelector('video')!.pause());
    await page.waitForTimeout(300);

    const clip = { x: 0, y: 0, width: 1280, height: 720 };
    const a = await page.screenshot({ clip });
    await page.evaluate(() => { document.querySelector('video')!.textTracks[0]!.mode = 'hidden'; });
    await page.waitForTimeout(200);
    const b = await page.screenshot({ clip });
    const captions = await screenshotDifference(page, a, b, 360, 720);
    expect(captions.inside).toBeGreaterThanOrEqual(200);
    expect(captions.outside).toBe(0);

    const c = await page.screenshot({ clip });
    await page.evaluate(() => document.querySelector('video')!.removeAttribute('controls'));
    await page.waitForTimeout(200);
    const d = await page.screenshot({ clip });
    const controls = await screenshotDifference(page, c, d, 640, 720);
    expect(controls.inside).toBeGreaterThanOrEqual(200);

    const centre = await page.screenshot({ clip: { x: 320, y: 180, width: 640, height: 360 } });
    expect((await lumaOfDataUrl(page, `data:image/png;base64,${centre.toString('base64')}`, 320, 180, 8, 6)).std).toBeGreaterThan(10);
    expect(errors).toEqual([]);
  });

  test('2.35:1 scope clip in a 16:9 box: letterbox rect and direct 2x output', async ({ context }) => {
    const worker = await serviceWorker(context);
    const url = await servePage(context, 'scope', playerPage([{ id: 'a', width: 1280, height: 720, clip: 'clip_scope_24.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    const t = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > 0);
    // 640x272 contained in 1280x720 -> 1280x544 at top 88; target == internal 2W x 2H -> direct.
    expect([t.inputWidth, t.inputHeight, t.outputWidth, t.outputHeight]).toEqual([640, 272, 1280, 544]);
    const r = await rects(page, '#a-video');
    expectClose(r.canvas!.left - r.video.left, 0);
    expectClose(r.canvas!.top - r.video.top, 88);
    expectClose(r.canvas!.width, 1280);
    expectClose(r.canvas!.height, 544);
    expect(r.backing).toEqual([1280, 544]);
    // 24 fps cadence measured from rVFC presentation timestamps.
    const later = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > 30 && tel.fps > 0);
    expect(later.fps).toBeGreaterThan(20);
    expect(later.fps).toBeLessThan(28);
    expect(errors).toEqual([]);
  });

  test('a DRM event or the off switch restores the original picture at once and removes the canvas', async ({ context }) => {
    const worker = await serviceWorker(context);
    const html = playerPage([{ id: 'a', width: 1200, height: 900, clip: 'clip_540p30.webm', loop: true, videoStyle: 'object-position:40% 60%;background:#123456' }]);
    const url = await servePage(context, 'drm', html);
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > 0);
    await expect.poll(() => pictureHidden(page)).toBe(true);

    // Synchronous rollback inside the event dispatch.
    const afterDrm = await page.evaluate(() => {
      const v = document.querySelector('video')!;
      v.dispatchEvent(new Event('encrypted'));
      return { objectPosition: v.style.objectPosition, background: v.style.background, canvases: document.querySelectorAll('canvas.webgpu-vsr-canvas').length };
    });
    expect(afterDrm).toEqual({ objectPosition: '40% 60%', background: 'rgb(18, 52, 86)', canvases: 0 });
    await expect.poll(async () => (await onlyVideo(worker, tabId))?.reason).toBe('drm');

    // Off switch (another page, fresh video): rollback as soon as the settings change arrives.
    const url2 = await servePage(context, 'off', html);
    const second = await openPage(context, url2);
    const tab2 = await tabIdFor(worker, url2);
    await waitForTelemetry(worker, tab2, (tel) => tel.presentedFrames > 0);
    await expect.poll(() => pictureHidden(second.page)).toBe(true);
    await patchSettings(worker, { enabled: false });
    await expect
      .poll(() => second.page.evaluate(() => [document.querySelector('video')!.style.objectPosition, document.querySelector('video')!.style.background, document.querySelectorAll('canvas.webgpu-vsr-canvas').length]), {
        timeout: 2000,
      })
      .toEqual(['40% 60%', 'rgb(18, 52, 86)', 0]);
    await expect.poll(async () => (await onlyVideo(worker, tab2))?.reason).toBe('extension-disabled');
    await patchSettings(worker, { enabled: true });
    await waitForTelemetry(worker, tab2, (tel) => tel.presentedFrames > 0);
    expect(errors).toEqual([]);
    expect(second.errors).toEqual([]);
  });

  test('each quality reports its model and a measured GPU time (HUD and service worker)', async ({ context }) => {
    const worker = await serviceWorker(context);
    await patchSettings(worker, { hudEnabled: true });
    const url = await servePage(context, 'quality', playerPage([{ id: 'a', width: 1200, height: 900, clip: 'clip_540p30.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    const hud = page.locator('#a .webgpu-vsr-hud');
    let lastInits = 0;
    for (const [level, model] of [
      [0, 'bicubic-q0'],
      [2, 'rtx-vsr-medium'],
      [1, 'rtx-vsr-low'],
    ] as const) {
      await patchSettings(worker, { qualityLevel: level });
      const t = await waitForTelemetry(worker, tabId, (tel) => tel.activeModel === model && Number.isFinite(tel.gpuTimeMeanMs) && tel.gpuTimeMeanMs > 0 && Number.isFinite(tel.gpuTimeP95Ms) && tel.gpuTimeP95Ms > 0);
      expect([t.outputWidth, t.outputHeight]).toEqual([1200, 675]);
      expect(t.gpuTimeMeanMs).toBeLessThan(50);
      expect(t.pipelineInits).toBeGreaterThan(lastInits); // each switch built a new pipeline
      lastInits = t.pipelineInits;
      await expect(hud).toContainText(`Model ${model}`);
      await expect(hud).toContainText('ms mean');
    }
    expect(errors).toEqual([]);
  });

  test('GPU device loss restores the video at once; the same session re-binds after recovery', async ({ context }) => {
    const worker = await serviceWorker(context);
    const url = await servePage(context, 'device-loss', playerPage([{ id: 'a', width: 1200, height: 900, clip: 'clip_540p30.webm', loop: true }]));
    const page = await context.newPage();
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    // The content script's isolated world is reachable through CDP only.
    const client = await context.newCDPSession(page);
    const worlds: { id: number; name: string }[] = [];
    client.on('Runtime.executionContextCreated', (e) => worlds.push(e.context));
    await client.send('Runtime.enable');
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.goto(url);
    const tabId = await tabIdFor(worker, url);
    const before = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > 10);
    await expect.poll(() => pictureHidden(page)).toBe(true);
    await page.evaluate(() => ((document.querySelector('canvas.webgpu-vsr-canvas') as HTMLCanvasElement & { mark?: number }).mark = 1));

    // Destroy the session's GPUDevice from the extension world (test-only access to the running
    // controller and session): the device manager sees an unexpected loss, exactly like a GPU reset.
    const world = worlds.filter((w) => w.name === 'WebGPU VSR - Video Super Resolution').pop();
    expect(world).toBeDefined();
    const lost = await client.send('Runtime.evaluate', {
      contextId: world!.id,
      returnByValue: true,
      expression: `(() => {
        const controller = globalThis[Symbol.for('webgpu-vsr.content-controller')];
        const session = [...controller.entries.values()][0].handle;
        session.gpu.device.destroy();
        return true;
      })()`,
    });
    expect(lost.result.value).toBe(true);
    const lostAt = Date.now();
    await expect.poll(() => page.evaluate(() => document.querySelector('video')!.style.objectPosition), { timeout: 500, intervals: [20] }).toBe('');
    expect(Date.now() - lostAt).toBeLessThan(500);

    // Recovery (1 s backoff in gpuDeviceManager), then the same session and canvas present again.
    const after = await waitForTelemetry(worker, tabId, (tel) => tel.pipelineInits === before.pipelineInits + 1 && tel.presentedFrames > before.presentedFrames + 10);
    expect(after.activeModel).toBe('rtx-vsr-low');
    await expect.poll(() => pictureHidden(page)).toBe(true);
    expect(await page.evaluate(() => [document.querySelectorAll('canvas.webgpu-vsr-canvas').length, (document.querySelector('canvas.webgpu-vsr-canvas') as HTMLCanvasElement & { mark?: number }).mark])).toEqual([1, 1]);
    expect(errors).toEqual([]);
  });

  test('a page that removes the canvas or re-renders the container never leaves a blank player', async ({ context }) => {
    const worker = await serviceWorker(context);
    await context.addInitScript(LUMA_HELPER);
    const url = await servePage(context, 'canvas-removed', playerPage([{ id: 'a', width: 1280, height: 720, clip: 'clip_540p30.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    const state = () =>
      page.evaluate(() => {
        const v = document.querySelector('video')!;
        const c = v.previousElementSibling;
        return { objectPosition: v.style.objectPosition, canvasNextToVideo: !!c && c.classList.contains('webgpu-vsr-canvas'), canvases: document.querySelectorAll('canvas.webgpu-vsr-canvas').length };
      });
    const mounted = { objectPosition: HIDDEN_OBJECT_POSITION, canvasNextToVideo: true, canvases: 1 };
    let last = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 10);
    expect(await state()).toEqual(mounted);

    // 1. The page removes the overlay canvas: the frame loop puts it back next to the video.
    await page.evaluate(() => document.querySelector('canvas.webgpu-vsr-canvas')!.remove());
    last = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > last.presentedFrames + 10);
    await expect.poll(state).toEqual(mounted);
    expect((await screenLuma(page, '#a')).std).toBeGreaterThan(10); // a picture, not the #202020 box

    // 2. The player re-renders its container and re-appends the same <video> (no detach event).
    await page.evaluate(() => {
      const p = document.getElementById('a')!;
      const v = p.querySelector('video')!;
      p.textContent = '';
      p.appendChild(v);
      void v.play();
    });
    last = await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > last.presentedFrames + 10);
    await expect.poll(state).toEqual(mounted);
    expect((await screenLuma(page, '#a')).std).toBeGreaterThan(10);
    expect((await onlyVideo(worker, tabId))?.telemetry?.pipelineInits).toBe(1); // same session, no rebuild

    // 3. Paused: removal restores the native video at once; periodic evaluation remounts and redraws SR.
    await page.evaluate(() => document.querySelector('video')!.pause());
    await page.waitForTimeout(300);
    expect((await state()).objectPosition).toBe(HIDDEN_OBJECT_POSITION);
    const restored = await page.evaluate(async () => {
      document.querySelector('canvas.webgpu-vsr-canvas')!.remove();
      await new Promise((r) => setTimeout(r, 0)); // mutation records are delivered after the task
      return document.querySelector('video')!.style.objectPosition;
    });
    expect(restored).toBe('');
    await expect.poll(state, { timeout: 5000 }).toEqual(mounted);
    expect((await screenLuma(page, '#a')).std).toBeGreaterThan(10);
    expect(errors).toEqual([]);
  });

  test('a paused player resize re-renders the current picture within two seconds', async ({ context }) => {
    const worker = await serviceWorker(context);
    await context.addInitScript(LUMA_HELPER);
    const url = await servePage(context, 'paused-resize', playerPage([{ id: 'a', width: 1280, height: 720, clip: 'clip_540p30.webm', loop: true }]));
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    await waitForTelemetry(worker, tabId, (t) => t.presentedFrames > 10);
    await page.evaluate(() => document.querySelector('video')!.pause());
    await page.waitForTimeout(300);
    expect(await pictureHidden(page)).toBe(true);
    await page.evaluate(() => {
      const player = document.getElementById('a')!;
      player.style.width = '1200px';
      player.style.height = '675px';
    });
    await expect.poll(async () => {
      const backing = await page.evaluate(() => {
        const canvas = document.querySelector('canvas.webgpu-vsr-canvas') as HTMLCanvasElement;
        return [canvas.width, canvas.height];
      });
      return { backing, hidden: await pictureHidden(page), nonBlank: (await screenLuma(page, '#a')).std > 10 };
    }, { timeout: 2000 }).toEqual({ backing: [1200, 675], hidden: true, nonBlank: true });
    expect(errors).toEqual([]);
  });

  test("the popup's site switch also stops a cross-origin player iframe; badge and reports agree", async ({ context }) => {
    const worker = await serviceWorker(context);
    // An embedded player on another origin (its own content script instance, its own frame report).
    const player = playerPage([{ id: 'p', width: 1200, height: 675, clip: 'clip_540p30.webm', loop: true }]);
    await serveStatic(context, { '/embed/player.html': player }, SECOND_ORIGIN);
    const host = `<!doctype html><html><body style="margin:0;background:#000"><iframe id="f" src="${SECOND_ORIGIN}/embed/player.html" style="border:0;width:1200px;height:675px"></iframe></body></html>`;
    const url = await servePage(context, 'embed-host', host);
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    const frameVideos = () =>
      worker.evaluate(async (id) => {
        const all = (await chrome.storage.session.get(null)) as Record<string, { host: string; videos: { state: string; reason: string | null }[] }>;
        return Object.entries(all)
          .filter(([k]) => k.startsWith(`tab:${id}:frame:`) && !k.endsWith(':frame:0'))
          .flatMap(([, f]) => f.videos.map((v) => `${f.host}:${v.state}:${v.reason ?? ''}`));
      }, tabId);
    const badge = () => worker.evaluate((id) => chrome.action.getBadgeText({ tabId: id }), tabId);
    const iframePictureHidden = () => pictureHidden(page.frames().find((frame) => frame.url().startsWith(SECOND_ORIGIN))!);
    await expect.poll(frameVideos, { timeout: 20000 }).toEqual(['127.0.0.1:active:']);
    await expect.poll(iframePictureHidden).toBe(true);
    await expect.poll(badge).toBe('ON');

    // "Off for this site" on the tab (what the popup writes: the tab's top-level host).
    await patchSettings(worker, { siteOverrides: { [new URL(TEST_ORIGIN).hostname]: false } });
    await expect.poll(frameVideos).toEqual(['127.0.0.1:ineligible:site-disabled']);
    await expect.poll(iframePictureHidden).toBe(false);
    await expect.poll(() => page.frameLocator('#f').locator('canvas.webgpu-vsr-canvas').count()).toBe(0);
    await expect.poll(badge).toBe('OFF');

    // Back on: the embedded player is processed again.
    await patchSettings(worker, { siteOverrides: {} });
    await expect.poll(frameVideos, { timeout: 20000 }).toEqual(['127.0.0.1:active:']);
    await expect.poll(badge).toBe('ON');
    expect(errors).toEqual([]);
  });

  test('theater-mode re-parenting keeps the session; a removal beyond 5 s destroys it', async ({ context }) => {
    const worker = await serviceWorker(context);
    await patchSettings(worker, { hudEnabled: true });
    const url = await servePage(
      context,
      'theater',
      playerPage([
        { id: 'a', width: 800, height: 450, clip: 'clip_scope_24.webm', loop: true },
        { id: 'b', width: 800, height: 450 },
      ]),
    );
    const { page, errors } = await openPage(context, url);
    const tabId = await tabIdFor(worker, url);
    type Held = Window & { held: HTMLVideoElement };
    await page.evaluate(() => ((window as unknown as Held).held = document.getElementById('a-video') as HTMLVideoElement));
    const canvasParent = () => page.evaluate(() => document.querySelector('canvas.webgpu-vsr-canvas')?.parentElement?.id ?? null);
    const canvasCount = () => page.evaluate(() => document.querySelectorAll('canvas.webgpu-vsr-canvas').length);
    const objectPosition = () => page.evaluate(() => (window as unknown as Held).held.style.objectPosition);

    const hudState = () =>
      page.evaluate(() => {
        const h = document.querySelector('.webgpu-vsr-hud') as HTMLElement | null;
        const panel = h?.shadowRoot?.querySelector('.panel');
        return h && panel ? `${h.parentElement?.id}:${getComputedStyle(panel).visibility}` : null;
      });
    const first = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > 10);
    expect(await canvasParent()).toBe('a');
    expect(await objectPosition()).toBe(HIDDEN_OBJECT_POSITION);
    await expect.poll(hudState).toBe('a:visible');

    // Detach (theater toggle unmounts the player): the native video is restored at once, the
    // session survives the grace period (canvas kept), the video is not reported meanwhile.
    await page.evaluate(() => (window as unknown as Held).held.remove());
    expect(await objectPosition()).toBe('');
    await expect.poll(async () => (await topFrameVideos(worker, tabId)).length).toBe(0);
    await page.waitForTimeout(1000);
    expect(await canvasCount()).toBe(1);

    // Re-insert into another container within 5 s (removal paused it; the player resumes it).
    await page.evaluate(async () => {
      const v = (window as unknown as Held).held;
      document.getElementById('b')!.appendChild(v);
      await v.play();
    });
    await expect.poll(canvasParent).toBe('b');
    const second = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > first.presentedFrames + 10);
    // The diagnostic HUD followed the video and is shown again (it is hidden while detached).
    await expect.poll(hudState).toBe('b:visible');
    // Same session (frame counter continued) and no pipeline re-initialisation.
    expect(second.pipelineInits).toBe(first.pipelineInits);
    await expect.poll(objectPosition).toBe(HIDDEN_OBJECT_POSITION);
    const r = await rects(page, '#a-video');
    const b = await page.evaluate(() => {
      const box = document.getElementById('b')!.getBoundingClientRect();
      return { left: box.left, top: box.top };
    });
    // 640x272 contained in 800x450 -> 800x340 at top 55, inside container b.
    expectClose(r.video.left, b.left);
    expectClose(r.canvas!.left - r.video.left, 0);
    expectClose(r.canvas!.top - r.video.top, 55);
    expectClose(r.canvas!.width, 800);
    expectClose(r.canvas!.height, 340);

    // A synchronous move while playing (no detach event at all) is followed on the next frame.
    await page.evaluate(() => document.getElementById('a')!.appendChild((window as unknown as Held).held));
    await expect.poll(canvasParent).toBe('a');
    const third = await waitForTelemetry(worker, tabId, (tel) => tel.presentedFrames > second.presentedFrames + 10);
    expect(third.pipelineInits).toBe(first.pipelineInits);
    expect(await page.evaluate(() => document.querySelectorAll('#b canvas').length)).toBe(0);

    // Removed for longer than the grace period: destroyed, canvas gone, picture restored.
    const removedAt = Date.now();
    await page.evaluate(() => (window as unknown as Held).held.remove());
    await page.waitForTimeout(3000);
    expect(await canvasCount()).toBe(1);
    await expect.poll(canvasCount, { timeout: 10000 }).toBe(0);
    expect(Date.now() - removedAt).toBeGreaterThanOrEqual(4500);
    expect(await objectPosition()).toBe('');
    expect(errors).toEqual([]);
  });
});
