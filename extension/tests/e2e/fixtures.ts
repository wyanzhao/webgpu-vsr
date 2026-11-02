import { test as base, chromium, type BrowserContext, type Page, type Route } from '@playwright/test';
import { mkdtempSync, readFileSync, rmSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, resolve, normalize, sep } from 'node:path';

export const EXTENSION_ROOT = resolve(import.meta.dirname, '..', '..');
export const DIST_DIR = join(EXTENSION_ROOT, 'dist');
export const HARNESS_DIR = join(EXTENSION_ROOT, 'dist-harness');
export const FIXTURES_DIR = join(EXTENSION_ROOT, 'tests', 'fixtures');

/** Origin used for every test page; http://localhost is a secure context, so WebGPU is exposed. */
export const TEST_ORIGIN = 'http://localhost:4173';
/** A second (cross-origin, also secure: loopback) origin for embedded-player iframes. */
export const SECOND_ORIGIN = 'http://127.0.0.1:4173';

export const GPU_ARGS = ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'];

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
  '.raw': 'application/octet-stream',
  '.webm': 'video/webm',
  '.vtt': 'text/vtt; charset=utf-8',
  '.mp4': 'video/mp4',
  '.css': 'text/css',
  '.png': 'image/png',
  '.wgsl': 'text/plain; charset=utf-8',
};

/**
 * Serves files under `origin` (default TEST_ORIGIN) from fixed roots:
 *   /dist/...      -> extension/dist
 *   /harness/...   -> extension/dist-harness
 *   /fixtures/...  -> extension/tests/fixtures
 *   /media/...     -> extension/tests/media (VP9 test clips generated with ffmpeg testsrc2)
 *   /weights/...   -> extension/public/weights
 * Supports HTTP Range requests so <video> elements can seek.
 */
export async function serveStatic(context: BrowserContext, extraHtml: Record<string, string> = {}, origin = TEST_ORIGIN): Promise<void> {
  const roots: Record<string, string> = {
    dist: DIST_DIR,
    harness: HARNESS_DIR,
    fixtures: FIXTURES_DIR,
    media: join(EXTENSION_ROOT, 'tests', 'media'),
    weights: join(EXTENSION_ROOT, 'public', 'weights'),
  };
  await context.route(`${origin}/**`, async (route: Route) => {
    const url = new URL(route.request().url());
    const path = decodeURIComponent(url.pathname);
    if (extraHtml[path] !== undefined) {
      await route.fulfill({ status: 200, contentType: MIME['.html'], body: extraHtml[path] });
      return;
    }
    const [, top, ...rest] = path.split('/');
    const root = roots[top ?? ''];
    if (!root) {
      await route.fulfill({ status: 404, body: `no route for ${path}` });
      return;
    }
    const file = normalize(join(root, ...rest));
    if (!file.startsWith(root + sep) || !existsSync(file) || !statSync(file).isFile()) {
      await route.fulfill({ status: 404, body: `missing ${path}` });
      return;
    }
    const body = readFileSync(file);
    const contentType = MIME[extname(file)] ?? 'application/octet-stream';
    const range = route.request().headers()['range'];
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      const start = m && m[1] ? Number(m[1]) : 0;
      const end = m && m[2] ? Math.min(Number(m[2]), body.length - 1) : body.length - 1;
      await route.fulfill({
        status: 206,
        contentType,
        headers: {
          'Accept-Ranges': 'bytes',
          'Content-Range': `bytes ${start}-${end}/${body.length}`,
          'Content-Length': String(end - start + 1),
        },
        body: body.subarray(start, end + 1),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType, headers: { 'Accept-Ranges': 'bytes' }, body });
  });
}

type ExtensionFixtures = {
  context: BrowserContext;
  extensionId: string;
  userDataDir: string;
};

/** Persistent Chromium context with the unpacked extension from extension/dist loaded. */
export const extensionTest = base.extend<ExtensionFixtures>({
  // eslint-disable-next-line no-empty-pattern
  userDataDir: async ({}, use) => {
    const dir = mkdtempSync(join(tmpdir(), 'webgpu-vsr-e2e-'));
    await use(dir);
    rmSync(dir, { recursive: true, force: true });
  },
  context: async ({ userDataDir }, use) => {
    if (!existsSync(join(DIST_DIR, 'manifest.json'))) {
      throw new Error('extension/dist is missing: run `npm run build` before the e2e suite');
    }
    const context = await launchExtensionContext(userDataDir);
    await serveStatic(context);
    await use(context);
    await context.close();
  },
  extensionId: async ({ context }, use) => {
    let [worker] = context.serviceWorkers();
    if (!worker) worker = await context.waitForEvent('serviceworker');
    await use(new URL(worker.url()).host);
  },
});

export async function launchExtensionContext(userDataDir: string): Promise<BrowserContext> {
  return chromium.launchPersistentContext(userDataDir, {
    channel: 'chromium',
    headless: true,
    args: [`--disable-extensions-except=${DIST_DIR}`, `--load-extension=${DIST_DIR}`, ...GPU_ARGS],
  });
}

type GpuFixtures = { gpuContext: BrowserContext; gpuPage: Page };

/** Plain Chromium page on TEST_ORIGIN with WebGPU enabled; used by GPU harness tests. */
export const gpuTest = base.extend<GpuFixtures>({
  // eslint-disable-next-line no-empty-pattern
  gpuContext: async ({}, use) => {
    const dir = mkdtempSync(join(tmpdir(), 'webgpu-vsr-gpu-'));
    const context = await chromium.launchPersistentContext(dir, { channel: process.env.PW_CHANNEL ?? 'chromium', headless: true, args: GPU_ARGS });
    await serveStatic(context);
    await use(context);
    await context.close();
    rmSync(dir, { recursive: true, force: true });
  },
  gpuPage: async ({ gpuContext }, use) => {
    const page = await gpuContext.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error' || msg.type() === 'warning') console.log(`[page ${msg.type()}] ${msg.text()}`);
    });
    await use(page);
  },
});

export { expect } from '@playwright/test';
