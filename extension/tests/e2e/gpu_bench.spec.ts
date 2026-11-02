/**
 * Production-path GPU benchmarks (VideoFrame -> importExternalTexture -> pipeline -> canvas,
 * 3 frames in flight through the FrameRing), timed with WebGPU timestamp queries.
 * Chromium is launched with WebGPU developer features so timestamps are not quantised to
 * 100 us. Results are printed as `[bench]` lines and recorded in docs/BENCHMARKS.md.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium, test as base, type BrowserContext, type Page } from '@playwright/test';
import { GPU_ARGS, TEST_ORIGIN, expect, serveStatic } from './fixtures';
import { call, openHarness } from '../harness/client';
import type { ModelId } from '../../src/pipeline/types';

const test = base.extend<{ benchContext: BrowserContext; benchPage: Page }>({
  // eslint-disable-next-line no-empty-pattern
  benchContext: async ({}, use) => {
    const dir = mkdtempSync(join(tmpdir(), 'vsr-bench-'));
    const context = await chromium.launchPersistentContext(dir, {
      channel: process.env.PW_CHANNEL ?? 'chromium',
      headless: true,
      args: [...GPU_ARGS, '--enable-webgpu-developer-features'],
    });
    await serveStatic(context);
    await use(context);
    await context.close();
    rmSync(dir, { recursive: true, force: true });
  },
  benchPage: async ({ benchContext }, use) => {
    const page = await benchContext.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log(`[page error] ${m.text()}`);
    });
    await use(page);
  },
});

function stats(times: number[]) {
  const s = [...times].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  return { n: s.length, median: q(0.5), p95: q(0.95), min: s[0], max: s[s.length - 1], mean: s.reduce((a, b) => a + b, 0) / s.length };
}

const CASES: { name: string; pipeline: ModelId | 'q0'; w: number; h: number; ow: number; oh: number }[] = [
  { name: 'low 960x540->1920x1080 (direct)', pipeline: 'low', w: 960, h: 540, ow: 1920, oh: 1080 },
  { name: 'medium 960x540->1920x1080 (direct)', pipeline: 'medium', w: 960, h: 540, ow: 1920, oh: 1080 },
  { name: 'low 1280x720->1920x1080 (resample)', pipeline: 'low', w: 1280, h: 720, ow: 1920, oh: 1080 },
  { name: 'medium 1280x720->1920x1080 (resample)', pipeline: 'medium', w: 1280, h: 720, ow: 1920, oh: 1080 },
  { name: 'q0 960x540->1920x1080', pipeline: 'q0', w: 960, h: 540, ow: 1920, oh: 1080 },
  { name: 'q0 1280x720->1920x1080', pipeline: 'q0', w: 1280, h: 720, ow: 1920, oh: 1080 },
  { name: 'low 1280x720->2560x1440 (direct)', pipeline: 'low', w: 1280, h: 720, ow: 2560, oh: 1440 },
  { name: 'medium 1280x720->2560x1440 (direct)', pipeline: 'medium', w: 1280, h: 720, ow: 2560, oh: 1440 },
  { name: 'low 1920x1080->3840x2160 (direct)', pipeline: 'low', w: 1920, h: 1080, ow: 3840, oh: 2160 },
  { name: 'medium 1920x1080->3840x2160 (direct)', pipeline: 'medium', w: 1920, h: 1080, ow: 3840, oh: 2160 },
  { name: 'q0 1920x1080->3840x2160', pipeline: 'q0', w: 1920, h: 1080, ow: 3840, oh: 2160 },
];

test.describe('GPU benchmarks (timestamp queries)', () => {
  test.setTimeout(600000);

  test('production path GPU time per frame', async ({ benchPage }) => {
    await openHarness(benchPage, TEST_ORIGIN);
    const info = await call(benchPage, 'info');
    console.log(`[bench] adapter ${info.vendor} ${info.architecture} features=${info.features.join(',')}`);
    const results: Record<string, ReturnType<typeof stats>> = {};
    for (const c of CASES) {
      const r = await call(benchPage, 'bench', { pipeline: c.pipeline, width: c.w, height: c.h, outWidth: c.ow, outHeight: c.oh, frames: 200, warmup: 40 });
      expect(r.times).toHaveLength(200);
      expect(r.timingSources).toEqual(['timestamp']);
      const s = stats(r.times);
      results[c.name] = s;
      console.log(
        `[bench] ${c.name}: median ${s.median.toFixed(3)} ms, p95 ${s.p95.toFixed(3)} ms, min ${s.min.toFixed(3)}, max ${s.max.toFixed(3)} ` +
          `(n=${s.n}, memory ${(r.memoryBytes / 1e6).toFixed(3)} MB = pipeline ${(r.pipelineBytes / 1e6).toFixed(3)} + ring ${(r.ringBytes / 1e6).toFixed(6)}, ` +
          `ring-full waits ${r.waits}, wall ${(r.wallMs / r.frames).toFixed(2)} ms/frame)`,
      );
      for (const t of r.times) expect(t).toBeGreaterThan(0);
    }
    // performance target: low 540p -> 1080p well inside the 6.5 ms design budget
    expect(results['low 960x540->1920x1080 (direct)'].median).toBeLessThan(6.5);
  });
});
