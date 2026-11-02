/**
 * Node-side helpers for the GPU harness page (used by tests/e2e/gpu_*.spec.ts).
 */
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Page } from '@playwright/test';
import type { HarnessApi } from './harness';
import { createWeightBundle } from '../../src/weights/loader';
import type { WeightBundle, WeightManifest } from '../../src/weights/types';
import type { ModelId } from '../../src/pipeline/types';
import type { RefWeights } from '../reference/network';

const HERE = dirname(fileURLToPath(import.meta.url));
const EXTENSION_ROOT = join(HERE, '..', '..');
export const HARNESS_PATH = '/gpu-harness.html';

/** Open the harness page on the secure test origin and wait until its API is installed. */
export async function openHarness(page: Page, origin: string): Promise<void> {
  if (!existsSync(join(EXTENSION_ROOT, 'dist-harness', 'harness.js'))) {
    throw new Error('dist-harness/harness.js is missing: run `npm run build:harness` first');
  }
  const html = readFileSync(join(HERE, 'index.html'), 'utf8');
  await page.route(`${origin}${HARNESS_PATH}`, (route) => route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html }));
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`${origin}${HARNESS_PATH}`);
  await page.waitForFunction(() => document.title === 'webgpu-vsr harness ready', undefined, { timeout: 30000 }).catch(() => {
    throw new Error(`harness did not start: ${errors.join('; ')}`);
  });
}

type Api = HarnessApi;
type Args<K extends keyof Api> = Parameters<Api[K]>;
type Ret<K extends keyof Api> = Awaited<ReturnType<Api[K]>>;

/** Call a harness entry point inside the page. */
export async function call<K extends keyof Api>(page: Page, name: K, ...args: Args<K>): Promise<Ret<K>> {
  return page.evaluate(
    async ([n, a]) => {
      const api = (globalThis as unknown as { mvsr: Record<string, (...x: unknown[]) => Promise<unknown>> }).mvsr;
      return api[n as string](...(a as unknown[]));
    },
    [name, args] as const,
  ) as Promise<Ret<K>>;
}

const bundleCache = new Map<ModelId, Promise<WeightBundle>>();

/** The shipped weight bundle, validated by the production loader. */
export function loadBundle(model: ModelId): Promise<WeightBundle> {
  let b = bundleCache.get(model);
  if (!b) {
    const dir = join(EXTENSION_ROOT, 'public', 'weights');
    const manifest = JSON.parse(readFileSync(join(dir, `${model}_manifest.json`), 'utf8')) as WeightManifest;
    const raw = readFileSync(join(dir, `${model}.bin`));
    b = createWeightBundle(model, manifest, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer);
    bundleCache.set(model, b);
  }
  return b;
}

export function refWeights(bundle: WeightBundle): RefWeights {
  return { get: (layer, kind, op) => bundle.tensor(layer, kind, op).data };
}

export function halvesFromBytes(bytes: Uint8Array): Float32Array {
  const copy = bytes.slice(0);
  return Float32Array.from(new Float16Array(copy.buffer, 0, copy.byteLength >> 1));
}

export function bytesFromHalves(values: ArrayLike<number>): Uint8Array {
  const h = new Float16Array(values.length);
  for (let i = 0; i < values.length; i++) h[i] = values[i];
  return new Uint8Array(h.buffer);
}

export interface DiffStats {
  count: number;
  maxAbs: number;
  meanAbs: number;
  p999: number;
  exactFraction: number;
}

export function diffStats(a: ArrayLike<number>, b: ArrayLike<number>): DiffStats {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  const d = new Float64Array(a.length);
  let sum = 0;
  let exact = 0;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i] - b[i]);
    if (!Number.isFinite(v)) throw new Error(`non-finite difference at ${i}: ${a[i]} vs ${b[i]}`);
    d[i] = v;
    sum += v;
    if (v === 0) exact++;
  }
  d.sort();
  return {
    count: a.length,
    maxAbs: d[d.length - 1],
    meanAbs: sum / a.length,
    p999: d[Math.min(d.length - 1, Math.floor(d.length * 0.999))],
    exactFraction: exact / a.length,
  };
}
