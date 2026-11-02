import { describe, expect, it } from 'vitest';
import {
  DIAGNOSTICS_STORAGE_KEY,
  MAX_SYNTHETIC_LATENCY_MS,
  defaultDiagnostics,
  loadDiagnostics,
  sanitizeDiagnostics,
  syntheticLatencyFor,
  watchDiagnostics,
} from '../../src/shared/diagnostics';

describe('developer diagnostics', () => {
  it('keeps known pipeline ids with positive finite latencies, clamped', () => {
    expect(
      sanitizeDiagnostics({
        syntheticGpuLatencyMs: {
          'rtx-vsr-medium': 40,
          'rtx-vsr-low': 5000,
          'bicubic-q0': 0,
          'rtx-vsr-high': 10,
          constructor: 3,
          __proto__: 4,
        },
        other: true,
      }),
    ).toEqual({ syntheticGpuLatencyMs: { 'rtx-vsr-medium': 40, 'rtx-vsr-low': MAX_SYNTHETIC_LATENCY_MS } });
    for (const junk of [undefined, null, 7, 'x', [], { syntheticGpuLatencyMs: [5] }, { syntheticGpuLatencyMs: { 'bicubic-q0': Number.NaN } }, { syntheticGpuLatencyMs: { 'bicubic-q0': -1 } }, { syntheticGpuLatencyMs: { 'bicubic-q0': '9' } }]) {
      expect(sanitizeDiagnostics(junk)).toEqual(defaultDiagnostics());
    }
  });

  it('looks up the latency of a pipeline (0 when unset)', () => {
    const d = sanitizeDiagnostics({ syntheticGpuLatencyMs: { 'bicubic-q0': 12.5 } });
    expect(syntheticLatencyFor(d, 'bicubic-q0')).toBe(12.5);
    expect(syntheticLatencyFor(d, 'rtx-vsr-low')).toBe(0);
    expect(syntheticLatencyFor(d, 'toString')).toBe(0);
  });

  it('loads from and watches chrome.storage.local (local area, own key only)', async () => {
    const area = { get: async (key: string) => ({ [key]: { syntheticGpuLatencyMs: { 'rtx-vsr-low': 30 } } }) };
    expect(await loadDiagnostics(area)).toEqual({ syntheticGpuLatencyMs: { 'rtx-vsr-low': 30 } });
    expect(await loadDiagnostics({ get: async () => ({}) })).toEqual(defaultDiagnostics());

    type Listener = (changes: Record<string, { newValue?: unknown }>, area: string) => void;
    const listeners = new Set<Listener>();
    const events = { addListener: (l: Listener) => listeners.add(l), removeListener: (l: Listener) => listeners.delete(l) };
    const seen: unknown[] = [];
    const stop = watchDiagnostics((d) => seen.push(d), events);
    const fire = (changes: Record<string, { newValue?: unknown }>, areaName: string) => listeners.forEach((l) => l(changes, areaName));
    fire({ [DIAGNOSTICS_STORAGE_KEY]: { newValue: { syntheticGpuLatencyMs: { 'rtx-vsr-medium': 40 } } } }, 'local');
    fire({ [DIAGNOSTICS_STORAGE_KEY]: { newValue: { syntheticGpuLatencyMs: { 'rtx-vsr-medium': 1 } } } }, 'session');
    fire({ settings: { newValue: {} } }, 'local');
    fire({ [DIAGNOSTICS_STORAGE_KEY]: {} }, 'local'); // key removed
    expect(seen).toEqual([{ syntheticGpuLatencyMs: { 'rtx-vsr-medium': 40 } }, defaultDiagnostics()]);
    stop();
    expect(listeners.size).toBe(0);
  });
});
