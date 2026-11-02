/**
 * Developer diagnostics (not user settings, never shown in the popup).
 *
 * Stored in `chrome.storage.local` under {@link DIAGNOSTICS_STORAGE_KEY}, written by hand from the
 * service-worker console or by the e2e suite, e.g.
 *
 *   chrome.storage.local.set({ diagnostics: { syntheticGpuLatencyMs: { 'rtx-vsr-medium': 40 } } })
 *
 * `syntheticGpuLatencyMs` adds a fixed number of milliseconds to every measured GPU time and to
 * the submit -> done latency of the named pipelines' frames before the budget controller (and the
 * telemetry) sees them, as if the GPU work took that much longer. It exercises the
 * automatic downgrade path on hardware that is fast enough to never trigger it; it does not slow
 * the GPU down. Remove the key (or set {}) to turn it off. Stored values are untrusted and
 * sanitised like the settings.
 */
import { QUALITY_PIPELINE_IDS } from './settings';

export const DIAGNOSTICS_STORAGE_KEY = 'diagnostics';
export const MAX_SYNTHETIC_LATENCY_MS = 1000;

export interface Diagnostics {
  /** Extra milliseconds per frame, keyed by pipeline id ('rtx-vsr-medium', 'rtx-vsr-low', 'bicubic-q0'). */
  syntheticGpuLatencyMs: Readonly<Record<string, number>>;
}

const PIPELINE_IDS: ReadonlySet<string> = new Set(Object.values(QUALITY_PIPELINE_IDS));

export function defaultDiagnostics(): Diagnostics {
  return { syntheticGpuLatencyMs: {} };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Keeps known pipeline ids with finite positive latencies, clamped to {@link MAX_SYNTHETIC_LATENCY_MS}. */
export function sanitizeDiagnostics(raw: unknown): Diagnostics {
  const out = defaultDiagnostics();
  if (!isRecord(raw) || !isRecord(raw.syntheticGpuLatencyMs)) return out;
  const latency: Record<string, number> = {};
  for (const [id, value] of Object.entries(raw.syntheticGpuLatencyMs)) {
    if (!PIPELINE_IDS.has(id) || typeof value !== 'number' || !Number.isFinite(value) || value <= 0) continue;
    latency[id] = Math.min(MAX_SYNTHETIC_LATENCY_MS, value);
  }
  out.syntheticGpuLatencyMs = latency;
  return out;
}

/** Synthetic latency for one pipeline (0 when none is configured). */
export function syntheticLatencyFor(diagnostics: Diagnostics, pipelineId: string): number {
  return Object.prototype.hasOwnProperty.call(diagnostics.syntheticGpuLatencyMs, pipelineId)
    ? (diagnostics.syntheticGpuLatencyMs[pipelineId] as number)
    : 0;
}

export interface DiagnosticsStorageArea {
  get(keys: string): Promise<Record<string, unknown>>;
}

export interface DiagnosticsChangedEvent {
  addListener(cb: (changes: Record<string, { newValue?: unknown }>, areaName: string) => void): void;
  removeListener(cb: (changes: Record<string, { newValue?: unknown }>, areaName: string) => void): void;
}

export async function loadDiagnostics(area: DiagnosticsStorageArea = chrome.storage.local as unknown as DiagnosticsStorageArea): Promise<Diagnostics> {
  const stored = await area.get(DIAGNOSTICS_STORAGE_KEY);
  return sanitizeDiagnostics(stored[DIAGNOSTICS_STORAGE_KEY]);
}

/** Calls `onChange` with sanitised diagnostics whenever the stored key changes. Returns an unsubscribe. */
export function watchDiagnostics(
  onChange: (diagnostics: Diagnostics) => void,
  events: DiagnosticsChangedEvent = chrome.storage.onChanged as unknown as DiagnosticsChangedEvent,
): () => void {
  const listener = (changes: Record<string, { newValue?: unknown }>, areaName: string): void => {
    if (areaName !== 'local') return;
    const change = changes[DIAGNOSTICS_STORAGE_KEY];
    if (!change) return;
    onChange(sanitizeDiagnostics(change.newValue));
  };
  events.addListener(listener);
  return () => events.removeListener(listener);
}
