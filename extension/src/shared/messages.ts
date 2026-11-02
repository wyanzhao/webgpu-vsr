/**
 * Typed runtime messaging between the service worker, popup and content scripts.
 *
 * Every receive path validates the message with {@link isExtensionMessage} (or the narrower
 * guards) before acting on it: messages cross process boundaries and content scripts run next to
 * untrusted pages, so nothing is trusted by shape alone.
 *
 * Request -> response pairs:
 *   GET_SETTINGS              -> GET_SETTINGS_RESPONSE   (service worker)
 *   UPDATE_SETTINGS {patch}   -> GET_SETTINGS_RESPONSE   (service worker; written settings)
 *   TELEMETRY_UPDATE          -> ACK                     (content -> service worker)
 *   GET_TAB_STATE {tabId}     -> TAB_STATE               (popup -> service worker)
 *   GET_FRAME_CONTEXT         -> FRAME_CONTEXT {topHost} (content -> service worker: the tab's
 *                                                         top-level hostname, for sub-frames)
 *   PING                      -> PONG                    (any)
 * Broadcast (no response): SETTINGS_CHANGED (service worker -> every frame of every tab).
 * Any request may be answered with ERROR.
 */

import { INELIGIBLE_REASONS, type EligibilityPoint, type IneligibleReason } from '../content/eligibility';
import { sanitizeSettingsPatch, type SettingsPatch, type UserSettings } from './settings';

export type { IneligibleReason };

export type TimingSource = 'timestamp' | 'cpu';

/** Live metrics of one processing session (IMPLEMENTATION_NOTES 5). */
export interface SessionTelemetry {
  inputWidth: number;
  inputHeight: number;
  outputWidth: number;
  outputHeight: number;
  fps: number;
  gpuTimeMeanMs: number;
  gpuTimeP95Ms: number;
  droppedFrames: number;
  activeModel: string;
  timingSource: TimingSource;
  /** Frames this session presented on the overlay canvas (monotonic for the session). */
  presentedFrames: number;
  /** Pipeline initialisations by this session (first start, resizes, tier switches, device recovery). */
  pipelineInits: number;
}

/**
 * - `eligible`: passed the gate; the GPU session is starting and has not presented a frame yet
 * - `active`: a GPU session presented at least one frame
 * - `ineligible`: failed the gate (`reason` set)
 * - `error`: the GPU session failed (`error` set); the native video is shown
 */
export type VideoState = 'eligible' | 'active' | 'ineligible' | 'error';

export interface VideoReport {
  /** Unique within its frame (`v1`, `v2`, ...). */
  id: string;
  state: VideoState;
  point: EligibilityPoint | null;
  reason: IneligibleReason | null;
  reasonText: string | null;
  sourceWidth: number;
  sourceHeight: number;
  targetWidth: number;
  targetHeight: number;
  telemetry: SessionTelemetry | null;
  error: string | null;
}

/** One frame's report as stored by the service worker. */
export interface FrameReport {
  frameId: number;
  host: string;
  url: string;
  videos: VideoReport[];
  updatedAt: number;
}

export interface TabState {
  tabId: number;
  /** Hostname of the tab's top frame as reported by its content script (null if none reported). */
  host: string | null;
  frames: FrameReport[];
}

export type GetSettingsMessage = { type: 'GET_SETTINGS' };
export type GetSettingsResponseMessage = { type: 'GET_SETTINGS_RESPONSE'; settings: UserSettings };
export type UpdateSettingsMessage = { type: 'UPDATE_SETTINGS'; patch: SettingsPatch };
export type SettingsChangedMessage = { type: 'SETTINGS_CHANGED'; settings: UserSettings };
export type TelemetryUpdateMessage = { type: 'TELEMETRY_UPDATE'; host: string; url: string; videos: VideoReport[] };
export type GetTabStateMessage = { type: 'GET_TAB_STATE'; tabId: number };
export type TabStateMessage = { type: 'TAB_STATE'; state: TabState; settings: UserSettings };
export type GetFrameContextMessage = { type: 'GET_FRAME_CONTEXT' };
/** `topHost`: hostname of the sender tab's top-level http(s) document, null for other schemes. */
export type FrameContextMessage = { type: 'FRAME_CONTEXT'; topHost: string | null };
export type PingMessage = { type: 'PING' };
export type PongMessage = { type: 'PONG'; version: string; context: 'service-worker' | 'content' };
export type AckMessage = { type: 'ACK' };
export type ErrorMessage = { type: 'ERROR'; error: string };

export type RequestMessage =
  | GetSettingsMessage
  | UpdateSettingsMessage
  | TelemetryUpdateMessage
  | GetTabStateMessage
  | GetFrameContextMessage
  | PingMessage;

export type ResponseMessage = GetSettingsResponseMessage | TabStateMessage | FrameContextMessage | PongMessage | AckMessage | ErrorMessage;

export type ExtensionMessage = RequestMessage | ResponseMessage | SettingsChangedMessage;

export type MessageType = ExtensionMessage['type'];

/** Successful response type for each request type. */
export interface ResponseFor {
  GET_SETTINGS: GetSettingsResponseMessage;
  UPDATE_SETTINGS: GetSettingsResponseMessage;
  TELEMETRY_UPDATE: AckMessage;
  GET_TAB_STATE: TabStateMessage;
  GET_FRAME_CONTEXT: FrameContextMessage;
  PING: PongMessage;
}

export const MAX_VIDEOS_PER_REPORT = 64;
const MAX_STRING = 2048;

// ---------------------------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isFiniteNonNeg(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0;
}

function isDimension(v: unknown): v is number {
  return isFiniteNonNeg(v) && Number.isInteger(v) && v <= 65536;
}

function isCount(v: unknown): v is number {
  return isFiniteNonNeg(v) && Number.isInteger(v);
}

function isShortString(v: unknown, max = MAX_STRING): v is string {
  return typeof v === 'string' && v.length <= max;
}

function isTabId(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0;
}

/** Strict structural check of {@link UserSettings} (exact field types and ranges). */
export function isUserSettings(v: unknown): v is UserSettings {
  if (!isRecord(v)) return false;
  if (typeof v.enabled !== 'boolean' || typeof v.hudEnabled !== 'boolean') return false;
  if (v.qualityLevel !== 0 && v.qualityLevel !== 1 && v.qualityLevel !== 2) return false;
  if (typeof v.sharpness !== 'number' || !(v.sharpness >= 0 && v.sharpness <= 1)) return false;
  if (!isRecord(v.siteOverrides)) return false;
  return Object.values(v.siteOverrides).every((x) => typeof x === 'boolean');
}

export function isSessionTelemetry(v: unknown): v is SessionTelemetry {
  if (!isRecord(v)) return false;
  return (
    isDimension(v.inputWidth) &&
    isDimension(v.inputHeight) &&
    isDimension(v.outputWidth) &&
    isDimension(v.outputHeight) &&
    isFiniteNonNeg(v.fps) &&
    isFiniteNonNeg(v.gpuTimeMeanMs) &&
    isFiniteNonNeg(v.gpuTimeP95Ms) &&
    isFiniteNonNeg(v.droppedFrames) &&
    Number.isInteger(v.droppedFrames) &&
    isShortString(v.activeModel, 64) &&
    (v.timingSource === 'timestamp' || v.timingSource === 'cpu') &&
    isCount(v.presentedFrames) &&
    isCount(v.pipelineInits)
  );
}

const REASON_SET: ReadonlySet<string> = new Set(INELIGIBLE_REASONS);

export function isIneligibleReason(v: unknown): v is IneligibleReason {
  return typeof v === 'string' && REASON_SET.has(v);
}

export function isVideoReport(v: unknown): v is VideoReport {
  if (!isRecord(v)) return false;
  if (!isShortString(v.id, 64) || v.id.length === 0) return false;
  if (v.state !== 'eligible' && v.state !== 'active' && v.state !== 'ineligible' && v.state !== 'error') return false;
  const pointOk = v.point === null || (typeof v.point === 'number' && Number.isInteger(v.point) && v.point >= 1 && v.point <= 9);
  if (!pointOk) return false;
  if (!(v.reason === null || isIneligibleReason(v.reason))) return false;
  if (!(v.reasonText === null || isShortString(v.reasonText, 256))) return false;
  if (v.state === 'ineligible' && (v.reason === null || v.point === null)) return false;
  if (v.state !== 'ineligible' && (v.reason !== null || v.point !== null)) return false;
  if (!isDimension(v.sourceWidth) || !isDimension(v.sourceHeight)) return false;
  if (!isDimension(v.targetWidth) || !isDimension(v.targetHeight)) return false;
  if (!(v.telemetry === null || isSessionTelemetry(v.telemetry))) return false;
  if (!(v.error === null || isShortString(v.error, 512))) return false;
  if (v.state === 'error' && v.error === null) return false;
  return true;
}

function isVideoReportList(v: unknown): v is VideoReport[] {
  return Array.isArray(v) && v.length <= MAX_VIDEOS_PER_REPORT && v.every(isVideoReport);
}

export function isFrameReport(v: unknown): v is FrameReport {
  return (
    isRecord(v) &&
    isTabId(v.frameId) &&
    isShortString(v.host, 253) &&
    isShortString(v.url) &&
    isVideoReportList(v.videos) &&
    isFiniteNonNeg(v.updatedAt)
  );
}

export function isTabState(v: unknown): v is TabState {
  return (
    isRecord(v) &&
    isTabId(v.tabId) &&
    (v.host === null || isShortString(v.host, 253)) &&
    Array.isArray(v.frames) &&
    v.frames.every(isFrameReport)
  );
}

/** Validates any extension message, including payloads. */
export function isExtensionMessage(v: unknown): v is ExtensionMessage {
  if (!isRecord(v) || typeof v.type !== 'string') return false;
  switch (v.type) {
    case 'GET_SETTINGS':
    case 'GET_FRAME_CONTEXT':
    case 'PING':
    case 'ACK':
      return true;
    case 'FRAME_CONTEXT':
      return v.topHost === null || (isShortString(v.topHost, 253) && v.topHost.length > 0);
    case 'GET_SETTINGS_RESPONSE':
    case 'SETTINGS_CHANGED':
      return isUserSettings(v.settings);
    case 'UPDATE_SETTINGS': {
      if (!isRecord(v.patch)) return false;
      const clean = sanitizeSettingsPatch(v.patch);
      // Reject patches carrying keys/values that do not survive validation.
      return clean !== null && JSON.stringify(clean) === JSON.stringify(normalizePatchForCompare(v.patch));
    }
    case 'TELEMETRY_UPDATE':
      return isShortString(v.host, 253) && isShortString(v.url) && isVideoReportList(v.videos);
    case 'GET_TAB_STATE':
      return isTabId(v.tabId);
    case 'TAB_STATE':
      return isTabState(v.state) && isUserSettings(v.settings);
    case 'PONG':
      return isShortString(v.version, 32) && (v.context === 'service-worker' || v.context === 'content');
    case 'ERROR':
      return isShortString(v.error);
    default:
      return false;
  }
}

/** Key order used by sanitizeSettingsPatch, so a valid patch compares equal to its clean form. */
function normalizePatchForCompare(patch: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of ['enabled', 'qualityLevel', 'sharpness', 'hudEnabled', 'siteOverrides']) {
    if (key in patch) out[key] = patch[key];
  }
  if (Object.keys(patch).some((k) => !(k in out))) out.__unknown = true;
  return out;
}

const REQUEST_TYPES: ReadonlySet<string> = new Set(['GET_SETTINGS', 'UPDATE_SETTINGS', 'TELEMETRY_UPDATE', 'GET_TAB_STATE', 'GET_FRAME_CONTEXT', 'PING']);

export function isRequestMessage(v: unknown): v is RequestMessage {
  return isExtensionMessage(v) && REQUEST_TYPES.has(v.type);
}

export function isSettingsChangedMessage(v: unknown): v is SettingsChangedMessage {
  return isExtensionMessage(v) && v.type === 'SETTINGS_CHANGED';
}

/** Checks that `v` is the expected successful response to a request of type `type`. */
export function isResponseFor<T extends keyof ResponseFor>(type: T, v: unknown): v is ResponseFor[T] {
  if (!isExtensionMessage(v)) return false;
  switch (type) {
    case 'GET_SETTINGS':
    case 'UPDATE_SETTINGS':
      return v.type === 'GET_SETTINGS_RESPONSE';
    case 'TELEMETRY_UPDATE':
      return v.type === 'ACK';
    case 'GET_TAB_STATE':
      return v.type === 'TAB_STATE';
    case 'GET_FRAME_CONTEXT':
      return v.type === 'FRAME_CONTEXT';
    case 'PING':
      return v.type === 'PONG';
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------------------------

export class MessageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MessageError';
  }
}

/** Minimal `chrome.runtime.sendMessage` surface, injectable for tests. */
export type RuntimeSend = (message: unknown) => Promise<unknown>;

/**
 * Sends a request to the extension runtime (service worker) and validates the response type.
 * Throws {@link MessageError} on an ERROR response or an invalid/missing response.
 */
export async function sendRequest<M extends RequestMessage>(
  message: M,
  send: RuntimeSend = (m) => chrome.runtime.sendMessage(m),
): Promise<ResponseFor[M['type']]> {
  const response = await send(message);
  if (isExtensionMessage(response) && response.type === 'ERROR') throw new MessageError(response.error);
  if (!isResponseFor(message.type, response)) {
    throw new MessageError(`invalid response to ${message.type}`);
  }
  return response as ResponseFor[M['type']];
}

// ---------------------------------------------------------------------------------------------
// Tab state presentation (toolbar badge)
// ---------------------------------------------------------------------------------------------

export interface BadgeSpec {
  text: '' | 'OFF' | 'ON' | 'ERR';
  color: string;
}

export const BADGE_COLORS = Object.freeze({ off: '#8E8E93', on: '#30D158', error: '#FF453A' });

/**
 * Toolbar badge for a tab:
 * - `OFF`: the extension is disabled, or the user disabled the tab's site;
 * - `ON`: at least one video in the tab (any frame) is being super-resolved (state `active`);
 * - `ERR`: no active video and at least one GPU session failed;
 * - empty otherwise (no video, or only ineligible / not-yet-started videos).
 */
export function badgeFor(settings: UserSettings, state: TabState | null): BadgeSpec {
  if (!settings.enabled) return { text: 'OFF', color: BADGE_COLORS.off };
  if (state?.host && settings.siteOverrides[state.host.toLowerCase()] === false) return { text: 'OFF', color: BADGE_COLORS.off };
  const videos = state ? state.frames.flatMap((f) => f.videos) : [];
  if (videos.some((v) => v.state === 'active')) return { text: 'ON', color: BADGE_COLORS.on };
  if (videos.some((v) => v.state === 'error')) return { text: 'ERR', color: BADGE_COLORS.error };
  return { text: '', color: BADGE_COLORS.off };
}

/** Builds a tab's state from stored frame reports (invalid entries are skipped). */
export function tabStateFromFrames(tabId: number, frames: readonly unknown[]): TabState {
  const valid = frames.filter(isFrameReport).sort((a, b) => a.frameId - b.frameId);
  const top = valid.find((f) => f.frameId === 0);
  return { tabId, host: top ? top.host : null, frames: valid };
}
