import { describe, expect, it } from 'vitest';
import {
  BADGE_COLORS,
  MAX_VIDEOS_PER_REPORT,
  MessageError,
  badgeFor,
  isExtensionMessage,
  isFrameReport,
  isRequestMessage,
  isResponseFor,
  isSessionTelemetry,
  isSettingsChangedMessage,
  isUserSettings,
  isVideoReport,
  sendRequest,
  tabStateFromFrames,
  type FrameReport,
  type SessionTelemetry,
  type TabState,
  type VideoReport,
} from '../../src/shared/messages';
import { defaultSettings } from '../../src/shared/settings';

const telemetry = (): SessionTelemetry => ({
  inputWidth: 960,
  inputHeight: 540,
  outputWidth: 1920,
  outputHeight: 1080,
  fps: 59.94,
  gpuTimeMeanMs: 4.2,
  gpuTimeP95Ms: 5.1,
  droppedFrames: 3,
  activeModel: 'rtx-vsr-low',
  timingSource: 'timestamp',
  presentedFrames: 1200,
  pipelineInits: 1,
});

const activeReport = (): VideoReport => ({
  id: 'v1',
  state: 'active',
  point: null,
  reason: null,
  reasonText: null,
  sourceWidth: 960,
  sourceHeight: 540,
  targetWidth: 1920,
  targetHeight: 1080,
  telemetry: telemetry(),
  error: null,
});

const ineligibleReport = (): VideoReport => ({
  id: 'v2',
  state: 'ineligible',
  point: 5,
  reason: 'no-upscale-needed',
  reasonText: 'Display size is less than 1.15x the source resolution',
  sourceWidth: 1920,
  sourceHeight: 1080,
  targetWidth: 1920,
  targetHeight: 1080,
  telemetry: null,
  error: null,
});

const frame = (frameId: number, videos: VideoReport[], host = 'example.com'): FrameReport => ({
  frameId,
  host,
  url: `https://${host}/`,
  videos,
  updatedAt: 1000,
});

describe('settings and telemetry guards', () => {
  it('isUserSettings requires exact types and ranges', () => {
    expect(isUserSettings(defaultSettings())).toBe(true);
    expect(isUserSettings({ ...defaultSettings(), qualityLevel: 3 })).toBe(false);
    expect(isUserSettings({ ...defaultSettings(), sharpness: 1.01 })).toBe(false);
    expect(isUserSettings({ ...defaultSettings(), enabled: 1 })).toBe(false);
    expect(isUserSettings({ ...defaultSettings(), siteOverrides: { 'a.com': 'yes' } })).toBe(false);
    expect(isUserSettings({ ...defaultSettings(), siteOverrides: null })).toBe(false);
  });

  it('isSessionTelemetry validates every field', () => {
    expect(isSessionTelemetry(telemetry())).toBe(true);
    expect(isSessionTelemetry({ ...telemetry(), timingSource: 'gpu' })).toBe(false);
    expect(isSessionTelemetry({ ...telemetry(), fps: -1 })).toBe(false);
    expect(isSessionTelemetry({ ...telemetry(), gpuTimeMeanMs: Number.NaN })).toBe(false);
    expect(isSessionTelemetry({ ...telemetry(), droppedFrames: 1.5 })).toBe(false);
    expect(isSessionTelemetry({ ...telemetry(), outputWidth: 1920.5 })).toBe(false);
    expect(isSessionTelemetry({ ...telemetry(), presentedFrames: -1 })).toBe(false);
    expect(isSessionTelemetry({ ...telemetry(), pipelineInits: 0.5 })).toBe(false);
    const { presentedFrames: _frames, ...noFrames } = telemetry();
    expect(isSessionTelemetry(noFrames)).toBe(false);
    const { activeModel: _omit, ...missing } = telemetry();
    expect(isSessionTelemetry(missing)).toBe(false);
  });

  it('isVideoReport enforces state/reason consistency', () => {
    expect(isVideoReport(activeReport())).toBe(true);
    expect(isVideoReport(ineligibleReport())).toBe(true);
    expect(isVideoReport({ ...ineligibleReport(), reason: null })).toBe(false);
    expect(isVideoReport({ ...ineligibleReport(), reason: 'because' })).toBe(false);
    expect(isVideoReport({ ...activeReport(), point: 3 })).toBe(false);
    expect(isVideoReport({ ...activeReport(), state: 'error' })).toBe(false); // error needs a message
    expect(isVideoReport({ ...activeReport(), state: 'error', telemetry: null, error: 'device lost' })).toBe(true);
    expect(isVideoReport({ ...activeReport(), id: '' })).toBe(false);
    expect(isVideoReport({ ...activeReport(), point: null, state: 'paused' })).toBe(false);
    expect(isVideoReport({ ...ineligibleReport(), point: 10 })).toBe(false);
  });
});

describe('isExtensionMessage', () => {
  const valid: unknown[] = [
    { type: 'GET_SETTINGS' },
    { type: 'PING' },
    { type: 'ACK' },
    { type: 'GET_SETTINGS_RESPONSE', settings: defaultSettings() },
    { type: 'SETTINGS_CHANGED', settings: defaultSettings() },
    { type: 'UPDATE_SETTINGS', patch: { enabled: false, siteOverrides: { 'a.com': null } } },
    { type: 'UPDATE_SETTINGS', patch: {} },
    { type: 'TELEMETRY_UPDATE', host: 'example.com', url: 'https://example.com/', videos: [activeReport(), ineligibleReport()] },
    { type: 'GET_TAB_STATE', tabId: 12 },
    { type: 'GET_FRAME_CONTEXT' },
    { type: 'FRAME_CONTEXT', topHost: 'blog.example' },
    { type: 'FRAME_CONTEXT', topHost: null },
    { type: 'TAB_STATE', state: { tabId: 12, host: 'example.com', frames: [frame(0, [])] }, settings: defaultSettings() },
    { type: 'PONG', version: '1.0.0', context: 'service-worker' },
    { type: 'ERROR', error: 'nope' },
  ];
  it.each(valid.map((m) => [(m as { type: string }).type, m]))('accepts a valid %s', (_t, m) => {
    expect(isExtensionMessage(m)).toBe(true);
  });

  const invalid: Array<[string, unknown]> = [
    ['null', null],
    ['array', [{ type: 'PING' }]],
    ['unknown type', { type: 'GET_STATE' }],
    ['settings missing', { type: 'SETTINGS_CHANGED' }],
    ['bad settings', { type: 'GET_SETTINGS_RESPONSE', settings: { enabled: true } }],
    ['patch not object', { type: 'UPDATE_SETTINGS', patch: 'enabled' }],
    ['patch unknown key', { type: 'UPDATE_SETTINGS', patch: { contentMode: 'anime' } }],
    ['patch out of range', { type: 'UPDATE_SETTINGS', patch: { sharpness: 4 } }],
    ['patch non-integer quality', { type: 'UPDATE_SETTINGS', patch: { qualityLevel: 1.5 } }],
    ['patch bad host', { type: 'UPDATE_SETTINGS', patch: { siteOverrides: { 'a b': true } } }],
    ['telemetry bad video', { type: 'TELEMETRY_UPDATE', host: 'a.com', url: 'https://a.com', videos: [{ id: 'v1' }] }],
    ['telemetry too many', { type: 'TELEMETRY_UPDATE', host: 'a.com', url: 'https://a.com', videos: Array.from({ length: MAX_VIDEOS_PER_REPORT + 1 }, activeReport) }],
    ['negative tab id', { type: 'GET_TAB_STATE', tabId: -1 }],
    ['fractional tab id', { type: 'GET_TAB_STATE', tabId: 1.5 }],
    ['frame context host missing', { type: 'FRAME_CONTEXT' }],
    ['frame context empty host', { type: 'FRAME_CONTEXT', topHost: '' }],
    ['frame context host not string', { type: 'FRAME_CONTEXT', topHost: 7 }],
    ['frame context host too long', { type: 'FRAME_CONTEXT', topHost: 'a'.repeat(254) }],
    ['pong bad context', { type: 'PONG', version: '1', context: 'page' }],
    ['error not string', { type: 'ERROR', error: 42 }],
  ];
  it.each(invalid)('rejects %s', (_name, m) => {
    expect(isExtensionMessage(m)).toBe(false);
  });

  it('distinguishes requests, broadcasts and responses', () => {
    expect(isRequestMessage({ type: 'PING' })).toBe(true);
    expect(isRequestMessage({ type: 'GET_TAB_STATE', tabId: 1 })).toBe(true);
    expect(isRequestMessage({ type: 'GET_FRAME_CONTEXT' })).toBe(true);
    expect(isRequestMessage({ type: 'FRAME_CONTEXT', topHost: null })).toBe(false);
    expect(isRequestMessage({ type: 'PONG', version: '1', context: 'content' })).toBe(false);
    expect(isRequestMessage({ type: 'SETTINGS_CHANGED', settings: defaultSettings() })).toBe(false);
    expect(isSettingsChangedMessage({ type: 'SETTINGS_CHANGED', settings: defaultSettings() })).toBe(true);
    expect(isSettingsChangedMessage({ type: 'SETTINGS_CHANGED', settings: {} })).toBe(false);
  });

  it('isResponseFor pairs each request with its response type', () => {
    const settingsResponse = { type: 'GET_SETTINGS_RESPONSE', settings: defaultSettings() };
    expect(isResponseFor('GET_SETTINGS', settingsResponse)).toBe(true);
    expect(isResponseFor('UPDATE_SETTINGS', settingsResponse)).toBe(true);
    expect(isResponseFor('PING', settingsResponse)).toBe(false);
    expect(isResponseFor('TELEMETRY_UPDATE', { type: 'ACK' })).toBe(true);
    expect(isResponseFor('GET_TAB_STATE', { type: 'ACK' })).toBe(false);
    expect(isResponseFor('GET_FRAME_CONTEXT', { type: 'FRAME_CONTEXT', topHost: 'a.com' })).toBe(true);
    expect(isResponseFor('GET_FRAME_CONTEXT', { type: 'ACK' })).toBe(false);
  });
});

describe('sendRequest', () => {
  it('returns a validated response', async () => {
    const response = await sendRequest({ type: 'GET_SETTINGS' }, async () => ({ type: 'GET_SETTINGS_RESPONSE', settings: defaultSettings() }));
    expect(response.settings).toEqual(defaultSettings());
  });

  it('throws MessageError with the ERROR text', async () => {
    await expect(sendRequest({ type: 'PING' }, async () => ({ type: 'ERROR', error: 'denied' }))).rejects.toThrow(new MessageError('denied'));
  });

  it('throws on a mismatched or malformed response', async () => {
    await expect(sendRequest({ type: 'PING' }, async () => ({ type: 'ACK' }))).rejects.toBeInstanceOf(MessageError);
    await expect(sendRequest({ type: 'PING' }, async () => undefined)).rejects.toThrow('invalid response to PING');
  });

  it('passes the exact message to the transport', async () => {
    const sent: unknown[] = [];
    await sendRequest({ type: 'GET_TAB_STATE', tabId: 7 }, async (m) => {
      sent.push(m);
      return { type: 'TAB_STATE', state: { tabId: 7, host: null, frames: [] }, settings: defaultSettings() };
    });
    expect(sent).toEqual([{ type: 'GET_TAB_STATE', tabId: 7 }]);
  });
});

describe('tab state and badge', () => {
  it('tabStateFromFrames drops invalid frames, sorts, and takes the host from frame 0', () => {
    const state = tabStateFromFrames(4, [frame(7, [activeReport()], 'cdn.example.net'), { frameId: 'x' }, frame(0, [], 'example.com')]);
    expect(state.tabId).toBe(4);
    expect(state.host).toBe('example.com');
    expect(state.frames.map((f) => f.frameId)).toEqual([0, 7]);
    expect(isFrameReport(state.frames[1])).toBe(true);
    expect(tabStateFromFrames(4, [frame(3, [])]).host).toBeNull();
  });

  const tab = (videos: VideoReport[], host = 'example.com'): TabState => ({ tabId: 1, host, frames: [frame(0, videos, host)] });

  it('OFF when disabled globally, even with an active video', () => {
    expect(badgeFor({ ...defaultSettings(), enabled: false }, tab([activeReport()]))).toEqual({ text: 'OFF', color: BADGE_COLORS.off });
  });

  it('OFF when the tab site is disabled by the user', () => {
    expect(badgeFor({ ...defaultSettings(), siteOverrides: { 'example.com': false } }, tab([activeReport()])).text).toBe('OFF');
    expect(badgeFor({ ...defaultSettings(), siteOverrides: { 'other.com': false } }, tab([activeReport()])).text).toBe('ON');
  });

  it('ON when any frame has an active video', () => {
    const state: TabState = { tabId: 1, host: 'example.com', frames: [frame(0, [ineligibleReport()]), frame(5, [activeReport()], 'player.example.net')] };
    expect(badgeFor(defaultSettings(), state)).toEqual({ text: 'ON', color: BADGE_COLORS.on });
  });

  it('ERR when a session failed and none is active; empty otherwise', () => {
    const failed: VideoReport = { ...activeReport(), state: 'error', telemetry: null, error: 'device lost' };
    expect(badgeFor(defaultSettings(), tab([failed, ineligibleReport()]))).toEqual({ text: 'ERR', color: BADGE_COLORS.error });
    expect(badgeFor(defaultSettings(), tab([ineligibleReport()])).text).toBe('');
    expect(badgeFor(defaultSettings(), null).text).toBe('');
  });
});
