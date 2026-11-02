import { describe, expect, it, vi } from 'vitest';
import {
  EligibilityLatches,
  collectEligibilityInput,
  INELIGIBLE_REASONS,
  REASON_POINT,
  REASON_TEXT,
  evaluateEligibility,
  isHdrTransfer,
  mediaSourceIsSameOrigin,
  type EligibilityInput,
  type FrameProbeResult,
} from '../../src/content/eligibility';
import { FakeEnv, FakeNode } from './fake_dom';
import { defaultSettings } from '../../src/shared/settings';

/**
 * A 960x540 same-origin SDR video shown in a 1280x720 box at DPR 2: target 1920x1080 after the
 * 1080p ceiling (2560x1440 * 0.75), i.e. a 2.0x upscale. Passes all nine points.
 */
function passing(overrides: Partial<EligibilityInput> = {}): EligibilityInput {
  return {
    drm: { mediaKeys: false, encryptedEventLatched: false },
    presentation: { pictureInPicture: false, videoIsFullscreenElement: false },
    geometry: {
      connected: true,
      layoutSupported: true,
      display: 'block',
      visibility: 'visible',
      opacity: 1,
      hasLayoutBox: true,
      clientWidth: 1280,
      clientHeight: 720,
      videoWidth: 960,
      videoHeight: 540,
      objectFit: 'contain',
      objectPosition: '50% 50%',
      transform: 'none',
      scale: 'none',
      rotate: 'none',
      translate: 'none',
      currentCSSZoom: 1,
      ancestors: [],
      renderedWidth: 1280,
      renderedHeight: 720,
      offsetWidth: 1280,
      offsetHeight: 720,
    },
    origin: { sourceUrl: 'https://example.com/v.webm', documentOrigin: 'https://example.com', crossOriginAttribute: null, taintLatched: false },
    target: { width: 1920, height: 1080 },
    settings: defaultSettings(),
    hostname: 'example.com',
    topHostname: 'example.com',
    siteRuleEnabled: true,
    probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
    gpuAvailable: () => true,
    budgetHealthy: () => true,
    ...overrides,
  };
}

function withGeometry(g: Partial<EligibilityInput['geometry']>): EligibilityInput {
  const base = passing();
  return { ...base, geometry: { ...base.geometry, ...g } };
}

function withOrigin(o: Partial<EligibilityInput['origin']>): EligibilityInput {
  const base = passing();
  return { ...base, origin: { ...base.origin, ...o } };
}

describe('reason table', () => {
  it('every reason has a point in 1..9 and human text', () => {
    for (const r of INELIGIBLE_REASONS) {
      expect(REASON_POINT[r]).toBeGreaterThanOrEqual(1);
      expect(REASON_POINT[r]).toBeLessThanOrEqual(9);
      expect(REASON_TEXT[r].length).toBeGreaterThan(5);
    }
    expect(new Set(Object.values(REASON_POINT))).toEqual(new Set([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  });
});

describe('baseline', () => {
  it('a fully valid input is eligible and carries source/target sizes', () => {
    expect(evaluateEligibility(passing())).toEqual({
      eligible: true,
      point: null,
      reason: null,
      message: null,
      sourceWidth: 960,
      sourceHeight: 540,
      targetWidth: 1920,
      targetHeight: 1080,
    });
  });
});

describe('point 1: DRM / EME', () => {
  it('fails when MediaKeys are attached', () => {
    const r = evaluateEligibility(passing({ drm: { mediaKeys: true, encryptedEventLatched: false } }));
    expect([r.eligible, r.point, r.reason, r.message]).toEqual([false, 1, 'drm', REASON_TEXT.drm]);
  });
  it('fails when an encrypted/waitingforkey event was latched', () => {
    expect(evaluateEligibility(passing({ drm: { mediaKeys: false, encryptedEventLatched: true } })).reason).toBe('drm');
  });
});

describe('point 2: presentation mode', () => {
  it('fails in Picture-in-Picture', () => {
    const r = evaluateEligibility(passing({ presentation: { pictureInPicture: true, videoIsFullscreenElement: false } }));
    expect([r.point, r.reason]).toEqual([2, 'picture-in-picture']);
  });
  it('fails when the video element itself is fullscreen', () => {
    const r = evaluateEligibility(passing({ presentation: { pictureInPicture: false, videoIsFullscreenElement: true } }));
    expect([r.point, r.reason]).toEqual([2, 'native-fullscreen']);
  });
});

describe('point 3: DOM geometry / visibility', () => {
  it.each([
    ['detached', { connected: false }, 'not-connected'],
    ['display none', { display: 'none' }, 'not-rendered'],
    ['visibility hidden', { visibility: 'hidden' }, 'not-rendered'],
    ['no layout box', { hasLayoutBox: false }, 'not-rendered'],
    ['no decoded size', { videoWidth: 0 }, 'no-video-dimensions'],
    ['no decoded height', { videoHeight: 0 }, 'no-video-dimensions'],
    ['narrower than 128', { clientWidth: 127 }, 'too-small'],
    ['shorter than 128', { clientHeight: 127 }, 'too-small'],
    ['unknown object-fit', { objectFit: 'bogus' }, 'unsupported-layout'],
    ['unparsable object-position', { objectPosition: 'calc(1em + 3vw) 0' }, 'unsupported-layout'],
  ] as const)('%s -> %s', (_name, g, reason) => {
    const r = evaluateEligibility(withGeometry(g));
    expect([r.point, r.reason]).toEqual([3, reason]);
  });

  it('rejects the video own transform even when its bounding size is unchanged', () => {
    expect(evaluateEligibility(withGeometry({ transform: 'matrix(1, 0, 0, 1, 20, 0)' })).reason).toBe('unsupported-layout');
  });

  it.each([{ renderedWidth: 1280.51 }, { renderedHeight: 720.51 }])('rejects ancestor scaling on either axis: %o', (geometry) => {
    expect(evaluateEligibility(withGeometry(geometry)).reason).toBe('unsupported-layout');
  });

  it('accepts untransformed layout sizes within half a CSS pixel', () => {
    expect(evaluateEligibility(withGeometry({ renderedWidth: 1280.5, renderedHeight: 719.5 })).eligible).toBe(true);
    expect(evaluateEligibility(passing()).eligible).toBe(true);
  });

  it('passes at exactly 128x128 with cover / off-centre position', () => {
    const r = evaluateEligibility({
      ...withGeometry({ clientWidth: 128, clientHeight: 128, objectFit: 'cover', objectPosition: '0% 100%', videoWidth: 64, videoHeight: 64 }),
      target: { width: 128, height: 128 },
    });
    expect(r.eligible).toBe(true);
  });
});

describe('point 4: CORS / taint', () => {
  it('fails for a cross-origin URL without a crossorigin attribute', () => {
    const r = evaluateEligibility(withOrigin({ sourceUrl: 'https://cdn.other.net/v.mp4' }));
    expect([r.point, r.reason]).toEqual([4, 'cross-origin']);
  });
  it('passes for a cross-origin URL with crossorigin="anonymous"', () => {
    expect(evaluateEligibility(withOrigin({ sourceUrl: 'https://cdn.other.net/v.mp4', crossOriginAttribute: 'anonymous' })).eligible).toBe(true);
    expect(evaluateEligibility(withOrigin({ sourceUrl: 'https://cdn.other.net/v.mp4', crossOriginAttribute: '' })).eligible).toBe(true);
  });
  it('fails when the session latched a SecurityError, even same-origin', () => {
    expect(evaluateEligibility(withOrigin({ taintLatched: true })).reason).toBe('tainted');
  });
  it('fails when the frame probe reports a tainted frame', () => {
    expect(evaluateEligibility(passing({ probeFrame: () => ({ tainted: true, transfer: null }) })).reason).toBe('tainted');
  });
  it('same-origin rules for media URLs', () => {
    const o = 'https://example.com';
    expect(mediaSourceIsSameOrigin('https://example.com/a.webm', o)).toBe(true);
    expect(mediaSourceIsSameOrigin('/relative.webm', o)).toBe(true);
    expect(mediaSourceIsSameOrigin('blob:https://example.com/0b4c-11', o)).toBe(true);
    expect(mediaSourceIsSameOrigin('blob:https://evil.com/0b4c-11', o)).toBe(false);
    expect(mediaSourceIsSameOrigin('data:video/webm;base64,AAAA', o)).toBe(true);
    expect(mediaSourceIsSameOrigin(null, o)).toBe(true); // srcObject stream
    expect(mediaSourceIsSameOrigin('http://example.com/a.webm', o)).toBe(false); // scheme differs
    expect(mediaSourceIsSameOrigin('https://example.com:8443/a.webm', o)).toBe(false); // port differs
  });
});

describe('point 5: resolution / upscale threshold', () => {
  it('fails below 1.15x on the limiting axis', () => {
    // 1280x720 into 1440x810: exactly 1.125x
    const r = evaluateEligibility({ ...withGeometry({ videoWidth: 1280, videoHeight: 720 }), target: { width: 1440, height: 810 } });
    expect([r.point, r.reason]).toEqual([5, 'no-upscale-needed']);
  });
  it('passes at exactly 1.15x', () => {
    // 1000x500 -> 1150x575 is exactly 1.15 on both axes
    const r = evaluateEligibility({ ...withGeometry({ videoWidth: 1000, videoHeight: 500 }), target: { width: 1150, height: 575 } });
    expect(r.eligible).toBe(true);
  });
  it('uses the limiting axis: a stretched fill with no vertical upscale fails', () => {
    // 640x480 -> 1280x480: 2x horizontally but 1x vertically
    const r = evaluateEligibility({ ...withGeometry({ videoWidth: 640, videoHeight: 480 }), target: { width: 1280, height: 480 } });
    expect(r.reason).toBe('no-upscale-needed');
  });
  it('fails when the 1080p ceiling leaves a 1080p source unscaled', () => {
    const r = evaluateEligibility({ ...withGeometry({ videoWidth: 1920, videoHeight: 1080 }), target: { width: 1920, height: 1080 } });
    expect(r.reason).toBe('no-upscale-needed');
  });
  it('fails for an empty target', () => {
    expect(evaluateEligibility(passing({ target: { width: 0, height: 0 } })).reason).toBe('no-upscale-needed');
  });
});

describe('point 6: dynamic range', () => {
  it.each(['pq', 'hlg', 'PQ', 'smpte2084', 'arib-std-b67'])('rejects transfer %s', (transfer) => {
    const r = evaluateEligibility(passing({ probeFrame: () => ({ tainted: false, transfer }) }));
    expect([r.point, r.reason]).toEqual([6, 'hdr']);
  });
  it.each(['bt709', 'smpte170m', 'iec61966-2-1', null])('accepts transfer %s', (transfer) => {
    expect(evaluateEligibility(passing({ probeFrame: () => ({ tainted: false, transfer }) })).eligible).toBe(true);
  });
  it('isHdrTransfer is case-insensitive and null-safe', () => {
    expect(isHdrTransfer('HLG')).toBe(true);
    expect(isHdrTransfer(undefined)).toBe(false);
  });
});

describe('point 7: user / site settings', () => {
  it('fails when the extension is disabled', () => {
    const r = evaluateEligibility(passing({ settings: { ...defaultSettings(), enabled: false } }));
    expect([r.point, r.reason]).toEqual([7, 'extension-disabled']);
  });
  it('fails when the user disabled the site (case-insensitive host)', () => {
    const r = evaluateEligibility(passing({ hostname: 'Example.com', settings: { ...defaultSettings(), siteOverrides: { 'example.com': false } } }));
    expect([r.point, r.reason]).toEqual([7, 'site-disabled']);
  });
  it('fails when the site rule is disabled and the user did not override it', () => {
    expect(evaluateEligibility(passing({ siteRuleEnabled: false })).reason).toBe('site-rule-disabled');
  });
  it('an explicit user enable wins over a disabled site rule', () => {
    expect(evaluateEligibility(passing({ siteRuleEnabled: false, settings: { ...defaultSettings(), siteOverrides: { 'example.com': true } } })).eligible).toBe(true);
  });
  it('an override for another host does not apply', () => {
    expect(evaluateEligibility(passing({ settings: { ...defaultSettings(), siteOverrides: { 'other.com': false } } })).eligible).toBe(true);
  });
  it("the tab's site switch also stops a cross-origin player iframe (popup per-site switch)", () => {
    const iframe = { hostname: 'www.youtube-nocookie.com', topHostname: 'blog.example' };
    const off = { ...defaultSettings(), siteOverrides: { 'blog.example': false } };
    const r = evaluateEligibility(passing({ ...iframe, settings: off }));
    expect([r.point, r.reason]).toEqual([7, 'site-disabled']);
    // other tabs embedding the same player are not affected
    expect(evaluateEligibility(passing({ ...iframe, topHostname: 'news.example', settings: off })).eligible).toBe(true);
    // the player's own host turned off stays off on every embedding page, even one switched on
    const playerOff = { ...defaultSettings(), siteOverrides: { 'www.youtube-nocookie.com': false, 'blog.example': true } };
    expect(evaluateEligibility(passing({ ...iframe, settings: playerOff })).reason).toBe('site-disabled');
    // the embedding site switched ON does not override the player's own site rule
    const on = { ...defaultSettings(), siteOverrides: { 'blog.example': true } };
    expect(evaluateEligibility(passing({ ...iframe, siteRuleEnabled: false, settings: on })).reason).toBe('site-rule-disabled');
    // unknown top-level site: only the frame's own host counts
    expect(evaluateEligibility(passing({ ...iframe, topHostname: null, settings: off })).eligible).toBe(true);
  });
});

describe('point 8: WebGPU device', () => {
  it('fails when no device is available', () => {
    const r = evaluateEligibility(passing({ gpuAvailable: () => false }));
    expect([r.point, r.reason]).toEqual([8, 'webgpu-unavailable']);
  });
});

describe('point 9: frame budget', () => {
  it('fails when throttled', () => {
    const r = evaluateEligibility(passing({ budgetHealthy: () => false }));
    expect([r.point, r.reason]).toEqual([9, 'frame-budget']);
  });
});

describe('ordering', () => {
  /** Input where points listed in `failing` fail and every other point passes. */
  function failing(points: number[]): EligibilityInput {
    let input = passing();
    const set = new Set(points);
    if (set.has(1)) input = { ...input, drm: { mediaKeys: true, encryptedEventLatched: false } };
    if (set.has(2)) input = { ...input, presentation: { pictureInPicture: true, videoIsFullscreenElement: false } };
    if (set.has(3)) input = { ...input, geometry: { ...input.geometry, clientWidth: 100 } };
    if (set.has(4)) input = { ...input, origin: { ...input.origin, taintLatched: true } };
    if (set.has(5)) input = { ...input, target: { width: 1000, height: 562 } };
    if (set.has(6)) input = { ...input, probeFrame: () => ({ tainted: false, transfer: 'pq' }) };
    if (set.has(7)) input = { ...input, settings: { ...input.settings, enabled: false } };
    if (set.has(8)) input = { ...input, gpuAvailable: () => false };
    if (set.has(9)) input = { ...input, budgetHealthy: () => false };
    return input;
  }

  it('each point fails alone with its own number', () => {
    for (let p = 1; p <= 9; p++) expect(evaluateEligibility(failing([p])).point).toBe(p);
  });

  it('with all nine failing, point 1 is reported', () => {
    expect(evaluateEligibility(failing([1, 2, 3, 4, 5, 6, 7, 8, 9])).point).toBe(1);
  });

  it('the earliest failing point wins for every suffix of failures', () => {
    for (let first = 1; first <= 9; first++) {
      const pts = Array.from({ length: 10 - first }, (_, i) => first + i);
      expect(evaluateEligibility(failing(pts)).point).toBe(first);
    }
  });

  it('pairs report the smaller point regardless of which', () => {
    expect(evaluateEligibility(failing([9, 4])).point).toBe(4);
    expect(evaluateEligibility(failing([6, 5])).point).toBe(5);
    expect(evaluateEligibility(failing([8, 7])).point).toBe(7);
    expect(evaluateEligibility(failing([3, 2])).point).toBe(2);
  });

  it('probes are lazy: not called once an earlier point fails, and the frame is probed once', () => {
    const probeFrame = vi.fn((): FrameProbeResult => ({ tainted: false, transfer: 'bt709' }));
    const gpuAvailable = vi.fn(() => true);
    const budgetHealthy = vi.fn(() => true);
    evaluateEligibility({ ...failing([3]), probeFrame, gpuAvailable, budgetHealthy });
    expect(probeFrame).not.toHaveBeenCalled();
    expect(gpuAvailable).not.toHaveBeenCalled();
    expect(budgetHealthy).not.toHaveBeenCalled();

    evaluateEligibility({ ...failing([7]), probeFrame, gpuAvailable, budgetHealthy });
    expect(probeFrame).toHaveBeenCalledTimes(1); // used for points 4 and 6
    expect(gpuAvailable).not.toHaveBeenCalled();

    evaluateEligibility({ ...passing(), probeFrame, gpuAvailable, budgetHealthy });
    expect(probeFrame).toHaveBeenCalledTimes(2);
    expect(gpuAvailable).toHaveBeenCalledTimes(1);
    expect(budgetHealthy).toHaveBeenCalledTimes(1);
  });
});

describe('EligibilityLatches', () => {
  class FakeTarget {
    listeners = new Map<string, Set<() => void>>();
    addEventListener(type: string, l: () => void): void {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type)!.add(l);
    }
    removeEventListener(type: string, l: () => void): void {
      this.listeners.get(type)?.delete(l);
    }
    dispatch(type: string): void {
      this.listeners.get(type)?.forEach((l) => l());
    }
  }

  it('latches DRM on encrypted and waitingforkey, notifies, and unsubscribes', () => {
    const latches = new EligibilityLatches();
    const a = new FakeTarget();
    const b = new FakeTarget();
    const onLatched = vi.fn();
    const stopA = latches.watchDrmEvents(a as unknown as EventTarget, onLatched);
    latches.watchDrmEvents(b as unknown as EventTarget);
    expect(latches.isDrmLatched(a)).toBe(false);
    a.dispatch('encrypted');
    expect(latches.isDrmLatched(a)).toBe(true);
    expect(onLatched).toHaveBeenCalledTimes(1);
    b.dispatch('waitingforkey');
    expect(latches.isDrmLatched(b)).toBe(true);

    stopA();
    expect(a.listeners.get('encrypted')?.size).toBe(0);
    expect(a.listeners.get('waitingforkey')?.size).toBe(0);
  });

  it('taint and DRM latches are per video and cleared by reset', () => {
    const latches = new EligibilityLatches();
    const v1 = {};
    const v2 = {};
    latches.latchTaint(v1);
    latches.latchDrm(v1);
    expect([latches.isTaintLatched(v1), latches.isTaintLatched(v2)]).toEqual([true, false]);
    latches.reset(v1);
    expect([latches.isTaintLatched(v1), latches.isDrmLatched(v1)]).toEqual([false, false]);
  });
});

describe('live transform collection', () => {
  it.each([
    ['clip', 'rect(0px, 400px, 300px, 0px)'],
    ['clipPath', 'circle(40%)'],
    ['maskImage', 'linear-gradient(black, transparent)'],
    ['webkitMaskImage', 'linear-gradient(black, transparent)'],
    ['mixBlendMode', 'multiply'],
    ['objectViewBox', 'inset(10%)'],
    ['backdropFilter', 'blur(4px)'],
    ['webkitBackdropFilter', 'blur(4px)'],
    ['position', 'fixed'],
  ] as const)('rejects video effect %s', (property, value) => {
    const fx = new FakeEnv();
    const parent = new FakeNode();
    parent.connectedRoot = true;
    const video = new FakeNode();
    parent.appendChild(video);
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    const collect = () => collectEligibilityInput(video as unknown as HTMLVideoElement, {
      settings: defaultSettings(), hostname: 'example.com', topHostname: 'example.com', siteRuleEnabled: true,
      latches: new EligibilityLatches(), probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
      gpuAvailable: () => true, budgetHealthy: () => true,
      win: { document: fx.doc as unknown as Document, getComputedStyle: fx.env().getComputedStyle,
        devicePixelRatio: 1, location: { origin: 'https://example.com' } as Location },
    });
    // Absent optional properties are supported.
    expect(evaluateEligibility(collect()).eligible).toBe(true);
    video.computed[property] = value;
    expect(evaluateEligibility(collect()).reason).toBe('unsupported-layout');
    video.computed[property] = property === 'mixBlendMode' ? 'normal' : property === 'clip' ? 'auto' : property === 'position' ? 'static' : 'none';
    expect(evaluateEligibility(collect()).eligible).toBe(true);
  });

  it.each([
    ['inset box-shadow', { boxShadow: 'inset 0px 0px 8px black' }, 'block'],
    ['negative outline-offset', { outlineStyle: 'solid', outlineOffset: '-0.5px' }, 'block'],
    ...['flex', 'inline-flex', 'grid', 'inline-grid'].map((display) => [`${display} order`, { order: '-1' }, display] as const),
  ] as const)('rejects %s at the eligibility gate', (_name, effect, display) => {
    const fx = new FakeEnv();
    const parent = new FakeNode();
    parent.connectedRoot = true;
    const video = new FakeNode();
    parent.appendChild(video);
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    parent.computed.display = display;
    const collect = () => collectEligibilityInput(video as unknown as HTMLVideoElement, {
      settings: defaultSettings(), hostname: 'example.com', topHostname: 'example.com', siteRuleEnabled: true,
      latches: new EligibilityLatches(), probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
      gpuAvailable: () => true, budgetHealthy: () => true,
      win: { document: fx.doc as unknown as Document, getComputedStyle: fx.env().getComputedStyle,
        devicePixelRatio: 1, location: { origin: 'https://example.com' } as Location },
    });
    expect(evaluateEligibility(collect()).eligible).toBe(true);
    Object.assign(video.computed, effect);
    expect(evaluateEligibility(collect()).reason).toBe('unsupported-layout');
  });

  it('rejects native video fullscreen inside nested open shadow roots', () => {
    const fx = new FakeEnv();
    const host = new FakeNode();
    host.connectedRoot = true;
    const innerHost = Object.assign(new FakeNode(), { parentNode: { host }, connectedRoot: true });
    const shadowParent = Object.assign(new FakeNode(), { parentNode: { host: innerHost }, connectedRoot: true });
    const video = new FakeNode();
    shadowParent.appendChild(video);
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    host.shadowRoot = { fullscreenElement: innerHost };
    innerHost.shadowRoot = { fullscreenElement: video };
    fx.doc.fullscreenElement = host;
    const input = collectEligibilityInput(video as unknown as HTMLVideoElement, {
      settings: defaultSettings(), hostname: 'example.com', topHostname: 'example.com', siteRuleEnabled: true,
      latches: new EligibilityLatches(), probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
      gpuAvailable: () => true, budgetHealthy: () => true,
      win: { document: fx.doc as unknown as Document, getComputedStyle: fx.env().getComputedStyle,
        devicePixelRatio: 1, location: { origin: 'https://example.com' } as Location },
    });
    expect(evaluateEligibility(input).reason).toBe('native-fullscreen');
  });

  it.each(['0', '0.5'])('gates the video own computed opacity: %s', (opacity) => {
    const fx = new FakeEnv();
    const parent = new FakeNode();
    parent.connectedRoot = true;
    const video = Object.assign(new FakeNode(), { mediaKeys: null, currentSrc: '' });
    parent.appendChild(video);
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 0, top: 0, width: 1280, height: 720 };
    video.computed.opacity = opacity;
    const input = collectEligibilityInput(video as unknown as HTMLVideoElement, {
      settings: defaultSettings(), hostname: 'example.com', topHostname: 'example.com', siteRuleEnabled: true,
      latches: new EligibilityLatches(), probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
      gpuAvailable: () => true, budgetHealthy: () => true,
      win: { document: Object.assign(fx.doc, { documentElement: parent }) as unknown as Document,
        getComputedStyle: fx.env().getComputedStyle, devicePixelRatio: 1, location: { origin: 'https://example.com' } as Location },
    });
    expect(evaluateEligibility(input).reason).toBe(opacity === '0' ? 'not-rendered' : null);
  });

  it.each([
    ['mirrored shadow host', false, true, 'unsupported-layout'],
    ['mirrored slot ancestor', true, true, 'unsupported-layout'],
    ['untransformed shadow host', false, false, null],
    ['named slot', true, false, 'unsupported-layout'],
  ] as const)('checks composed ancestry for an off-centre video: %s', (_name, slotted, mirrored, reason) => {
    const fx = new FakeEnv();
    const host = new FakeNode();
    host.connectedRoot = true;
    const shadowParent = Object.assign(new FakeNode(), { parentNode: { host }, connectedRoot: true });
    const video = Object.assign(new FakeNode(), { mediaKeys: null, currentSrc: '' });
    if (slotted) {
      host.appendChild(video);
      const slot = new FakeNode();
      slot.setAttribute('name', 'player');
      video.setAttribute('slot', 'player');
      shadowParent.appendChild(slot);
      Object.assign(video, { assignedSlot: slot });
      if (mirrored) shadowParent.computed.transform = 'matrix(-1, 0, 0, 1, 0, 0)';
    } else {
      shadowParent.appendChild(video);
      if (mirrored) host.computed.transform = 'matrix(-1, 0, 0, 1, 0, 0)';
    }
    video.videoWidth = 640;
    video.videoHeight = 360;
    video.rect = { left: 80, top: 40, width: 1280, height: 800 };
    video.computed.objectPosition = '20% 30%';
    const input = collectEligibilityInput(video as unknown as HTMLVideoElement, {
      settings: defaultSettings(), hostname: 'example.com', topHostname: 'example.com', siteRuleEnabled: true,
      latches: new EligibilityLatches(), probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
      gpuAvailable: () => true, budgetHealthy: () => true,
      win: { document: Object.assign(fx.doc, { documentElement: host }) as unknown as Document,
        getComputedStyle: fx.env().getComputedStyle, devicePixelRatio: 1, location: { origin: 'https://example.com' } as Location },
    });
    expect(evaluateEligibility(input).reason).toBe(reason);
  });

  it('live collection rejects independent transforms, ancestor flips and zoom while allowing ancestor translations', () => {
    const cases = [
    ['video scale: -1 1', 'video', { scale: '-1 1' }, false],
    ['video rotate', 'video', { rotate: '180deg' }, false],
    ['video translate', 'video', { translate: '10px' }, false],
    ['ancestor scaleX(-1)', 'parent', { transform: 'matrix(-1, 0, 0, 1, 0, 0)' }, false],
    ['ancestor zoom: 2', 'parent', { zoom: '2' }, false],
    ['ancestor scale', 'parent', { scale: '1 1' }, false],
    ['ancestor rotate', 'parent', { rotate: '0deg' }, false],
    ['ancestor translateZ(0)', 'parent', { transform: 'matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)' }, true],
    ['ancestor translate(10px, 5px)', 'parent', { transform: 'matrix(1, 0, 0, 1, 10, 5)' }, true],
    ] as const;
    for (const [name, target, style, eligible] of cases) {
      const fx = new FakeEnv();
      const root = new FakeNode();
      root.connectedRoot = true;
      const parent = new FakeNode();
      root.appendChild(parent);
      const video = Object.assign(new FakeNode(), { mediaKeys: null, currentSrc: '' });
      parent.appendChild(video);
      video.videoWidth = 640;
      video.videoHeight = 360;
      video.rect = { left: 0, top: 0, width: 1280, height: 720 };
      Object.assign((target === 'video' ? video : parent).computed, style);
      const input = collectEligibilityInput(video as unknown as HTMLVideoElement, {
        settings: defaultSettings(), hostname: 'example.com', topHostname: 'example.com', siteRuleEnabled: true,
        latches: new EligibilityLatches(), probeFrame: () => ({ tainted: false, transfer: 'bt709' }),
        gpuAvailable: () => true, budgetHealthy: () => true,
        win: { document: Object.assign(fx.doc, { documentElement: root }) as unknown as Document,
          getComputedStyle: fx.env().getComputedStyle, devicePixelRatio: 1, location: { origin: 'https://example.com' } as Location },
      });
      expect(evaluateEligibility(input).reason, name).toBe(eligible ? null : 'unsupported-layout');
    }
    expect(evaluateEligibility(withGeometry({ currentCSSZoom: 2 })).reason).toBe('unsupported-layout');
    expect(evaluateEligibility(withGeometry({ ancestors: [{ transform: 'none', scale: 'none', rotate: 'none', zoom: 1, currentCSSZoom: 2 }] })).reason).toBe('unsupported-layout');
    expect(evaluateEligibility(withGeometry({ layoutSupported: false })).reason).toBe('unsupported-layout');
  });
});
