import { describe, expect, it } from 'vitest';
import { captureLayoutSnapshot, compareLayoutSnapshots } from '../../src/content/invariance';
import { CanvasOverlay } from '../../src/content/overlay';
import { VideoHud } from '../../src/content/hud';
import { FakeEnv, FakeNode } from './fake_dom';

function fixture(): { fx: FakeEnv; root: FakeNode; parent: FakeNode; video: FakeNode; sibling: FakeNode; snapshot: () => ReturnType<typeof captureLayoutSnapshot> } {
  const fx = new FakeEnv();
  const root = new FakeNode();
  root.connectedRoot = true;
  const parent = new FakeNode();
  root.appendChild(parent);
  const video = new FakeNode();
  const sibling = new FakeNode();
  parent.appendChild(video);
  parent.appendChild(sibling);
  video.videoWidth = 640;
  video.videoHeight = 360;
  video.rect = { left: 0, top: 0, width: 1280, height: 720 };
  Object.assign(fx.doc, { documentElement: root });
  return { fx, root, parent, video, sibling,
    snapshot: () => captureLayoutSnapshot(video as unknown as HTMLVideoElement, fx.env()) };
}

describe('complete insertion invariance', () => {
  it('allows video auto-to-zero physical and logical offsets at patch installation without snapshotting later syncs', () => {
    const { fx, video } = fixture();
    const env = fx.env();
    const offsets = ['top', 'right', 'bottom', 'left', 'inset-block-start', 'inset-block-end',
      'inset-inline-start', 'inset-inline-end', 'inset', 'inset-block', 'inset-inline'];
    for (const property of offsets) {
      expect(env.getComputedStyle(video as unknown as Element).getPropertyValue(property)).toBe('auto');
    }
    let enumerations = 0;
    const original = env.getComputedStyle;
    env.getComputedStyle = (element) => {
      const style = original(element);
      return { ...style, item: (index: number) => { enumerations++; return style.item(index); } } as CSSStyleDeclaration;
    };
    const overlay = new CanvasOverlay(video as unknown as HTMLVideoElement, { env });
    try {
      expect(overlay.layoutSupported).toBe(true);
      expect(overlay.hasInsertionLayoutFailure).toBe(false);
      for (const property of offsets) {
        expect(env.getComputedStyle(video as unknown as Element).getPropertyValue(property)).toBe('0px');
      }
      const readsAfterInstallation = enumerations;
      expect(readsAfterInstallation).toBeGreaterThan(0);
      overlay.markFramePresented();
      overlay.sync();
      overlay.sync();
      expect(enumerations).toBe(readsAfterInstallation);
      expect(overlay.videoHidden).toBe(true);
    } finally {
      overlay.destroy();
    }
    expect(video.style.getPropertyValue('position')).toBe('');
  });

  it.each(['top', 'right', 'bottom', 'left', 'inset-block-start', 'inset-block-end',
    'inset-inline-start', 'inset-inline-end'])(
    'allows a patch-induced resolved video %s change with unchanged geometry', (property) => {
      const { fx, video } = fixture();
      const name = property.replace(/-([a-z])/g, (_match, ch: string) => ch.toUpperCase());
      video.computedRule = () => ({ [name]: video.style.getPropertyValue('position') === 'relative' ? '1px' : 'auto' });
      const before = video.getBoundingClientRect();
      const overlay = new CanvasOverlay(video as unknown as HTMLVideoElement, { env: fx.env() });
      try {
        expect(overlay.hasInsertionLayoutFailure).toBe(false);
        expect(overlay.layoutSupported).toBe(true);
        expect(video.getBoundingClientRect()).toEqual(before);
      } finally {
        overlay.destroy();
      }
    },
  );

  it.each(['inset', 'inset-block', 'inset-inline'])(
    'rolls back and latches a patch-induced nonzero video %s change', (property) => {
      const { fx, parent, video, sibling } = fixture();
      const name = property.replace(/-([a-z])/g, (_match, ch: string) => ch.toUpperCase());
      video.computedRule = () => ({ [name]: video.style.getPropertyValue('position') === 'relative' ? '1px' : 'auto' });
      const overlay = new CanvasOverlay(video as unknown as HTMLVideoElement, { env: fx.env() });
      try {
        expect(overlay.hasInsertionLayoutFailure).toBe(true);
        expect(overlay.layoutSupported).toBe(false);
        expect(parent.children).toEqual([video, sibling]);
        expect(video.style.getPropertyValue('position')).toBe('');
        overlay.sync();
        expect(parent.children).toEqual([video, sibling]);
        expect(overlay.videoHidden).toBe(false);
      } finally {
        overlay.destroy();
      }
    },
  );

  describe.each(['canvas', 'HUD'] as const)('%s insertion resolved-value invariance', (kind) => {
    const insert = (fx: FakeEnv, parent: FakeNode, video: FakeNode): CanvasOverlay | VideoHud =>
      kind === 'canvas' ? new CanvasOverlay(video as unknown as HTMLVideoElement, { env: fx.env() }) :
        new VideoHud(parent as unknown as HTMLElement, {
          video: video as unknown as HTMLVideoElement, document: fx.doc as unknown as Document,
          getComputedStyle: fx.env().getComputedStyle,
        });

    it('keeps the node inserted when ancestor margin-inline-end resolves differently with an unchanged rect', () => {
      const { fx, parent, video, snapshot } = fixture();
      video.computed.position = 'relative';
      parent.style.setProperty('margin-inline-end', '0px');
      parent.afterInsertion = () => parent.style.setProperty('margin-inline-end', '-24px');
      const before = snapshot();
      const rect = parent.getBoundingClientRect();
      const node = insert(fx, parent, video);
      try {
        expect(node.hasInsertionLayoutFailure).toBe(false);
        expect(parent.children).toHaveLength(3);
        expect(parent.getBoundingClientRect()).toEqual(rect);
        expect(fx.env().getComputedStyle(parent as unknown as Element).getPropertyValue('margin-inline-end')).toBe('-24px');
        expect(compareLayoutSnapshots(before, snapshot())).toBeNull();
        parent.rect.left += 0.51;
        expect(compareLayoutSnapshots(before, snapshot())).toEqual({ element: parent, property: 'rect' });
      } finally {
        node.destroy();
      }
    });

    it.each([['opacity', '1', '0'], ['filter', 'none', 'opacity(0)'], ['display', 'block', 'none']])(
      'still rolls back an ancestor %s change with an unchanged rect', (property, initial, changed) => {
        const { fx, parent, video, sibling } = fixture();
        video.computed.position = 'relative';
        parent.computedRule = () => ({ [property]: parent.children.length > 2 ? changed : initial });
        const rect = parent.getBoundingClientRect();
        const node = insert(fx, parent, video);
        try {
          expect(node.hasInsertionLayoutFailure).toBe(true);
          expect(parent.children).toEqual([video, sibling]);
          expect(parent.getBoundingClientRect()).toEqual(rect);
          expect(fx.env().getComputedStyle(parent as unknown as Element).getPropertyValue(property)).toBe(initial);
        } finally {
          node.destroy();
        }
      },
    );
  });

  it('allows enumerated video inset shorthands to resolve all-auto components to all-zero components', () => {
    const { fx, video } = fixture();
    video.computedRule = () => {
      const positioned = video.style.getPropertyValue('position') === 'relative';
      return { inset: positioned ? '0px 0px 0px 0px' : 'auto',
        insetBlock: positioned ? '0px 0px' : 'auto auto', insetInline: positioned ? '0px 0px' : 'auto auto' };
    };
    const overlay = new CanvasOverlay(video as unknown as HTMLVideoElement, { env: fx.env() });
    try {
      expect(overlay.layoutSupported).toBe(true);
      expect(overlay.hasInsertionLayoutFailure).toBe(false);
      expect(video.style.getPropertyValue('position')).toBe('relative');
    } finally {
      overlay.destroy();
    }
  });

  it('enumerates every computed property and walks assigned slots and shadow hosts through the document element', () => {
    const { fx, root, parent, video, sibling, snapshot } = fixture();
    const slot = new FakeNode();
    slot.parentNode = { host: root };
    video.assignedSlot = slot;
    const outside = new FakeNode();
    root.parentElement = outside;
    const before = snapshot();
    expect([...before.elements.keys()]).toEqual([video, sibling, slot, root]);
    expect(before.elements.get(video as unknown as Element)?.properties.get('object-fit')).toBe('contain');
    const css = fx.env().getComputedStyle(root as unknown as Element);
    expect(before.elements.get(root as unknown as Element)?.properties.size).toBe(css.length);
    root.computed.filter = 'blur(1px)';
    expect(compareLayoutSnapshots(before, snapshot())).toEqual({ element: root, property: 'filter' });
    expect(before.elements.has(parent as unknown as Element)).toBe(false);
  });

  it('ignores only named video properties and detects sibling styles, new properties and rect edge changes', () => {
    const { video, sibling, snapshot } = fixture();
    const before = snapshot();
    const ignore = new Set(['position', 'object-position', 'background-color', 'background-image']);
    video.computed.position = 'absolute';
    video.computed.objectPosition = '0px 100000px';
    video.style.setProperty('background-color', 'transparent');
    video.style.setProperty('background-image', 'none');
    expect(compareLayoutSnapshots(before, snapshot(), ignore)).toBeNull();
    sibling.computed.objectPosition = '10px 10px';
    expect(compareLayoutSnapshots(before, snapshot(), ignore)).toEqual({ element: sibling, property: 'object-position' });
    sibling.computed.objectPosition = '50% 50%';
    sibling.style.setProperty('color', 'red');
    expect(compareLayoutSnapshots(before, snapshot(), ignore)).toEqual({ element: sibling, property: 'color' });
    sibling.style.removeProperty('color');
    sibling.rect.left += 0.5;
    expect(compareLayoutSnapshots(before, snapshot(), ignore)).toBeNull();
    sibling.rect.width += 0.01;
    expect(compareLayoutSnapshots(before, snapshot(), ignore)).toEqual({ element: sibling, property: 'rect' });
  });

  it('keeps an unaffected overlay and HUD inserted, excludes owned nodes and never snapshots on ordinary sync or HUD updates', () => {
    const { fx, parent, video, snapshot } = fixture();
    // This test exercises insertion without installing the separately checked static-video patch.
    video.computed.position = 'relative';
    const before = snapshot();
    let enumerations = 0;
    const env = fx.env();
    const original = env.getComputedStyle;
    env.getComputedStyle = (element) => {
      const style = original(element);
      return { ...style, item: (index: number) => { enumerations++; return style.item(index); } } as CSSStyleDeclaration;
    };
    const overlay = new CanvasOverlay(video as unknown as HTMLVideoElement, { env });
    const hud = new VideoHud(parent as unknown as HTMLElement, {
      video: video as unknown as HTMLVideoElement, document: fx.doc as unknown as Document,
      getComputedStyle: env.getComputedStyle,
    });
    try {
      expect(overlay.hasInsertionLayoutFailure).toBe(false);
      expect(overlay.isMounted).toBe(true);
      expect(hud.element.parentElement).toBe(parent);
      expect(compareLayoutSnapshots(before, snapshot())).toBeNull();
      expect(snapshot().elements.has(overlay.canvas)).toBe(false);
      expect(snapshot().elements.has(hud.element)).toBe(false);
      expect(hud.hasInsertionLayoutFailure).toBe(false);
      const readsAfterInsertion = enumerations;
      overlay.sync();
      overlay.markFramePresented();
      overlay.sync();
      hud.moveTo(parent as unknown as HTMLElement);
      hud.reposition();
      hud.update({ id: 'v1', state: 'eligible', point: null, reason: null, reasonText: null,
        sourceWidth: 640, sourceHeight: 360, targetWidth: 1280, targetHeight: 720, telemetry: null, error: null });
      expect(enumerations).toBe(readsAfterInsertion);
    } finally {
      hud.destroy();
      overlay.destroy();
    }
  });

  it('removes a HUD synchronously when moving it changes an ancestor style and refuses another move', () => {
    const { fx, root, parent, video } = fixture();
    const hud = new VideoHud(parent as unknown as HTMLElement, {
      video: video as unknown as HTMLVideoElement, document: fx.doc as unknown as Document,
      getComputedStyle: fx.env().getComputedStyle,
    });
    const next = new FakeNode();
    root.appendChild(next);
    next.appendChild(video);
    let insertions = 0;
    next.afterInsertion = () => {
      insertions++;
      next.computedRule = () => ({ opacity: next.children.length > 1 ? '0' : '1' });
    };
    hud.moveTo(next as unknown as HTMLElement);
    expect(hud.element.parentElement).toBeNull();
    expect(next.children).toEqual([video]);
    expect(hud.hasInsertionLayoutFailure).toBe(true);
    hud.moveTo(next as unknown as HTMLElement);
    expect(insertions).toBe(1);
    expect(fx.env().getComputedStyle(next as unknown as Element).opacity).toBe('1');
    expect(video.style.writes).toBe(0);
  });
});
