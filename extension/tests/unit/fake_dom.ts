/**
 * Minimal fake DOM for the overlay and session unit tests: inline styles with priorities, a
 * node tree with sibling order and connection state, and an {@link OverlayEnvironment} whose
 * ResizeObserver / MutationObserver / IntersectionObserver / timers are driven by the test.
 */
import type { OverlayEnvironment } from '../../src/content/overlay';

export class FakeStyle {
  props = new Map<string, { value: string; priority: string }>();
  writes = 0;
  managedWrites: Array<{ name: string; transition: string; priority: string }> = [];
  constructor(private readonly onChange: () => void = () => {}) {}
  private recordManagedWrite(name: string): void {
    if (['object-position', 'background-color', 'background-image', 'position'].includes(name)) {
      this.managedWrites.push({ name, transition: this.getPropertyValue('transition-property'),
        priority: this.getPropertyPriority('transition-property') });
    }
  }
  setProperty(name: string, value: string, priority = ''): void {
    this.recordManagedWrite(name);
    this.writes++;
    this.props.set(name, { value, priority });
    this.onChange();
  }
  getPropertyValue(name: string): string {
    return this.props.get(name)?.value ?? '';
  }
  get length(): number {
    return this.props.size;
  }
  item(index: number): string {
    return [...this.props.keys()][index] ?? '';
  }
  getPropertyPriority(name: string): string {
    return this.props.get(name)?.priority ?? '';
  }
  removeProperty(name: string): string {
    this.recordManagedWrite(name);
    const old = this.getPropertyValue(name);
    this.props.delete(name);
    this.onChange();
    return old;
  }
}

export type Computed = Partial<Record<'boxShadow' | 'outlineStyle' | 'outlineOffset' | 'order' | 'top' | 'right' | 'bottom' | 'left' | 'inset' | 'webkitBackdropFilter' | 'position' | 'display' | 'visibility' | 'opacity' | 'transform' | 'scale' | 'rotate' | 'translate' | 'zoom' | 'zIndex' | 'objectFit' | 'objectPosition' | 'borderLeftWidth' | 'borderTopWidth' | 'borderRightWidth' | 'borderBottomWidth' | 'paddingLeft' | 'paddingTop' | 'paddingRight' | 'paddingBottom' | 'clip' | 'clipPath' | 'maskImage' | 'webkitMaskImage' | 'mixBlendMode' | 'objectViewBox' | 'backdropFilter' | 'filter' | 'borderTopLeftRadius' | 'borderTopRightRadius' | 'borderBottomRightRadius' | 'borderBottomLeftRadius', string>>;

export class FakeNode {
  style = new FakeStyle(() => {
    this.attrs.style = [...this.style.props].map(([name, { value, priority }]) =>
      `${name}: ${value}${priority ? ` !${priority}` : ''};`).join(' ');
  });
  computed: Computed = {};
  computedRule: (() => Computed) | null = null;
  stylesheetImportant: Computed = {};
  /** Simulates computed values that even inline important declarations cannot override. */
  computedOverrides: Computed = {};
  positionShift = { left: 0, top: 0 };
  /** Simulates a page / environment where the positioning patch cannot take effect. */
  forceStaticPosition = false;
  children: FakeNode[] = [];
  parentElement: FakeNode | null = null;
  parentNode: FakeNode | { host: FakeNode } | null = null;
  assignedSlot: FakeNode | null = null;
  shadowRoot: { fullscreenElement: FakeNode | null } | null = null;
  attachedShadow: FakeNode | null = null;
  shadowMode: ShadowRootMode | null = null;
  className = '';
  ownerDocument?: Document;
  attrs: Record<string, string> = {};
  rect = { left: 0, top: 0, width: 0, height: 0 };
  /** Canvas fixtures derive their viewport rect from CSS unless a test supplies a fake rect. */
  isCanvas = false;
  canvasRect: FakeNode['rect'] | null = null;
  checkVisibility?: (options?: CheckVisibilityOptions) => boolean;
  scrollLeft = 0;
  scrollTop = 0;
  width = 300;
  height = 150;
  videoWidth = 0;
  videoHeight = 0;
  connectedRoot = false;
  listeners = new Map<string, Set<() => void>>();
  /** Tests can install selector-dependent style/rect effects after a DOM insertion. */
  afterInsertion: ((node: FakeNode) => void) | null = null;
  /** What getContext() returns (a fake GPUCanvasContext for canvases). */
  context: unknown = null;

  getContext(_type: string): unknown {
    return this.context;
  }

  attachShadow(options: ShadowRootInit): FakeNode {
    this.shadowMode = options.mode;
    this.attachedShadow = new FakeNode();
    this.attachedShadow.parentNode = { host: this };
    if (options.mode === 'open') this.shadowRoot = Object.assign(this.attachedShadow, { fullscreenElement: null });
    return this.attachedShadow;
  }

  get isConnected(): boolean {
    return this.connectedRoot || (this.parentElement !== null && this.parentElement.isConnected);
  }
  get nextSibling(): FakeNode | null {
    const p = this.parentElement;
    return p ? (p.children[p.children.indexOf(this) + 1] ?? null) : null;
  }
  get previousSibling(): FakeNode | null {
    const p = this.parentElement;
    return p ? (p.children[p.children.indexOf(this) - 1] ?? null) : null;
  }
  get clientWidth(): number {
    return Math.round(this.rect.width);
  }
  get clientHeight(): number {
    return Math.round(this.rect.height);
  }
  get offsetWidth(): number {
    return Math.round(this.rect.width);
  }
  get offsetHeight(): number {
    return Math.round(this.rect.height);
  }
  getClientRects(): FakeNode['rect'][] {
    return this.isConnected ? [this.rect] : [];
  }
  getAttribute(n: string): string | null {
    return this.attrs[n] ?? null;
  }
  hasAttribute(n: string): boolean {
    return Object.hasOwn(this.attrs, n);
  }
  removeAttribute(n: string): void {
    delete this.attrs[n];
    if (n === 'style') this.style.props.clear();
  }
  setAttribute(n: string, v: string): void {
    this.attrs[n] = v;
  }
  insertBefore(node: FakeNode, ref: FakeNode | null): FakeNode {
    node.remove();
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(node);
    else this.children.splice(i, 0, node);
    node.parentElement = this;
    node.parentNode = this;
    this.afterInsertion?.(node);
    return node;
  }
  appendChild(node: FakeNode): FakeNode {
    return this.insertBefore(node, null);
  }
  remove(): void {
    if (!this.parentElement) return;
    this.parentElement.children.splice(this.parentElement.children.indexOf(this), 1);
    if (this.parentNode === this.parentElement) this.parentNode = null;
    this.parentElement = null;
  }
  contains(other: FakeNode): boolean {
    let n: FakeNode | null = other;
    while (n) {
      if (n === this) return true;
      n = n.parentElement;
    }
    return false;
  }
  get containingBlock(): FakeNode | null {
    let block = this.parentElement;
    while (block && (block.computed.display === 'contents' ||
      ((block.style.getPropertyValue('position') || block.computed.position || 'static') === 'static' &&
        (!block.computed.transform || block.computed.transform === 'none')))) {
      block = block.parentElement;
    }
    return block;
  }
  getBoundingClientRect(): FakeNode['rect'] {
    if (!this.isCanvas) {
      const rect = { ...this.rect };
      if (this.style.getPropertyValue('position') === 'relative') {
        rect.left += this.positionShift.left;
        rect.top += this.positionShift.top;
      }
      for (let node = this.parentElement; node; node = node.parentElement) {
        if (node.style.getPropertyValue('position') === 'relative') {
          rect.left += node.positionShift.left;
          rect.top += node.positionShift.top;
        }
      }
      return rect;
    }
    if (this.canvasRect) return this.canvasRect;
    const block = this.containingBlock;
    const rect = block?.getBoundingClientRect();
    const num = (value: string | undefined): number => parseFloat(value || '0');
    return {
      left: (rect?.left ?? 0) + num(block?.computed.borderLeftWidth) - (block?.scrollLeft ?? 0) + num(this.style.getPropertyValue('left')),
      top: (rect?.top ?? 0) + num(block?.computed.borderTopWidth) - (block?.scrollTop ?? 0) + num(this.style.getPropertyValue('top')),
      width: num(this.style.getPropertyValue('width')),
      height: num(this.style.getPropertyValue('height')),
    };
  }
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
  listenerCount(): number {
    return [...this.listeners.values()].reduce((n, s) => n + s.size, 0);
  }
}

export class FakeEnv {
  /** Context handed to every created element (canvases ask for 'webgpu'). */
  canvasContext: unknown = null;
  doc = Object.assign(new FakeNode(), {
    fullscreenElement: null as FakeNode | null,
    defaultView: { innerWidth: 1920, innerHeight: 1080 },
    createElement: (tag: string) => Object.assign(new FakeNode(), {
      context: this.canvasContext, isCanvas: tag === 'canvas', ownerDocument: this.doc as unknown as Document,
    }),
  });
  elementsFromPoint: (x: number, y: number) => FakeNode[] = () => {
    const elements: FakeNode[] = [];
    const visit = (node: FakeNode): void => {
      if (node.videoWidth > 0) elements.push(node);
      node.children.forEach(visit);
    };
    visit(this.doc);
    return elements;
  };
  dpr = 1;
  observers: Array<{ cb: () => void; targets: FakeNode[]; disconnected: boolean }> = [];
  mutationObservers: Array<{ cb: () => void; targets: FakeNode[]; disconnected: boolean }> = [];
  intersectionObserverAvailable = true;
  intersectionObservers: Array<{ cb: (entries: IntersectionObserverEntry[]) => void; targets: FakeNode[]; unobserved: FakeNode[]; disconnected: boolean }> = [];
  windowListeners = new Map<string, Set<() => void>>();
  timers: Array<{ id: number; fn: () => void; ms: number }> = [];
  private seq = 0;

  env(): OverlayEnvironment {
    return {
      document: this.doc as unknown as Document,
      getComputedStyle: (el) => {
        const node = el as unknown as FakeNode;
        const c = node.computed;
        const initial: Record<string, string> = {
          position: 'static',
          boxShadow: 'none', outlineStyle: 'none', outlineOffset: '0px', order: '0',
          top: 'auto', right: 'auto', bottom: 'auto', left: 'auto', inset: 'auto',
          insetBlockStart: 'auto', insetBlockEnd: 'auto', insetInlineStart: 'auto', insetInlineEnd: 'auto',
          insetBlock: 'auto', insetInline: 'auto',
          clipPath: 'none', maskImage: 'none', mixBlendMode: 'normal', filter: 'none',
          objectPosition: '50% 50%',
          display: 'block',
          visibility: 'visible',
          opacity: '1',
          transform: 'none',
          rotate: 'none', scale: 'none', translate: 'none', clip: 'auto', zoom: '1',
          zIndex: 'auto',
          objectFit: 'contain',
          borderLeftWidth: '0px',
          borderTopWidth: '0px',
          borderRightWidth: '0px',
          borderBottomWidth: '0px',
          paddingLeft: '0px',
          paddingTop: '0px',
          paddingRight: '0px',
          paddingBottom: '0px',
          borderTopLeftRadius: '0px',
          borderTopRightRadius: '0px',
          borderBottomRightRadius: '0px',
          borderBottomLeftRadius: '0px',
        };
        const reset = node.style.getPropertyValue('all') === 'initial' && node.style.getPropertyPriority('all') === 'important';
        const values: Record<string, string> = { ...initial, ...(reset ? {} : c) };
        const names = new Set([...Object.keys(values), ...Object.keys(node.stylesheetImportant),
          ...[...node.style.props.keys()].map((name) => name.replace(/-([a-z])/g, (_match, ch: string) => ch.toUpperCase()))]);
        for (const name of names) {
          const cssName = name.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`);
          values[name] = (node.style.getPropertyPriority(cssName) === 'important' ? node.style.getPropertyValue(cssName) :
            reset ? initial[name] : node.stylesheetImportant[name as keyof Computed]) ||
            (reset ? initial[name] : node.style.getPropertyValue(cssName)) || values[name] || '';
        }
        if (node.forceStaticPosition) values.position = 'static';
        // Chrome resolves relative-positioned auto offsets to used values, unlike static offsets.
        if (values.position === 'relative') {
          for (const [start, end] of [['top', 'bottom'], ['left', 'right']] as const) {
            if (values[start] === 'auto' && values[end] === 'auto') {
              values[start] = '0px';
              values[end] = '0px';
            }
          }
          for (const name of ['insetBlockStart', 'insetBlockEnd', 'insetInlineStart', 'insetInlineEnd',
            'inset', 'insetBlock', 'insetInline']) {
            if (values[name] === 'auto') values[name] = '0px';
          }
        }
        const computed: Record<string, string | undefined> = { ...values, ...node.computedOverrides, ...node.computedRule?.() };
        const cssNames = Object.keys(computed).map((name) => name.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`));
        return { ...computed, length: cssNames.length, item: (index: number) => cssNames[index] ?? '',
          getPropertyValue: (name: string) => computed[name.replace(/-([a-z])/g, (_match, ch: string) => ch.toUpperCase())] ?? '',
        } as unknown as CSSStyleDeclaration;
      },
      elementsFromPoint: (x, y) => this.elementsFromPoint(x, y) as unknown as Element[],
      devicePixelRatio: () => this.dpr,
      createResizeObserver: (cb) => {
        const o = { cb, targets: [] as FakeNode[], disconnected: false };
        this.observers.push(o);
        return { observe: (t: Element) => o.targets.push(t as unknown as FakeNode), disconnect: () => (o.disconnected = true) };
      },
      createMutationObserver: (cb) => {
        const o = { cb, targets: [] as FakeNode[], disconnected: false };
        this.mutationObservers.push(o);
        return { observe: (t: Node) => o.targets.push(t as unknown as FakeNode), disconnect: () => (o.disconnected = true) };
      },
      createIntersectionObserver: (cb) => {
        if (!this.intersectionObserverAvailable) return null;
        const o = { cb, targets: [] as FakeNode[], unobserved: [] as FakeNode[], disconnected: false };
        this.intersectionObservers.push(o);
        return {
          observe: (t: Element) => o.targets.push(t as unknown as FakeNode),
          unobserve: (t: Element) => {
            const node = t as unknown as FakeNode;
            o.unobserved.push(node);
            o.targets = o.targets.filter((target) => target !== node);
          },
          disconnect: () => (o.disconnected = true),
        };
      },
      addWindowListener: (type, l) => {
        if (!this.windowListeners.has(type)) this.windowListeners.set(type, new Set());
        this.windowListeners.get(type)!.add(l);
      },
      removeWindowListener: (type, l) => this.windowListeners.get(type)?.delete(l),
      setTimeout: (fn, ms) => {
        const id = ++this.seq;
        this.timers.push({ id, fn, ms });
        return id;
      },
      clearTimeout: (h) => {
        this.timers = this.timers.filter((t) => t.id !== h);
      },
    };
  }

  runTimers(): void {
    const due = this.timers;
    this.timers = [];
    due.forEach((t) => t.fn());
  }

  fireResize(): void {
    this.observers.filter((o) => !o.disconnected).forEach((o) => o.cb());
  }

  fireIntersections(entries: Array<{ target: FakeNode; intersectionRect: FakeNode['rect']; boundingClientRect?: FakeNode['rect'] }>): void {
    const rect = (r: FakeNode['rect']): DOMRectReadOnly => ({ ...r, x: r.left, y: r.top,
      right: r.left + r.width, bottom: r.top + r.height, toJSON: () => r });
    for (const observer of this.intersectionObservers.filter((o) => !o.disconnected)) {
      const records = entries.filter((entry) => observer.targets.includes(entry.target)).map((entry) => ({
        target: entry.target as unknown as Element,
        boundingClientRect: rect(entry.boundingClientRect ?? entry.target.getBoundingClientRect()),
        intersectionRect: rect(entry.intersectionRect),
        rootBounds: null,
        intersectionRatio: 0,
        isIntersecting: entry.intersectionRect.width > 0 && entry.intersectionRect.height > 0,
        time: 0,
      }));
      if (records.length) observer.cb(records);
    }
  }

  /** Delivers child-list mutation records to the observers of `target` (as the browser would). */
  fireChildList(target: FakeNode): void {
    this.mutationObservers.filter((o) => !o.disconnected && o.targets.includes(target)).forEach((o) => o.cb());
  }
}
