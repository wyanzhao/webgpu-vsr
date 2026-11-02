/** Full layout/style checks, used only around extension node insertion and patch installation. */
export interface LayoutSnapshotEnvironment {
  document: Document;
  getComputedStyle(element: Element): CSSStyleDeclaration;
}

const extensionNodes = new WeakSet<Element>();

// CSSOM resolves these to layout-dependent used values, which may change without moving a box.
// Compare bounding-rect edges for geometry instead of comparing these property strings.
const LAYOUT_RESOLVED_PROPERTIES = new Set([
  'width', 'height', 'inline-size', 'block-size',
  'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
  'margin-block-start', 'margin-block-end', 'margin-inline-start', 'margin-inline-end',
  'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'padding-block-start', 'padding-block-end', 'padding-inline-start', 'padding-inline-end',
  'top', 'right', 'bottom', 'left',
  'inset-block-start', 'inset-block-end', 'inset-inline-start', 'inset-inline-end',
  'transform-origin', 'perspective-origin', 'line-height',
]);

export function registerExtensionNode(element: Element): void {
  extensionNodes.add(element);
}

interface ElementSnapshot {
  edges: number[];
  properties: Map<string, string>;
}

export interface LayoutSnapshot {
  video: HTMLVideoElement;
  elements: Map<Element, ElementSnapshot>;
}

export interface LayoutDifference {
  element: Element;
  property: string;
}

export function captureLayoutSnapshot(video: HTMLVideoElement, env: LayoutSnapshotEnvironment): LayoutSnapshot {
  const elements = new Set<Element>([video]);
  for (const sibling of Array.from(video.parentElement?.children ?? [])) elements.add(sibling);
  for (let ancestor: Element | null = video; ancestor;) {
    if (ancestor === env.document.documentElement) break;
    ancestor = ancestor.assignedSlot ?? ancestor.parentElement ??
      (ancestor.parentNode && 'host' in ancestor.parentNode ? (ancestor.parentNode as ShadowRoot).host : null);
    if (ancestor) elements.add(ancestor);
  }
  const snapshots = new Map<Element, ElementSnapshot>();
  for (const element of elements) {
    if (extensionNodes.has(element)) continue;
    const rect = element.getBoundingClientRect();
    const style = env.getComputedStyle(element);
    const properties = new Map<string, string>();
    for (let i = 0; i < style.length; i++) {
      const name = style.item(i);
      properties.set(name, style.getPropertyValue(name));
    }
    snapshots.set(element, { edges: [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height], properties });
  }
  return { video, elements: snapshots };
}

/** Used-value properties are skipped for all elements; owned-property overrides apply only to the video. */
export function compareLayoutSnapshots(
  before: LayoutSnapshot,
  after: LayoutSnapshot,
  ignore: ReadonlySet<string> = new Set(),
): LayoutDifference | null {
  for (const [element, previous] of before.elements) {
    const current = after.elements.get(element);
    if (!current) return { element, property: 'element' };
    if (previous.edges.some((edge, i) => !Number.isFinite(current.edges[i]! - edge) ||
      Math.abs(current.edges[i]! - edge) > 0.5)) return { element, property: 'rect' };
    const names = new Set([...previous.properties.keys(), ...current.properties.keys()]);
    for (const name of names) {
      if (LAYOUT_RESOLVED_PROPERTIES.has(name) || (element === before.video && ignore.has(name))) continue;
      if ((previous.properties.get(name) ?? '') !== (current.properties.get(name) ?? '')) return { element, property: name };
    }
  }
  for (const element of after.elements.keys()) {
    if (!before.elements.has(element)) return { element, property: 'element' };
  }
  return null;
}
