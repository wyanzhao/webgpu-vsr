/**
 * Site rules (public/sites/*.json): per-platform thumbnail/preview exclusion
 * selectors and SPA navigation events. Rules are packaged with the extension and fetched by the
 * content script through chrome.runtime.getURL (they are web-accessible resources).
 *
 * File schema (validated by {@link parseSiteRule}):
 * {
 *   "id": "youtube",                     // also the file name: sites/<id>.json
 *   "hosts": ["www.youtube.com"],        // exact hostnames
 *   "hostSuffixes": [".youtube.com"],    // matches any subdomain (and the bare domain)
 *   "enabled": true,                     // false disables processing unless the user overrides
 *   "ignoreSelectors": ["ytd-thumbnail video"],
 *   "spaEvents": ["yt-navigate-finish"], // window events that trigger an immediate rescan
 *   "notes": "..."
 * }
 */

import { normalizeHostname } from '../shared/settings';

export interface SiteRule {
  id: string;
  hosts: string[];
  hostSuffixes: string[];
  enabled: boolean;
  ignoreSelectors: string[];
  spaEvents: string[];
  notes: string;
}

/** Rule files shipped in public/sites, most specific first; `generic` matches everything. */
export const SITE_RULE_IDS = ['youtube', 'bilibili', 'generic'] as const;
export type SiteRuleId = (typeof SITE_RULE_IDS)[number];

/** Events every page listens to for SPA navigations (in addition to a rule's own events). */
export const BASE_SPA_EVENTS: readonly string[] = ['popstate', 'hashchange', 'pageshow'];

export class SiteRuleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SiteRuleError';
  }
}

function stringArray(raw: unknown, field: string, max: number): string[] {
  if (raw === undefined) return [];
  if (!Array.isArray(raw) || raw.length > max) throw new SiteRuleError(`${field} must be an array of at most ${max} strings`);
  return raw.map((v, i) => {
    if (typeof v !== 'string' || v.trim() === '' || v.length > 512) throw new SiteRuleError(`${field}[${i}] must be a non-empty string`);
    return v.trim();
  });
}

/** Validates an untrusted rule object; throws {@link SiteRuleError} describing the first problem. */
export function parseSiteRule(raw: unknown): SiteRule {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SiteRuleError('rule must be an object');
  const r = raw as Record<string, unknown>;
  if (typeof r.id !== 'string' || !/^[a-z0-9-]{1,64}$/.test(r.id)) throw new SiteRuleError('id must match [a-z0-9-]{1,64}');
  if (typeof r.enabled !== 'boolean') throw new SiteRuleError('enabled must be a boolean');
  const hosts = stringArray(r.hosts, 'hosts', 256).map((h) => {
    const n = normalizeHostname(h);
    if (n === null) throw new SiteRuleError(`invalid host ${h}`);
    return n;
  });
  const hostSuffixes = stringArray(r.hostSuffixes, 'hostSuffixes', 256).map((s) => {
    const n = normalizeHostname(s.replace(/^\./, ''));
    if (n === null) throw new SiteRuleError(`invalid host suffix ${s}`);
    return `.${n}`;
  });
  const spaEvents = stringArray(r.spaEvents, 'spaEvents', 32);
  for (const e of spaEvents) {
    if (!/^[A-Za-z0-9_:.-]+$/.test(e)) throw new SiteRuleError(`invalid event name ${e}`);
  }
  if (r.notes !== undefined && typeof r.notes !== 'string') throw new SiteRuleError('notes must be a string');
  return {
    id: r.id,
    hosts,
    hostSuffixes,
    enabled: r.enabled,
    ignoreSelectors: stringArray(r.ignoreSelectors, 'ignoreSelectors', 256),
    spaEvents,
    notes: typeof r.notes === 'string' ? r.notes : '',
  };
}

/** True when the rule applies to `hostname` (exact host, or subdomain/bare domain of a suffix). */
export function ruleMatchesHost(rule: Pick<SiteRule, 'hosts' | 'hostSuffixes'>, hostname: string): boolean {
  const host = normalizeHostname(hostname);
  if (host === null) return false;
  if (rule.hosts.includes(host)) return true;
  return rule.hostSuffixes.some((suffix) => host.endsWith(suffix) || host === suffix.slice(1));
}

/**
 * Picks the rule for a hostname: the first matching non-generic rule, else the rule with id
 * `generic`, else a permissive built-in default.
 */
export function selectSiteRule(rules: readonly SiteRule[], hostname: string): SiteRule {
  for (const rule of rules) {
    if (rule.id !== 'generic' && ruleMatchesHost(rule, hostname)) return rule;
  }
  return rules.find((r) => r.id === 'generic') ?? defaultSiteRule();
}

/** Rule used when no rule file could be loaded: process everything, exclude nothing. */
export function defaultSiteRule(): SiteRule {
  return { id: 'generic', hosts: [], hostSuffixes: [], enabled: true, ignoreSelectors: [], spaEvents: [], notes: '' };
}

/** Which rule file to fetch for a hostname without loading all of them. */
export function siteRuleIdForHost(hostname: string): SiteRuleId {
  const host = normalizeHostname(hostname) ?? '';
  if (host === 'youtube.com' || host.endsWith('.youtube.com') || host === 'youtube-nocookie.com' || host.endsWith('.youtube-nocookie.com')) {
    return 'youtube';
  }
  if (host === 'bilibili.com' || host.endsWith('.bilibili.com')) return 'bilibili';
  return 'generic';
}

/** Something with Element.matches (real elements or test fakes). */
export interface Matchable {
  matches(selector: string): boolean;
}

/**
 * True when the element matches any ignore selector of the rule. Invalid selectors are skipped
 * (Element.matches throws SyntaxError for them) instead of disabling the whole rule.
 */
export function isExcludedByRule(element: Matchable, rule: Pick<SiteRule, 'ignoreSelectors'>): boolean {
  for (const selector of rule.ignoreSelectors) {
    try {
      if (element.matches(selector)) return true;
    } catch {
      // invalid selector: ignore this entry only
    }
  }
  return false;
}

export type FetchJson = (url: string) => Promise<unknown>;

/**
 * Loads the rule for `hostname`: fetches sites/<id>.json (and sites/generic.json when the
 * specific file fails), validates it, and falls back to {@link defaultSiteRule}.
 */
export async function loadSiteRule(
  hostname: string,
  resolveUrl: (path: string) => string,
  fetchJson: FetchJson,
): Promise<{ rule: SiteRule; errors: string[] }> {
  const errors: string[] = [];
  const id = siteRuleIdForHost(hostname);
  const candidates: SiteRuleId[] = id === 'generic' ? ['generic'] : [id, 'generic'];
  for (const candidate of candidates) {
    try {
      const rule = parseSiteRule(await fetchJson(resolveUrl(`sites/${candidate}.json`)));
      if (rule.id !== candidate) throw new SiteRuleError(`sites/${candidate}.json declares id ${rule.id}`);
      if (candidate !== 'generic' && !ruleMatchesHost(rule, hostname)) continue;
      return { rule, errors };
    } catch (err) {
      errors.push(`sites/${candidate}.json: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return { rule: defaultSiteRule(), errors };
}
