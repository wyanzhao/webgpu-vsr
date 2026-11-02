import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  SITE_RULE_IDS,
  SiteRuleError,
  defaultSiteRule,
  isExcludedByRule,
  loadSiteRule,
  parseSiteRule,
  ruleMatchesHost,
  selectSiteRule,
  siteRuleIdForHost,
  type SiteRule,
} from '../../src/content/site_rules';

const SITES_DIR = resolve(__dirname, '..', '..', 'public', 'sites');
const readRule = (id: string): unknown => JSON.parse(readFileSync(join(SITES_DIR, `${id}.json`), 'utf8'));
const shipped: SiteRule[] = SITE_RULE_IDS.map((id) => parseSiteRule(readRule(id)));

describe('shipped rule files', () => {
  it('exist exactly for the known ids and validate', () => {
    const files = readdirSync(SITES_DIR).filter((f) => f.endsWith('.json')).sort();
    expect(files).toEqual([...SITE_RULE_IDS].map((id) => `${id}.json`).sort());
    for (const id of SITE_RULE_IDS) expect(parseSiteRule(readRule(id)).id).toBe(id);
  });

  it('are all enabled and the generic rule matches no specific host', () => {
    expect(shipped.every((r) => r.enabled)).toBe(true);
    const generic = shipped.find((r) => r.id === 'generic')!;
    expect(generic.hosts).toEqual([]);
    expect(generic.hostSuffixes).toEqual([]);
  });

  it.each([
    ['www.youtube.com', 'youtube'],
    ['m.youtube.com', 'youtube'],
    ['youtube.com', 'youtube'],
    ['music.youtube.com', 'youtube'],
    ['www.youtube-nocookie.com', 'youtube'],
    ['WWW.YOUTUBE.COM', 'youtube'],
    ['www.bilibili.com', 'bilibili'],
    ['live.bilibili.com', 'bilibili'],
    ['player.bilibili.com', 'bilibili'],
    ['bilibili.com', 'bilibili'],
    ['example.com', 'generic'],
    ['notyoutube.com', 'generic'],
    ['youtube.com.evil.net', 'generic'],
    ['fakebilibili.com', 'generic'],
    ['localhost', 'generic'],
  ])('%s -> %s (selectSiteRule and siteRuleIdForHost agree)', (host, id) => {
    expect(selectSiteRule(shipped, host).id).toBe(id);
    expect(siteRuleIdForHost(host)).toBe(id);
  });

  it('YouTube rule excludes thumbnails/previews and listens for yt-navigate-finish', () => {
    const yt = shipped.find((r) => r.id === 'youtube')!;
    expect(yt.spaEvents).toContain('yt-navigate-finish');
    const thumbnail = { matches: (s: string) => s === 'ytd-thumbnail video' };
    const mainPlayer = { matches: (s: string) => s === '#movie_player video' };
    expect(isExcludedByRule(thumbnail, yt)).toBe(true);
    expect(isExcludedByRule(mainPlayer, yt)).toBe(false);
  });
});

describe('parseSiteRule', () => {
  const valid = { id: 'x', enabled: true, hosts: ['A.example.com'], hostSuffixes: ['example.org'], ignoreSelectors: [' .p video '] };

  it('normalizes supported rule fields and fills their defaults', () => {
    expect(parseSiteRule(valid)).toEqual({
      id: 'x',
      enabled: true,
      hosts: ['a.example.com'],
      hostSuffixes: ['.example.org'],
      ignoreSelectors: ['.p video'],
      spaEvents: [],
      notes: '',
    });
  });

  it.each([
    ['non-object', 5],
    ['bad id', { ...valid, id: 'Has Space' }],
    ['missing enabled', { ...valid, enabled: undefined }],
    ['bad host', { ...valid, hosts: ['exa mple.com'] }],
    ['hosts not array', { ...valid, hosts: 'a.com' }],
    ['empty selector', { ...valid, ignoreSelectors: [''] }],
    ['bad event name', { ...valid, spaEvents: ['bad event'] }],
    ['notes not string', { ...valid, notes: 1 }],
  ])('rejects %s', (_name, raw) => {
    expect(() => parseSiteRule(raw)).toThrow(SiteRuleError);
  });
});

describe('matching and exclusion', () => {
  it('suffix matches subdomains and the bare domain but not look-alikes', () => {
    const rule = { hosts: [], hostSuffixes: ['.example.com'] };
    expect(ruleMatchesHost(rule, 'a.b.example.com')).toBe(true);
    expect(ruleMatchesHost(rule, 'example.com')).toBe(true);
    expect(ruleMatchesHost(rule, 'badexample.com')).toBe(false);
    expect(ruleMatchesHost(rule, 'not a host')).toBe(false);
  });

  it('selectSiteRule falls back to the built-in default without a generic rule', () => {
    expect(selectSiteRule([], 'example.com')).toEqual(defaultSiteRule());
  });

  it('invalid selectors are skipped instead of disabling the rule', () => {
    const el = {
      matches(s: string): boolean {
        if (s.startsWith('::')) throw new SyntaxError('invalid');
        return s === '.preview video';
      },
    };
    expect(isExcludedByRule(el, { ignoreSelectors: ['::bogus(', '.preview video'] })).toBe(true);
    expect(isExcludedByRule(el, { ignoreSelectors: ['::bogus('] })).toBe(false);
    expect(isExcludedByRule(el, { ignoreSelectors: [] })).toBe(false);
  });
});

describe('loadSiteRule', () => {
  const resolveUrl = (p: string): string => `chrome-extension://id/${p}`;
  const fromDisk = async (url: string): Promise<unknown> => readRule(url.replace('chrome-extension://id/sites/', '').replace('.json', ''));

  it('loads the specific rule for a matching host', async () => {
    const requested: string[] = [];
    const { rule, errors } = await loadSiteRule('www.youtube.com', resolveUrl, async (u) => {
      requested.push(u);
      return fromDisk(u);
    });
    expect(rule.id).toBe('youtube');
    expect(errors).toEqual([]);
    expect(requested).toEqual(['chrome-extension://id/sites/youtube.json']);
  });

  it('falls back to generic.json when the specific file fails, reporting the error', async () => {
    const { rule, errors } = await loadSiteRule('live.bilibili.com', resolveUrl, async (u) => {
      if (u.endsWith('bilibili.json')) throw new Error('HTTP 404');
      return fromDisk(u);
    });
    expect(rule.id).toBe('generic');
    expect(errors).toEqual(['sites/bilibili.json: HTTP 404']);
  });

  it('rejects a file whose id does not match its name', async () => {
    const { rule, errors } = await loadSiteRule('example.com', resolveUrl, async () => ({ ...(readRule('youtube') as object) }));
    expect(rule).toEqual(defaultSiteRule());
    expect(errors[0]).toContain('declares id youtube');
  });

  it('returns the permissive default when nothing loads', async () => {
    const { rule, errors } = await loadSiteRule('example.com', resolveUrl, async () => {
      throw new Error('offline');
    });
    expect(rule).toEqual(defaultSiteRule());
    expect(errors).toEqual(['sites/generic.json: offline']);
  });
});
