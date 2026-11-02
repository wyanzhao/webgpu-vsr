import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test as base, type BrowserContext, type Page, type Worker } from '@playwright/test';
import { TEST_ORIGIN, expect, launchExtensionContext } from './fixtures';

/**
 * Popup -> UPDATE_SETTINGS -> chrome.storage.local, and persistence of those settings across
 * (a) service-worker termination and (b) a full browser restart with the same profile.
 * The browser is managed here (not by extensionTest) because the test relaunches it.
 */
const test = base.extend<{ userDataDir: string }>({
  // eslint-disable-next-line no-empty-pattern
  userDataDir: async ({}, use) => {
    const dir = mkdtempSync(join(tmpdir(), 'webgpu-vsr-popup-'));
    await use(dir);
    rmSync(dir, { recursive: true, force: true });
  },
});

const SITE_PAGE = '<!doctype html><title>site</title><p>no video here</p>';

async function launch(userDataDir: string): Promise<{ context: BrowserContext; worker: Worker; extensionId: string }> {
  const context = await launchExtensionContext(userDataDir);
  await context.route(`${TEST_ORIGIN}/site/**`, (route) => route.fulfill({ status: 200, contentType: 'text/html', body: SITE_PAGE }));
  const worker = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
  return { context, worker, extensionId: new URL(worker.url()).host };
}

async function openPopup(context: BrowserContext, extensionId: string, query = ''): Promise<Page> {
  const page = await context.newPage();
  await page.goto(`chrome-extension://${extensionId}/popup.html${query}`);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  return page;
}

async function storedSettings(worker: Worker): Promise<unknown> {
  return worker.evaluate(async () => (await chrome.storage.local.get('settings')).settings);
}

const EXPECTED = { enabled: false, qualityLevel: 2, sharpness: 0.8, hudEnabled: true, siteOverrides: { localhost: false } };

async function expectPopupShows(page: Page): Promise<void> {
  await expect(page.locator('#enabled')).not.toBeChecked();
  await expect(page.locator('#quality')).toHaveValue('2');
  await expect(page.locator('#sharpness')).toHaveValue('0.8');
  await expect(page.locator('#sharpness-value')).toHaveText('0.80');
  await expect(page.locator('#hud')).toBeChecked();
  await expect(page.locator('#status')).toHaveText('Off');
}

test('popup writes settings to chrome.storage.local and they persist across SW termination and browser restart', async ({ userDataDir }) => {
  let { context, worker, extensionId } = await launch(userDataDir);
  try {
    // A real site tab so the per-site switch has a hostname.
    const site = await context.newPage();
    await site.goto(`${TEST_ORIGIN}/site/index.html`);
    const tabId = await worker.evaluate(async (url) => (await chrome.tabs.query({ url }))[0]?.id ?? -1, `${TEST_ORIGIN}/site/index.html`);
    expect(tabId).toBeGreaterThanOrEqual(0);

    const popup = await openPopup(context, extensionId, `?tabId=${tabId}`);
    await expect(popup.locator('#enabled')).toBeChecked();
    await expect(popup.locator('#quality')).toHaveValue('1');
    await expect(popup.locator('#site-host')).toHaveText('localhost');
    await expect(popup.locator('#site-enabled')).toBeEnabled();
    await expect(popup.locator('#site-enabled')).toBeChecked();

    // Each control round-trips through the service worker into storage.
    await popup.locator('#quality').selectOption('2');
    await expect.poll(async () => ((await storedSettings(worker)) as { qualityLevel: number }).qualityLevel).toBe(2);
    await popup.locator('#sharpness').fill('0.8');
    await expect.poll(async () => ((await storedSettings(worker)) as { sharpness: number }).sharpness).toBe(0.8);
    await popup.locator('#hud').click();
    await expect.poll(async () => ((await storedSettings(worker)) as { hudEnabled: boolean }).hudEnabled).toBe(true);
    await popup.locator('#site-enabled').click();
    await expect.poll(async () => ((await storedSettings(worker)) as { siteOverrides: object }).siteOverrides).toEqual({ localhost: false });
    await expect.poll(() => worker.evaluate((id) => chrome.action.getBadgeText({ tabId: id }), tabId)).toBe('OFF');
    await popup.locator('#enabled').click();
    await expect.poll(() => storedSettings(worker)).toEqual(EXPECTED);
    await expectPopupShows(popup);

    // (a) Terminate the service worker through the DevTools protocol (Target.closeTarget), then
    // talk to the extension again: a fresh worker must serve the persisted settings. A marker
    // global set before termination proves the new worker is a new execution context.
    await worker.evaluate(() => {
      (self as unknown as { __webgpuVsrMarker?: string }).__webgpuVsrMarker = 'before-termination';
    });
    expect(await worker.evaluate(() => (self as unknown as { __webgpuVsrMarker?: string }).__webgpuVsrMarker ?? null)).toBe('before-termination');
    const cdp = await context.newCDPSession(popup);
    const swTargets = async (): Promise<Array<{ targetId: string; url: string }>> =>
      (await cdp.send('Target.getTargets')).targetInfos.filter((t) => t.type === 'service_worker' && t.url.startsWith(`chrome-extension://${extensionId}/`));
    const [original] = await swTargets();
    expect(original).toBeDefined();
    expect((await cdp.send('Target.closeTarget', { targetId: original!.targetId })).success).toBe(true);
    // Stopped: the worker target is gone until something wakes the extension up again.
    await expect.poll(async () => (await swTargets()).length).toBe(0);

    await popup.reload();
    await expect(popup.locator('body')).toHaveAttribute('data-ready', 'true');
    await expectPopupShows(popup);
    expect(await popup.evaluate(() => chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }))).toEqual({ type: 'GET_SETTINGS_RESPONSE', settings: EXPECTED });
    // The worker is running again (Chrome keeps the target id of a registration's version; the
    // marker below shows that the global scope is new).
    await expect.poll(async () => (await swTargets()).length).toBe(1);
    worker = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
    expect(await worker.evaluate(() => (self as unknown as { __webgpuVsrMarker?: string }).__webgpuVsrMarker ?? null)).toBeNull();
    expect(await storedSettings(worker)).toEqual(EXPECTED);

    // (b) Full browser restart with the same profile.
    await context.close();
    ({ context, worker, extensionId } = await launch(userDataDir));
    expect(await storedSettings(worker)).toEqual(EXPECTED); // onInstalled/onStartup did not clobber
    const again = await openPopup(context, extensionId);
    await expectPopupShows(again);

    // And the popup still writes after the restart.
    await again.locator('#enabled').click();
    await expect.poll(async () => ((await storedSettings(worker)) as { enabled: boolean }).enabled).toBe(true);
    await expect(again.locator('#status')).not.toHaveText('Off');
  } finally {
    await context.close();
  }
});
