import { defineConfig } from '@playwright/test';

// Browser contexts are created by the fixtures in tests/e2e/fixtures.ts
// (persistent context + unpacked extension, or a WebGPU harness page).
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120000,
  retries: 0,
  workers: 1,
  reporter: [['list']],
  use: { trace: 'retain-on-failure' },
});
