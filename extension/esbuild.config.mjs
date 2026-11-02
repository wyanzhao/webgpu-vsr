import * as esbuild from 'esbuild';
import { cpSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, 'dist');
const harnessDir = resolve(__dirname, 'dist-harness');
const isWatch = process.argv.includes('--watch');
const withHarness = process.argv.includes('--harness');

rmSync(distDir, { recursive: true, force: true });
mkdirSync(distDir, { recursive: true });

const buildOptions = {
  bundle: true,
  minify: !isWatch,
  sourcemap: isWatch ? 'inline' : false,
  logLevel: 'info',
  loader: { '.wgsl': 'text', '.json': 'json' },
  target: ['chrome128', 'es2022'],
};

const entries = [
  { ...buildOptions, entryPoints: [resolve(__dirname, 'src/background/index.ts')], outfile: resolve(distDir, 'background.js'), format: 'esm' },
  { ...buildOptions, entryPoints: [resolve(__dirname, 'src/content/index.ts')], outfile: resolve(distDir, 'content.js'), format: 'iife' },
  { ...buildOptions, entryPoints: [resolve(__dirname, 'src/popup/popup.ts')], outfile: resolve(distDir, 'popup.js'), format: 'iife' },
];

// GPU test harness page (served to Playwright over http://localhost); not part of the extension.
const harnessEntry = resolve(__dirname, 'tests/harness/harness.ts');
if (withHarness && existsSync(harnessEntry)) {
  rmSync(harnessDir, { recursive: true, force: true });
  mkdirSync(harnessDir, { recursive: true });
  entries.push({ ...buildOptions, minify: false, entryPoints: [harnessEntry], outfile: resolve(harnessDir, 'harness.js'), format: 'iife' });
}

for (const entry of entries) {
  if (isWatch) {
    const ctx = await esbuild.context(entry);
    await ctx.watch();
  } else {
    await esbuild.build(entry);
  }
}

cpSync(resolve(__dirname, 'manifest.json'), resolve(distDir, 'manifest.json'));
cpSync(resolve(__dirname, 'src/popup/popup.html'), resolve(distDir, 'popup.html'));
cpSync(resolve(__dirname, 'src/popup/popup.css'), resolve(distDir, 'popup.css'));
for (const dir of ['icons', 'sites', 'weights']) {
  const src = resolve(__dirname, 'public', dir);
  mkdirSync(resolve(distDir, dir), { recursive: true });
  if (existsSync(src)) cpSync(src, resolve(distDir, dir), { recursive: true });
}

console.log('Build completed successfully. Assets staged to dist/');
