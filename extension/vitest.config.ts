import { defineConfig } from 'vitest/config';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/unit/**/*.test.ts'],
    testTimeout: 60000,
  },
  resolve: {
    alias: { '@': resolve(__dirname, 'src') },
  },
  plugins: [
    {
      name: 'wgsl-raw-loader',
      transform(_code, id) {
        if (id.endsWith('.wgsl')) {
          return { code: `export default ${JSON.stringify(readFileSync(id, 'utf-8'))};`, map: null };
        }
        return undefined;
      },
    },
  ],
});
