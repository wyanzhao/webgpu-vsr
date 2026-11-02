#!/usr/bin/env node
/**
 * Generates the extension icons (public/icons/icon-{16,32,48,128}.png) procedurally:
 * a rounded square with a violet-to-blue gradient, a white play triangle and a four-point
 * "enhance" sparkle. Shapes are rasterised analytically with 8x8 supersampling and written as
 * RGBA PNGs (zlib + CRC32 from Node's standard library; no dependencies).
 *
 * Usage: node scripts/generate_icons.mjs [outDir]
 */
import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.argv[2] ?? resolve(here, '..', 'public', 'icons'));
const SIZES = [16, 32, 48, 128];
const SS = 8;

const CRC_TABLE = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function encodePng(size, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0; // filter: none
    rgba.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// Shapes in unit coordinates [0,1]^2.
function inRoundedSquare(x, y) {
  const m = 0.04;
  const r = 0.22;
  const lo = m + r;
  const hi = 1 - m - r;
  const cx = Math.min(Math.max(x, lo), hi);
  const cy = Math.min(Math.max(y, lo), hi);
  return (x - cx) ** 2 + (y - cy) ** 2 <= r * r && x >= m && x <= 1 - m && y >= m && y <= 1 - m;
}

function inTriangle(x, y) {
  // Play triangle, slightly left of centre.
  const ax = 0.33, ay = 0.27, bx = 0.33, by = 0.73, cx = 0.74, cy = 0.5;
  const s1 = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
  const s2 = (cx - bx) * (y - by) - (cy - by) * (x - bx);
  const s3 = (ax - cx) * (y - cy) - (ay - cy) * (x - cx);
  return (s1 <= 0 && s2 <= 0 && s3 <= 0) || (s1 >= 0 && s2 >= 0 && s3 >= 0);
}

function inSparkle(x, y) {
  // Four-point star centred top-right: |dx|^0.5 + |dy|^0.5 <= r^0.5 (astroid).
  const dx = Math.abs(x - 0.76);
  const dy = Math.abs(y - 0.24);
  return Math.sqrt(dx) + Math.sqrt(dy) <= Math.sqrt(0.13);
}

function renderIcon(size) {
  const rgba = Buffer.alloc(size * size * 4);
  const top = [94, 92, 230]; // #5E5CE6
  const bottom = [10, 132, 255]; // #0A84FF
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let bg = 0;
      let fg = 0;
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const x = (px + (sx + 0.5) / SS) / size;
          const y = (py + (sy + 0.5) / SS) / size;
          if (!inRoundedSquare(x, y)) continue;
          bg++;
          if (inTriangle(x, y) || (size >= 32 && inSparkle(x, y))) fg++;
        }
      }
      const n = SS * SS;
      const alpha = bg / n;
      const t = (px + py) / (2 * size);
      const base = top.map((c, i) => c + (bottom[i] - c) * t);
      const f = bg > 0 ? fg / bg : 0;
      const color = base.map((c) => c + (255 - c) * f);
      const o = (py * size + px) * 4;
      rgba[o] = Math.round(color[0]);
      rgba[o + 1] = Math.round(color[1]);
      rgba[o + 2] = Math.round(color[2]);
      rgba[o + 3] = Math.round(alpha * 255);
    }
  }
  return rgba;
}

mkdirSync(outDir, { recursive: true });
for (const size of SIZES) {
  const file = resolve(outDir, `icon-${size}.png`);
  const png = encodePng(size, renderIcon(size));
  writeFileSync(file, png);
  console.log(`wrote ${file} (${png.length} bytes)`);
}
