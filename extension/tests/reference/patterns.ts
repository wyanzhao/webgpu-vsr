/**
 * Deterministic test images shared by the GPU harness and the Node-side reference.
 */

/** 32-bit LCG (Numerical Recipes constants). */
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Natural-looking RGBA8 image: smooth gradients, a sinusoidal texture, hard edges and
 * per-pixel noise, so every network stage sees non-trivial, full-range data.
 */
export function makeRgba8(width: number, height: number, seed: number): Uint8Array {
  const rnd = lcg(seed);
  const out = new Uint8Array(4 * width * height);
  const cx = width * (0.3 + 0.4 * rnd());
  const cy = height * (0.3 + 0.4 * rnd());
  const r0 = Math.min(width, height) * 0.25;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const inside = (x - cx) ** 2 + (y - cy) ** 2 < r0 * r0;
      const stripe = Math.sin(x * 0.45 + y * 0.12) * Math.cos(y * 0.31);
      const base = [
        0.15 + 0.7 * (x / Math.max(1, width - 1)),
        0.5 + 0.35 * stripe,
        0.85 - 0.7 * (y / Math.max(1, height - 1)),
      ];
      const p = 4 * (y * width + x);
      for (let c = 0; c < 3; c++) {
        let v = inside ? 1 - base[c] : base[c];
        if ((x >> 3) % 5 === 0 && c === 1) v = 0.05;
        v += (rnd() - 0.5) * 0.12;
        out[p + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
      }
      out[p + 3] = 255;
    }
  }
  return out;
}

export function toBase64(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function fromBase64(b64: string): Uint8Array {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
