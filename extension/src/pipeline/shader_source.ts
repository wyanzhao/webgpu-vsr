/**
 * WGSL specialisation: every kernel is ONE template in src/shaders; this module prepends
 * `enable f16;` when f16 types are used, the shared declarations and the specialisation constants, resolves
 * `#if FLAG / #else / #endif` blocks and appends the source/output I/O functions of the
 * requested variant (external texture vs buffer input, canvas vs buffer output). All other
 * shader code is shared between variants.
 */
import commonWgsl from '../shaders/common.wgsl';
import foldWgsl from '../shaders/fold.wgsl';
import convWgsl from '../shaders/conv.wgsl';
import headWgsl from '../shaders/sr_head.wgsl';
import resampleWgsl from '../shaders/resample.wgsl';
import { BASELINE_COEFFICIENTS, type OutputKind, type SourceKind } from './types';

export { BASELINE_COEFFICIENTS };

/** WGSL literals of float32(v / 255) for v = 0..255 (exact decimal forms of the f32 values). */
const UNORM8_TABLE = Array.from({ length: 256 }, (_, v) => {
  const f = Math.fround(v / 255);
  // 9 significant digits round-trip a float32 exactly
  return `${Number(f.toPrecision(9))}`;
}).join(', ');

/** Minimal `#if NAME` / `#if !NAME` / `#else` / `#endif` preprocessor (nesting allowed). */
export function preprocess(source: string, flags: Record<string, boolean>): string {
  const out: string[] = [];
  const stack: { active: boolean; parentActive: boolean; seenElse: boolean }[] = [];
  let active = true;
  source.split('\n').forEach((line, i) => {
    const t = line.trim();
    const m = /^#if\s+(!?)([A-Z0-9_]+)$/.exec(t);
    if (m) {
      if (!(m[2] in flags)) throw new Error(`line ${i + 1}: undefined shader flag ${m[2]}`);
      stack.push({ active, parentActive: active, seenElse: false });
      active = active && (m[1] === '!' ? !flags[m[2]] : flags[m[2]]);
      return;
    }
    if (t === '#else') {
      const top = stack[stack.length - 1];
      if (!top || top.seenElse) throw new Error(`line ${i + 1}: unexpected #else`);
      top.seenElse = true;
      active = top.parentActive && !active;
      return;
    }
    if (t === '#endif') {
      const top = stack.pop();
      if (!top) throw new Error(`line ${i + 1}: unexpected #endif`);
      active = top.parentActive;
      return;
    }
    if (t.startsWith('#')) throw new Error(`line ${i + 1}: unknown directive ${t}`);
    if (active) out.push(line);
  });
  if (stack.length) throw new Error('unterminated #if');
  return out.join('\n');
}

function constBlock(consts: Record<string, number>): string {
  return Object.entries(consts)
    .map(([k, v]) => {
      if (!Number.isInteger(v) || v < 0) throw new Error(`constant ${k} must be a non-negative integer, got ${v}`);
      return `const ${k}: u32 = ${v}u;`;
    })
    .join('\n');
}

function assemble(body: string, consts: Record<string, number>, flags: Record<string, boolean>, io: string, extra = ''): string {
  const code = [commonWgsl, constBlock(consts), extra, io, preprocess(body, flags)].join('\n\n');
  const usesF16 = /\bf16\b/.test(code.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, ''));
  return usesF16 ? `enable f16;\n\n${code}` : code;
}

function halfLiteral(v: number): string {
  if (Math.f16round(v) !== v) throw new Error(`${v} is not exactly representable in f16`);
  return `${v}`;
}

/**
 * Source access functions: src_rgb (texture/rgba8 only; colour in [0, 1]) and src_i (the
 * preprocessed I = [2R-1, 2G-1, 2B-1, 0] as exact halves held in f32).
 */
export function sourceSnippet(kind: SourceKind, binding: number): string {
  switch (kind) {
    case 'texture':
      return `@group(0) @binding(${binding}) var SRC_TEX: texture_external;
fn src_rgb(x: i32, y: i32) -> vec3<f32> {
  return textureLoad(SRC_TEX, vec2<i32>(x, y)).rgb;
}
fn src_i(x: i32, y: i32) -> vec4<f32> {
  return preprocess_i(src_rgb(x, y));
}`;
    case 'rgba8':
      return `@group(0) @binding(${binding}) var<storage, read> SRC_PX: array<u32>;
// float32(v / 255) correctly rounded, as dlpp_preProcessE computes it (div.rn.f64 + cvt.rn.f32)
const UNORM8 = array<f32, 256>(${UNORM8_TABLE});
fn src_rgb(x: i32, y: i32) -> vec3<f32> {
  let p = SRC_PX[u32(y) * P.src_w + u32(x)];
  return vec3<f32>(UNORM8[p & 255u], UNORM8[(p >> 8u) & 255u], UNORM8[(p >> 16u) & 255u]);
}
fn src_i(x: i32, y: i32) -> vec4<f32> {
  return preprocess_i(src_rgb(x, y));
}`;
    case 'half4':
      return `@group(0) @binding(${binding}) var<storage, read> SRC_H4: array<vec4<f16>>;
fn src_i(x: i32, y: i32) -> vec4<f32> {
  return vec4<f32>(SRC_H4[u32(y) * P.src_w + u32(x)]);
}`;
  }
}

export interface FoldSpec {
  kind: 'fold';
  source: SourceKind;
  /** cat0 pixel stride in channels (2*K0). */
  catStride: number;
}

export interface ConvSpec {
  kind: 'conv';
  /** 3x3 input channels, pixel stride and channel offset in the SRC buffer. */
  cin: number;
  inStride: number;
  inOffset: number;
  /** Output channels and per-thread / per-tile blocking. */
  k: number;
  kb: number;
  tx: number;
  ty: number;
  /** Input channels staged per workgroup-memory chunk. */
  cch: number;
  outStride: number;
  outOffset: number;
  upsample: boolean;
  /** RB part B: 1x1 projection input channels and channel offset inside DST. */
  proj?: { cin: number; offset: number };
  /** Fused 2x2 pool target (pixel stride and channel offset). */
  pool?: { stride: number; offset: number };
}

export interface HeadSpec {
  kind: 'head';
  k0: number;
  featStride: number;
  source: SourceKind;
  output: OutputKind;
  storageFormat: GPUTextureFormat;
}

export interface ResampleSpec {
  kind: 'resample';
  /** 'planar' = VSR internal FP16 image; texture/rgba8 = Quality 0 source frame. */
  source: 'planar' | 'texture' | 'rgba8';
  output: OutputKind;
  storageFormat: GPUTextureFormat;
}

export type ShaderSpec = FoldSpec | ConvSpec | HeadSpec | ResampleSpec;

function vec4Units(name: string, channels: number): number {
  if (channels % 4 !== 0) throw new Error(`${name} = ${channels} is not a multiple of 4`);
  return channels / 4;
}

function storageTextureDecl(binding: number, format: GPUTextureFormat): string {
  if (format !== 'bgra8unorm' && format !== 'rgba8unorm') throw new Error(`unsupported storage format ${format}`);
  return `@group(0) @binding(${binding}) var OUT_TEX: texture_storage_2d<${format}, write>;`;
}

/** Output pixels from which a conv may use 16 channels per thread. */
export const WIDE_BLOCK_MIN_PIXELS = 32768;

/**
 * Tiling for a conv layer with K output channels: 128 threads, all K channels per
 * workgroup, each thread a 2x2 pixel block x KB channels. Large plain/up-conv layers with
 * K >= 32 use KB = 16 (more reuse of every staged input value), and so do large encoder
 * projection layers (rbB), whose 1x1 input has at most K channels. Decoder projection layers
 * (tbB, 2K-channel 1x1 input), K = 16 layers and small levels use KB = 8 (register
 * pressure, enough workgroups to fill the GPU). `projectionCin` is the 1x1 input width of a
 * projection layer, 0 for a plain conv. Chosen from timestamp measurements on the M5 Max
 * and the M6 (docs/BENCHMARKS.md).
 */
export function convTiling(k: number, projectionCin = 0, outPixels = 0): { kb: number; tx: number; ty: number } {
  const large = outPixels >= WIDE_BLOCK_MIN_PIXELS;
  if (projectionCin > 0) {
    return large && projectionCin <= k && k >= 32 ? { kb: 16, tx: 16, ty: 8 } : k === 64 ? { kb: 8, tx: 8, ty: 8 } : { kb: 8, tx: 16, ty: k === 16 ? 16 : 8 };
  }
  switch (k) {
    case 16:
      return { kb: 8, tx: 16, ty: 16 };
    case 32:
      return large ? { kb: 16, tx: 16, ty: 16 } : { kb: 8, tx: 16, ty: 8 };
    case 64:
      return large ? { kb: 16, tx: 16, ty: 8 } : { kb: 8, tx: 8, ty: 8 };
    default:
      throw new Error(`no conv tiling for K = ${k}`);
  }
}

/** Largest channel chunk (multiple of 4 dividing cin) whose halo tile fits the budget. */
export function convChunk(cin: number, tileW: number, tileH: number, budgetBytes = 16384): number {
  const perChannel = (tileW + 2) * (tileH + 2) * 2;
  let best = 4;
  for (let c = 4; c <= cin; c += 4) {
    if (cin % c === 0 && c * perChannel <= budgetBytes) best = c;
  }
  return best;
}

export function buildShader(spec: ShaderSpec): string {
  switch (spec.kind) {
    case 'fold':
      return assemble(foldWgsl, { CAT_STRIDE4: vec4Units('catStride', spec.catStride) }, {}, sourceSnippet(spec.source, 2));
    case 'conv': {
      const { kb, k } = spec;
      if (k % kb !== 0 || kb % 4 !== 0) throw new Error(`invalid channel blocking K=${k} KB=${kb}`);
      if (spec.tx % 2 !== 0 || spec.ty % 2 !== 0) throw new Error('tile sizes must be even');
      if (spec.cin % spec.cch !== 0) throw new Error(`chunk ${spec.cch} does not divide ${spec.cin}`);
      if (spec.pool && !spec.proj) throw new Error('pool epilogue requires the projection variant');
      const consts: Record<string, number> = {
        CIN4: vec4Units('cin', spec.cin),
        IN_STRIDE4: vec4Units('inStride', spec.inStride),
        IN_OFF4: vec4Units('inOffset', spec.inOffset),
        K: k,
        KB: kb,
        KG: k / kb,
        TX: spec.tx,
        TY: spec.ty,
        CCH4: vec4Units('cch', spec.cch),
        OUT_STRIDE4: vec4Units('outStride', spec.outStride),
        OUT_OFF4: vec4Units('outOffset', spec.outOffset),
      };
      if (spec.proj) {
        consts.CINX4 = vec4Units('proj.cin', spec.proj.cin);
        consts.X_OFF4 = vec4Units('proj.offset', spec.proj.offset);
      }
      if (spec.pool) {
        consts.POOL_STRIDE4 = vec4Units('pool.stride', spec.pool.stride);
        consts.POOL_OFF4 = vec4Units('pool.offset', spec.pool.offset);
      }
      return assemble(convWgsl, consts, { UPSAMPLE: spec.upsample, PROJ: !!spec.proj, POOL: !!spec.pool }, '');
    }
    case 'head': {
      const c = BASELINE_COEFFICIENTS.map((row) => `vec4<f32>(${row.map(halfLiteral).join(', ')})`);
      const extra = `const BASE_C0: vec4<f32> = ${c[0]};\nconst BASE_C1: vec4<f32> = ${c[1]};`;
      const out =
        spec.output === 'canvas'
          ? `${storageTextureDecl(5, spec.storageFormat)}
fn store_out(x: u32, y: u32, v: vec4<f32>) {
  let c = clamp(v.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  textureStore(OUT_TEX, vec2<u32>(x, y), vec4<f32>(floor(c * 255.0) / 255.0, 1.0));
}`
          : `@group(0) @binding(5) var<storage, read_write> OUT_PLANAR: array<f16>;
fn store_out(x: u32, y: u32, v: vec4<f32>) {
  let i = y * P.out_w + x;
  let plane = P.out_w * P.out_h;
  OUT_PLANAR[i] = f16(v.r);
  OUT_PLANAR[plane + i] = f16(v.g);
  OUT_PLANAR[2u * plane + i] = f16(v.b);
}`;
      const consts = { K04: vec4Units('k0', spec.k0), FEAT_STRIDE4: vec4Units('featStride', spec.featStride) };
      return assemble(headWgsl, consts, {}, `${sourceSnippet(spec.source, 4)}\n${out}`, extra);
    }
    case 'resample': {
      const src =
        spec.source === 'planar'
          ? `@group(0) @binding(1) var<storage, read> SRC_PLANAR: array<f16>;
fn load_src(x: i32, y: i32) -> vec3<f32> {
  let i = u32(y) * P.aux0 + u32(x);
  return vec3<f32>(f32(SRC_PLANAR[i]), f32(SRC_PLANAR[P.aux1 + i]), f32(SRC_PLANAR[2u * P.aux1 + i]));
}`
          : `${sourceSnippet(spec.source, 1)}
fn load_src(x: i32, y: i32) -> vec3<f32> {
  return hr3(src_rgb(x, y));
}`;
      const out =
        spec.output === 'canvas'
          ? `${storageTextureDecl(2, spec.storageFormat)}
fn store_out(x: u32, y: u32, bytes: vec3<f32>) {
  textureStore(OUT_TEX, vec2<u32>(x, y), vec4<f32>(bytes / 255.0, 1.0));
}`
          : `@group(0) @binding(2) var<storage, read_write> OUT_PX: array<u32>;
fn store_out(x: u32, y: u32, bytes: vec3<f32>) {
  let b = vec3<u32>(bytes);
  OUT_PX[y * P.out_w + x] = b.x | (b.y << 8u) | (b.z << 16u) | (255u << 24u);
}`;
      return assemble(resampleWgsl, {}, {}, `${src}\n${out}`);
    }
  }
}

/** Stable cache key of a specialisation (identical specs share one pipeline). */
export function shaderKey(spec: ShaderSpec): string {
  const stable = (v: unknown): string => {
    if (v === null || typeof v !== 'object') return JSON.stringify(v) ?? 'undefined';
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o)
      .filter((k) => o[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stable(o[k])}`)
      .join(',')}}`;
  };
  return stable(spec);
}

/** Values of the WGSL LayerParams uniform (common.wgsl); omitted fields are zero. */
export interface LayerParamsData {
  inW?: number;
  inH?: number;
  outW?: number;
  outH?: number;
  w3?: number;
  b3?: number;
  w1?: number;
  b1?: number;
  srcW?: number;
  srcH?: number;
  aux0?: number;
  aux1?: number;
  f0?: number;
  f1?: number;
  f2?: number;
  f3?: number;
}

export const LAYER_PARAMS_BYTES = 64;

export function encodeLayerParams(p: LayerParamsData): ArrayBuffer {
  const buf = new ArrayBuffer(LAYER_PARAMS_BYTES);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  const ints = [p.inW, p.inH, p.outW, p.outH, p.w3, p.b3, p.w1, p.b1, p.srcW, p.srcH, p.aux0, p.aux1];
  ints.forEach((v, i) => {
    const x = v ?? 0;
    if (!Number.isInteger(x) || x < 0 || x > 0xffffffff) throw new RangeError(`LayerParams field ${i} = ${x} is not a u32`);
    u[i] = x;
  });
  [p.f0, p.f1, p.f2, p.f3].forEach((v, i) => {
    f[12 + i] = v ?? 0;
  });
  return buf;
}

export const FRAME_PARAMS_BYTES = 16;

/** FrameParams uniform: residual gain g = 2*sharpness and the exact-composition flag. */
export function encodeFrameParams(sharpness: number, frame = 0): ArrayBuffer {
  if (!(sharpness >= 0 && sharpness <= 1)) throw new RangeError(`sharpness must be in [0, 1], got ${sharpness}`);
  const buf = new ArrayBuffer(FRAME_PARAMS_BYTES);
  const gain = 2 * sharpness;
  new Float32Array(buf, 0, 1)[0] = gain;
  new Uint32Array(buf, 4, 3).set([gain === 1 ? 1 : 0, frame >>> 0, 0]);
  return buf;
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

/** Compile (once per device) the compute pipeline of a specialisation. */
export function getComputePipeline(device: GPUDevice, spec: ShaderSpec): Promise<GPUComputePipeline> {
  let cache = pipelineCache.get(device);
  if (!cache) {
    cache = new Map();
    pipelineCache.set(device, cache);
  }
  const key = shaderKey(spec);
  let p = cache.get(key);
  if (!p) {
    const module = device.createShaderModule({ label: key, code: buildShader(spec) });
    p = device.createComputePipelineAsync({ label: `${spec.kind}`, layout: 'auto', compute: { module, entryPoint: 'main' } });
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return p;
}
