/**
 * Compiles every WGSL specialisation the pipelines can create with a real WGSL compiler:
 * Dawn (the `webgpu` package) on the Metal backend. Each module must report zero
 * compilation errors and produce a compute pipeline.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { buildShader, preprocess, shaderKey, type ShaderSpec } from '../../src/pipeline/shader_source';
import { buildVsrProgram } from '../../src/pipeline/vsr_pipeline';
import { planVsrMemory } from '../../src/pipeline/memory';
import { BASELINE_COEFFICIENTS, type ModelId, type OutputKind, type SourceKind } from '../../src/pipeline/types';
import { vec4Offset } from '../../src/weights/loader';
import type { WeightBundle } from '../../src/weights/types';
import { nodeBundle, nodeGpuContext } from '../harness/node_gpu';

const MODELS: ModelId[] = ['low', 'medium'];
const SOURCES: SourceKind[] = ['texture', 'rgba8', 'half4'];
const OUTPUTS: OutputKind[] = ['canvas', 'buffer'];
const FORMATS: GPUTextureFormat[] = ['bgra8unorm', 'rgba8unorm'];

async function allSpecs(): Promise<Map<string, ShaderSpec>> {
  const specs = new Map<string, ShaderSpec>();
  const add = (s: ShaderSpec) => specs.set(shaderKey(s), s);
  const bundles = new Map<ModelId, WeightBundle>();
  for (const m of MODELS) bundles.set(m, await nodeBundle(m));
  for (const model of MODELS) {
    const b = bundles.get(model)!;
    for (const [w, h, ow, oh] of [
      [960, 540, 1920, 1080],
      [1280, 720, 1920, 1080],
    ]) {
      const plan = planVsrMemory(model, w, h, ow, oh);
      for (const source of SOURCES)
        for (const output of OUTPUTS)
          for (const fmt of FORMATS) {
            for (const step of buildVsrProgram(model, plan, (l, k, o) => vec4Offset(b.tensor(l, k, o)), source, output, fmt)) add(step.spec);
          }
    }
  }
  for (const source of ['texture', 'rgba8'] as const)
    for (const output of OUTPUTS) for (const storageFormat of FORMATS) add({ kind: 'resample', source, output, storageFormat });
  return specs;
}

describe('WGSL compilation on Dawn/Metal', () => {
  let device: GPUDevice;
  let specs: Map<string, ShaderSpec>;

  beforeAll(async () => {
    const ctx = await nodeGpuContext();
    device = ctx.device;
    expect(ctx.hasF16).toBe(true);
    specs = await allSpecs();
  });

  it('covers every kernel kind and I/O variant', () => {
    const kinds = new Map<string, number>();
    for (const s of specs.values()) kinds.set(s.kind, (kinds.get(s.kind) ?? 0) + 1);
    expect(kinds.get('fold')).toBe(SOURCES.length * 2); // x2: cat0 strides of low and medium
    expect(kinds.get('resample')).toBe(3 * OUTPUTS.length * FORMATS.length); // planar, texture, rgba8
    expect(kinds.get('head')).toBeGreaterThanOrEqual(2 * SOURCES.length * 2);
    expect(kinds.get('conv')).toBeGreaterThan(20);
  });

  it('every specialisation compiles with zero errors and creates a compute pipeline', async () => {
    let compiled = 0;
    for (const [key, spec] of specs) {
      const code = buildShader(spec);
      expect(code.startsWith('enable f16;')).toBe(spec.kind !== 'resample' || spec.source === 'planar');
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ label: key, code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      expect(errors.map((e) => `${key} ${e.lineNum}:${e.linePos} ${e.message}`)).toEqual([]);
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      expect(pipeline).toBeTruthy();
      expect(await device.popErrorScope()).toBeNull();
      compiled++;
    }
    expect(compiled).toBe(specs.size);
    console.log(`[wgsl] ${compiled} specialisations compiled with zero errors`);
  }, 120000);

  it('the compiler is real: an invalid module reports errors', async () => {
    const module = device.createShaderModule({ code: buildShader({ kind: 'resample', source: 'planar', output: 'buffer', storageFormat: 'bgra8unorm' }).replace('fn main(', 'fn main(((') });
    const info = await module.getCompilationInfo();
    expect(info.messages.filter((m) => m.type === 'error').length).toBeGreaterThan(0);
  });

  it('bakes the stored scale-2 baseline coefficients into the head shader', () => {
    const code = buildShader({ kind: 'head', k0: 16, featStride: 32, source: 'texture', output: 'canvas', storageFormat: 'bgra8unorm' });
    for (const [i, row] of BASELINE_COEFFICIENTS.entries()) {
      expect(code).toContain(`const BASE_C${i}: vec4<f32> = vec4<f32>(${row.join(', ')});`);
    }
  });
});

describe('template preprocessor', () => {
  it('selects #if / #else branches, including nested ones', () => {
    const src = ['a', '#if X', 'b', '#if Y', 'c', '#else', 'd', '#endif', '#else', 'e', '#endif', 'f'].join('\n');
    expect(preprocess(src, { X: true, Y: true })).toBe('a\nb\nc\nf');
    expect(preprocess(src, { X: true, Y: false })).toBe('a\nb\nd\nf');
    expect(preprocess(src, { X: false, Y: true })).toBe('a\ne\nf');
    expect(preprocess('#if !X\nn\n#else\np\n#endif', { X: false })).toBe('n');
    expect(preprocess('#if !X\nn\n#else\np\n#endif', { X: true })).toBe('p');
  });
  it('rejects undefined flags and unbalanced directives', () => {
    expect(() => preprocess('#if Z\n#endif', {})).toThrow(/undefined shader flag/);
    expect(() => preprocess('#if X\n', { X: true })).toThrow(/unterminated/);
    expect(() => preprocess('#endif', {})).toThrow(/unexpected #endif/);
    expect(() => preprocess('#define Q', {})).toThrow(/unknown directive/);
  });
  it('rejects invalid specialisations', () => {
    const base = { kind: 'conv', cin: 16, inStride: 16, inOffset: 0, k: 16, kb: 8, tx: 16, ty: 16, cch: 16, outStride: 16, outOffset: 0, upsample: false } as const;
    expect(() => buildShader({ ...base, kb: 6 })).toThrow(/channel blocking/);
    expect(() => buildShader({ ...base, cch: 12 })).toThrow(/does not divide/);
    expect(() => buildShader({ ...base, inStride: 18 })).toThrow(/multiple of 4/);
    expect(() => buildShader({ ...base, pool: { stride: 32, offset: 0 } })).toThrow(/requires the projection/);
  });
});

describe('resample shader feature requirements', () => {
  it.each(['texture', 'rgba8'] as const)('Quality 0 %s has no f16 types or extension', (source) => {
    for (const output of OUTPUTS) {
      const code = buildShader({ kind: 'resample', source, output, storageFormat: 'rgba8unorm' });
      expect(code).not.toMatch(/^enable f16;/m);
      expect(code.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '')).not.toMatch(/\bf16\b/);
      expect(code).toContain('CUBIC: array<u32>');
      expect(code).toContain('unpack2x16float(CUBIC[i])');
    }
  });

  it('the planar VSR resample still enables f16 for storage', () => {
    const code = buildShader({ kind: 'resample', source: 'planar', output: 'buffer', storageFormat: 'rgba8unorm' });
    expect(code.startsWith('enable f16;')).toBe(true);
    expect(code).toContain('SRC_PLANAR: array<f16>');
  });
});
