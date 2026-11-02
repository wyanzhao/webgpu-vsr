/**
 * RTX VSR folded low/medium network as one WebGPU compute pass per frame
 * (docs/IMPLEMENTATION_NOTES.md sections 3 and 4).
 *
 * Dispatch order: fold(src -> cat0.U) . for l = 0..3 { rbA(cat_l.U -> mid),
 * rbB(mid + cat_l.U -> cat_l.skip, pooled -> cat_{l+1}.U | x4) } . rb4A(x4 -> mid) .
 * rb4B(mid + x4 -> b4) . for L = 3..0 { up(prev -> cat_L.U), tbA(cat_L -> mid),
 * tbB(mid + cat_L -> cat_L.U in place) } . head(cat0.U + src -> canvas | internal) .
 * [resample(internal -> canvas)]. The internal image aliases `mid` (memory.ts).
 */
import { GpuMemoryLedger, INTERNAL_ALIAS, UNIFORM_SLOT_BYTES, planFits, planVsrMemory, vsrDispatchCount, type VsrBufferName, type VsrMemoryPlan } from './memory';
import {
  convChunk,
  convTiling,
  encodeFrameParams,
  encodeLayerParams,
  getComputePipeline,
  type ConvSpec,
  type FoldSpec,
  type HeadSpec,
  type LayerParamsData,
  type ResampleSpec,
  type ShaderSpec,
} from './shader_source';
import { resampleGroups, resampleParams, sourceResource, targetResource } from './resample_pipeline';
import type { FrameSlot } from './ring';
import type { FrameSource, FrameTarget, ModelId, OutputKind, SRComputePipeline, SourceKind } from './types';
import { FOLD_CHANNELS, MODEL_CHANNELS, type LayerName, type TensorOp, type WeightBundle } from '../weights/types';
import { loadWeightBundle, uploadWeightBundle, vec4Offset } from '../weights/loader';
import { cubicWeightTable } from './cubic_weights';

export interface VsrPipelineOptions {
  model: ModelId;
  storageFormat: GPUTextureFormat;
  /** Weight bundle, or a loader invoked once on first initialize. */
  weights?: WeightBundle | (() => Promise<WeightBundle>);
  /** Source variant ('texture' in production). */
  source?: SourceKind;
  /** Output variant ('canvas' in production; 'buffer' = planar FP16 / packed RGBA8). */
  output?: OutputKind;
}

/** Base URL of the bundled weights: the extension origin when running inside Chrome. */
export function defaultWeightsBaseUrl(): string {
  const rt = (globalThis as { chrome?: { runtime?: { getURL?: (p: string) => string } } }).chrome?.runtime;
  return rt?.getURL ? rt.getURL('weights/') : '/weights/';
}

export type BufferRef = VsrBufferName | 'weights';

export interface VsrStep {
  label: string;
  spec: ShaderSpec;
  params: LayerParamsData;
  groups: [number, number];
  /** Buffers bound at bindings 1.. for conv steps (weights, src, dst, [pool]). */
  buffers?: BufferRef[];
}

interface CompiledStep extends VsrStep {
  slot: number;
  pipeline: GPUComputePipeline;
  bindGroup?: GPUBindGroup;
}

export class VsrPipeline implements SRComputePipeline {
  readonly id: string;
  readonly scaleFactor = 2;
  readonly model: ModelId;
  private readonly source: SourceKind;
  private readonly output: OutputKind;
  private device: GPUDevice | null = null;
  private bundle: WeightBundle | null = null;
  private weightLoader: (() => Promise<WeightBundle>) | null;
  private weightBuffer: GPUBuffer | null = null;
  private readonly weightLedger = new GpuMemoryLedger();
  private readonly ledger = new GpuMemoryLedger();
  private buffers = new Map<VsrBufferName, GPUBuffer>();
  private allocated: VsrMemoryPlan | null = null;
  private plan: VsrMemoryPlan | null = null;
  private steps: CompiledStep[] = [];
  private sharpness = 0.5;
  private frameCounter = 0;

  constructor(private readonly options: VsrPipelineOptions) {
    this.model = options.model;
    this.id = `rtx-vsr-${options.model}`;
    this.source = options.source ?? 'texture';
    this.output = options.output ?? 'canvas';
    const w = options.weights;
    if (w && typeof w !== 'function') {
      this.bundle = w;
      this.weightLoader = null;
    } else {
      this.weightLoader = w ?? (() => loadWeightBundle(options.model, { baseUrl: defaultWeightsBaseUrl() }));
    }
  }

  /** Dispatch labels in recording order (fold, rb0A, rb0B, ..., head[, resample]). */
  get stepLabels(): string[] {
    return this.steps.map((s) => s.label);
  }

  /** Current memory plan (after initialize). */
  get memoryPlan(): VsrMemoryPlan | null {
    return this.plan;
  }

  get direct(): boolean {
    return this.plan?.direct ?? true;
  }

  async initialize(device: GPUDevice, inWidth: number, inHeight: number, outWidth: number, outHeight: number): Promise<void> {
    if (!device.features.has('shader-f16')) throw new Error('VSR models require the shader-f16 feature (use Quality 0)');
    if (this.device !== device) {
      this.releaseGpu();
      this.device = device;
    }
    if (!this.bundle) {
      this.bundle = await this.weightLoader!();
      if (this.bundle.model !== this.model) throw new Error(`weights are for '${this.bundle.model}', pipeline is '${this.model}'`);
    }
    if (!this.weightBuffer) {
      this.weightBuffer = this.weightLedger.adopt(uploadWeightBundle(device, this.bundle), `weights-${this.model}`);
    }
    const plan = planVsrMemory(this.model, inWidth, inHeight, outWidth, outHeight);
    if (!this.allocated || !planFits(this.allocated, plan)) {
      this.ledger.releaseAll();
      this.buffers.clear();
      for (const b of plan.buffers) {
        const usage =
          b.name === 'params' || b.name === 'frame'
            ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        const buffer = this.ledger.createBuffer(device, { label: `vsr-${b.name}`, size: b.bytes, usage });
        if (b.name === 'cubic') device.queue.writeBuffer(buffer, 0, cubicWeightTable());
        this.buffers.set(b.name, buffer);
      }
      this.allocated = plan;
    }
    this.plan = plan;
    const steps = buildVsrProgram(this.model, plan, (l, k, o) => this.offset(l, k, o), this.source, this.output, this.options.storageFormat);
    device.pushErrorScope('validation');
    let initialized = false;
    let scopeError: GPUError | null;
    try {
      const pipelines = await Promise.all(steps.map((s) => getComputePipeline(device, s.spec)));
      const params = this.buffers.get('params')!;
      this.steps = steps.map((s, i) => {
        device.queue.writeBuffer(params, i * UNIFORM_SLOT_BYTES, encodeLayerParams(s.params));
        const compiled: CompiledStep = { ...s, slot: i, pipeline: pipelines[i] };
        if (s.buffers) {
          compiled.bindGroup = device.createBindGroup({
            label: s.label,
            layout: pipelines[i].getBindGroupLayout(0),
            entries: [this.paramsEntry(i), ...s.buffers.map((name, j) => ({ binding: j + 1, resource: { buffer: this.buffer(name) } }))],
          });
        }
        return compiled;
      });
      device.queue.writeBuffer(this.buffers.get('frame')!, 0, encodeFrameParams(this.sharpness));
      initialized = true;
    } finally {
      const scope = device.popErrorScope();
      // Cleanup must not replace the original compilation or binding error.
      scopeError = await (initialized ? scope : scope.catch(() => null));
    }
    if (scopeError) throw new Error(`VSR pipeline initialisation failed: ${scopeError.message}`);
  }

  execute(
    commandEncoder: GPUCommandEncoder,
    externalTexture: GPUExternalTexture,
    targetCanvasView: GPUTextureView,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    this.encodeFrame(commandEncoder, { kind: 'texture', texture: externalTexture }, { kind: 'canvas', view: targetCanvasView }, timestampWrites);
  }

  /** Execute with a frame-ring slot: its uniform carries this frame's parameters. */
  executeSlot(commandEncoder: GPUCommandEncoder, externalTexture: GPUExternalTexture, targetCanvasView: GPUTextureView, slot: FrameSlot): void {
    this.encodeFrame(
      commandEncoder,
      { kind: 'texture', texture: externalTexture },
      { kind: 'canvas', view: targetCanvasView },
      slot.timestampWrites,
      slot.uniformBuffer,
    );
  }

  /**
   * Record the whole network into one compute pass. `frameUniform` (a ring slot buffer) is
   * filled with the current frame parameters; without it the pipeline's own uniform is used.
   * `stopAfter` ends the pass after the named dispatch (see stepLabels) to inspect
   * intermediate activations.
   */
  encodeFrame(
    encoder: GPUCommandEncoder,
    source: FrameSource,
    target: FrameTarget,
    timestampWrites?: GPUComputePassTimestampWrites,
    frameUniform?: GPUBuffer,
    stopAfter?: string,
  ): void {
    const device = this.device;
    if (!device || !this.plan || !this.steps.length) throw new Error('VsrPipeline.execute before initialize');
    if (source.kind !== this.source) throw new Error(`pipeline expects a '${this.source}' source, got '${source.kind}'`);
    if (target.kind !== this.output) throw new Error(`pipeline expects a '${this.output}' target, got '${target.kind}'`);
    this.frameCounter = (this.frameCounter + 1) >>> 0;
    const frame = frameUniform ?? this.buffers.get('frame')!;
    if (frameUniform) device.queue.writeBuffer(frameUniform, 0, encodeFrameParams(this.sharpness, this.frameCounter));
    const pass = encoder.beginComputePass(timestampWrites ? { label: this.id, timestampWrites } : { label: this.id });
    if (stopAfter !== undefined && !this.steps.some((s) => s.label === stopAfter)) throw new Error(`no dispatch named ${stopAfter}`);
    for (const step of this.steps) {
      const bindGroup = step.bindGroup ?? this.frameBindGroup(step, source, target, frame);
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(step.groups[0], step.groups[1]);
      if (step.label === stopAfter) break;
    }
    pass.end();
  }

  setSharpness(sharpness: number): void {
    const data = encodeFrameParams(sharpness);
    this.sharpness = sharpness;
    const frame = this.buffers.get('frame');
    if (this.device && frame) this.device.queue.writeBuffer(frame, 0, data);
  }

  memoryBytes(): number {
    return this.ledger.totalBytes + this.weightLedger.totalBytes;
  }

  /** Per-allocation breakdown (labels and bytes). */
  memoryBreakdown(): { label: string; bytes: number }[] {
    return [...this.weightLedger.breakdown(), ...this.ledger.breakdown()];
  }

  /** The buffer holding the internal planar FP16 image (resample mode only). */
  internalBuffer(): GPUBuffer | null {
    return this.plan && !this.plan.direct ? (this.buffers.get(INTERNAL_ALIAS) ?? null) : null;
  }

  /** Read access to an activation buffer (testing and diagnostics). */
  activationBuffer(name: VsrBufferName): GPUBuffer | null {
    return this.buffers.get(name) ?? null;
  }

  destroy(): void {
    this.releaseGpu();
    this.device = null;
  }

  private releaseGpu(): void {
    this.ledger.releaseAll();
    this.weightLedger.releaseAll();
    this.weightBuffer = null;
    this.buffers.clear();
    this.allocated = null;
    this.plan = null;
    this.steps = [];
  }

  private buffer(name: BufferRef): GPUBuffer {
    const b = name === 'weights' ? this.weightBuffer : this.buffers.get(name);
    if (!b) throw new Error(`buffer ${name} is not allocated`);
    return b;
  }

  private paramsEntry(slot: number): GPUBindGroupEntry {
    return { binding: 0, resource: { buffer: this.buffer('params'), offset: slot * UNIFORM_SLOT_BYTES, size: UNIFORM_SLOT_BYTES } };
  }

  private frameBindGroup(step: CompiledStep, source: FrameSource, target: FrameTarget, frame: GPUBuffer): GPUBindGroup {
    const device = this.device!;
    const entries: GPUBindGroupEntry[] = [this.paramsEntry(step.slot)];
    const kind = step.spec.kind;
    if (kind === 'fold') {
      entries.push({ binding: 1, resource: { buffer: this.buffer('cat0') } }, { binding: 2, resource: sourceResource(source) });
    } else if (kind === 'head') {
      const out: GPUBindingResource = this.plan!.direct ? targetResource(target) : { buffer: this.buffer(INTERNAL_ALIAS) };
      entries.push(
        { binding: 1, resource: { buffer: frame } },
        { binding: 2, resource: { buffer: this.buffer('weights') } },
        { binding: 3, resource: { buffer: this.buffer('cat0') } },
        { binding: 4, resource: sourceResource(source) },
        { binding: 5, resource: out },
      );
    } else if (kind === 'resample') {
      entries.push(
        { binding: 1, resource: { buffer: this.buffer(INTERNAL_ALIAS) } },
        { binding: 2, resource: targetResource(target) },
        { binding: 3, resource: { buffer: this.buffer('cubic') } },
      );
    } else {
      throw new Error(`step ${step.label} has no static bind group`);
    }
    return device.createBindGroup({ label: step.label, layout: step.pipeline.getBindGroupLayout(0), entries });
  }

  private offset(layer: LayerName, kind: 'weight' | 'bias', op: TensorOp): number {
    return vec4Offset(this.bundle!.tensor(layer, kind, op));
  }
}

/** Tensor offset (vec4<f16> units) lookup used to fill the dispatch parameters. */
export type TensorOffsetFn = (layer: LayerName, kind: 'weight' | 'bias', op: TensorOp) => number;

/**
 * The complete per-frame dispatch program for a memory plan: shader specialisation,
 * uniform parameters, workgroup counts and buffer bindings of every dispatch, in order.
 */
export function buildVsrProgram(
model: ModelId,
plan: VsrMemoryPlan,
offsetOf: TensorOffsetFn,
source: SourceKind,
output: OutputKind,
storageFormat: GPUTextureFormat,
): VsrStep[] {
  const K = MODEL_CHANNELS[model];
  const L = plan.levels;
  const cs = plan.catStride;
  const xs = plan.x4b4Stride;
  const W = plan.input.width;
  const H = plan.input.height;
  const steps: VsrStep[] = [];
  const catName = (l: number) => `cat${l}` as VsrBufferName;
  const conv = (
    label: string,
    spec: Omit<ConvSpec, 'kind' | 'kb' | 'tx' | 'ty' | 'cch'>,
    inLevel: number,
    outLevel: number,
    layer: LayerName,
    op3: TensorOp,
    buffers: BufferRef[],
  ) => {
    const t = convTiling(spec.k, spec.proj?.cin ?? 0, L[outLevel].width * L[outLevel].height);
    const tileIn = spec.upsample ? { w: t.tx / 2, h: t.ty / 2 } : { w: t.tx, h: t.ty };
    const full: ConvSpec = { kind: 'conv', ...spec, ...t, cch: convChunk(spec.cin, tileIn.w, tileIn.h) };
    const params: LayerParamsData = {
      inW: L[inLevel].width,
      inH: L[inLevel].height,
      outW: L[outLevel].width,
      outH: L[outLevel].height,
      w3: offsetOf(layer, 'weight', op3),
      b3: offsetOf(layer, 'bias', op3),
      w1: spec.proj ? offsetOf(layer, 'weight', '1x1') : 0,
    };
    steps.push({
      label,
      spec: full,
      params,
      groups: [Math.ceil(L[outLevel].width / t.tx), Math.ceil(L[outLevel].height / t.ty)],
      buffers: ['weights', ...buffers],
    });
  };

  const fold: FoldSpec = { kind: 'fold', source, catStride: cs[0] };
  steps.push({
    label: 'fold',
    spec: fold,
    params: { outW: L[0].width, outH: L[0].height, srcW: W, srcH: H },
    groups: [Math.ceil(L[0].width / 16), Math.ceil(L[0].height / 16)],
  });

  for (let i = 0; i < 4; i++) {
    const layer = `rb${i}` as LayerName;
    const cin = i === 0 ? FOLD_CHANNELS : K[i - 1];
    conv(`${layer}A`, { cin, inStride: cs[i], inOffset: 0, k: K[i], outStride: K[i], outOffset: 0, upsample: false }, i, i, layer, '3x3_0', [
      catName(i),
      'mid',
    ]);
    const poolStride = i < 3 ? cs[i + 1] : xs;
    conv(
      `${layer}B`,
      {
        cin: K[i],
        inStride: K[i],
        inOffset: 0,
        k: K[i],
        outStride: cs[i],
        outOffset: K[i],
        upsample: false,
        proj: { cin, offset: 0 },
        pool: { stride: poolStride, offset: 0 },
      },
      i,
      i,
      layer,
      '3x3_1',
      ['mid', catName(i), i < 3 ? catName(i + 1) : 'x4b4'],
    );
  }
  conv('rb4A', { cin: K[3], inStride: xs, inOffset: 0, k: K[4], outStride: K[4], outOffset: 0, upsample: false }, 4, 4, 'rb4', '3x3_0', [
    'x4b4',
    'mid',
  ]);
  conv(
    'rb4B',
    { cin: K[4], inStride: K[4], inOffset: 0, k: K[4], outStride: xs, outOffset: K[3], upsample: false, proj: { cin: K[3], offset: 0 } },
    4,
    4,
    'rb4',
    '3x3_1',
    ['mid', 'x4b4'],
  );
  for (let j = 0; j < 4; j++) {
    const lv = 3 - j;
    const layer = `tb${j}` as LayerName;
    const prev = j === 0 ? { buf: 'x4b4' as VsrBufferName, stride: xs, off: K[3], ch: K[4] } : { buf: catName(lv + 1), stride: cs[lv + 1], off: 0, ch: K[lv + 1] };
    conv(
      `${layer}U`,
      { cin: prev.ch, inStride: prev.stride, inOffset: prev.off, k: K[lv], outStride: cs[lv], outOffset: 0, upsample: true },
      lv + 1,
      lv,
      layer,
      '3x3_u',
      [prev.buf, catName(lv)],
    );
    conv(`${layer}A`, { cin: 2 * K[lv], inStride: cs[lv], inOffset: 0, k: K[lv], outStride: K[lv], outOffset: 0, upsample: false }, lv, lv, layer, '3x3_0', [
      catName(lv),
      'mid',
    ]);
    conv(
      `${layer}B`,
      { cin: K[lv], inStride: K[lv], inOffset: 0, k: K[lv], outStride: cs[lv], outOffset: 0, upsample: false, proj: { cin: 2 * K[lv], offset: 0 } },
      lv,
      lv,
      layer,
      '3x3_1',
      ['mid', catName(lv)],
    );
  }
  const headOutput: OutputKind = plan.direct ? output : 'buffer';
  const head: HeadSpec = { kind: 'head', k0: K[0], featStride: cs[0], source, output: headOutput, storageFormat };
  steps.push({
    label: 'head',
    spec: head,
    params: {
      inW: L[0].width,
      inH: L[0].height,
      outW: 2 * W,
      outH: 2 * H,
      w3: offsetOf('sr', 'weight', '3x3'),
      b3: offsetOf('sr', 'bias', '3x3'),
      w1: offsetOf('sr', 'weight', '1x1'),
      b1: offsetOf('sr', 'bias', '1x1'),
      srcW: W,
      srcH: H,
    },
    groups: [L[0].width / 8, L[0].height / 8],
  });
  if (!plan.direct) {
    const spec: ResampleSpec = { kind: 'resample', source: 'planar', output, storageFormat };
    const ow = plan.output.width;
    const oh = plan.output.height;
    steps.push({ label: 'resample', spec, params: resampleParams(2 * W, 2 * H, 2 * W, ow, oh), groups: resampleGroups(ow, oh) });
  }
  if (steps.length !== vsrDispatchCount(plan.direct)) {
    throw new Error(`internal: unexpected dispatch count ${steps.length}`);
  }
  return steps;
}
