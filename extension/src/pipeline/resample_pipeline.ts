/**
 * Quality 0: Catmull-Rom (a = -0.5) resample of the source frame (layout-1 preprocess
 * H(R), H(G), H(B) per tap) straight to the canvas, one dispatch per frame. The same
 * resample kernel template is used by the VSR pipeline for its internal -> target pass.
 */
import { resampleScale } from './geometry';
import { GpuMemoryLedger, UNIFORM_SLOT_BYTES, planBicubicMemory } from './memory';
import { encodeFrameParams, encodeLayerParams, getComputePipeline, type LayerParamsData, type ResampleSpec } from './shader_source';
import type { FrameSource, FrameTarget, OutputKind, SRComputePipeline } from './types';
import type { FrameSlot } from './ring';
import { cubicWeightTable } from './cubic_weights';

/** LayerParams of a resample dispatch from a (valid) source of sw x sh. */
export function resampleParams(sw: number, sh: number, stride: number, ow: number, oh: number): LayerParamsData {
  const s = resampleScale(sw, sh, ow, oh);
  return { inW: sw, inH: sh, outW: ow, outH: oh, srcW: sw, srcH: sh, aux0: stride, aux1: stride * sh, f0: s.x, f1: s.y };
}

export function resampleGroups(ow: number, oh: number): [number, number] {
  return [Math.ceil(ow / 32), Math.ceil(oh / 8)];
}

export function sourceResource(src: FrameSource): GPUBindingResource {
  return src.kind === 'texture' ? src.texture : { buffer: src.buffer };
}

export function targetResource(t: FrameTarget): GPUBindingResource {
  return t.kind === 'canvas' ? t.view : { buffer: t.buffer };
}

export interface BicubicOptions {
  storageFormat: GPUTextureFormat;
  /** Source variant: 'texture' in production, 'rgba8' for exact testing. */
  source?: 'texture' | 'rgba8';
  output?: OutputKind;
}

export class BicubicPipeline implements SRComputePipeline {
  readonly id = 'bicubic-q0';
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private ledger = new GpuMemoryLedger();
  private params: GPUBuffer | null = null;
  private cubic: GPUBuffer | null = null;
  private size = { inW: 0, inH: 0, outW: 0, outH: 0 };
  private sharpness = 0.5;
  private readonly source: 'texture' | 'rgba8';
  private readonly output: OutputKind;

  constructor(private readonly options: BicubicOptions) {
    this.source = options.source ?? 'texture';
    this.output = options.output ?? 'canvas';
  }

  get scaleFactor(): number {
    return this.size.inW > 0 ? this.size.outW / this.size.inW : 0;
  }

  get spec(): ResampleSpec {
    return { kind: 'resample', source: this.source, output: this.output, storageFormat: this.options.storageFormat };
  }

  async initialize(device: GPUDevice, inWidth: number, inHeight: number, outWidth: number, outHeight: number): Promise<void> {
    if (this.device !== device) {
      this.ledger.releaseAll();
      this.device = device;
      const bytes = Object.fromEntries(planBicubicMemory().map((b) => [b.name, b.bytes]));
      this.params = this.ledger.createBuffer(device, {
        label: 'q0-params',
        size: bytes.params as number,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.cubic = this.ledger.createBuffer(device, {
        label: 'q0-cubic',
        size: bytes.cubic as number,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.cubic, 0, cubicWeightTable());
      this.pipeline = await getComputePipeline(device, this.spec);
    }
    this.size = { inW: inWidth, inH: inHeight, outW: outWidth, outH: outHeight };
    device.queue.writeBuffer(this.params!, 0, encodeLayerParams(resampleParams(inWidth, inHeight, inWidth, outWidth, outHeight)));
  }

  execute(
    commandEncoder: GPUCommandEncoder,
    externalTexture: GPUExternalTexture,
    targetCanvasView: GPUTextureView,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    this.encodeFrame(commandEncoder, { kind: 'texture', texture: externalTexture }, { kind: 'canvas', view: targetCanvasView }, timestampWrites);
  }

  /** Execute with the timestamp writes of a frame-ring slot. */
  executeSlot(commandEncoder: GPUCommandEncoder, externalTexture: GPUExternalTexture, targetCanvasView: GPUTextureView, slot: FrameSlot): void {
    this.encodeFrame(commandEncoder, { kind: 'texture', texture: externalTexture }, { kind: 'canvas', view: targetCanvasView }, slot.timestampWrites);
  }

  encodeFrame(encoder: GPUCommandEncoder, source: FrameSource, target: FrameTarget, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.device || !this.pipeline || !this.params || !this.cubic) throw new Error('BicubicPipeline.execute before initialize');
    if (source.kind !== this.source) throw new Error(`pipeline expects a '${this.source}' source, got '${source.kind}'`);
    if (target.kind !== this.output) throw new Error(`pipeline expects a '${this.output}' target, got '${target.kind}'`);
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params, offset: 0, size: UNIFORM_SLOT_BYTES } },
        { binding: 1, resource: sourceResource(source) },
        { binding: 2, resource: targetResource(target) },
        { binding: 3, resource: { buffer: this.cubic } },
      ],
    });
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    const [gx, gy] = resampleGroups(this.size.outW, this.size.outH);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  }

  /** Quality 0 has no learned residual: the setting is validated and kept for reporting. */
  setSharpness(sharpness: number): void {
    encodeFrameParams(sharpness);
    this.sharpness = sharpness;
  }

  get currentSharpness(): number {
    return this.sharpness;
  }

  memoryBytes(): number {
    return this.ledger.totalBytes;
  }

  destroy(): void {
    this.ledger.releaseAll();
    this.device = null;
    this.pipeline = null;
    this.params = null;
    this.cubic = null;
  }
}
