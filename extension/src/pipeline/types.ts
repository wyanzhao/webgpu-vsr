/**
 * Shared GPU pipeline contracts (docs/IMPLEMENTATION_NOTES.md section 5).
 *
 * GpuContext and SRComputePipeline are implemented exactly as specified; the
 * extra members below are additive.
 */

export type ModelId = 'low' | 'medium';

/**
 * Stored SR-head baseline coefficients at scale 2 (constructor constant bank 0x628,
 * semantics/FLOAT-RECONSTRUCTION.md), tap order [-1, 0, 1, 2], indexed by output parity.
 * shader_source.ts bakes this table into the head shader.
 */
export const BASELINE_COEFFICIENTS: readonly (readonly number[])[] = [
  [-0.03515625, 0.26171875, 0.87890625, -0.10546875],
  [-0.10546875, 0.87890625, 0.26171875, -0.03515625],
];

export type QualityTier = 'medium' | 'low' | 'quality0_bicubic';

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasF16: boolean;
  hasTimestamps: boolean;
  /** 'bgra8unorm' if bgra8unorm-storage is available, else 'rgba8unorm'. */
  storageFormat: GPUTextureFormat;
}

export interface SRComputePipeline {
  /** 'rtx-vsr-low' | 'rtx-vsr-medium' | 'bicubic-q0' */
  readonly id: string;
  /** Internal scale (2 for VSR, outW/inW for bicubic). */
  readonly scaleFactor: number;
  initialize(device: GPUDevice, inWidth: number, inHeight: number, outWidth: number, outHeight: number): Promise<void>;
  execute(
    commandEncoder: GPUCommandEncoder,
    externalTexture: GPUExternalTexture,
    targetCanvasView: GPUTextureView,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void;
  setSharpness(sharpness: number): void;
  /** Bytes of every GPU allocation it owns. */
  memoryBytes(): number;
  destroy(): void;
}

/** Where a pipeline reads its source frame from. Production uses 'texture'. */
export type SourceKind =
  /** GPUExternalTexture (importExternalTexture of a VideoFrame). */
  | 'texture'
  /** Tightly packed RGBA8 bytes, one u32 per pixel (R in the low byte). */
  | 'rgba8'
  /** Pre-normalised half4 I = [2R-1, 2G-1, 2B-1, 0] per pixel (exact-testing only). */
  | 'half4';

/** Where the final image goes. Production uses 'canvas'. */
export type OutputKind =
  /** write-only storage texture view (canvas swap-chain texture). */
  | 'canvas'
  /** Planar FP16 RGB buffer (VSR head) or packed RGBA8 u32 buffer (resample). */
  | 'buffer';

/** Source binding handed to a pipeline for one frame. */
export type FrameSource =
  | { kind: 'texture'; texture: GPUExternalTexture }
  | { kind: 'rgba8'; buffer: GPUBuffer }
  | { kind: 'half4'; buffer: GPUBuffer };

/** Destination binding for one frame. */
export type FrameTarget = { kind: 'canvas'; view: GPUTextureView } | { kind: 'buffer'; buffer: GPUBuffer };

export type TimingSource = 'timestamp' | 'cpu';

export interface FrameTiming {
  /** GPU execution time (timestamp-query) or CPU submit -> onSubmittedWorkDone time. */
  gpuTimeMs: number;
  timingSource: TimingSource;
  /** submit -> onSubmittedWorkDone (queue wait + execution + callback delivery), ms. */
  latencyMs: number;
}
