/**
 * Weight bundle format written by tools/convert_weights.py and the expected low/medium
 * topology (docs/IMPLEMENTATION_NOTES.md section 3).
 */
import type { ModelId } from '../pipeline/types';

export type TensorDType = 'float16' | 'opaque8';
export type LayerName = 'rb0' | 'rb1' | 'rb2' | 'rb3' | 'rb4' | 'tb0' | 'tb1' | 'tb2' | 'tb3' | 'sr';
export type TensorKind = 'weight' | 'bias' | 'amax_output';
export type TensorOp = '3x3_0' | '3x3_1' | '1x1' | '3x3_u' | '3x3';

export interface ManifestTensor {
  offset: number;
  size: number;
  dtype: TensorDType;
  /** Registry order [K, C, R, S] (weights) or [K] (biases); null for opaque8. */
  logical_shape: number[] | null;
  /** Physical order [K, R, S, C] (weights) or [K] (biases); null for opaque8. */
  physical_shape: number[] | null;
  physical_layout?: string;
  layer: LayerName;
  kind: TensorKind;
  op: TensorOp;
  sha256: string;
}

export interface WeightManifest {
  model: ModelId;
  format_version: number;
  byte_order: string;
  alignment_bytes: number;
  encoder_channels: number[];
  total_size_bytes: number;
  raw_payload_bytes: number;
  bundle_sha256: string;
  tensor_count: number;
  tensors: Record<string, ManifestTensor>;
}

/** Encoder widths K0..K4 (notes 3: low [16,32,32,64,64], medium [32,64,64,64,64]). */
export const MODEL_CHANNELS: Record<ModelId, readonly number[]> = {
  low: [16, 32, 32, 64, 64],
  medium: [32, 64, 64, 64, 64],
};

/** Channels of the folded input F (2x2 space-to-depth of half4). */
export const FOLD_CHANNELS = 16;
/** SR head hidden width and learned output channels (3 colours x 16 shuffle phases). */
export const HEAD_HIDDEN = 64;
export const HEAD_OUT = 48;

export function tensorName(model: ModelId, layer: LayerName, kind: TensorKind, op: TensorOp): string {
  return `Flatten_DLPP_${model}_2x_${layer}_${kind}_${op}`;
}

export interface ExpectedTensor {
  name: string;
  layer: LayerName;
  kind: TensorKind;
  op: TensorOp;
  dtype: TensorDType;
  /** Physical shape [K,R,S,C] or [K]; null for opaque8 calibration entries. */
  physicalShape: number[] | null;
}

/** Input channels of encoder block i (rb0 consumes the 16-channel fold). */
export function encoderInputChannels(model: ModelId, i: number): number {
  const K = MODEL_CHANNELS[model];
  return i === 0 ? FOLD_CHANNELS : K[i - 1];
}

/** Decoder block j (0..3) works at level L = 3 - j. */
export function decoderLevel(j: number): number {
  return 3 - j;
}

/** Channels of the decoder input that tb_j upsamples (b for tb0, d_{j-1} otherwise). */
export function decoderPrevChannels(model: ModelId, j: number): number {
  const K = MODEL_CHANNELS[model];
  return j === 0 ? K[4] : K[decoderLevel(j) + 1];
}

/** Complete expected tensor list (99 entries) for a model. */
export function expectedTensors(model: ModelId): ExpectedTensor[] {
  const K = MODEL_CHANNELS[model];
  const out: ExpectedTensor[] = [];
  const add = (layer: LayerName, kind: TensorKind, op: TensorOp, shape: number[] | null) =>
    out.push({ name: tensorName(model, layer, kind, op), layer, kind, op, dtype: shape ? 'float16' : 'opaque8', physicalShape: shape });
  const block = (layer: LayerName, k: number, cin: number, cmid: number) => {
    add(layer, 'weight', '3x3_0', [k, 3, 3, cin]);
    add(layer, 'bias', '3x3_0', [k]);
    add(layer, 'weight', '3x3_1', [k, 3, 3, cmid]);
    add(layer, 'bias', '3x3_1', [k]);
    add(layer, 'weight', '1x1', [k, 1, 1, cin]);
    add(layer, 'bias', '1x1', [k]);
    for (const op of ['3x3_0', '3x3_1', '1x1'] as const) add(layer, 'amax_output', op, null);
  };
  for (let i = 0; i < 5; i++) block(`rb${i}` as LayerName, K[i], encoderInputChannels(model, i), K[i]);
  for (let j = 0; j < 4; j++) {
    const layer = `tb${j}` as LayerName;
    const kl = K[decoderLevel(j)];
    block(layer, kl, 2 * kl, kl);
    add(layer, 'weight', '3x3_u', [kl, 3, 3, decoderPrevChannels(model, j)]);
    add(layer, 'bias', '3x3_u', [kl]);
    add(layer, 'amax_output', '3x3_u', null);
  }
  add('sr', 'weight', '3x3', [HEAD_HIDDEN, 3, 3, K[0]]);
  add('sr', 'bias', '3x3', [HEAD_HIDDEN]);
  add('sr', 'weight', '1x1', [HEAD_OUT, 1, 1, HEAD_HIDDEN]);
  add('sr', 'bias', '1x1', [HEAD_OUT]);
  add('sr', 'amax_output', '3x3', null);
  add('sr', 'amax_output', '1x1', null);
  return out;
}

/** A typed view of one FP16 tensor inside the bundle. */
export interface TensorView {
  name: string;
  /** Physical shape [K,R,S,C] or [K]. */
  shape: readonly number[];
  /** Offset of the first element in FP16 elements from the start of the bundle. */
  elementOffset: number;
  /** Element values (a view into the bundle bytes). */
  data: Float16Array;
}

export interface WeightBundle {
  model: ModelId;
  channels: readonly number[];
  manifest: WeightManifest;
  /** The raw bundle bytes (uploaded verbatim as one storage buffer). */
  bytes: ArrayBuffer;
  tensor(layer: LayerName, kind: 'weight' | 'bias', op: TensorOp): TensorView;
}
