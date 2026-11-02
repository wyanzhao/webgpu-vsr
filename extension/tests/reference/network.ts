/**
 * Whole-network reference of the folded low/medium RTX VSR model
 * (docs/IMPLEMENTATION_NOTES.md section 3, steps 1-10).
 */
import { MODEL_CHANNELS, type LayerName, type TensorOp } from '../../src/weights/types';
import type { ModelId } from '../../src/pipeline/types';
import { alignedSize } from '../../src/pipeline/geometry';
import { addHalf, concat, conv1x1, conv3x3, fold, padReplicate, pool2x2, srHead, type Tensor } from './ops';

/** Weight provider: values of an FP16 tensor in physical (KRSC) order. */
export interface RefWeights {
  get(layer: LayerName, kind: 'weight' | 'bias', op: TensorOp): ArrayLike<number>;
}

export interface NetworkOptions {
  /** Residual gain g = 2*sharpness (default 1). */
  gain?: number;
  /**
   * Deliberately wrong variants used as negative controls in parity tests:
   *   'swap-concat'  : cat = [skip, U] instead of [U, skip]
   *   'no-projection': RB(x) = ELU(C3_1(ELU(C3_0(x)))) without the 1x1 skip projection
   */
  negativeControl?: 'swap-concat' | 'no-projection';
}

export interface NetworkTrace {
  fold: Tensor;
  skips: Tensor[];
  bottleneck: Tensor;
  decoder: Tensor[];
  /** Planar RGB halves, 2W x 2H. */
  output: Float32Array;
}

function residualBlock(x: Tensor, w: RefWeights, layer: LayerName, k: number, opts: NetworkOptions): Tensor {
  const m = conv3x3(x, w.get(layer, 'weight', '3x3_0'), w.get(layer, 'bias', '3x3_0'), k, 'elu');
  const e = conv3x3(m, w.get(layer, 'weight', '3x3_1'), w.get(layer, 'bias', '3x3_1'), k, 'elu');
  if (opts.negativeControl === 'no-projection') return e;
  const p = conv1x1(x, w.get(layer, 'weight', '1x1'), null, k);
  return addHalf(e, p);
}

/**
 * Run the network on the half4 source I (W x H, values [2R-1, 2G-1, 2B-1, 0] as halves).
 */
export function runNetwork(model: ModelId, w: RefWeights, src: Tensor, opts: NetworkOptions = {}): NetworkTrace {
  const K = MODEL_CHANNELS[model];
  const a = alignedSize(src.w, src.h);
  const f = fold(padReplicate(src, a.width, a.height));
  const skips: Tensor[] = [];
  let x = f;
  for (let i = 0; i < 4; i++) {
    const s = residualBlock(x, w, `rb${i}` as LayerName, K[i], opts);
    skips.push(s);
    x = pool2x2(s);
  }
  const bottleneck = residualBlock(x, w, 'rb4', K[4], opts);
  let d = bottleneck;
  const decoder: Tensor[] = [];
  for (let j = 0; j < 4; j++) {
    const L = 3 - j;
    const layer = `tb${j}` as LayerName;
    const u = conv3x3(d, w.get(layer, 'weight', '3x3_u'), w.get(layer, 'bias', '3x3_u'), K[L], 'elu', true);
    const cat = opts.negativeControl === 'swap-concat' ? concat(skips[L], u) : concat(u, skips[L]);
    d = residualBlock(cat, w, layer, K[L], opts);
    decoder.push(d);
  }
  const output = srHead(
    d,
    src,
    w.get('sr', 'weight', '3x3'),
    w.get('sr', 'bias', '3x3'),
    w.get('sr', 'weight', '1x1'),
    w.get('sr', 'bias', '1x1'),
    opts.gain ?? 1,
  );
  return { fold: f, skips, bottleneck, decoder, output };
}
