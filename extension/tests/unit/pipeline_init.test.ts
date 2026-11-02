import { describe, expect, it, vi } from 'vitest';
import { VsrPipeline } from '../../src/pipeline/vsr_pipeline';
import { BicubicPipeline } from '../../src/pipeline/resample_pipeline';
import { nodeBundle } from '../harness/node_gpu';

function rejectingDevice(error: Error, popRejects: boolean) {
  const pushErrorScope = vi.fn();
  const popErrorScope = vi.fn(() => popRejects ? Promise.reject(new Error('pop failed')) : Promise.resolve({ message: 'validation failed' }));
  const device = {
    features: new Set(['shader-f16']),
    createBuffer: (desc: GPUBufferDescriptor) => ({ size: desc.size, destroy() {} }),
    queue: { writeBuffer() {} },
    createShaderModule: () => ({}),
    createComputePipelineAsync: () => Promise.reject(error),
    pushErrorScope,
    popErrorScope,
  } as unknown as GPUDevice;
  return { device, pushErrorScope, popErrorScope };
}

describe('pipeline initialisation error scopes', () => {
  it.each([false, true])('pops the VSR scope and preserves the compilation error (popRejects=%s)', async (popRejects) => {
    const error = new Error('compilation failed');
    const f = rejectingDevice(error, popRejects);
    const pipe = new VsrPipeline({ model: 'low', storageFormat: 'rgba8unorm', weights: await nodeBundle('low') });
    try {
      await expect(pipe.initialize(f.device, 32, 32, 64, 64)).rejects.toBe(error);
      expect(f.pushErrorScope).toHaveBeenCalledTimes(1);
      expect(f.popErrorScope).toHaveBeenCalledTimes(1);
    } finally {
      pipe.destroy();
    }
  });

  it('the bicubic pipeline propagates compilation failure without leaving an error scope', async () => {
    const error = new Error('compilation failed');
    const f = rejectingDevice(error, false);
    const pipe = new BicubicPipeline({ storageFormat: 'rgba8unorm', source: 'rgba8', output: 'buffer' });
    try {
      await expect(pipe.initialize(f.device, 32, 32, 64, 64)).rejects.toBe(error);
      expect(f.pushErrorScope.mock.calls.length).toBe(f.popErrorScope.mock.calls.length);
    } finally {
      pipe.destroy();
    }
  });
});
