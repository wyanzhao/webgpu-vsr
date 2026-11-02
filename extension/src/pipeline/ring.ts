/**
 * Triple-buffered frame ring (docs/IMPLEMENTATION_NOTES.md section 4).
 *
 * Each of the 3 slots owns a frame-uniform buffer, and when timestamp-query is available a
 * 2-entry query set, a resolve buffer and a MAP_READ readback buffer. acquire() returns
 * null when every slot is in flight: the caller drops that frame (the ring counts it), so
 * the GPU never has more than 3 frames queued. Activations are not triplicated: the queue
 * serialises submissions, so one activation pool is correct.
 */
import { GpuMemoryLedger, UNIFORM_SLOT_BYTES } from './memory';
import type { FrameTiming } from './types';

export const RING_SLOTS = 3;
const TIMESTAMP_BYTES = 16;

export interface FrameSlot {
  readonly index: number;
  /** Per-slot frame uniform (FrameParams); pipelines write it with queue.writeBuffer. */
  readonly uniformBuffer: GPUBuffer;
  /** Pass timestamp writes (begin -> 0, end -> 1) when timestamps are available. */
  readonly timestampWrites?: GPUComputePassTimestampWrites;
}

interface SlotState extends FrameSlot {
  querySet?: GPUQuerySet;
  resolveBuffer?: GPUBuffer;
  readbackBuffer?: GPUBuffer;
  inFlight: boolean;
  resolved: boolean;
}

export interface FrameRingOptions {
  slots?: number;
  /** Clock for CPU fallback timing (injectable). */
  now?: () => number;
}

export class FrameRing {
  private readonly slots: SlotState[] = [];
  private readonly ledger = new GpuMemoryLedger();
  private readonly now: () => number;
  private dropped = 0;
  private submittedFrames = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly useTimestamps: boolean,
    options: FrameRingOptions = {},
  ) {
    this.now = options.now ?? (() => performance.now());
    const n = options.slots ?? RING_SLOTS;
    for (let i = 0; i < n; i++) {
      const uniformBuffer = this.ledger.createBuffer(device, {
        label: `ring-${i}-frame`,
        size: UNIFORM_SLOT_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const slot: SlotState = { index: i, uniformBuffer, inFlight: false, resolved: false };
      if (useTimestamps) {
        const querySet = this.ledger.createQuerySet(device, { label: `ring-${i}-ts`, type: 'timestamp', count: 2 });
        slot.querySet = querySet;
        slot.resolveBuffer = this.ledger.createBuffer(device, {
          label: `ring-${i}-resolve`,
          size: TIMESTAMP_BYTES,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        slot.readbackBuffer = this.ledger.createBuffer(device, {
          label: `ring-${i}-readback`,
          size: TIMESTAMP_BYTES,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        (slot as { timestampWrites?: GPUComputePassTimestampWrites }).timestampWrites = {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        };
      }
      this.slots.push(slot);
    }
  }

  get size(): number {
    return this.slots.length;
  }

  get inFlight(): number {
    return this.slots.filter((s) => s.inFlight).length;
  }

  get droppedFrames(): number {
    return this.dropped;
  }

  get completedFrames(): number {
    return this.submittedFrames;
  }

  /** A free slot, or null (frame dropped and counted) when all slots are in flight. */
  acquire(): FrameSlot | null {
    if (this.destroyed) throw new Error('FrameRing used after destroy()');
    const slot = this.slots.find((s) => !s.inFlight);
    if (!slot) {
      this.dropped++;
      return null;
    }
    slot.inFlight = true;
    slot.resolved = false;
    return slot;
  }

  /** Record the timestamp resolve + readback copy (call after the compute pass ends). */
  encodeResolve(encoder: GPUCommandEncoder, handle: FrameSlot): void {
    const slot = this.state(handle);
    if (!slot.querySet || !slot.resolveBuffer || !slot.readbackBuffer) return;
    encoder.resolveQuerySet(slot.querySet, 0, 2, slot.resolveBuffer, 0);
    encoder.copyBufferToBuffer(slot.resolveBuffer, 0, slot.readbackBuffer, 0, TIMESTAMP_BYTES);
    slot.resolved = true;
  }

  /**
   * Submit the frame's command buffers and resolve with its timing once the GPU finished;
   * the slot becomes free at that point.
   */
  async submit(handle: FrameSlot, commandBuffers: GPUCommandBuffer[]): Promise<FrameTiming> {
    const slot = this.state(handle);
    if (!slot.inFlight) throw new Error(`ring slot ${slot.index} was not acquired`);
    const t0 = this.now();
    try {
      this.device.queue.submit(commandBuffers);
      await this.device.queue.onSubmittedWorkDone();
      const cpuMs = this.now() - t0;
      let timing: FrameTiming = { gpuTimeMs: cpuMs, timingSource: 'cpu', latencyMs: cpuMs };
      if (slot.resolved && slot.readbackBuffer) {
        await slot.readbackBuffer.mapAsync(GPUMapMode.READ);
        const ts = new BigUint64Array(slot.readbackBuffer.getMappedRange().slice(0));
        slot.readbackBuffer.unmap();
        const ns = Number(ts[1] - ts[0]);
        if (ts[1] > ts[0] && Number.isFinite(ns)) timing = { gpuTimeMs: ns / 1e6, timingSource: 'timestamp', latencyMs: cpuMs };
      }
      this.submittedFrames++;
      return timing;
    } finally {
      slot.inFlight = false;
      slot.resolved = false;
    }
  }

  /** Release a slot that was acquired but never submitted (e.g. encoding failed). */
  abandon(handle: FrameSlot): void {
    const slot = this.state(handle);
    slot.inFlight = false;
    slot.resolved = false;
  }

  memoryBytes(): number {
    return this.ledger.totalBytes;
  }

  destroy(): void {
    this.destroyed = true;
    this.ledger.releaseAll();
    this.slots.length = 0;
  }

  private state(handle: FrameSlot): SlotState {
    const slot = this.slots[handle.index];
    if (!slot || slot !== handle) throw new Error('foreign frame slot');
    return slot;
  }
}
