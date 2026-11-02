import Foundation
import Metal

// Benchmark mode: deterministic synthetic frames, warm-up, then N timed frames. Each frame is
// one command buffer holding one serial compute pass; per-frame GPU time is
// MTLCommandBuffer.gpuEndTime - gpuStartTime. Up to three frames are in flight (triple
// buffering, like the extension's frame ring); the queue serialises them on the single
// activation pool through Metal's hazard tracking.

struct BenchResult: Codable {
    let model: String
    let input: String
    let output: String
    let outputFormat: String
    let frames: Int
    let warmup: Int
    let dispatchesPerFrame: Int
    let gpuMedianMs: Double
    let gpuP95Ms: Double
    let gpuMeanMs: Double
    let gpuMinMs: Double
    let gpuMaxMs: Double
    let wallFps: Double
    let activationBytes: Int
    let weightBufferBytes: Int
    let sourceBytes: Int
    let peakAllocatedBufferBytes: Int
    let deviceAllocatedBytes: Int
}

func percentile(_ sorted: [Double], _ q: Double) -> Double {
    precondition(!sorted.isEmpty)
    // Nearest-rank percentile.
    let rank = Int((q * Double(sorted.count)).rounded(.up))
    return sorted[min(max(rank - 1, 0), sorted.count - 1)]
}

func runBenchmark(ctx: MetalContext, model: LoadedModel, width: Int, height: Int, frames: Int,
                  warmup: Int, format: OutputFormat) throws -> BenchResult {
    guard frames >= 1, warmup >= 0 else { throw BenchError("frames must be >= 1 and warmup >= 0") }
    let engine = try VSREngine(ctx: ctx, model: model, width: width, height: height, format: format)
    let sources = try (0..<3).map { try makeSourceBuffer(ctx.device, syntheticFrame(width: width, height: height, seed: UInt32(100 + $0))) }
    for i in 0..<warmup { try runFrame(engine, source: sources[i % 3]) }

    let inFlight = DispatchSemaphore(value: 3)
    let lock = NSLock()
    var times = [Double](repeating: 0, count: frames)
    var failure: String?
    let group = DispatchGroup()
    let start = DispatchTime.now().uptimeNanoseconds
    for i in 0..<frames {
        inFlight.wait()
        guard let cb = ctx.queue.makeCommandBuffer() else { throw BenchError("no command buffer") }
        engine.encode(cb, source: sources[i % 3])
        group.enter()
        cb.addCompletedHandler { done in
            lock.lock()
            if let err = done.error { failure = "\(err)" }
            times[i] = (done.gpuEndTime - done.gpuStartTime) * 1000
            lock.unlock()
            inFlight.signal()
            group.leave()
        }
        cb.commit()
    }
    group.wait()
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    if let f = failure { throw BenchError("command buffer failed: \(f)") }
    let sorted = times.sorted()
    let g = engine.geometry
    let sourceBytes = sources.reduce(0) { $0 + $1.length }
    return BenchResult(
        model: model.id, input: "\(width)x\(height)", output: "\(g.outW)x\(g.outH)", outputFormat: format.rawValue,
        frames: frames, warmup: warmup, dispatchesPerFrame: engine.dispatchCount,
        gpuMedianMs: percentile(sorted, 0.5), gpuP95Ms: percentile(sorted, 0.95),
        gpuMeanMs: times.reduce(0, +) / Double(frames), gpuMinMs: sorted.first!, gpuMaxMs: sorted.last!,
        wallFps: Double(frames) / elapsed,
        activationBytes: engine.activationBytes, weightBufferBytes: engine.weightBufferBytes, sourceBytes: sourceBytes,
        peakAllocatedBufferBytes: engine.allocatedBytes + sourceBytes,
        deviceAllocatedBytes: ctx.device.currentAllocatedSize)
}

func formatMB(_ bytes: Int) -> String { String(format: "%.1f MiB", Double(bytes) / 1_048_576) }

func printTable(_ rows: [BenchResult]) {
    let header = ["model", "input", "output", "frames", "median ms", "p95 ms", "mean ms", "fps (wall)", "peak buffers", "activations", "weights"]
    var lines = [header]
    for r in rows {
        lines.append([r.model, r.input, r.output, "\(r.frames)", String(format: "%.3f", r.gpuMedianMs),
                      String(format: "%.3f", r.gpuP95Ms), String(format: "%.3f", r.gpuMeanMs),
                      String(format: "%.1f", r.wallFps), formatMB(r.peakAllocatedBufferBytes),
                      formatMB(r.activationBytes), formatMB(r.weightBufferBytes)])
    }
    let widths = (0..<header.count).map { c in lines.map { $0[c].count }.max()! }
    for (i, l) in lines.enumerated() {
        print(l.enumerated().map { $0.element.padding(toLength: widths[$0.offset], withPad: " ", startingAt: 0) }.joined(separator: "  "))
        if i == 0 { print(widths.map { String(repeating: "-", count: $0) }.joined(separator: "  ")) }
    }
}

struct StepProfile: Codable {
    let step: String
    let gpuMs: Double
    let gmacs: Double
    let tflops: Double
}

/// --profile: per-dispatch GPU time. One full frame first makes every intermediate valid,
/// then each dispatch is re-encoded `repeats` times in its own command buffer.
func profileSteps(ctx: MetalContext, model: LoadedModel, width: Int, height: Int, repeats: Int,
                  format: OutputFormat) throws -> [StepProfile] {
    let engine = try VSREngine(ctx: ctx, model: model, width: width, height: height, format: format)
    let src = try makeSourceBuffer(ctx.device, syntheticFrame(width: width, height: height, seed: 100))
    try runFrame(engine, source: src)
    var out: [StepProfile] = []
    for (i, label) in engine.stepLabels.enumerated() {
        guard let cb = ctx.queue.makeCommandBuffer() else { throw BenchError("no command buffer") }
        engine.encode(cb, source: src, steps: i..<(i + 1), repeats: repeats)
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw BenchError("profile \(label): \(e)") }
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(repeats)
        let macs = engine.stepMACs[i]
        out.append(StepProfile(step: label, gpuMs: ms, gmacs: macs / 1e9, tflops: macs > 0 ? 2 * macs / (ms * 1e-3) / 1e12 : 0))
    }
    return out
}

func printProfile(_ model: String, _ input: String, _ rows: [StepProfile]) {
    print("profile \(model) \(input):")
    let total = rows.reduce(0) { $0 + $1.gpuMs }
    for r in rows {
        print(String(format: "  %-8@ %8.3f ms  %6.2f GMAC  %6.2f TFLOP/s  %5.1f%%", r.step, r.gpuMs, r.gmacs, r.tflops, 100 * r.gpuMs / total))
    }
    let macs = rows.reduce(0) { $0 + $1.gmacs }
    print(String(format: "  total    %8.3f ms  %6.2f GMAC  %6.2f TFLOP/s", total, macs, 2 * macs * 1e9 / (total * 1e-3) / 1e12))
}
