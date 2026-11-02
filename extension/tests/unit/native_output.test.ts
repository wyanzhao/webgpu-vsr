import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

const sources = resolve(__dirname, '../../../native-benchmark/Sources');
const hasSwift = spawnSync('swiftc', ['--version']).status === 0;

// Exercise the actual CLI and table formatter without requiring a Metal device or model files.
const stubs = `
struct BenchError: Error { let message: String; init(_ message: String) { self.message = message } }
enum OutputFormat: String { case half, rgba8 }
let allModelIds = ["low"]
struct Device { let name = "test device" }
struct MetalContext {
    let device = Device()
    init(metallib: URL) throws {}
}
struct Spec { let family = "test" }
struct LoadedModel {
    let id = "low", spec = Spec(), widths = [1], shuffle = 2, hidden = 1, headOutputs = 4, parameterCount = 1
    static func load(id: String, modelsDir: URL) throws -> LoadedModel { LoadedModel() }
}
struct CheckResult: Codable { let passed: Bool }
class Validator {
    let results = [CheckResult(passed: true)]
    init(ctx: MetalContext, models: [LoadedModel], verbose: Bool, headFixturesDir: URL) {}
    func runAll() throws -> Bool { print("validation: 1/1 checks passed"); return true }
}
func profileSteps(ctx: MetalContext, model: LoadedModel, width: Int, height: Int, repeats: Int, format: OutputFormat) throws -> [StepProfile] {
    [StepProfile(step: "head", gpuMs: 1, gmacs: 2, tflops: 4)]
}
func printProfile(_ model: String, _ input: String, _ rows: [StepProfile]) { print("profile table") }
struct QualityRow: Codable { let method: String }
func runQuality(ctx: MetalContext, modelsDir: URL, dir: URL, outDir: URL?, models: [String]) throws -> [QualityRow] { [] }
func printQuality(_ rows: [QualityRow]) {}
func runBenchmark(ctx: MetalContext, model: LoadedModel, width: Int, height: Int, frames: Int, warmup: Int, format: OutputFormat) throws -> BenchResult {
    BenchResult(model: "low", input: "16x16", output: "32x32", outputFormat: format.rawValue,
                frames: frames, warmup: warmup, dispatchesPerFrame: 1,
                gpuMedianMs: 1, gpuP95Ms: 1, gpuMeanMs: 1, gpuMinMs: 1, gpuMaxMs: 1, wallFps: 1000,
                activationBytes: 1048576, weightBufferBytes: 1048576, sourceBytes: 0,
                peakAllocatedBufferBytes: 2097152, deviceAllocatedBytes: 2097152)
}
`;

describe.skipIf(!hasSwift)('native CLI output', () => {
  let temporary: string;
  let binary: string;
  beforeAll(() => {
    temporary = mkdtempSync(join(tmpdir(), 'webgpu-vsr-output-'));
    binary = join(temporary, 'bench');
    const benchmark = readFileSync(join(sources, 'Benchmark.swift'), 'utf8');
    const result = benchmark.slice(benchmark.indexOf('struct BenchResult'), benchmark.indexOf('func percentile'));
    const table = benchmark.slice(benchmark.indexOf('func formatMB'), benchmark.indexOf('struct StepProfile'));
    const profile = benchmark.slice(benchmark.indexOf('struct StepProfile'), benchmark.indexOf('/// --profile'));
    const main = readFileSync(join(sources, 'main.swift'), 'utf8').replace('import Metal', '');
    const source = join(temporary, 'main.swift');
    writeFileSync(source, `import Foundation\n${result}\n${table}\n${profile}\n${stubs}\n${main}`);
    execFileSync('swiftc', ['-module-cache-path', join(temporary, 'cache'), source, '-o', binary], { timeout: 60000 });
  }, 60000);
  afterAll(() => { if (temporary) rmSync(temporary, { recursive: true, force: true }); });

  it('--json - writes only JSON to stdout and routes human output to stderr', () => {
    for (const mode of [[], ['--validate', '--profile', '--all', '--verbose']]) {
      const result = spawnSync(binary, [...mode, '--frames', '1', '--json', '-'], { encoding: 'utf8' });
      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({ device: 'test device', benchmarks: expect.arrayContaining([expect.objectContaining({ model: 'low' })]) });
      expect(result.stderr).toContain('device: test device');
      expect(result.stderr).toContain('peak buffers');
    }
    const normal = spawnSync(binary, ['--frames', '1'], { encoding: 'utf8' });
    expect(normal.stdout).toContain('device: test device');
    expect(normal.stdout).toContain('peak buffers');
    expect(normal.stderr).toBe('');
  });

  it('labels binary memory values as MiB', () => {
    const result = spawnSync(binary, ['--frames', '1'], { encoding: 'utf8' });
    expect(result.status).toBe(0);
    expect(result.stdout).toContain('1.0 MiB');
    expect(result.stdout).toContain('2.0 MiB');
    expect(result.stdout).not.toMatch(/\bMB\b/);
  });

  it('includes profile measurements and model/input metadata in --profile --json - stdout', () => {
    for (const frames of [[], ['--frames', '3']]) {
      const result = spawnSync(binary, ['--profile', '--model', 'low', '--input', '16x16', '--format', 'half', ...frames, '--json', '-'], { encoding: 'utf8' });
      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        device: 'test device',
        profile: {
          model: 'low', input: '16x16', outputFormat: 'half', repeats: frames.length ? 3 : 20,
          rows: [{ step: 'head', gpuMs: 1, gmacs: 2, tflops: 4 }],
        },
      });
      expect(result.stderr).toContain('profile table');
      expect(result.stderr).toContain('device: test device');
    }
  });
});
