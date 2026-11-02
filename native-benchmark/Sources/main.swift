import Foundation
import Metal

// vsr-bench: native Metal benchmark and validator for the reconstructed RTX VSR
// (SDK 1.1) float models low / medium / high / ultra.

let usage = """
usage: vsr-bench [options]

  --model low|medium|high|ultra   model for a single benchmark run (default low)
  --input WxH                     source size (default 960x540); output is 2W x 2H
  --frames N                      timed frames (default 200; --all default 60)
  --warmup N                      untimed warm-up frames (default 10)
  --format half|rgba8             head output: planar FP16 or packed RGBA8 (default rgba8)
  --all                           benchmark every model at 960x540, 1280x720, 1920x1080
  --validate                      run all correctness checks; exit 1 on any failure
  --profile                       per-dispatch GPU time / throughput for --model at --input
  --quality DIR                   score every model and two CPU baselines on DIR/<set>/*.png
                                  against DIR/ref (see tools/make_quality_set.sh)
  --quality-out DIR               with --quality, write each clip's first upscaled frame here
  --json PATH                     also write results as JSON (PATH '-' = stdout)
  --models-dir DIR                exported model specs (default <exe dir>/models)
  --metallib PATH                 compiled kernels (default <exe dir>/vsr_kernels.metallib)
  --head-fixtures DIR             research head fixtures for --validate
                                  (default <exe dir>/../../extension/tests/fixtures/head)
  --verbose                       extra diagnostics
  --help
"""

struct Options {
    var model = "low"
    var input = (960, 540)
    var frames: Int?
    var warmup = 10
    var format = OutputFormat.rgba8
    var all = false
    var validate = false
    var profile = false
    var quality: URL?
    var qualityOut: URL?
    var json: String?
    var modelsDir: URL?
    var metallib: URL?
    var headFixtures: URL?
    var verbose = false
}

func parseSize(_ s: String) throws -> (Int, Int) {
    let parts = s.lowercased().split(separator: "x")
    guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]), w > 0, h > 0, w <= 8192, h <= 8192 else {
        throw BenchError("bad size '\(s)' (expected WxH)")
    }
    return (w, h)
}

func parseOptions(_ args: [String]) throws -> Options {
    var o = Options()
    var i = 0
    func value() throws -> String {
        i += 1
        guard i < args.count else { throw BenchError("missing value for \(args[i - 1])") }
        return args[i]
    }
    while i < args.count {
        let a = args[i]
        switch a {
        case "--model":
            o.model = try value()
            guard allModelIds.contains(o.model) else { throw BenchError("unknown model '\(o.model)'") }
        case "--input": o.input = try parseSize(try value())
        case "--frames":
            guard let n = Int(try value()), n >= 1 else { throw BenchError("--frames needs a positive integer") }
            o.frames = n
        case "--warmup":
            guard let n = Int(try value()), n >= 0 else { throw BenchError("--warmup needs a non-negative integer") }
            o.warmup = n
        case "--format":
            guard let f = OutputFormat(rawValue: try value()) else { throw BenchError("--format must be half or rgba8") }
            o.format = f
        case "--all": o.all = true
        case "--validate": o.validate = true
        case "--profile": o.profile = true
        case "--quality": o.quality = URL(fileURLWithPath: try value())
        case "--quality-out": o.qualityOut = URL(fileURLWithPath: try value())
        case "--json": o.json = try value()
        case "--models-dir": o.modelsDir = URL(fileURLWithPath: try value())
        case "--metallib": o.metallib = URL(fileURLWithPath: try value())
        case "--head-fixtures": o.headFixtures = URL(fileURLWithPath: try value())
        case "--verbose": o.verbose = true
        case "--help", "-h":
            print(usage)
            exit(0)
        default: throw BenchError("unknown argument '\(a)'\n\(usage)")
        }
        i += 1
    }
    return o
}

struct ProfileResult: Codable {
    let model: String
    let input: String
    let outputFormat: String
    let repeats: Int
    let rows: [StepProfile]
}

struct Report: Codable {
    let device: String
    let os: String
    let validation: [CheckResult]?
    let validationPassed: Bool?
    let benchmarks: [BenchResult]?
    let profile: ProfileResult?
    let quality: [QualityRow]?
}

func main() -> Int32 {
    do {
        let o = try parseOptions(Array(CommandLine.arguments.dropFirst()))
        // Keep all human diagnostics (including validation / profiling) off JSON stdout.
        let jsonOutput: FileHandle?
        if o.json == "-" {
            fflush(stdout)
            let saved = dup(STDOUT_FILENO)
            guard saved >= 0 else { throw BenchError("cannot preserve JSON stdout") }
            jsonOutput = FileHandle(fileDescriptor: saved, closeOnDealloc: true)
            guard dup2(STDERR_FILENO, STDOUT_FILENO) >= 0 else { throw BenchError("cannot redirect human output") }
        } else {
            jsonOutput = nil
        }
        defer {
            if let output = jsonOutput {
                fflush(stdout)
                dup2(output.fileDescriptor, STDOUT_FILENO)
            }
        }
        let exeDir = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0]))
            .resolvingSymlinksInPath().deletingLastPathComponent()
        let ctx = try MetalContext(metallib: o.metallib ?? exeDir.appendingPathComponent("vsr_kernels.metallib"))
        let modelsDir = o.modelsDir ?? exeDir.appendingPathComponent("models")
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        print("device: \(ctx.device.name), \(osVersion)")

        var validation: [CheckResult]?
        var passed: Bool?
        var bench: [BenchResult]?
        var profile: ProfileResult?
        var quality: [QualityRow]?

        if o.validate {
            let models = try allModelIds.map { try LoadedModel.load(id: $0, modelsDir: modelsDir) }
            for m in models {
                print("model \(m.id): \(m.spec.family), widths \(m.widths), head p\(m.shuffle) hidden \(m.hidden) K \(m.headOutputs), \(m.parameterCount) parameters")
            }
            let fixtures = o.headFixtures ?? exeDir.appendingPathComponent("../../extension/tests/fixtures/head").standardizedFileURL
            let v = Validator(ctx: ctx, models: models, verbose: o.verbose, headFixturesDir: fixtures)
            passed = try v.runAll()
            validation = v.results
        }
        if o.profile {
            let m = try LoadedModel.load(id: o.model, modelsDir: modelsDir)
            let rows = try profileSteps(ctx: ctx, model: m, width: o.input.0, height: o.input.1,
                                        repeats: o.frames ?? 20, format: o.format)
            printProfile(m.id, "\(o.input.0)x\(o.input.1)", rows)
            profile = ProfileResult(model: m.id, input: "\(o.input.0)x\(o.input.1)", outputFormat: o.format.rawValue,
                                    repeats: o.frames ?? 20, rows: rows)
        }
        if let dir = o.quality {
            let rows = try runQuality(ctx: ctx, modelsDir: modelsDir, dir: dir, outDir: o.qualityOut, models: allModelIds)
            printQuality(rows)
            quality = rows
        }
        if o.all || !(o.validate || o.profile || o.quality != nil) {
            let ids = o.all ? allModelIds : [o.model]
            let sizes = o.all ? [(960, 540), (1280, 720), (1920, 1080)] : [o.input]
            let frames = o.frames ?? (o.all ? 60 : 200)
            var rows: [BenchResult] = []
            for id in ids {
                let m = try LoadedModel.load(id: id, modelsDir: modelsDir)
                for (w, h) in sizes {
                    let r = try runBenchmark(ctx: ctx, model: m, width: w, height: h, frames: frames,
                                             warmup: o.warmup, format: o.format)
                    if o.verbose {
                        print(String(format: "  %@ %@: median %.3f ms p95 %.3f ms", r.model, r.input, r.gpuMedianMs, r.gpuP95Ms))
                    }
                    rows.append(r)
                }
            }
            printTable(rows)
            bench = rows
        }
        if let path = o.json {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try enc.encode(Report(device: ctx.device.name, os: osVersion, validation: validation,
                                             validationPassed: passed, benchmarks: bench, profile: profile, quality: quality))
            if path == "-" {
                jsonOutput?.write(data)
                jsonOutput?.write(Data([10]))
            } else {
                try data.write(to: URL(fileURLWithPath: path))
                print("wrote \(path)")
            }
        }
        if passed == false { return 1 }
        return 0
    } catch {
        FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
        return 2
    }
}

exit(main())
