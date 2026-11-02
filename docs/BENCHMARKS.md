# WebGPU pipeline benchmarks

Measured results only. Timings below were produced on the machines and browser / runtime
paths identified with each table or study; no timing is extrapolated between machines or
resolutions. Model operation counts and allocation accounting are identified separately.

## M5 Max environment

| Item | Value |
|---|---|
| Machine | Apple M5 Max (40-core GPU), macOS 27.2 (26B123) |
| Browser | Playwright 1.64.1 bundled Chromium 156 (channel `chromium`, new headless) |
| Flags | `--enable-unsafe-webgpu --enable-features=WebGPU --ignore-gpu-blocklist --enable-webgpu-developer-features` |
| Adapter | vendor `apple`, architecture `metal-3`; device features `shader-f16`, `timestamp-query`, `bgra8unorm-storage` |
| Canvas storage format | `bgra8unorm` |

`--enable-webgpu-developer-features` only disables Chromium's 100 µs timestamp
quantisation; it does not change the shaders or the pipeline.

## Method

`cd extension && npm run build:harness && npx playwright test tests/e2e/gpu_bench.spec.ts`

* Production path, per frame: `new VideoFrame(RGBA pixels)` → `device.importExternalTexture`
  → `pipeline.executeSlot(...)` (one compute pass for the whole network) → canvas
  swap-chain texture (`STORAGE_BINDING`), exactly as the content script records it.
* Frames are submitted through the `FrameRing` (3 slots, at most 3 frames in flight); when
  all slots are busy the benchmark waits for the oldest frame instead of dropping it.
* GPU time = end-of-pass minus beginning-of-pass timestamp of the frame's compute pass
  (`timestampWrites` of the ring slot), resolved and read back per slot.
* 40 warm-up frames, then 200 measured frames per case.

## GPU time per frame

Cool-machine measurements on 2026-11-01:

| Case | Path | Median (ms) | p95 (ms) |
|---|---|---:|---:|
| low, 960x540 → 1920x1080 | head → canvas (direct) | **2.150** | 2.180 |
| medium, 960x540 → 1920x1080 | head → canvas (direct) | **4.848** | 4.943 |
| low, 1280x720 → 1920x1080 | head → internal 2560x1440 → Catmull-Rom → canvas | **3.411** | 3.439 |
| medium, 1280x720 → 1920x1080 | head → internal 2560x1440 → Catmull-Rom → canvas | **7.974** | 8.036 |
| Quality 0, 960x540 → 1920x1080 | Catmull-Rom from the external texture | **0.255** | 0.256 |
| Quality 0, 1280x720 → 1920x1080 | Catmull-Rom from the external texture | **0.255** | 0.256 |

Larger outputs under the 4K presentation ceiling, measured on 2026-11-02 with the same
command and machine (the first six cases re-read within 1 % of the table above):

| Case | Path | Median (ms) | p95 (ms) | Memory (MB) |
|---|---|---:|---:|---:|
| low, 1280x720 → 2560x1440 | head → canvas (direct) | **3.337** | 3.413 | 34.100 |
| medium, 1280x720 → 2560x1440 | head → canvas (direct) | **7.861** | 7.929 | 66.934 |
| low, 1920x1080 → 3840x2160 | head → canvas (direct) | **6.692** | 7.014 | 74.528 |
| medium, 1920x1080 → 3840x2160 | head → canvas (direct) | **17.233** | 20.239 | 146.356 |
| Quality 0, 1920x1080 → 3840x2160 | Catmull-Rom from the external texture | **1.012** | 1.025 | 0.124 |

A 720p source presented directly at 2560x1440 costs no more than the 1080p resample path
and needs less memory (no internal image). Low 1080p → 4K fits a 60 fps frame; medium fits
30 fps. The session's deadline rule steps medium down to low when it does not keep up.

The Catmull-Rom kernel reads its tap weights from the exact single-rounding table
([cubic_weights.ts](../extension/src/pipeline/cubic_weights.ts), Implementation Notes, section 3, item 11) instead of evaluating the
polynomials. A separate measurement of the resample cases gives low 720p 3.515,
medium 720p 8.040 and Quality 0 0.255-0.256 ms medians.

Against the original design budget of 6.5 ms
per frame, low passes at both input sizes and medium passes at 960x540. Medium
1280x720 → 1920x1080 measured **7.97 ms** (7.974 ms median); its 6.5 ms target is
**not a release requirement**. This measurement uses the encoder projection tiling
configuration described in the [latency study](#latency-study-medium-1280x720--1920x1080-2026-11-01).

These are back-to-back (saturated) costs. Under paced playback the GPU clock drops and the
same frames read 3-7x longer timestamps ([Implementation Notes, section 6.7](IMPLEMENTATION_NOTES.md#67-measured-in-the-e2e-environment-playwright-chromium-156-headless-m5-max)); the session's downgrade
rule therefore judges whether frames complete inside their slot, not their paced GPU time.
The head shares horizontal baseline rows and uses 2-pixel register blocking for its 1x1
layer. Projection accumulators stay in f32 registers. Large non-projection layers with
K >= 32 and large encoder projection layers rb0B / rb1B use 16 output channels per thread.

### Per-dispatch profile (medium, 1280x720 → 2560x1440 direct)

Measured with Dawn's Node bindings (`webgpu` 0.6.2, Metal, timestamp quantisation
disabled) by ending the compute pass after each dispatch (`encodeFrame(..., stopAfter)`)
and differencing the median of 8 runs; whole-frame median 7.876 ms.

| Dispatch | ms | Dispatch | ms | Dispatch | ms |
|---|---:|---|---:|---|---:|
| fold | 0.017 | rb4A | 0.054 | tb2U | 0.338 |
| rb0A | 0.242 | rb4B | 0.050 | tb2A | 0.858 |
| rb0B | 0.525 | tb0U | 0.046 | tb2B | 0.510 |
| rb1A | 0.239 | tb0A | 0.108 | tb3U | 0.643 |
| rb1B | 0.536 | tb0B | 0.072 | tb3A | 0.850 |
| rb2A | 0.157 | tb1U | 0.122 | tb3B | 0.516 |
| rb2B | 0.176 | tb1A | 0.303 | head | 1.222 |
| rb3A / rb3B | 0.061 / 0.062 | tb1B | 0.164 | | |

The decoder blocks at levels 0 and 1 (tb2, tb3) and the head take 4.9 of the 7.876 ms. The
largest 3x3 layers (tb2A: 4.34 GMAC, tb3A: 4.34 GMAC) run at about 5 TMAC/s, i.e. the
kernels are arithmetic bound with f32 accumulation as required by the FP16 numerics
policy (enforced per layer by `gpu_parity.spec.ts` (h) and (i), see below). (The same method gives unstable per-dispatch numbers for the low model at 540p,
whose dispatches are too short for the GPU clock to settle, so they are not reported.)

## M6 base: branded Chrome benchmark

From `extension/`, after building the harness:

```sh
PW_CHANNEL=chrome npx playwright test tests/e2e/gpu_bench.spec.ts
```

Measured on 2026-11-01 on a Mac mini with Apple M6 (12-core GPU), 16 GB, macOS 27.0.1,
using branded Google Chrome 156.0.8078.79 headless (Playwright channel `chrome`). The same
`tests/e2e/gpu_bench.spec.ts` cases, warm-up, sample count and GPU timestamp method were
used. These are separate branded-Chrome measurements, not the Dawn Node latency study.

| Case | Median (ms) | p95 (ms) |
|---|---:|---:|
| low, 960x540 → 1920x1080 | 5.052 | 5.146 |
| medium, 960x540 → 1920x1080 | 13.133 | 13.205 |
| low, 1280x720 → 1920x1080 | 8.696 | 8.873 |
| medium, 1280x720 → 1920x1080 | 22.296 | 22.487 |
| Quality 0, 960x540 → 1920x1080 | 0.718 | 0.732 |
| Quality 0, 1280x720 → 1920x1080 | 0.720 | 0.749 |

For paced playback and the observed M6 controller downgrades, see
[Validation](VALIDATION.md#m6-base-real-site-results).

## Latency study: medium 1280x720 → 1920x1080 (2026-11-01)

Medium 720p → 1080p uses f32 accumulation and explicit half rounding, enforced by
`gpu_parity` (h)/(i). The configurations below compare latency under that numerics contract.

**Measurement setup.** The M5 Max (MacBook Pro) is unsuitable for A/B work: under sustained
load its GPU clock drops within seconds (an identical pipeline read 8.0 ms, then 11.3 ms right
after; the Playwright suite later read 18 ms), and other applications share the GPU. All
comparisons were therefore run on a Mac mini (Apple M6, 12-core GPU, 16 GB, macOS 27.0.1),
through Dawn's Node bindings with timestamp quantisation disabled: one pipeline resident at a
time, 70 sequential single-frame submissions per configuration, alternating A/B measurements.
Across these measurements, the M6 medians were identical to 0.01 ms and its output
hashes were bit-identical to the M5 Max. Medium 720p reads 22.58 ms there.

**Where the time goes.** A WGSL microbenchmark of independent f32 FMA chains measures the
devices' sustained throughput on those chains: 2.16 TFMA/s on the M6, 6.35 TFMA/s on the M5 Max (while a local VM
was loading the CPU). Medium 720p performs about 36.6 GMAC (34.4 G after the compiler merges
the duplicated products of the nearest-2x up-convolutions). The plain 3x3 layers run at
1.72 TMAC/s on the M6, about 80 % of peak, and the whole frame at about 72-75 % of that measured throughput on both
machines. These operation counts describe the model work, not an extrapolated frame time.

**Kernel configuration comparisons (M6, per-layer A/B, same session):**

| Configuration | Result |
|---|---|
| f32 weight buffer (no f16→f32 conversion in the inner loop) | frame +8.4 % (tb3A +5 %, head +10 %): weight traffic matters, conversions do not |
| Patch held in f32 registers | frame 2.3x slower on the M5 Max (register spills) |
| Two tap rows of the patch at a time (8 instead of 16 patch registers) | frame +7 % |
| 2-4 quads per thread (more weight reuse), compiler-unrolled | tb3A +84-88 % |
| 2 quads per thread, hand-unrolled | tb3A +24 %, tb2A +14-22 % |
| Input chunk size: 4, 32, 64 channels per barrier | 4 channels: within ±1.5 %; one chunk: +9-33 % (occupancy) |
| Workgroup tile / KB sweep on tb3A, tb2A | best is the production tiling (others +0.6 % to +20 %) |
| Row-wise Winograd F(2,3) (1.5x fewer multiplies, f32 transformed weights) | KB 16: 3.1x slower; KB 8: tb3A +14 %, tb2A +22 %; KB 4: +35-37 % |
| KB 16 for projection layers | rb0B −8 %, rb1B −8 %; tb2B / tb3B +13-15 % (2K-channel 1x1 input) |
| WebGPU matrix units | not available: Chrome 156 and Dawn expose `subgroups` but no subgroup-matrix feature without unsafe flags |

The production configuration uses KB 16 for the large encoder projection layers (`convTiling`, 1x1
input no wider than K): medium 720p 22.58 → 22.31 ms (−1.2 %), low 720p 8.73 → 8.71 ms,
medium 540p 13.19 → 13.11 ms, low 540p unchanged, outputs bit-identical. The kernels are
bound by the f32 multiply-add rate plus weight traffic; reducing multiplies (Winograd, folding
the up-convolution weights per output phase) needs f32 transformed weights whose extra
traffic costs more than the arithmetic saved.

Medium 1280x720 → 1920x1080 exceeds 6.5 ms under the exact-numerics contract in these
measurements. The cool-machine M5 Max benchmark measures 7.97 ms; 6.5 ms is not a release
requirement. The session uses the
[deadline-based downgrade rule](IMPLEMENTATION_NOTES.md#63-budget-controller-budgetcontroller),
not the historical target as a runtime threshold. The M5 paced e2e test retained medium;
the [M6 real-site checks](VALIDATION.md#m6-base-real-site-results) stepped down at 720p60.

## GPU memory

Bytes reported by `memoryBytes()` (every buffer / query set the pipeline allocates,
including the uploaded weight bundle) plus the frame ring
(`tests/unit/memory.test.ts`, `tests/e2e/gpu_bench.spec.ts`):

| Case | Pipeline (MB) | Ring (bytes) | Total (MB) |
|---|---:|---:|---:|
| low, 960x540 → 1920x1080 | 19.300 | 912 | **19.301** |
| medium, 960x540 → 1920x1080 | 37.859 | 912 | 37.860 |
| low, 1280x720 → 1920x1080 | 48.804 | 912 | 48.805 |
| medium, 1280x720 → 1920x1080 | 74.101 | 912 | 74.102 |
| Quality 0 (any size) | 0.123144 | 912 | 0.124 |

(MB = 10^6 bytes.) There is no VRAM ceiling; the low
960x540 → 1920x1080 total is pinned by `tests/unit/memory.test.ts`. Medium and the 720p resample
cases are larger: medium doubles
the level-0 width (K0 = 32), and the resample path stores the 3 x 2560 x 1440 FP16
internal image (22.1 MB), which is aliased into the enlarged `mid` scratch buffer. Every
resampling pipeline (VSR resample path, Quality 0) also owns the 122,888-byte exact
Catmull-Rom weight table.

## Numerical parity (same build)

`npx playwright test tests/e2e/gpu_parity.spec.ts`

| Check | Result |
|---|---|
| Fold (half4 and RGBA8 sources, 64x64, 100x60, 37x19, low + medium) | bit-exact |
| Ordered 2x2 pool, levels 0-3, low + medium (76,800 halves) | bit-exact; the naive f32 mean differs in 20,287 of them |
| Up-convolution mapping on impulses | bit-exact |
| Folded SR head vs research oracle (low C16 + medium C32, 6 cases each) | 147,456 / 147,456 halves bit-exact |
| Full network vs TS reference, 64x64, 100x60, 160x96, 256x144 (low + medium) | max \|Δ\| 9.766e-4, mean 2.07e-5 – 2.45e-5, p99.9 9.766e-4, 96.4 – 97.0 % bit-exact |
| Negative controls (100x60) | swap concat: mean 4.3e-2 (low) / 1.36e-1 (medium); no projection: 1.7e-2 / 1.9e-2 |
| Production texture → canvas vs buffer variant (low/medium direct, low resample, Q0) | byte-exact |
| Quality 0 resampler vs reference (5 sizes incl. 64x36 → 1920x1080, a downscale and 1280x720 → 854x480, whose column 194 has the one double-rounding fraction) | byte-exact |
| (h) every RB / TB convolution (rbA, rbB + projection, rb4, tbU, tbA, tbB; real weights, 160x96) evaluated by the reference on the GPU's own layer input | 0 - 0.21 % of a layer's halves differ (limit 1 %), each within one rounding flip; shader mutations that break the policy make 11 - 63 % differ while (e) still passes |
| (i) SR head (hidden 3x3, 1x1, z rounding; real weights) on the GPU's own d3, 160x96 | 21 / 184,320 outputs differ for low and for medium (limit 0.1 %), max \|Δ\| 9.77e-4; mutated (low / medium): no z rounding 521 / 919, half-precision hidden sums 6,242 / 4,279 |

Tolerance used for the full network: max ≤ 2^-8, p99.9 ≤ 2^-9, mean ≤ 2^-12.
