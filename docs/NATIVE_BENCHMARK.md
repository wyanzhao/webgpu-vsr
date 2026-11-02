# Native Metal benchmark: `native-benchmark/`

`vsr-bench` is a standalone macOS command-line tool that runs the four reconstructed
RTX Video Super Resolution float models of NVIDIA RTX Video SDK 1.1 (`low`, `medium`,
`high`, `ultra`) as Metal compute kernels on Apple Silicon. It reports:

- correctness against an independent CPU reference, with bit-exact checks on the stages that
  have an exact half-precision contract;
- GPU time per frame (median and p95) and allocated buffer bytes at 960x540, 1280x720 and
  1920x1080 input (1920x1080 -> 3840x2160 is the 4K target for high and ultra).

The browser extension ships only `low` and `medium`, through WebGPU. `high` and `ultra` run
at native resolution without the 2x2 fold, which is about 7.1x and 15.6x the work of `low`,
based on the 1920x1080-input GMAC table below (187.3 and 409.9 versus 26.3 GMAC).
They exist only in this tool (see IMPLEMENTATION_NOTES.md section 5).

## Build and run

Requirements: Apple Silicon Mac, Xcode command-line tools with the Metal toolchain
(`xcrun metal`), Python 3 (standard library only), the RTX VSR research dataset (its directory
in the `RTX_VSR_RESEARCH_DIR` environment variable; `make` reads it to export the high and
ultra models), and the shipped bundles in `extension/public/weights`.

```sh
cd native-benchmark
make            # export model specs and weights, compile the kernels and the CLI into build/
make validate   # run every correctness check; the exit status is non-zero on any failure
make bench      # full benchmark table (same as ./build/vsr-bench --all)
make clean      # remove build/
```

`make` (the `all` target) does three things:

1. `tools/export_models.py build` reads `variant-graphs/<model>.json`,
   `all-tensors-derived.json` and the raw parameters. It writes `build/models/<model>.json`
   for all four models and packs `build/weights/{high,ultra}.bin` plus their manifests.
   `low` and `medium` keep using the shipped bundles in `extension/public/weights`; the script
   only verifies them. It never writes into the research directory. One exporter run produces
   all of these files: the Makefile states them as one multi-target pattern rule, which the
   macOS `/usr/bin/make` (GNU Make 3.81) runs once for all targets, also under `make -j`.
   Grouped targets `&:` require make 4.3 and are unsuitable for macOS GNU Make 3.81.
2. `xcrun -sdk macosx metal -std=metal3.2 -O3 -fno-fast-math -ffp-contract=off` compiles
   `shaders/vsr_kernels.metal`, and `xcrun metallib` turns the result into
   `build/vsr_kernels.metallib`.
3. `swiftc -O -wmo` builds `Sources/*.swift` into `build/vsr-bench`.

Warnings are errors in both the Metal and the Swift build.

### Command-line interface

```
vsr-bench [options]
  --model low|medium|high|ultra   model for a single benchmark run (default low)
  --input WxH                     source size (default 960x540); output is 2W x 2H
  --frames N                      timed frames (default 200; --all default 60; --profile repeats, default 20)
  --warmup N                      untimed warm-up frames (default 10)
  --format half|rgba8             head output: planar FP16 or packed RGBA8 (default rgba8)
  --all                           benchmark every model at 960x540, 1280x720, 1920x1080
  --validate                      run all correctness checks; exit 1 on any failure
  --profile                       per-dispatch GPU time / throughput for --model at --input
  --json PATH                     also write results as JSON (PATH '-' = stdout)
  --models-dir DIR                exported model specs (default <exe dir>/models)
  --metallib PATH                 compiled kernels (default <exe dir>/vsr_kernels.metallib)
  --head-fixtures DIR             research head fixtures for --validate
                                  (default <exe dir>/../../extension/tests/fixtures/head)
  --verbose
```

Examples: `./build/vsr-bench --model low --input 960x540 --frames 200`,
`./build/vsr-bench --all --json results.json`,
`./build/vsr-bench --profile --model ultra --input 1920x1080`.

With `--json -`, stdout contains only JSON; the device header, human table and other
diagnostics go to stderr. Otherwise human output continues to use stdout. Memory values
in the human table use MiB (1 MiB = 1,048,576 bytes); JSON fields remain byte counts.

With `--profile`, the JSON report includes a `profile` object with `model`, `input`
(`WxH`), `outputFormat` (`half` or `rgba8`), `repeats` (the `--frames` value, default 20)
and `rows`. Each row contains `step`, `gpuMs` (per-dispatch time in milliseconds),
`gmacs` (billions of multiply-accumulates) and `tflops` (TFLOP/s). The object is omitted
without `--profile`. For example, `--profile --json -` writes these measurements to
stdout and the human profile table to stderr.

Exit status: 0 on success, 1 when `--validate` finds a failure, 2 on a usage or runtime
error.

## Network contracts implemented

All four models share one U-Net: five residual blocks (RBs) in the encoder with four ordered
pools, and four decoder blocks. Each decoder block applies a nearest-2x 3x3 upsampling
convolution with ELU, concatenates `[upsampled, skip]`, and runs an RB. The SR head follows.
The definitions below use `H(x)` for rounding to IEEE half with round-to-nearest-even.

- RB: `RB(x) = H(H(ELU(C3_1(H(ELU(C3_0(x)))))) + H(C1(x)))`. `C1` is the 1x1 skip projection
  and has no bias. Its stored bias is verified to be exactly zero.
- Pool: `a = H(TL*0.25); a = H(fma(TR,0.25,a)); a = H(fma(BL,0.25,a)); out = H(fma(BR,0.25,a))`.
- Upsampling convolution: the source tap is `floor((out + tap - 1)/2)`, and out-of-range
  taps are zero.
- Head: `h = H(ELU(C3(d3)))`, then `z = H(C1(h) + b1)`, then
  `T = H(0.5*((e+ - e-)/(e+ + e-) + 1))` with `e+ = exp2(z*log2e)` and `e- = exp2(-z*log2e)`.
  The baseline uses the original source at scale 2: taps
  `floor((o+0.5)/2-0.5)-1 .. +2`, clamped. Each tap is normalised as
  `N = H(H(I+1)*0.5)`. The stored coefficient tables `[-0.03515625, 0.26171875,
  0.87890625, -0.10546875]` (phase 0) and their reverse (phase 1) are read from the
  constructor constant banks at offset 0x628. The baseline is a horizontal-first
  two-accumulator half dot D. Output: `out = H(H(B + T) - 0.5)`.

| | low / medium (folded) | high / ultra (unfolded) |
|---|---|---|
| alignment | 32 | 16 |
| network input | 2x2 fold, `F[Y,X,8dy+4dx+c] = I[2Y+dy,2X+dx,c]`, 16 ch at TW/2 x TH/2 | `[2R-1, 2G-1, 2B-1, 0, 0, 0, 0, 0]`, C=8 at TW x TH |
| widths K0..K4 | low [16,32,32,64,64], medium [32,64,64,64,64] | high [32,32,64,64,64], ultra [32,64,128,128,128] |
| head | `sr`: 3x3 K0->64 + ELU, 1x1 64->48 | `sr_2x`: 3x3 32->16 + ELU, 1x1 16->12 |
| learned shuffle p | 4, `k = 16*color + 4*dy + dx`, feature `floor(o/4)` | 2, `k = 4*color + 2*dy + dx`, feature `floor(o/2)` |
| baseline scale | 2 (original source) | 2 (original source) |

The exporter derives every shape from the variant graph: the chain of each conv's
`logical_KCRS`, the concat order `["upsampled_conv_features", "encoder_skip"]`, the pool
stride, the shuffle factor and phase string, the baseline coordinates, and the scale-2 head
selection. It checks each shape against the registry and fails on any disagreement. The
Swift loader re-checks the KRSC physical shape, the bundle SHA-256 and FP16 finiteness. For
high and ultra, the head is `sr_2x`: RUNTIME-CONTRACT.md shows that the public
`EvaluateFeature` path sets network scale 2.0 and selects head index `int(2.0) - 2 = 0`.
The packed bundles hold the 66 FP16 tensors (33 weight and bias pairs) that this path uses, verified against
`all-tensors-derived.json`. They are 1,297,856 B for high and 4,829,184 B for ultra.

## Implementation

### Kernels (`shaders/vsr_kernels.metal`)

Each frame is one serial compute pass with 24 dispatches. The engine fuses everything that
the WebGPU plan (IMPLEMENTATION_NOTES.md section 4) fuses, and also fuses the whole head:

| dispatch | kernel | work |
|---|---|---|
| `prep` | `vsr_prep` | RGBA8 -> half via a 256-entry LUT (`H(2*(v/255)-1)` in f32), edge replication, fold or C=8 pad -> `x0` |
| `rbL.A` (L = 0..4) | `vsr_conv3x3` | `H(ELU(C3_0(x_L) + b))` -> `mid` |
| `rbL.B` | `vsr_conv3x3` + PROJ + POOL | RB merge -> skip half of `cat_L`; ordered pool -> U half of `cat_{L+1}` (or `x4`). `rb4.B` -> `b4`, no pool |
| `tbJ.up` | `vsr_conv3x3` + UPSAMPLE | nearest-2x 3x3 + ELU -> U half of `cat_L` |
| `tbJ.A` | `vsr_conv3x3` | 3x3 over `cat_L` (2K channels) + ELU -> `mid` |
| `tbJ.B` | `vsr_conv3x3` + PROJ | RB merge -> U half of `cat_L` in place (or `dout`, see below) |
| `head` | `vsr_head` | hidden layer and logits in threadgroup memory, then T, baseline, composition and the RGBA8 (`dlpp_postProcessE`: clamp, truncating `v*255`) / planar-half store |

`vsr_conv3x3` is an implicit GEMM over `K x (9*C)` on `simdgroup_matrix<half,8,8>`, using
`simdgroup_multiply_accumulate` with `simdgroup_float8x8` accumulators. It works as follows:

- A threadgroup owns a 16x8 pixel tile. It stages an 18x10 halo patch into threadgroup
  memory 32 channels at a time, zero-filling the borders. Upsampling convolutions stage the
  replicated low-resolution patch, which gives the floor mapping exactly.
- The A operand is 8 consecutive pixels x 8 channels of the patch. The B operand is one
  contiguous 8x8 tile of host-repacked weights, in `[K/8][9][C/8][8][8]` layout.
- The bias initialises the accumulator as an 8-row replicated f32 tile. ELU and the half
  rounding run element-wise on the fragments, so the code never needs the undocumented
  mapping from lanes to elements.
- The 1x1 projection is a second GEMM in the same kernel. The two branches merge with one
  half add.
- The pool runs in the epilogue on the tile that was just stored.
- The in-place write in `tbJ.B` follows a threadgroup barrier. The barrier makes sure every
  projection read of the tile completes before the first store.

Each specialisation receives all 14 function constants. An earlier build left the
UPSAMPLE constant unset for the head, and the head silently used the upsampling loader. The
full-network check caught it.

### Tiling (tuned on this M5 Max)

| simdgroup tile (M-blocks x N-blocks) | high 4K | ultra 4K |
|---|---|---|
| 4 x 4 (16 accumulators) | 289 ms (1.3 TFLOP/s) | 620 ms |
| 4 x 2 (8 accumulators) | 80.6 ms (4.6 TFLOP/s) | 221 ms |
| 2 x 2 / 4 x 1 (4 accumulators) | 30.5-31 ms (12.1 TFLOP/s) | 67 ms (12.2 TFLOP/s), with the grid.z split below |

Register pressure dominates. The chosen policy gives each simdgroup 4 accumulators: M-blocks
x N-blocks = 2 x 2 (4 x 1 if K were 8). A threadgroup has at most 16 simdgroups (512
threads); 8 and 32 were both slower. Wider K is split across `grid.z` (K = 64 into 2 parts,
K = 128 into 4 parts), and each split pools only its own channels. A split decoder stage
cannot write in place, so it writes `d` to a separate `dout` buffer instead. The head keeps
all hidden channels in one threadgroup, with 4 x 1 tiles: 32 simdgroups for 64 hidden
channels, 8 for 16.

A register-only MMA microbenchmark peaks at 16.4 TFLOP/s on this GPU. The convolution
stages run at 11-12.8 TFLOP/s, about 75% of that peak.

### Memory

All tensors are FP16 NHWC with a row pitch padded to 16 pixels and a row count padded to 8,
so every tile has a full allocation. The padding is never read as data. The buffers are:

- `x0`;
- `cat_0..3`, holding `[U | skip]`;
- `x4` and `b4`;
- `mid`, the shared scratch for every `ELU(C3_0)`;
- `dout`, only when a decoder stage is split;
- the output;
- the packed weights.

The extension uses a single activation pool, because the queue serialises frames. The
benchmark keeps up to 3 command buffers in flight, and Metal hazard tracking serialises
them on the same pool. "Peak buffers" in the table below is the sum of every MTLBuffer the
run allocates: activations, weights and 3 synthetic source frames.

## Numerics policy and validation (`--validate`)

The numerics follow IMPLEMENTATION_NOTES.md section 3:

- weights and stored activations are FP16;
- convolutions accumulate in f32, with the bias added in f32;
- ELU is computed in f32;
- each convolution output is rounded to half once;
- the RB branches merge with one half add (HIGH-CLOSURE.md: packed `HADD2`).

The tests/reference TypeScript model uses the same RB rounding. The kernels are compiled
with `-ffp-contract=off`, so the only fused operations are the explicit `fma()` calls of the
pool and the dot D.

The CPU reference (`Sources/CPUReference.swift`) is a separate implementation. It uses plain
loops over dense, unpadded tensors and the original KRSC weights. Its upsampling tensor and
its concat are explicit, its pool and D use native `Float16` multiply and fused
`addingProduct`, and it evaluates `tanh` in Double. The checks are:

| check | what | result on M5 Max |
|---|---|---|
| fold / preprocess | `x0`, bit-exact, 64x64 and 100x60 (unaligned: edge replication) | 0 mismatches for all 4 models. The control (zero padding plus transposed fold, or a sign-split C=8 input) gives 6,070-32,176 mismatches |
| pool | the GPU's pooled `x_{l+1}` against the CPU ordered FMA of the GPU's own `s_l`, levels 0-3, 96x80 | 0 / 15,552 (low), 0 / 30,528 (medium), 0 / 86,400 (high), 0 / 111,360 (ultra). The f32-mean control gives 4,092-29,812 mismatches |
| head oracle | zero learned head weights, so `z = bias` exactly (pattern of `semantics/emulator/check_folded_heads.py`). Black, ramp and impulse sources at the research fixture sizes (32x32 folded, 16x16 for the 2x head) plus a 100x60 ramp; zero bias and unique bias `H((k-24)/128)` / `H((k-6)/32)` | bit-exact: 217,728 halves (low, medium) and 162,432 halves (high, ultra), 0 mismatches. The swapped-phase-coefficient and transposed-k-phase controls give 37,536-168,576 mismatches |
| RGBA8 store | the direct 2x path's `dlpp_postProcessE` byte store (PREPROCESS-SEMANTICS.md, RESAMPLE-SEMANTICS.md): clamp, then `trunc(v*255)` of the half output, bit-exact, 100x60. The extension's direct canvas store is the same | 0 / 96,000 bytes for every model. The round-half-up control `floor(v*255+0.5)` (the resample stage's rule) gives 35,899-36,303 mismatches |
| research head fixtures | low, medium: the folded head with zero learned weights against the research-emulator values in `extension/tests/fixtures/head` (`tools/gen_head_fixtures.py`, 6 cases, 32x32). The fixtures' exact half sources are fed through the engine's 256-entry source LUT (one byte code per distinct value), which prep and the head baseline both read | 73,728 / 73,728 halves bit-exact per model; the fixture coefficient banks equal the model's. The other-bias control (each output against the same source's fixture with the other bias mode) gives 72,192 mismatches |
| layers | every RB / TB convolution (rbA, rbB + projection merge, tbU, tbA, tbB) evaluated by the CPU on the GPU's own input of that layer, 96x80, plus the head on the GPU's own `d3`. At most 1 % of a layer's halves may differ, each within one rounding flip (one half ulp of the output, or of e and p before `H(e + H(p))`, plus 2^-16 for cancelling f32 sums); the head at most 0.1 %, max 2^-10 | worst layer 0.174 % (low), 0.347 % (medium), 0.221 % (high), 0.225 % (ultra), all within one flip; head on GPU `d3` 12 / 8 / 6 / 4 of 92,160 differ, max 0.000977 |
| full network | GPU against CPU on synthetic frames at 64x64, 96x80, 100x60 (unaligned), 320x180 and 960x540 | max \|err\| = 0.000977 (2^-10; one FP16 ULP in [0.5, 1) is 2^-11) and mean 2.1e-5 to 3.1e-5 for every model and size. Tolerance: max 1/256, mean 1/8192 |
| network controls | reversed concat `[skip, U]`, and the upsampling with a half-pixel shift `floor((o+1)/2)` | max 0.33-0.69 / mean 0.050-0.27, and max 0.092-0.26 / mean 0.0076-0.024. Both are far outside the tolerance, and both must fail it for the check to pass |

The f32 summation order differs between the GPU and the CPU. Sometimes that flips a half
rounding inside a layer. Even so, the decoder output `d3` stays within 0.002-0.008 absolute,
and the final image has a measured max error of 0.000977 (2^-10). The end-to-end absolute
tolerances (max 1/256, mean 1/8192) cannot see the rounding
sequence itself, which is why the layer check exists: it evaluates each layer from the GPU's
own input, so flips cannot accumulate, and a broken rounding policy shows up as a large
fraction of differing halves.

Mutation tests (run by hand with `--metallib`) confirm that the suite catches real kernel
bugs:

- reordering the pool FMAs to TL, BL, TR, BR fails all 4 pool checks, with exit status 1;
- dropping the -1 from the baseline tap origin fails 24 checks, with exit status 1;
- merging the unrounded f32 projection into the HADD (`half(float(e) + acc)` instead of
  `e + H(acc)`) fails the 4 layer checks (rb0.B 18-20 % of the halves differ), 42/46, exit
  status 1, while every whole-network check still passes (max 0.000977);
- rounding `exp(v)` to half inside the ELU fails the 4 layer checks (rb0.A 31-37 %), 42/46.

The head checks are tied to the research values for the folded models (the fixture check
above). The `head oracle` check additionally covers the unfolded 2x head of high and ultra and
a 100x60 size, with expected values from this tool's own CPU composition.

`make validate` takes about 28 s. Most of that is the CPU reference at 960x540: 1.7 s for
low, 2.2 s for medium, 3.6 s for high and 5.8 s for ultra, using all CPU cores.

## Benchmark method

- Frames: 3 deterministic synthetic RGBA8 frames, rotated. Each mixes gradients, a zone
  plate, hard-edged blocks and hashed noise.
- Warm-up: 10 frames.
- Timed frames: N (60 for `--all`, 200 for a single run). Each frame is one command buffer
  with one serial compute pass.
- Timing: per-frame GPU time is `MTLCommandBuffer.gpuEndTime - gpuStartTime`. The median
  and p95 are nearest-rank.
- Pipelining: up to 3 frames are in flight. "fps (wall)" is N divided by the wall-clock
  time of the timed loop.
- Output: the presentation format (RGBA8) at exactly 2W x 2H. All benchmark sizes are below
  the SDK's area cap (8,294,400 pixels for IDs 3-6, RUNTIME-CONTRACT.md), so the SDK would
  also run them as a single tile.

## Measured results (Apple M5 Max, 40-core GPU, macOS 27.2)

These numbers come from the verification run
`make clean && make && make validate && ./build/vsr-bench --all`, with 60 timed
frames per configuration and the run following straight after the validation (46/46 checks
passed, `make validate` 27.1 s):

| model | input -> output | median ms | p95 ms | fps (wall) | peak buffers | activations | weights |
|---|---|---:|---:|---:|---:|---:|---:|
| low | 960x540 -> 1920x1080 | 1.721 | 1.747 | 573.9 | 36.9 MiB | 30.1 MiB | 0.9 MiB |
| low | 1280x720 -> 2560x1440 | 2.770 | 2.779 | 360.3 | 65.0 MiB | 53.6 MiB | 0.9 MiB |
| low | 1920x1080 -> 3840x2160 | 5.462 | 5.500 | 182.8 | 143.9 MiB | 119.3 MiB | 0.9 MiB |
| medium | 960x540 -> 1920x1080 | 4.246 | 4.272 | 235.1 | 58.4 MiB | 50.8 MiB | 1.7 MiB |
| medium | 1280x720 -> 2560x1440 | 6.975 | 6.995 | 143.2 | 103.1 MiB | 90.9 MiB | 1.7 MiB |
| medium | 1920x1080 -> 3840x2160 | 14.397 | 14.452 | 69.4 | 227.3 MiB | 201.9 MiB | 1.7 MiB |
| high | 960x540 -> 1920x1080 | 8.427 | 8.496 | 118.5 | 149.5 MiB | 142.3 MiB | 1.3 MiB |
| high | 1280x720 -> 2560x1440 | 14.720 | 14.902 | 68.1 | 263.1 MiB | 251.2 MiB | 1.3 MiB |
| high | 1920x1080 -> 3840x2160 | 34.031 | 34.666 | 29.4 | 592.8 MiB | 567.8 MiB | 1.3 MiB |
| ultra | 960x540 -> 1920x1080 | 19.539 | 19.762 | 51.2 | 191.6 MiB | 181.0 MiB | 4.7 MiB |
| ultra | 1280x720 -> 2560x1440 | 34.704 | 35.320 | 28.8 | 334.6 MiB | 319.4 MiB | 4.7 MiB |
| ultra | 1920x1080 -> 3840x2160 | 83.105 | 85.415 | 12.1 | 749.9 MiB | 721.5 MiB | 4.7 MiB |

The RGBA8 store truncates according to the postProcessE rule above. A separate run gives
low 1.717 / 2.760 / 5.471 ms, medium 4.241 / 6.981 / 14.398 ms, high 8.430 / 15.146 /
37.092 ms and ultra 22.792 / 38.977 / 81.335 ms medians, within run-to-run variation.

These timings vary by up to about 20% between runs. They are slowest after minutes of
sustained load, such as the run straight after `make validate` above, and other GPU work on
the machine also affects them. A separate run on a less loaded GPU, with 30 frames, gave the
following medians:

| input | low | medium | high | ultra |
|---|---:|---:|---:|---:|
| 960x540 | 1.667 ms | 4.130 ms | 8.245 ms | 17.877 ms |
| 1280x720 | 2.662 ms | 6.848 ms | 14.169 ms | 32.233 ms |
| 1920x1080 | 5.330 ms | 14.161 ms | 30.998 ms | 73.539 ms |

Work per frame at 1920x1080 input, in multiply-accumulates (`--profile`):

| model | GMAC per frame |
|---|---:|
| low | 26.3 |
| medium | 81.3 |
| high | 187.3 |
| ultra | 409.9 |

Takeaways:

- Real time at 1080p output (16.7 ms per frame): low, medium and high meet it at 960x540
  input; ultra does not (17.9-22.8 ms in the runs above).
- Real time at 4K output: low and medium meet it. High needs about 31-37 ms and ultra
  74-86 ms per frame.
- Memory: the unfolded models at native 4K need 0.6-0.75 GB of activations, because level
  0 is 1920x1088 at 32-64 FP16 channels.

Per-dispatch profile, high at 1920x1080 -> 3840x2160 (`--profile`, each dispatch repeated
10 times). The decoder's level-0 block (`tb3`) and `rb0.B` dominate:

```
tb3.A  6.12 ms 38.5 GMAC 12.6 TFLOP/s | tb3.B 3.80 ms | tb3.up 3.00 ms | rb0.B 3.16 ms
head   1.96 ms 10.0 GMAC 10.3 TFLOP/s | total 30.8 ms, 187.3 GMAC, 12.2 TFLOP/s
```

## Limitations

- Correctness means agreement with the reconstructed contract and the independent CPU
  reference: bit-exact on the exact-half stages, and end-to-end absolute tolerances of
  max 1/256 and mean 1/8192. The measured max error is 0.000977 (2^-10). NVIDIA hardware
  output was never available. The research artifacts give no bit-exactness
  guarantee for the HMMA F16 accumulation or the MUFU exp of the NVIDIA kernels, so the
  f32-accumulation policy is a documented choice.
- The tool covers the float models only. The quantized variants (qlow..qultra) and the 3x
  and 4x heads of high and ultra are not on the public 2x path and are not implemented.
- The source is a linear RGBA8 buffer, not a video texture. Decode, colour conversion and
  presentation are out of scope; the head writes the final RGBA8 image directly.
- The M5 GPU's neural accelerators, reachable through Metal 4 tensor operations, are not
  used. The kernels use `simdgroup_matrix`.
