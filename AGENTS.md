# AGENTS.md

Guidance for coding agents and contributors working in this repository.

## Project

WebGPU VSR is a Chrome Manifest V3 extension for Apple Silicon Macs. It upscales web video in
real time by running a reconstruction of the RTX Video Super Resolution low and medium networks
as WebGPU compute shaders, and draws the result on a canvas placed under the page's own
`<video>`. `native-benchmark/` is a separate Swift/Metal tool that validates and benchmarks the
same networks (plus high and ultra) natively. The project is for personal use; see `README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `extension/src/background/` | MV3 service worker: settings, per-tab reports, badge |
| `extension/src/content/` | Content script: discovery, eligibility gate, session loop, overlay, HUD, insertion invariance |
| `extension/src/pipeline/` | WebGPU device manager, VSR and resample pipelines, memory plan, frame ring, budget controller |
| `extension/src/shaders/` | WGSL kernels (fold, conv, SR head, resample), specialised by `shader_source.ts` |
| `extension/src/weights/` | Weight bundle loader and types |
| `extension/src/popup/`, `extension/src/shared/` | Popup UI; settings, messages, diagnostics shared by all contexts |
| `extension/public/` | Icons, site rules, committed weight bundles (`weights/*.bin` + manifests) |
| `extension/tests/unit/` | Vitest unit tests (fake DOM, fake GPU devices, Dawn-backed GPU tests) |
| `extension/tests/e2e/` | Playwright tests in Chromium with WebGPU (overlay, downgrade, parity, benchmark) |
| `extension/tests/reference/`, `extension/tests/harness/` | CPU reference network and the GPU test harness |
| `native-benchmark/` | Swift CLI `vsr-bench`, Metal kernels, Makefile |
| `tools/` | Weight conversion/verification and fixture generation (need the research dataset) |
| `docs/` | `IMPLEMENTATION_NOTES.md` (binding contracts), `BENCHMARKS.md`, `NATIVE_BENCHMARK.md`, `VALIDATION.md` |

## Commands

Run from `extension/` unless noted.

```sh
npm ci                      # install dependencies
npm run build               # build the extension into extension/dist
npm run typecheck           # tsc --noEmit
npm run lint                # eslint src tests
npm run test:unit           # vitest; Dawn-backed GPU tests need a real GPU
npm run test:e2e            # builds the harness, then Playwright (Chromium with WebGPU)
npx playwright install chromium   # once, before the first e2e run
PW_CHANNEL=chrome npx playwright test tests/e2e/gpu_bench.spec.ts   # benchmark the installed Chrome
```

Native benchmark (from `native-benchmark/`; needs Xcode's Metal toolchain):

```sh
export RTX_VSR_RESEARCH_DIR=/path/to/rtx-vsr-research   # used to export the high/ultra models
make && make validate && make bench
```

Weight bundles (from the repository root; the research dataset is not part of the repository):

```sh
RTX_VSR_RESEARCH_DIR=/path/to/rtx-vsr-research python3 tools/convert_weights.py
RTX_VSR_RESEARCH_DIR=/path/to/rtx-vsr-research python3 tools/verify_weights.py
```

## Before you finish a change

- `npm run build`, `npm run typecheck`, `npm run lint` and `npm run test:unit` must pass.
- Changes to the content script, overlay, session or pipelines also need `npm run test:e2e`.
- Changes to shaders, numerics or weights also need the GPU parity suite
  (`tests/e2e/gpu_parity.spec.ts`) and, for shared kernels, `make validate` in
  `native-benchmark/`.
- GPU unit tests and all Playwright tests need a real GPU and a desktop session. They cannot
  run inside a headless sandbox; do not try to launch the browser there.
- Every bug fix comes with a test that fails without the fix.

## Contracts that must not regress

The full contracts are in `docs/IMPLEMENTATION_NOTES.md`; read the relevant section before
editing. The most important ones:

- **Numerics:** shader math runs in f32 with explicit binary16 rounding (`hr4`) at the points the
  contract names; convolutions accumulate in f32. Do not switch to native f16 arithmetic: Metal
  does not honour f16 rounding. Per-layer parity tests enforce this.
- **Overlay:** the native picture is hidden only after a validated frame is presented, only by
  shifting `object-position` (never opacity, visibility or filters on the video), and is restored
  synchronously on every exit path with the page's own inline styles restored exactly. The canvas
  sits immediately before the video with the video's z-index; the parent is never repositioned.
- **Fail closed:** layouts, styles or DOM reactions the canvas cannot reproduce exactly are
  rejected as `unsupported-layout`, and insertion of any extension node is rolled back if it
  changes the computed style or geometry of the video, its siblings or its ancestors.
- **Lifecycle:** every `VideoFrame` is closed; GPU resources are released on destroy; device
  loss follows the bounded recovery state machine in `pipeline/device.ts`.
- **Performance:** no per-frame layout reads or style writes; geometry is re-synchronised at
  most every 500 ms or on events.

## Conventions

- TypeScript strict mode, ES modules, no new runtime dependencies.
- Match the surrounding code's style and comment density; comments explain why, not history.
- Keep documentation factual: measured numbers come from a command that can be rerun, and no
  local machine paths or personal data go into tracked files.
- The weight bundles are NVIDIA-derived parameters for personal use; do not redistribute them.
