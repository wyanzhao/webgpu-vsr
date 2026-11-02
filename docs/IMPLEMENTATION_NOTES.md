# Implementation Notes & Binding Contracts (webgpu-vsr)

This file defines the binding implementation contracts of the extension and the native benchmark.
The numerical contracts follow the primary RTX VSR research dataset (the directory named by
`RTX_VSR_RESEARCH_DIR`; variant-graphs/*.json, semantics/FLOAT-RECONSTRUCTION.md,
semantics/HIGH-CLOSURE.md, PIXEL-FOLD.md, PREPROCESS-SEMANTICS.md, RUNTIME-CONTRACT.md).

## 1. Environment facts (measured on the build machine)

- Apple M5 Max, 40-core GPU, macOS 27.2, Xcode with Metal toolchain (`xcrun metal` works).
- Node 26 (`Math.f16round`, `Float16Array` are available), Python 3.15 without numpy.
- The minimum Chrome version is 135, which provides `Math.f16round` and `Float16Array`.
  Quality 0 runs without `shader-f16`; only the neural models require that feature.
- Playwright 1.64 bundled Chromium 156. Use `chromium.launchPersistentContext(userDataDir,
  { channel: 'chromium', headless: true, args: [...] })`; new-headless mode supports
  extensions and WebGPU on Metal. Branded Google Chrome ignores `--load-extension`.
- WebGPU exists only in secure contexts: `about:blank` has no `navigator.gpu`. Use
  `page.route('http://localhost:<port>/**', ...)` to serve test pages, or extension pages.
- Adapter: vendor `apple`, arch `metal-3`. Features include `shader-f16`, `timestamp-query`,
  `bgra8unorm-storage`, `subgroups`. Limits: `maxStorageBuffersPerShaderStage = 10`,
  `maxComputeWorkgroupStorageSize = 32768`, `maxComputeInvocationsPerWorkgroup = 1024`.
- Playwright Chromium has no proprietary codecs: test videos are VP9/WebM generated with
  `ffmpeg -c:v libvpx-vp9` (ffmpeg is at /opt/homebrew/bin/ffmpeg).

## 2. Repository layout and ownership

```
tools/convert_weights.py, tools/verify_weights.py           weight conversion and verification
extension/public/weights/{low,medium}.bin + *_manifest.json  weight bundles and manifests
extension/manifest.json, src/background, src/popup, src/shared, public/icons, public/sites
extension/src/content/{index,discovery,eligibility,overlay,session,hud,site_rules}.ts
extension/src/pipeline/{device,types,ring,timing,geometry,memory,resample_pipeline,vsr_pipeline,shader_source,cubic_weights}.ts
extension/src/shaders/*.wgsl, extension/src/weights/{loader,types}.ts
extension/tests/{unit,e2e,reference,harness,fixtures}
native-benchmark/                                            Swift + MSL
```

## 3. Network contract (low / medium, folded p4/r2 family)

Notation: `H(x)` = round to IEEE half (round-to-nearest-even). Feature maps are FP16 NHWC,
index `(y*W + x)*C + c`. FP16 weights are physical **KRSC**: index `((k*R + r)*S + s)*C + c`
(registry `logical_shape` is `[K, C, R, S]`; manifest `physical_shape` is `[K, R, S, C]`).
All convolutions are cross-correlation, stride 1. 3x3 convs use zero padding of 1.

Channel widths: low `K = [16, 32, 32, 64, 64]`, medium `K = [32, 64, 64, 64, 64]`.
Parameter names: `Flatten_DLPP_<model>_2x_<layer>_<kind>_<op>` with layer in
`rb0..rb4, tb0..tb3, sr`, kind in `weight|bias|amax_output`, op in `3x3_0, 3x3_1, 1x1, 3x3_u, 3x3`.
`amax_output` (opaque8) entries are not consumed by the float models.

1. **Geometry / padding.** Source `W x H`. Aligned `TW = 32*ceil(W/32)`, `TH = 32*ceil(H/32)`.
   The area beyond the source is filled by **edge replication** (clamp the source coordinate).
2. **Preprocess** (`dlpp_preProcessE`): `I[y,x] = half4(H(2R-1), H(2G-1), H(2B-1), 0)` with
   R,G,B in [0,1] (8-bit sources: v/255). Alpha discarded.
3. **Fold** (exact permutation, no arithmetic): `F[Y,X, 8*dy + 4*dx + c] = I[2Y+dy, 2X+dx, c]`,
   F is `TH/2 x TW/2 x 16`. Level-0 resolution `P0 = (TH/2) x (TW/2)`.
4. **RB** (encoder rb0..rb4, decoder inner RBs):
   `RB(x) = ELU(C3_1(ELU(C3_0(x)))) + C1(x)`; `C3_*` include their biases; `C1` (1x1 projection)
   is **biasless** (its stored bias is exactly zero — never add it). No activation after the sum.
   ELU(v) = v if v >= 0 else exp(v) - 1.
5. **Encoder**: `x0 = F`; for i in 0..3: `s_i = RB_i(x_i)` (kept as skip), `x_{i+1} = Pool(s_i)`;
   `b = RB_4(x_4)`. Resolutions: level i is `P0 / 2^i`.
6. **Pool** (ordered half FMA, row-major TL, TR, BL, BR):
   `a = H(TL*0.25); a = H(fma(TR,0.25,a)); a = H(fma(BL,0.25,a)); out = H(fma(BR,0.25,a))`.
7. **Decoder** tb_j (j = 0..3) at level `L = 3 - j`, previous `d_{-1} = b`:
   `U = ELU(Cu(nearest2x(d_{j-1})))` where Cu is the `3x3_u` conv evaluated at the upsampled
   resolution with source coordinate `src = floor((out + tap - 1) / 2)` per axis (tap 0..2),
   out-of-range source → 0 (zero padding, no clamping). Equivalent: even `2m → [m-1, m, m]`,
   odd `2m+1 → [m, m, m+1]`. Then `cat = [U (K_L channels), s_L (K_L channels)]` (U first) and
   `d_j = RB_tbj(cat)` with input 2*K_L channels, output K_L channels.
   tb widths: tb0 K3, tb1 K2, tb2 K1, tb3 K0.
8. **SR head** on `d_3` (level 0, K0 channels):
   `h = ELU(C3_sr(d_3))` with 64 hidden channels (sr_weight_3x3 `[64, K0, 3, 3]`),
   `z = C1_sr(h)` with 48 outputs **and its non-zero bias** (sr_weight_1x1 `[48, 64, 1, 1]`).
   Internal output is `2W x 2H` (crop of `2TW x 2TH`). For output pixel (ox, oy):
   `fy = floor(oy/4), fx = floor(ox/4)` (level-0 feature coords), channel
   `k = 16*color + 4*(oy % 4) + (ox % 4)`, `T = H((1 + tanh(z[fy,fx,k])) / 2)`.
   Compute tanh as the SASS does: `ep = exp2(z*log2e), em = exp2(-z*log2e)`,
   `t = (ep - em) / (ep + em)` in f32, `T = H(0.5 * (t + 1))`.
9. **Baseline B** from the ORIGINAL (unfolded, padded) source at scale 2, NOT 4:
   `fxs = (ox + 0.5)/2 - 0.5`, `ix = floor(fxs)`, taps `ix-1..ix+2` clamped to `[0, TW-1]`
   (same for y with TH). Source normalisation `N = H(H(I + 1) * 0.5)` per RGB channel.
   Stored coefficients (phase = `ox % 2` horizontally, `oy % 2` vertically):
   phase 0: `[-0.03515625, 0.26171875, 0.87890625, -0.10546875]`,
   phase 1: `[-0.10546875, 0.87890625, 0.26171875, -0.03515625]` (all exactly representable).
   Half dot `D(v,w)`: `a = H(v0*w0); a = H(fma(v2,w2,a)); b = H(v1*w1); b = H(fma(v3,w3,b)); D = H(a+b)`.
   Four horizontal D (rows iy-1..iy+2), then vertical D over those → B.
10. **Composition**: `out = H(H(B + T) - 0.5)`; final presentation clamps to [0,1].
    `sharpness` setting (default 0.5) scales the learned residual: gain `g = 2*sharpness`;
    when `g == 1` use the exact expression above, otherwise `out = H(B + g*(T - 0.5))`
    (f32 arithmetic followed by one half rounding).
11. **Resample** (`dlpp_ResampleAndComposeFP16E`), used when the output size differs from
    the internal `2W x 2H` (scale `float(2W)/float(OW)`), and always for Quality 0 (source =
    preprocess layout 1: planar `H(R), H(G), H(B)` in [0,1] at W x H, scale `W/OW`):
    `c = (u + 0.5)*scale - 0.5`, taps `floor(c)-1..+2` clamped to `[0, valid-1]` (edge
    replication); Catmull-Rom cubic, **a = -0.5** — the PTX constants are 1.5, -2.5, 2.0, 0.5,
    -0.5, 1.0 (`P-1 = -0.5t^3 + t^2 - 0.5t`, `P0 = 1.5t^3 - 2.5t^2 + 1`,
    `P1 = -1.5t^3 + 2t^2 + 0.5t`, `P2 = 0.5t^3 - 0.5t^2`). a = -0.75 would be
    wrong for this stage (-0.75 belongs only to the SR-head baseline's stored table). Fractions are
    converted to half, polynomials evaluated with half FMA, each separable `wx*wy` product
    is a half multiply, RGB accumulation over the 16 taps is f32 FMA
    (see `semantics/resample.ptx` lines 50-700 for the exact order). 8-bit store: clamp,
    `floor(v*255 + 0.5)`. Every `fma.rn.f16x2` of the weight chains rounds ONCE; WGSL has no
    half FMA, and an f32 `fma` followed by a rounding to half rounds twice, which gives a
    different weight for exactly one of the 15,361 half fractions in [0, 1]
    (t = 0.022247314453125: w1 = 0.9990234375 instead of 0.99853515625; reached e.g. by
    1280 -> 854 wide, column 194: 78 differing bytes in the unit test's 854x480 frame). The kernel
    therefore reads the four weights of its half t from a 15,361-entry table indexed by the half
    bit pattern of t (`src/pipeline/cubic_weights.ts`, single rounding evaluated exactly in
    float64; 122,888 bytes, one storage buffer per resampling pipeline). The direct 2x
    presentation (`dlpp_postProcessE`) truncates instead: clamp, `trunc(v*255)`.
12. Output ceiling: let target = video display rect in device pixels clamped to
    3840x2160 (long side ≤ 3840, short side ≤ 2160, aspect preserved). If target ≥ internal
    `2W x 2H` in both axes, present the internal image directly (compositor scales the rest);
    otherwise resample internal → target.

Numerics policy: weights and stored activations FP16; convolution accumulation in **f32**
then rounded to half once per layer output (bias add in f32 before rounding); ELU in f32.
Fold, pool, baseline D, T and composition use the exact half sequences above and are
tested bit-exactly. Whole-network parity is tested against the TypeScript reference
(tests/reference) with an explicit tolerance and a negative control. The whole-network
tolerance cannot see the rounding sequence itself (an unrounded projection, an unrounded e,
half-precision partial sums or a half-rounded exp() keep the network max at one half ulp), so
`gpu_parity.spec.ts` (h) also checks every RB / TB convolution with the real weights, each
evaluated by the reference on the GPU's own input of that layer: at most 1 % of a layer's
halves may differ (measured at most 0.21 %; those violations make 11-63 % differ), each by at
most one rounding flip. (i) does the same for the SR head on the GPU's own d3 (hidden 3x3, 1x1,
z rounding): at most 0.1 % of the outputs (measured 0.011 %; no z rounding 0.28-0.50 %,
half-precision hidden sums 2.3-3.4 %).

## 4. GPU memory plan (aliasing) and frame ring

Level `ℓ` spatial size `Pℓ = P0 / 4^ℓ`. Buffers (FP16 unless stated):

| Buffer | Size (elements) | Role |
|---|---|---|
| `cat_ℓ` (ℓ=0..3) | `Pℓ * 2*Kℓ` | interleaved `[U | skip]`. U-half holds encoder input `x_ℓ` (x0 = F), later the upconv U, later the decoder output `d`. Skip-half holds `s_ℓ`. |
| `x4b4` | `P4 * (K3 + K4)` | disjoint bottleneck input `x4` / output `b4` slices |
| `mid` | `max(max_ℓ Pℓ*Kℓ, 3 * 2W * 2H)` on the resample path; otherwise `max_ℓ Pℓ*Kℓ` | scratch for every `ELU(C3_0(...))`; after the last scratch read, aliases the unpadded planar RGB `internal` head output when resampling |
| `cubic` | 15,361 x 4 halves (122,888 B) | exact resample weights per half fraction (resample path and Quality 0 only) |
| weights | bundle size | one storage buffer, tensor offsets passed as element offsets |

Dispatch order per frame (all in ONE compute pass; WebGPU orders dispatches):
fold(tex→cat0.U) · for ℓ=0..3 { rbA(cat_ℓ.U → mid), rbB(mid + cat_ℓ.U → cat_ℓ.skip, pooled →
cat_{ℓ+1}.U or x4) } · rb4A(x4→mid) · rb4B(mid+x4→b4) · for L=3..0 { up(prev→cat_L.U),
tbA(cat_L→mid), tbB(mid + cat_L → cat_L.U in place) } · head(cat_0.U + source → canvas | internal)
· [resample(internal → canvas)].
In-place `tbB`/`rbB` writes are race-free because the buffer bound read_write is only read at
the thread's own pixel (1x1 projection); all 3x3 neighbourhood reads go to a different buffer.
Never bind one GPUBuffer twice in a dispatch with a writable binding (usage-scope conflict).

VRAM: there is no footprint ceiling. Footprints are measured and reported in [docs/BENCHMARKS.md](BENCHMARKS.md#gpu-memory); low 960x540 → 1920x1080
totals 19.301 MB including weights and the frame ring, pinned by tests/unit/memory.test.ts.

Latency: the 6.5 ms per-frame target is not a release requirement for medium
1280x720 → 1920x1080. The cool-machine M5 Max measurement is 7.97 ms (7.974 ms median); the
[latency study](BENCHMARKS.md#latency-study-medium-1280x720--1920x1080-2026-11-01)
records why 6.5 ms is out of reach under the exact-numerics contract. Low meets the target
at both measured sizes; medium meets it at 960x540.

Frame ring (triple buffering): 3 slots, each owning its uniform/param buffer, a 2-entry
timestamp `GPUQuerySet` (when available), resolve + readback buffers and an in-flight flag.
A frame acquires a free slot or is dropped (counted) — the GPU never has more than 3 frames
queued. Activations are not triplicated: the WebGPU queue serialises submissions, so a single
activation pool is correct.

## 5. Interface contracts

```ts
// src/pipeline/types.ts
export type ModelId = 'low' | 'medium';
export type QualityTier = 'medium' | 'low' | 'quality0_bicubic';
export interface GpuContext {
  adapter: GPUAdapter; device: GPUDevice;
  hasF16: boolean; hasTimestamps: boolean;
  storageFormat: GPUTextureFormat;   // 'bgra8unorm' if bgra8unorm-storage else 'rgba8unorm'
}
export interface SRComputePipeline {
  readonly id: string;            // 'rtx-vsr-low' | 'rtx-vsr-medium' | 'bicubic-q0'
  readonly scaleFactor: number;   // internal scale (2 for VSR, outW/inW for bicubic)
  initialize(device: GPUDevice, inWidth: number, inHeight: number,
             outWidth: number, outHeight: number): Promise<void>;
  execute(commandEncoder: GPUCommandEncoder, externalTexture: GPUExternalTexture,
          targetCanvasView: GPUTextureView, timestampWrites?: GPUComputePassTimestampWrites): void;
  setSharpness(sharpness: number): void;
  memoryBytes(): number;          // bytes of every GPU allocation it owns
  destroy(): void;
}
```

Settings (`src/shared/settings.ts`): `{ enabled: boolean; qualityLevel: 0 | 1 | 2;
sharpness: number; hudEnabled: boolean; siteOverrides: Record<string, boolean> }`,
defaults `{ enabled: true, qualityLevel: 1, sharpness: 0.5, hudEnabled: false, siteOverrides: {} }`.
Quality 0 = Catmull-Rom bicubic, 1 = low, 2 = medium. High/ultra (quality 3/4) exist only in
the native benchmark: the folded browser path cannot run them in budget. A content-type
mode (anime / live action) is not exposed: SDK 1.1 has no content-type variants.

Telemetry (`SessionTelemetry`): `inputWidth, inputHeight, outputWidth, outputHeight, fps,
gpuTimeMeanMs, gpuTimeP95Ms, droppedFrames, activeModel` (+ `timingSource: 'timestamp'|'cpu'`).

## 6. Session & downgrade

Files: `src/content/session.ts` (VideoSRSession, GpuSessionFactory, pure policy helpers),
`src/pipeline/timing.ts` (BudgetController, FrameRateEstimator), `src/shared/diagnostics.ts`,
tests `tests/unit/{timing,session,diagnostics}.test.ts`, `tests/e2e/{video_overlay,downgrade}.spec.ts`.

### 6.1 Wiring and ownership

- Bootstrap installs storage-change and message listeners before its first asynchronous read.
  `chrome.storage.local` is the authoritative settings source; content scripts ignore unversioned
  `SETTINGS_CHANGED` broadcast payloads, so out-of-order broadcasts cannot undo a newer storage
  change. Storage changes take effect immediately, including changes buffered during initialisation
  and applied before discovery starts.
- `bootstrapContentScript()` always installs `GpuSessionFactory` (an injected factory replaces it).
  The controller requires a factory. A video is `eligible` from passing the gate until its session presents the first frame (the HUD header
  reads `ELIGIBLE · starting`).
- Gate point 8 = `factory.isGpuAvailable()`: WebGPU entry point present and no
  `WebGpuUnavailableError` from `acquire()` and no failed device recovery in this frame (both latch
  for the frame's lifetime). It stays true while a lost device is being recovered, so a recovering
  session is not destroyed by an unrelated re-evaluation.
- Gate point 9 = `handle.isBudgetHealthy()`. A throttling session stops, restores the video and
  calls `requestEvaluation()`; the controller latches `frame-budget` for that video until the
  quality, the enable switch or the source changes.
- Gate point 7 also honours the tab's site: an explicit per-site "off" for the tab's top-level
  hostname (what the popup's switch writes) disables every frame of the tab, so a cross-origin
  player iframe stops too; an "off" for the frame's own host applies on every embedding page.
  The HUD is also removed for global, frame-site, top-level-site and built-in site-rule disablement,
  even when an earlier gate point supplies the reported reason. Other ineligible states retain it.
  Settings changes apply disablement immediately to detached videos too: their sessions and HUDs
  are destroyed during the grace period, while discovery keeps tracking them for reattachment.
  Sub-frames learn the top-level hostname from the service worker (`GET_FRAME_CONTEXT` ->
  `FRAME_CONTEXT { topHost }`, from `sender.tab.url`), falling back to the outermost
  `location.ancestorOrigins` entry. A top-level "on" does not override a frame's site rule.
- One `CanvasOverlay` per session; it survives detach/reattach and device loss (only GPU objects are
  recreated). The HUD stays owned by the controller (`syncHud`), which ignores late reports of a
  detached video so the hidden HUD survives the 5 s grace period, and re-anchors it to the video on
  reattach (the detach cleared its anchor, which hides it).
- Every periodic discovery scan checks extension-context liveness before telemetry deduplication,
  including a `PING` to detect an invalidated context whose runtime ID still exists. A missing runtime
  ID or an "Extension context invalidated" messaging error stops discovery, destroys sessions and
  HUDs, removes canvases and restores native styles, even for paused videos with unchanged telemetry.
- `gpuDeviceManager` starts recovery before notifying lost-listeners, so a listener can await
  `whenRecovered()`.

### 6.2 Frame loop

- Driven only by `requestVideoFrameCallback`; extra single renders on `seeked` and `loadeddata`
  while paused (paused-MSE workaround), after a pipeline swap, after reattach and after device
  re-bind. The next rVFC is requested before the frame is processed.
- Order per frame: watchdog check -> `ring.acquire()` (null: drop, counted, fed to the controller)
  -> `new VideoFrame(video)` -> size check -> `importExternalTexture` -> one compute pass into
  `getCurrentTexture()` -> `encodeResolve` -> `ring.submit` -> `frame.close()` in
  `finally`. Every early exit releases the ring slot.
- `new VideoFrame` errors: `SecurityError` (or a tainted/cross-origin message) latches taint and
  stops for that source; `InvalidStateError` (no decoded frame) skips the callback; anything else is a
  session error. `video.mediaKeys` is checked per frame, `encrypted`/`waitingforkey` stop at once
  (the controller's own DRM listener runs first and destroys the session synchronously).
  A source change (currentSrc/src URL or srcObject identity) clears DRM, taint and HDR latches
  and destroys the old session before running
  eligibility again. An eligible new source gets a fresh session and keeps the native picture
  visible until its first validated frame, even if the old source stopped before a deferred evaluation.
- Pipeline input = the VideoFrame's display size; output = the canvas backing store actually
  applied by the overlay. The overlay sizes it with `computeCanvasResolution(target, W, H,
  internalScale)` (`internalScale` 2 for VSR, none for Quality 0), which is the `decideOutput()`
  rule; `session.test.ts` checks the two agree on 126 source/display/tier combinations. Using the
  applied backing (not the instantaneous target) keeps the overlay's fullscreen-transition debounce
  from causing a re-initialisation per animation frame. Any mismatch (first frame, source
  resolution change, backing change) restores the video and builds a new pipeline; frames are
  skipped until it is ready.
- Every frame runs inside `validation` + `out-of-memory` error scopes. Only the first submitted
  frame's clean scopes validate a pipeline. Every completed frame can restore SR after its own
  scopes resolve, provided the pipeline is validated and its recorded presentation epoch still
  matches the overlay. Stale completions cannot hide the native picture or block later frames;
  the newest stale completion requests a replacement while paused. Any GPU error fails the session
  (video restored). Covered with a fake device in `tests/unit/session_frames.test.ts`.
- Every backing-store size change increments the presentation epoch and restores the native picture,
  including while first-frame validation is pending. Paused resize, seek, sharpness and style changes,
  canvas remounts after presentation, visibility/intersection validation recovery (even with
  unchanged placement and dimensions), and pipeline swaps request still-frame redraws. Requests
  coalesce while the ring is full and retry when a slot is released; the pending request clears
  only after submission. The controller's periodic
  evaluation calls `refreshGeometry()` for active sessions, so computed object-position changes are
  picked up even without frame callbacks. The frame loop also checks geometry at most every 500 ms;
  it does not sync on every frame or use an idle timer.
- The gate rejects the video's own transform, scale, rotate or translate, and ancestor transforms
  other than pure 2D translation, ancestor scale / rotate, or non-unit CSS zoom (including
  `currentCSSZoom` where available). Bounding sizes must still match the untransformed dimensions.
  Ancestor `translateZ(0)` is allowed because it leaves the 2D geometry unchanged. These layouts
  report `unsupported-layout`; the overlay does not attempt to reproduce transformed coordinates.
- Fullscreen resolution follows open shadow roots to the deepest fullscreen element. Fullscreen
  of the video restores the native picture and fails with `native-fullscreen`; a fullscreen
  composed ancestor remains container fullscreen. Slotted videos fail with `unsupported-layout`
  because a light-DOM sibling canvas cannot share their slot assignment.
- The video's own non-`none` `clip-path`, `mask-image`, `-webkit-mask-image` or `object-view-box`
  (where available), non-`auto` legacy `clip`, and non-`normal` `mix-blend-mode`, fail with
  `unsupported-layout` in both the gate and overlay. Fixed-position videos and non-`none`
  `backdrop-filter` / `-webkit-backdrop-filter` on the video are also left untouched.
  The canvas mirrors computed `filter`
  alongside opacity during `sync()` (500 ms cadence plus events), without per-frame style reads.
- Inset `box-shadow`, a non-`none` outline with negative `outline-offset`, and nonzero video
  `order` under flex, inline-flex, grid or inline-grid parents fail the shared unsupported-effects
  predicate at the eligibility gate and on every geometry sync. An effect added to an active
  session restores the native picture before another frame can hide it.
- Every managed picture-property write (hide, reassert, restore, and site-style suspend/reapply)
  and video-position patch write runs with inline `transition-property: none !important`. A
  computed-style read flushes each batch before the page's exact inline transition value and
  priority are restored. Restoration checks declaration ownership, preserving page rewrites.
  Repeated presentation without managed style changes adds no style flush or per-frame work.
- Around canvas insertion, video-position patch installation and HUD insertion or moves,
  synchronous before/after snapshots cover the video, every element sibling and every composed
  ancestor through `document.documentElement`. The ancestry walk follows `assignedSlot`, then
  `parentElement`, then the shadow-root host. Snapshots copy every computed property string via
  `CSSStyleDeclaration.length`, `item(i)` and `getPropertyValue()`, plus all four viewport rect
  edges. Only nodes registered by this extension are excluded; page nodes with matching class
  names remain checked, as do newly enumerated properties.
  CSSOM can return layout-resolved used values, so comparisons exclude `width`, `height`,
  `inline-size`, `block-size`, all physical/logical margin and padding longhands,
  `top/right/bottom/left`, the four logical inset longhands, `transform-origin`,
  `perspective-origin` and `line-height` on every element. Rect edges check geometry instead:
  a resolved `margin-inline-end` change from `0px` to `-24px` with unchanged bounds is allowed,
  while an edge difference greater than 0.5 CSS px rejects the operation. All other properties,
  including `opacity`, `filter` and `display`, remain compared even when bounds are unchanged.
  Additional owned-property exclusions apply only to the video during the operation.
  Position-patch installation excludes the video's intentional `static` to `relative` change
  and derived all-auto to all-zero resolution of the enumerated `inset`, `inset-block` and
  `inset-inline` shorthands. Every before component must be `auto` and every after component
  `0px`; other shorthand changes reject installation. Physical/logical offset longhands use
  the global used-value exclusion. Site offsets and z-index are separately validated with the
  owned patch suspended; remaining video, sibling and ancestor properties compare normally.
  A changed compared property or rect edge triggers synchronous rollback. Canvas insertion or
  position-patch rejection restores the native picture, removes the patch and canvas, and sets
  a controller `unsupported-layout` latch keyed by the video, its `src`, `currentSrc` and parent.
  HUD rejection removes the host and sets a separate per-video latch, including without WebGPU.
  Session teardown, ordinary evaluations, quality changes and HUD toggles retain the verdict;
  a source or parent change clears the corresponding latch. This covers structural selector
  effects such as `:first-child`, `:nth-child` and ancestor `:has()` changes.
  Snapshots cost O(elements × properties) and run only on actual insertion/reinsertion, moves
  or patch installation. Ordinary geometry syncs, frames, HUD text updates and no-op moves do
  not enumerate computed properties. Invalid geometry also restores the native picture.
- Object-fit and object-position parsing fails closed in both the gate and overlay: values that
  cannot be represented exactly report `unsupported-layout`, without substituting defaults.
  For an active session, the gate reads the site's values with the picture override suspended.
- Corner-radius tokenization preserves parentheses; only plain lengths and percentages are
  supported, and unparsed components (including `calc(...)`) report `unsupported-layout`.
  CSS overlapping-radii normalization first scales all outer border-box radii by the minimum
  of 1 and each side's length divided by the sum of its two adjacent radii. Content-box radii
  then subtract the adjacent border and padding, floored at zero on both elliptical axes.
  The canvas uses `inset(... round ...)` in its own
  coordinates, including negative insets for letterboxing, combined with object-fit crop insets.
  Square boxes retain the object-fit crop, with `clip-path: none !important` when nothing is clipped.
- Canvas drawing is clipped to the video content box. An empty visible intersection keeps the canvas
  hidden and refuses picture hiding. The overlay never patches the parent's position, so other page
  elements retain their containing blocks and stacking behavior. The absolute canvas remains
  immediately before the video and uses the containing block provided by the page. During `sync()`,
  its `left` and `top` are first set to `0px !important` and its viewport rect is measured. This
  origin includes the containing block's borders, translation and scroll; the desired viewport
  content-fit position minus that origin gives the canvas's sub-pixel `left` and `top`.
- Only a computed-static video is patched with `position: relative !important`. Non-`auto` computed
  offsets or z-index reject installation. On every `sync()` while the patch is installed, computed
  `top`, `right`, `bottom`, `left` and `z-index` must remain `auto` with our owned position patch
  synchronously suspended. Installation uses the same suspended read, because relative positioning
  resolves auto offsets to `0px` in Chrome. The exact page position declaration and priority are
  restored for this synchronous read, then the patch is reinstalled before paint. Non-auto site
  offsets or z-index cause patch removal, native picture restoration and unsupported layout.
  A removed position patch is reapplied once, adopting page rewrites for ownership-aware restoration. If computed position
  remains static, or installation moves any video viewport edge by more than 0.5 CSS px, the patch
  is rolled back. Before installation, paint order is sampled with `elementsFromPoint` at the
  rendered content rect's centre and four 25%/75% points, skipping points outside the viewport and
  excluding our canvas. After installation and on every later sync, the video's index and element
  directly above it (or none) must match the recorded order. A mismatch removes the patch and
  reports `unsupported-layout`. Position restores only while the inline value and priority remain
  owned by the overlay.
- After placement, `sync()` compares all four canvas viewport edges with the expected content-fit
  rect derived from the video's viewport border box, border and padding. An error greater than
  0.5 CSS px on any edge restores the native picture, keeps the canvas hidden and signals
  `unsupported-layout` through the session/controller path. Placement and video patch checks run
  only during geometry sync (500 ms cadence plus events), with no per-frame layout or stacking
  reads. The HUD stays anchored at the video's parent.
- Canvas styles are reset with inline `all: initial !important` first on every `sync()`, then all
  overlay longhands are reasserted with `!important`: clipping when needed (otherwise
  `clip-path: none`), `mask`, `-webkit-mask`, `mask-image`, `-webkit-mask-image`, `backdrop-filter`,
  `outline` and `box-shadow` are `none`; `mix-blend-mode: normal`, `isolation: auto`,
  `object-fit: fill`, `object-position: 50% 50%`, `zoom: 1`, `content-visibility: visible`, and
  the current visibility plus mirrored opacity and filter. Computed display must be `block`;
  transform, rotate, scale and translate must be `none`, clip must be `auto`, and zoom must be `1`
  where exposed. Shown visibility, clip-path, mask-image, mix-blend-mode, opacity, filter and
  object-fit must also match the set values;
  a mismatch restores native playback with `unsupported-layout`. These checks run only during
  geometry sync (500 ms cadence plus events), without per-frame style reads.
- A browser-measured visibility fail-safe observes both video and canvas with IntersectionObserver
  while mounted (when available), disconnecting on unmount, re-parenting and destroy. The expected
  visible canvas region is the video's intersection rect intersected with the visible content-fit
  viewport rect. The check is two-sided: the canvas intersection must cover it within 1 CSS px on
  every edge whenever its area is at least 1 square CSS px, and a nonempty canvas intersection must
  never exceed the expected region by more than 1 CSS px on any edge, including when the expected
  region is empty. Excess means the canvas is visible where the video is clipped. This detects
  unequal ancestor/viewport clipping without rejecting
  equal clipping of both targets. Entries whose bounds differ from current viewport bounds by more
  than 1 CSS px are stale: during sync each stale target is unobserved and observed again at most
  once to request a fresh initial entry. The canvas stays hidden and the native picture restored
  until fresh entries for both targets pass; a fresh failing pair signals `unsupported-layout`.
  Cached entries are also checked on the first picture hide, so stale bounds cannot authorize it.
  The comparison runs on observer callbacks and geometry sync, using the latest entries; fresh
  passing entries do not trigger re-observation. During sync, available `checkVisibility` checks opacity, CSS visibility
  and `content-visibility: auto`; an invisible canvas with a visible video also fails. A mismatch
  while the canvas is shown and the picture hidden restores the native video, hides the canvas and
  signals `unsupported-layout` through the placement self-check's path. There is no per-frame work;
  unavailable APIs skip their respective checks. These browser APIs do not establish pixel-level
  equivalence or detect all effects, such as masks on ancestors or other pixel-level effects.
- Picture hiding: after a validated frame is presented, `VideoPictureGuard` moves the decoded
  picture and poster out of the video box with `object-position: 0px 100000px !important`,
  `background-color: transparent !important` and `background-image: none !important`. Native
  controls and WebVTT captions stay visible and clicks still reach the video. The non-interactive
  canvas is immediately before the video with the video's z-index, own opacity and filter (mirrored only
  during geometry sync); a static video with `z-index: auto` and no offsets is made
  `position: relative !important` while mounted, with its exact inline position restored on
  unmount, re-parenting or destroy. Geometry reads the site's object-position with the guard's
  overrides synchronously suspended, then re-applies them before a paint can occur. Restore paths
  put back every saved inline value and priority (including page rewrites adopted while hidden).
- Re-parenting: discovery's detach/reattach events pause/resume the loop and re-anchor the overlay;
  a synchronous move with no detach, a page that removes the canvas, or a player that re-renders its
  container but keeps the `<video>` is caught per frame (`!overlay.isMounted`, i.e. the canvas is
  not the video's previous sibling in the anchored parent -> `overlay.reattach()`). The overlay never
  hides the video without a mounted canvas, and a child-list MutationObserver on the parent restores
  the video as soon as the canvas is gone (also while paused, when no frame callback runs). No GPU
  object is reallocated in either case.
- Dropped frames = ring-full drops + `presentedFrames` gaps (1 < delta <= 10 within 500 ms; larger
  jumps and pauses are seeks/resumes, not drops). `fps` = rVFC callbacks over the last second.
- No CPU pixel read-backs. The only GPU -> CPU traffic is the ring's 16-byte timestamp read-back.

### 6.3 Budget controller (`BudgetController`)

| Parameter | Value | Meaning |
|---|---|---|
| late frame | latency > 1000 / fps and GPU time > 0.70 x 1000 / fps | the frame missed its slot (`ring.submit` -> `onSubmittedWorkDone`) while its own GPU execution (timestamp query, or the CPU time without timestamps) used more than 70 % of it; fps from `FrameRateEstimator` (median of rVFC `mediaTime` deltas per presented frame, divided by `playbackRate`), 60 fps until 4 intervals are known |
| window | 60 frames | late flags of the current tier's recent frames (decision window excludes warm-up) |
| sustain | 60 | violation counter: +1 while more than half of the window is late, -1 otherwise |
| cooldown | 30 frames | warm-up of a new tier ignored by the decision (still shown in telemetry) |
| watchdog | 250 ms observed | the queue-head frame's clock starts at max(submit, previous completion) and advances only by observed main-thread time; exceeding it steps down at once |
| heartbeat / stall gap | 50 ms / 100 ms | while frames are in flight the session observes on every frame callback and a 50 ms timer; a longer gap than 100 ms between two observations is a main-thread stall and is not charged to the GPU |
| ring full | 15 consecutive | drops with all 3 slots in flight step down at once |

- Why lateness and not the mean GPU time: under paced playback the Apple GPU lowers its clock until
  the work fills about 75-80 % of each frame interval, so the timestamp time follows the clock
  governor, not the workload (6.7: low at 720p60 -> 1600x900 reads 12.5-13.9 ms paced, 3.4 ms back
  to back). A mean-GPU-time threshold would misclassify these on-time frames. A frame that
  completes inside its slot is on time at whatever clock the GPU chose; a GPU that cannot keep up
  at full clock misses its slots and is caught. The 70 % GPU-time condition keeps light frames that
  were late only because a busy main thread delayed their completion callback from counting.
  The GPU mean / p95 remain telemetry (`stats()`), with the late fraction of the window.
- Why observed time: the watchdog runs on the page's main thread. A 400 ms page busy loop
  can delay both frame callbacks and completion notifications; that blocked interval is not
  charged to the GPU (`downgrade.spec.ts`). A hung GPU is caught: after a stall the heartbeat keeps observing and 250 ms without progress step down;
  the ring-full rule catches a hung GPU on a page whose main thread is blocked most of the time.
- Step order medium -> low -> quality0_bicubic; a step below Quality 0 is `throttle`.
- Hysteresis: a step clears both windows and the counters, restarts the warm-up, and (timeout or
  ring-full steps) restarts the watchdog clock. Watchdog (timeout) steps occur at most once per
  250 ms of observed time (medium -> throttle in about 750 ms through watchdog steps alone).
  Ring-full steps are count-based: each step requires 15 consecutive drops, without a 250 ms
  minimum interval. Samples carry the tier of the pipeline that rendered them; samples of any
  other tier (frames in flight across a switch) are ignored.
- Never upgrades: the tier only moves down from the ceiling (the user's quality limited by
  shader-f16); `reset(ceiling)` on a quality or source change is the only way up. There is no automatic
  recovery to a higher tier because it would oscillate around the threshold.
- Synthetic latency (6.6) is added to the measured GPU time and to the latency before the
  controller sees them.

### 6.4 Tier switches

A quality change cancels any pending build for a different requested tier, including a return to
an already active tier. The active pipeline stays in use, no replacement is requested for that
same tier, and cancelled build failures are ignored.

The controller changes its tier at once; the session builds the new pipeline while the old one keeps
rendering, swaps after `initialize()` resolved, then sets the overlay's internal scale (VSR <->
Quality 0 changes the backing store), destroys the old pipeline (in-flight frames keep their buffers
alive) and requests the current frame immediately, retaining the request until a ring slot is
available. Telemetry `activeModel` is the pipeline actually
running. Weight bundles are fetched and SHA-256-validated once per content-script lifetime per
model and shared by later pipelines. Builds are revisioned; a stale build is destroyed on arrival.

### 6.5 Device loss

On `gpuDeviceManager.onLost`: restore the video immediately, cancel the loop, destroy pipeline and
ring, unconfigure the canvas, bump the GPU generation (stale completions are ignored), keep the
device reference (so the manager keeps recovering) and the controller tier. On `onRestored`:
reconfigure the same canvas, new ring, restart the loop (the pipeline is rebuilt from the next
frame). If recovery ends without a device, the frame is marked GPU-unavailable and the gate re-run.
Loss during initial acquisition is tracked before a session has GPU objects; the acquired lost
context is skipped and the session binds the recovered device once it owns its manager reference.
The manager has explicit `idle`, `recovering` and `failed` phases, a generation incremented on every
loss of its current context, and one attempt counter for the entire loss burst. Replacements lost
during acquisition or restoration notification continue the same recovery loop and budget, with
delays of 1 s, 2 s, 4 s, etc. (six attempts by default). A subsequent loss after notification also
keeps that budget. Every context adopted with a nonzero attempt counter, including a fresh
acquisition after release, starts a context- and generation-guarded 10 s stability window to reset the
counter; the stability clock/timers and backoff sleep are injectable for unit tests. Release and loss
cancel the stability timer, and callbacks from an older context cannot reset a newer one's budget.
Restoration checks the current context and generation before each listener and again after
notification. Exhaustion enters `failed`, reports failure once for the burst and resolves all
`whenRecovered()` waiters with a generation-tagged failure; acquisition cannot bypass exhaustion.
Sessions capture the loss generation when waiting, ignore older completion/restoration callbacks,
and bind only the manager's current live context for that generation. Only a current-generation
failure marks the frame GPU-unavailable and stops processing with the native picture restored.
Fake-device regressions cover losses during notification, nested-microtask losses before a session
completion callback, and the stability-window budget reset.
Covered end to end by destroying the live session's device through the content script's isolated
world (CDP) in `video_overlay.spec.ts`.

### 6.6 Telemetry and diagnostics

- `SessionTelemetry` includes `presentedFrames` (monotonic per session) and `pipelineInits`
  (initialisations: first start, resize, tier switch, device re-bind); both are validated by
  `isSessionTelemetry`. Sessions report at most twice per second; the first presented frame after a
  hide and a pipeline swap are reported as soon as that limit allows. The controller still forwards
  to the service worker about once per second, so the e2e suite observes fast tier
  changes through the HUD and presented frames through a page-world rVFC sampler.
- Developer diagnostics (not in the popup): `chrome.storage.local` key `diagnostics` =
  `{ syntheticGpuLatencyMs: { '<pipeline id>': ms } }`, ids `rtx-vsr-medium | rtx-vsr-low |
  bicubic-q0`, values clamped to (0, 1000]; sanitised like the settings and watched live. It
  inflates the GPU times and submit -> done latencies the controller and the telemetry see, as if
  the GPU work took that much longer; it does not slow the GPU.

### 6.7 Measured in the e2e environment (Playwright Chromium 156, headless, M5 Max)

Timestamp-query is available without developer flags (quantised to 100 us). Observed telemetry
means under paced playback (one frame per video frame, not back-to-back like BENCHMARKS.md):

| Case | GPU mean (ms) |
|---|---:|
| low, 960x540 30 fps -> 1200x675 (resample) | 6.0 - 6.3 |
| low, 1280x720 60 fps -> 1600x900 (resample) | 11.7 - 12.3 |
| medium, 1280x720 60 fps -> 1600x900 (resample) | about 10.4 (50.4 with 40 ms synthetic latency) |
| Quality 0, 1280x720 60 fps -> 1600x900 | 1.6 - 1.9 |

These are 3-7x the back-to-back benchmark times because the GPU runs at a low clock under the
paced load (production-pipeline measurement: low 1280x720 -> 1600x900 reads 12.6 ms
paced but 3.6 ms when an unrelated compute load keeps the GPU busy, and 3.4 ms back to back; paced
timestamps sit at 75-80 % of the submit interval from 6 to 16.7 ms intervals). Under the deadline rule (6.3) low stays active at 720p60: `downgrade.spec.ts` plays the 720p60 clip
in a 1600x900 player for 12 s at low (GPU mean 12.5-13.9 ms, p95 12.6-15.6 ms over five runs) and at
medium (mean 12.5-14.1 ms, p95 12.9-14.3 ms) with no tier change; the rVFC cadence was 46-59 fps in
those runs (headless Chromium's own cadence varies in that range on this clip). It also survives four
400 ms main-thread stalls at medium. The
downgrade tests still drive medium and low with synthetic latency; the 60 fps leak test runs
Quality 0 so it measures the frame loop alone, and counts the content script's VideoFrames through
its isolated world (every frame created is closed; a missing `close()` is detected directly, since
it does not reliably stall this Chromium build's playback).

### 6.8 Testing notes

- A WebGPU canvas is readable from the page world: `toDataURL()` returns the presented image at
  any time; `drawImage`/`createImageBitmap` return black outside the rendering step, but inside a
  page rVFC callback registered after the session's they read the frame just encoded.
- `MutationObserver` in an init script must observe `document` (documentElement does not exist yet).
- Removing a video from the document pauses it; the theater-mode test resumes it like a player does.

## Support contract and residual risk

The supported reference scenarios are:

- YouTube watch pages in default, theater and fullscreen modes, with H.264, VP9 and AV1,
  quality changes and SPA navigation.
- Bilibili in default, wide, web-fullscreen and fullscreen modes.
- Plain `<video controls>` players with WebVTT captions and cross-origin iframe players.
- Ordinary pages whose videos pass the eligibility gate.

Outside that contract, page CSS or DOM that reacts to extension-inserted nodes or inline values
is unsupported and may hide or alter playback. Examples include `:has()`, child-count selectors,
generated content keyed to the canvas or HUD, attribute selectors on extension-owned values,
and cousin layers revealed by insertion. The remedy is the popup's per-site switch: disabling
the extension for that site removes all extension nodes and restores native playback.

The fail-closed eligibility gate reduces exposure to unsupported layouts and effects. Composed
ancestry and sibling snapshots, synchronous rollback and source/parent-scoped latches reject
observed insertion or position-patch changes without repeatedly retrying them. The visibility
fail-safe restores the native picture when the canvas cannot safely cover the expected visible
region. These mechanisms reduce risk; they cannot establish invariance for every page CSS or
DOM reaction.

HUD lines, text and styles live in an open shadow root. Page selectors cannot see shadow content:
document-level `:has()`, `:empty`, `:not(:empty)` and child combinators cannot observe its updates.
Content and position updates leave the single host's light DOM and attributes unchanged;
insertion and moves remain snapshot-guarded. The open root lets Playwright locators inspect the
HUD content without exposing it to document-level CSS selectors.
Before its first inline-style write to a video, the overlay records whether a style attribute
already existed. Once owned declarations are removed, it removes an empty attribute only if
none existed originally, preserving page-owned empty attributes and any remaining declarations.
