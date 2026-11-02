// webgpu-vsr native benchmark: RTX VSR (SDK 1.1) float models as Metal compute kernels.
//
// Contract: docs/IMPLEMENTATION_NOTES.md section 3 (low/medium, folded p4/r2) and the
// research semantics (high/ultra, unfolded, public 2x head p2/r2). Storage is FP16 NHWC with
// a padded row pitch (pitch = roundup(W, 16), rows = roundup(H, 8)); pixels outside the valid
// W x H region are never read as data (3x3 staging zero-fills them), so the padding only
// gives every 16x8 tile a fully allocated footprint for simdgroup loads/stores.
//
// Numerics: FP16 weights and stored activations, convolution accumulation in f32 on
// simdgroup_matrix (half x half -> float), bias added in f32 (it initialises the
// accumulator), ELU in f32, one rounding to half per convolution output. A residual block
// merges its two branches with one half add: H(H(ELU(C3_1 + b)) + H(C1)). Fold, pool,
// baseline dot D, T and the composition use the exact half sequences of the contract
// (compiled with -ffp-contract=off so only the explicit fma() calls fuse).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------------------------------
// Shared structures (mirrored in Sources/MetalEngine.swift)
// ---------------------------------------------------------------------------------------

struct PrepParams {
    uint srcW, srcH;      // source frame (RGBA8)
    uint outW, outH;      // level-0 grid (folded: TW/2 x TH/2, unfolded: TW x TH)
    uint pitch;           // level-0 row pitch in pixels
    uint fold;            // 1 = space-to-depth 2x2 -> 16 channels, 0 = half4 + 4 zeros -> 8
};

struct ConvParams {
    uint inW, inH, inPitch, inStride;      // staged 3x3 input (low-res tensor when upsampling)
    uint outW, outH, outPitch, outStride;  // convolution domain and output tensor
    uint projStride;                       // 1x1 projection input (same level / pitch as out)
    uint poolPitch, poolStride;            // pooled output (level + 1)
    uint reserved;
};

struct HeadParams {
    uint featW, featH, featPitch, featStride;  // level-0 decoder output d3
    uint srcW, srcH;                           // original source frame
    uint outW, outH;                           // internal output 2W x 2H
    half coef[8];                              // stored scale-2 coefficients [phase][tap]
};

// ---------------------------------------------------------------------------------------
// Function constants
// ---------------------------------------------------------------------------------------

constant uint FC_CIN      [[function_constant(0)]];   // 3x3 input channels
constant uint FC_K        [[function_constant(1)]];   // 3x3 output channels
constant uint FC_CC       [[function_constant(2)]];   // staging channel chunk (<= 32)
constant uint FC_NB       [[function_constant(3)]];   // 8-wide output blocks per simdgroup
constant bool FC_UPSAMPLE [[function_constant(4)]];   // nearest2x source mapping
constant bool FC_PROJ     [[function_constant(5)]];   // + biasless 1x1 projection after ELU
constant uint FC_PROJ_C   [[function_constant(6)]];   // projection input channels
constant bool FC_POOL     [[function_constant(7)]];   // also emit the ordered-FMA 2x2 pool
constant bool FC_INPLACE  [[function_constant(8)]];   // output aliases projection input
constant uint FC_KZ       [[function_constant(9)]];   // head 1x1 outputs, padded to 8
constant uint FC_P        [[function_constant(10)]];  // head learned pixel-shuffle factor
constant uint FC_OUTMODE  [[function_constant(11)]];  // 0 planar half RGB, 1 RGBA8
constant uint FC_MB       [[function_constant(12)]];  // 8-pixel M-blocks per simdgroup (1, 2, 4, 8)
constant uint FC_NSPLIT   [[function_constant(13)]];  // output-channel split across grid.z
constant uint MGROUPS = 16 / FC_MB;                   // simdgroups sharing one N range
constant uint NBT = FC_K / 8 / FC_NSPLIT;             // 8-channel output blocks per threadgroup

constant uint TX = 16;                  // tile width  (pixels)
constant uint TY = 8;                   // tile height (pixels)
constant uint PX = TX + 2;              // staged patch width (halo 1)
constant uint PY = TY + 2;              // staged patch height
constant uint MAX_CC = 32;
constant uint PATCH_HALVES = PX * PY * MAX_CC;   // 5760 halves = 11.25 KiB

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

inline float elu(float v) { return v >= 0.0f ? v : precise::exp(v) - 1.0f; }

// Elementwise epilogues on accumulator fragments are position independent, so the
// (undocumented) lane -> element mapping of simdgroup_matrix is never needed.
inline void elu_inplace(thread simdgroup_float8x8 &m) {
    thread auto &e = m.thread_elements();
    e[0] = elu(e[0]);
    e[1] = elu(e[1]);
}

inline simdgroup_half8x8 to_half(thread simdgroup_float8x8 &m) {
    simdgroup_half8x8 h;
    thread auto &src = m.thread_elements();
    thread auto &dst = h.thread_elements();
    dst[0] = half(src[0]);
    dst[1] = half(src[1]);
    return h;
}

// Stage a (TY+2) x (TX+2) x FC_CC patch (channels [c0, c0+FC_CC)) of the 3x3 input into
// threadgroup memory, zero outside the valid region. With FC_UPSAMPLE the patch lives at the
// upsampled resolution (outW x outH) and pixel p reads low-res pixel floor(p/2): together
// with the zero border this is nearest2x followed by a zero-padded 3x3 cross-correlation,
// i.e. src = floor((out + tap - 1) / 2) with out-of-range sources contributing zero.
inline void stage_patch(threadgroup uint4 *patch, device const half *in,
                        thread const ConvParams &p, uint2 tile, uint c0, uint tid, uint nthreads) {
    const uint vecs = FC_CC / 8;
    const uint total = PX * PY * vecs;
    for (uint i = tid; i < total; i += nthreads) {
        const uint pix = i / vecs;
        const uint v = i - pix * vecs;
        const int gy = int(tile.y * TY + pix / PX) - 1;
        const int gx = int(tile.x * TX + pix % PX) - 1;
        uint4 val = uint4(0);
        if (FC_UPSAMPLE) {
            if (gx >= 0 && gy >= 0 && uint(gx) < p.outW && uint(gy) < p.outH) {
                const uint sp = (uint(gy) >> 1) * p.inPitch + (uint(gx) >> 1);
                val = *(device const uint4 *)(in + sp * p.inStride + c0 + v * 8);
            }
        } else {
            if (gx >= 0 && gy >= 0 && uint(gx) < p.inW && uint(gy) < p.inH) {
                const uint sp = uint(gy) * p.inPitch + uint(gx);
                val = *(device const uint4 *)(in + sp * p.inStride + c0 + v * 8);
            }
        }
        patch[i] = val;
    }
}

// Implicit GEMM over K x (9*C) for one 16x8 tile = 16 M-blocks of 8 pixels (M-block mb is
// pixel row mb/2, 8-pixel segment mb%2). Simdgroup sg owns M-blocks [g*MB, g*MB + MB) with
// g = sg % (16/MB), and output blocks nb0 .. nb0 + NB with nb0 = tile.z * NBT +
// (sg / (16/MB)) * NB (grid.z splits K when one threadgroup cannot hold all of it; the head
// never splits). Weights are pre-packed as [K/8][9][C/8][8 (c)][8 (k)] so every B operand
// is one contiguous 8x8 tile.
inline void conv3x3_tile(thread simdgroup_float8x8 (&acc)[8][4], threadgroup half *patch,
                         device const half *in, device const half *w3,
                         device const float *biasTile, thread const ConvParams &p, uint2 tile,
                         uint g, uint nb0, uint tid, uint nthreads) {
    for (uint n = 0; n < FC_NB; ++n) {
        simdgroup_float8x8 b;
        simdgroup_load(b, biasTile + (nb0 + n) * 8, FC_K);
        for (uint m = 0; m < FC_MB; ++m) acc[m][n] = b;
    }
    const uint CB = FC_CIN / 8;
    for (uint c0 = 0; c0 < FC_CIN; c0 += FC_CC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stage_patch((threadgroup uint4 *)patch, in, p, tile, c0, tid, nthreads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tap = 0; tap < 9; ++tap) {
            const uint r = tap / 3, s = tap % 3;
            for (uint c8 = 0; c8 < FC_CC / 8; ++c8) {
                simdgroup_half8x8 a[8];
                for (uint m = 0; m < FC_MB; ++m) {
                    const uint mb = FC_MB * g + m;
                    const uint ty = mb >> 1, xs = mb & 1;
                    simdgroup_load(a[m], patch + ((ty + r) * PX + xs * 8 + s) * FC_CC + c8 * 8, FC_CC);
                }
                const uint cb = c0 / 8 + c8;
                for (uint n = 0; n < FC_NB; ++n) {
                    simdgroup_half8x8 wb;
                    simdgroup_load(wb, w3 + (((nb0 + n) * 9 + tap) * CB + cb) * 64, 8);
                    for (uint m = 0; m < FC_MB; ++m)
                        simdgroup_multiply_accumulate(acc[m][n], a[m], wb, acc[m][n]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// 1. Preprocess + fold (or channel pad): source RGBA8 -> level-0 input x0.
//    I = half4(H(2R-1), H(2G-1), H(2B-1), 0) via a 256-entry LUT computed on the host
//    (f32 2*(v/255)-1, rounded to half); edge replication beyond the source.
//    Folded:   F[Y,X, 8*dy + 4*dx + c] = I[2Y+dy, 2X+dx, c]           (16 channels)
//    Unfolded: x0[y,x] = [I.r, I.g, I.b, 0, 0, 0, 0, 0]                 (8 channels)
// ---------------------------------------------------------------------------------------

kernel void vsr_prep(device const uchar4 *src [[buffer(0)]],
                     constant half *lut [[buffer(1)]],
                     device half *x0 [[buffer(2)]],
                     constant PrepParams &p [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.outW || gid.y >= p.outH) return;
    if (p.fold != 0) {
        device half4 *dst = (device half4 *)(x0 + (gid.y * p.pitch + gid.x) * 16);
        for (uint dy = 0; dy < 2; ++dy) {
            for (uint dx = 0; dx < 2; ++dx) {
                const uint sx = min(2 * gid.x + dx, p.srcW - 1);
                const uint sy = min(2 * gid.y + dy, p.srcH - 1);
                const uchar4 v = src[sy * p.srcW + sx];
                dst[2 * dy + dx] = half4(lut[v.r], lut[v.g], lut[v.b], 0.0h);
            }
        }
    } else {
        const uint sx = min(gid.x, p.srcW - 1);
        const uint sy = min(gid.y, p.srcH - 1);
        const uchar4 v = src[sy * p.srcW + sx];
        device half4 *dst = (device half4 *)(x0 + (gid.y * p.pitch + gid.x) * 8);
        dst[0] = half4(lut[v.r], lut[v.g], lut[v.b], 0.0h);
        dst[1] = half4(0.0h);
    }
}

// ---------------------------------------------------------------------------------------
// 2. Fused 3x3 convolution stage.
//    out = H(ELU(C3(in) + b))                                       (FC_PROJ = false)
//    out = H(H(ELU(C3(in) + b)) + H(C1(projIn)))   (biasless 1x1)    (FC_PROJ = true)
//    optional pool (ordered half FMA, TL, TR, BL, BR) of `out` into poolOut.
//    FC_UPSAMPLE: in is the low-res tensor, C3 runs on its nearest2x upsampling.
//    FC_INPLACE: out aliases projIn (decoder RB writes d into cat.U); a threadgroup
//    barrier separates every projection read of the tile from the first store. Only legal
//    with FC_NSPLIT == 1 (the host enforces it): with a K split, other threadgroups would
//    still read the same pixels' projection inputs.
// ---------------------------------------------------------------------------------------

kernel void vsr_conv3x3(device const half *in [[buffer(0)]],
                        device const half *w3 [[buffer(1)]],
                        device const float *biasTile [[buffer(2)]],
                        device const half *projIn [[buffer(3)]],
                        device const half *w1 [[buffer(4)]],
                        device half *out [[buffer(5)]],
                        device half *poolOut [[buffer(6)]],
                        constant ConvParams &p [[buffer(7)]],
                        uint3 tile3 [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint3 tpg [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half patch[PATCH_HALVES];
    const uint nthreads = tpg.x;
    const uint2 tile = tile3.xy;
    const uint g = sg % MGROUPS;
    const uint nb0 = tile3.z * NBT + (sg / MGROUPS) * FC_NB;
    simdgroup_float8x8 acc[8][4];
    const ConvParams lp = p;
    conv3x3_tile(acc, patch, in, w3, biasTile, lp, tile, g, nb0, tid, nthreads);
    // e = H(ELU(C3 + b)); with a projection the block output is H(e + H(C1(projIn))), the
    // packed-half merge of the two branches (HIGH-CLOSURE.md: HADD2 at a8c0/aa80/af10..b0d0).
    simdgroup_half8x8 res[8][4];
    for (uint m = 0; m < FC_MB; ++m) {
        for (uint n = 0; n < FC_NB; ++n) {
            elu_inplace(acc[m][n]);
            res[m][n] = to_half(acc[m][n]);
        }
    }

    if (FC_PROJ) {
        for (uint m = 0; m < FC_MB; ++m)
            for (uint n = 0; n < FC_NB; ++n) acc[m][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint c8 = 0; c8 < FC_PROJ_C / 8; ++c8) {
            simdgroup_half8x8 a[8];
            for (uint m = 0; m < FC_MB; ++m) {
                const uint mb = FC_MB * g + m;
                const uint pix = (tile.y * TY + (mb >> 1)) * p.outPitch + tile.x * TX + (mb & 1) * 8;
                simdgroup_load(a[m], projIn + pix * p.projStride + c8 * 8, p.projStride);
            }
            for (uint n = 0; n < FC_NB; ++n) {
                simdgroup_half8x8 wb;
                simdgroup_load(wb, w1 + ((nb0 + n) * (FC_PROJ_C / 8) + c8) * 64, 8);
                for (uint m = 0; m < FC_MB; ++m)
                    simdgroup_multiply_accumulate(acc[m][n], a[m], wb, acc[m][n]);
            }
        }
        for (uint m = 0; m < FC_MB; ++m) {
            for (uint n = 0; n < FC_NB; ++n) {
                // Same fragment shape, so element i of both fragments is the same (row, col).
                simdgroup_half8x8 pr = to_half(acc[m][n]);
                thread auto &e = res[m][n].thread_elements();
                thread auto &q = pr.thread_elements();
                e[0] = e[0] + q[0];
                e[1] = e[1] + q[1];
            }
        }
    }
    if (FC_INPLACE) threadgroup_barrier(mem_flags::mem_device);

    for (uint m = 0; m < FC_MB; ++m) {
        const uint mb = FC_MB * g + m;
        const uint pix = (tile.y * TY + (mb >> 1)) * p.outPitch + tile.x * TX + (mb & 1) * 8;
        for (uint n = 0; n < FC_NB; ++n)
            simdgroup_store(res[m][n], out + pix * p.outStride + (nb0 + n) * 8, p.outStride);
    }

    if (FC_POOL) {
        // Pools the channels this threadgroup produced: [tile.z*NBT*8, (tile.z+1)*NBT*8).
        threadgroup_barrier(mem_flags::mem_device);
        const uint vecs = NBT * 2;
        const uint cbase = tile3.z * NBT * 8;
        const uint total = (TX / 2) * (TY / 2) * vecs;
        const uint pw = p.outW / 2, ph = p.outH / 2;
        for (uint i = tid; i < total; i += nthreads) {
            const uint q = i / vecs;
            const uint v = i - q * vecs;
            const uint gx = tile.x * (TX / 2) + q % (TX / 2);
            const uint gy = tile.y * (TY / 2) + q / (TX / 2);
            if (gx >= pw || gy >= ph) continue;
            const uint r0 = (2 * gy) * p.outPitch + 2 * gx;
            const uint r1 = r0 + p.outPitch;
            const uint ch = cbase + v * 4;
            const half4 tl = *(device const half4 *)(out + r0 * p.outStride + ch);
            const half4 tr = *(device const half4 *)(out + (r0 + 1) * p.outStride + ch);
            const half4 bl = *(device const half4 *)(out + r1 * p.outStride + ch);
            const half4 br = *(device const half4 *)(out + (r1 + 1) * p.outStride + ch);
            half4 a = tl * 0.25h;
            a = fma(tr, half4(0.25h), a);
            a = fma(bl, half4(0.25h), a);
            a = fma(br, half4(0.25h), a);
            *(device half4 *)(poolOut + (gy * p.poolPitch + gx) * p.poolStride + ch) = a;
        }
    }
}

// ---------------------------------------------------------------------------------------
// 3. Fused SR head: h = H(ELU(C3(d3) + b)) (FC_K hidden channels, threadgroup memory),
//    z = H(C1(h) + b1) (FC_KZ channels, threadgroup memory), then for every output pixel
//    of the tile's footprint: k = P*P*color + P*(oy % P) + (ox % P) at feature
//    (oy / P, ox / P); T = H(0.5 * (tanh(z) + 1)) with tanh as (e+ - e-)/(e+ + e-) in f32;
//    baseline B: scale 2 from the ORIGINAL source, taps floor((o+0.5)/2-0.5) - 1 .. + 2
//    clamped, N = H(H(I + 1) * 0.5), horizontal-first two-accumulator half dot D;
//    out = H(H(B + T) - 0.5).
// ---------------------------------------------------------------------------------------

inline half3 dot_d(half3 v0, half3 v1, half3 v2, half3 v3, half w0, half w1, half w2, half w3) {
    half3 a = v0 * w0;
    a = fma(v2, half3(w2), a);
    half3 b = v1 * w1;
    b = fma(v3, half3(w3), b);
    return a + b;
}

inline half3 norm_px(uchar4 v, constant half *lut) {
    const half3 i = half3(lut[v.r], lut[v.g], lut[v.b]);
    const half3 s = i + 1.0h;
    return s * 0.5h;
}

kernel void vsr_head(device const half *feat [[buffer(0)]],
                     device const half *w3 [[buffer(1)]],
                     device const float *biasTile [[buffer(2)]],
                     device const half *w1 [[buffer(3)]],
                     device const float *bias1Tile [[buffer(4)]],
                     device const uchar4 *src [[buffer(5)]],
                     constant half *lut [[buffer(6)]],
                     device half *outHalf [[buffer(7)]],
                     device uchar4 *outRGBA [[buffer(8)]],
                     constant HeadParams &hp [[buffer(9)]],
                     uint3 tile3 [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint3 tpg [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint nsg [[simdgroups_per_threadgroup]]) {
    // patch (3x3 staging) and z share storage: z is written only after the hidden layer.
    threadgroup half shared[PATCH_HALVES > TX * TY * 48 ? PATCH_HALVES : TX * TY * 48];
    threadgroup half hbuf[TX * TY * 64];

    const uint nthreads = tpg.x;
    ConvParams p;
    p.inW = hp.featW; p.inH = hp.featH; p.inPitch = hp.featPitch; p.inStride = hp.featStride;
    p.outW = hp.featW; p.outH = hp.featH; p.outPitch = hp.featPitch; p.outStride = 0;
    p.projStride = 0; p.poolPitch = 0; p.poolStride = 0; p.reserved = 0;

    const uint2 tile = tile3.xy;
    simdgroup_float8x8 acc[8][4];
    {
        const uint g = sg % MGROUPS;
        const uint nb0 = (sg / MGROUPS) * FC_NB;
        conv3x3_tile(acc, shared, feat, w3, biasTile, p, tile, g, nb0, tid, nthreads);
        for (uint m = 0; m < FC_MB; ++m) {
            const uint mb = FC_MB * g + m;
            for (uint n = 0; n < FC_NB; ++n) {
                elu_inplace(acc[m][n]);
                simdgroup_store(to_half(acc[m][n]), hbuf + mb * 8 * FC_K + (nb0 + n) * 8, FC_K);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // z = H(C1(h) + b1): 16 M-blocks x KZ/8 output blocks, M-blocks distributed over simdgroups.
    for (uint mb = sg; mb < 16; mb += nsg) {
        simdgroup_float8x8 z[6];
        for (uint n = 0; n < FC_KZ / 8; ++n) simdgroup_load(z[n], bias1Tile + n * 8, FC_KZ);
        for (uint c8 = 0; c8 < FC_K / 8; ++c8) {
            simdgroup_half8x8 a;
            simdgroup_load(a, hbuf + mb * 8 * FC_K + c8 * 8, FC_K);
            for (uint n = 0; n < FC_KZ / 8; ++n) {
                simdgroup_half8x8 wb;
                simdgroup_load(wb, w1 + (n * (FC_K / 8) + c8) * 64, 8);
                simdgroup_multiply_accumulate(z[n], a, wb, z[n]);
            }
        }
        for (uint n = 0; n < FC_KZ / 8; ++n)
            simdgroup_store(to_half(z[n]), shared + mb * 8 * FC_KZ + n * 8, FC_KZ);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output footprint of this tile: (TX*P) x (TY*P) pixels of the 2W x 2H internal image.
    const uint fw = TX * FC_P, fh = TY * FC_P;
    const uint plane = hp.outW * hp.outH;
    for (uint i = tid; i < fw * fh; i += nthreads) {
        const uint ox = tile.x * fw + i % fw;
        const uint oy = tile.y * fh + i / fw;
        if (ox >= hp.outW || oy >= hp.outH) continue;
        const uint lx = ox / FC_P - tile.x * TX;
        const uint ly = oy / FC_P - tile.y * TY;
        const uint dx = ox % FC_P, dy = oy % FC_P;
        threadgroup const half *zp = shared + (ly * TX + lx) * FC_KZ + FC_P * dy + dx;

        // Baseline taps: ix = floor((ox + 0.5)/2 - 0.5) = (ox - 1) >> 1 (arithmetic shift).
        const int ix = (int(ox) - 1) >> 1;
        const int iy = (int(oy) - 1) >> 1;
        const uint phx = ox & 1, phy = oy & 1;
        const half wx0 = hp.coef[4 * phx + 0], wx1 = hp.coef[4 * phx + 1];
        const half wx2 = hp.coef[4 * phx + 2], wx3 = hp.coef[4 * phx + 3];
        const half wy0 = hp.coef[4 * phy + 0], wy1 = hp.coef[4 * phy + 1];
        const half wy2 = hp.coef[4 * phy + 2], wy3 = hp.coef[4 * phy + 3];
        uint xs[4];
        for (int t = 0; t < 4; ++t) xs[t] = uint(clamp(ix - 1 + t, 0, int(hp.srcW) - 1));
        half3 rows[4];
        for (int t = 0; t < 4; ++t) {
            const uint sy = uint(clamp(iy - 1 + t, 0, int(hp.srcH) - 1));
            device const uchar4 *row = src + sy * hp.srcW;
            rows[t] = dot_d(norm_px(row[xs[0]], lut), norm_px(row[xs[1]], lut),
                            norm_px(row[xs[2]], lut), norm_px(row[xs[3]], lut), wx0, wx1, wx2, wx3);
        }
        const half3 B = dot_d(rows[0], rows[1], rows[2], rows[3], wy0, wy1, wy2, wy3);

        half3 o;
        for (uint c = 0; c < 3; ++c) {
            // |z| >= 16 already gives t = +-1 exactly in f32; the clamp only prevents inf/inf.
            const float zf = clamp(float(zp[c * FC_P * FC_P]), -16.0f, 16.0f);
            const float ep = precise::exp2(zf * 1.44269504088896341f);
            const float em = precise::exp2(-zf * 1.44269504088896341f);
            const float t = precise::divide(ep - em, ep + em);
            const half T = half(0.5f * (t + 1.0f));
            const half s = B[c] + T;
            o[c] = s - 0.5h;
        }
        if (FC_OUTMODE == 0) {
            const uint idx = oy * hp.outW + ox;
            outHalf[idx] = o.r;
            outHalf[plane + idx] = o.g;
            outHalf[2 * plane + idx] = o.b;
        } else {
            // Direct 2x path = dlpp_postProcessE (PREPROCESS-SEMANTICS.md): clamp, multiply by
            // 255 and convert with truncation toward zero. The +0.5 round-half-up belongs to the
            // resample stage only (RESAMPLE-SEMANTICS.md). v * 255 is exact in f32 (a half times
            // an 8-bit integer), so the conversion is the only rounding.
            const float3 v = clamp(float3(o), 0.0f, 1.0f);
            const uint3 q = uint3(v * 255.0f);
            outRGBA[oy * hp.outW + ox] = uchar4(uchar(q.r), uchar(q.g), uchar(q.b), 255);
        }
    }
}
