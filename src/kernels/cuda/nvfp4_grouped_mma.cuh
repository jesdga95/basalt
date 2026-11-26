#pragma once
// Grouped expert stage functions at 2304 x 1152 that stream the weights through shared memory. NVFP4 gate/up (type 40) sums
// exact int32 sub-block dots on mma.m16n8k16.s32.s8.s8.s32 in row_dot_sub8's tree order (v1's bits at every row count).
// MXFP8 (type 46) and NVFP4 down sum each 32-value block on two mma.m16n8k16.f32.f16.f16.f32 from zero on exact f16 forms
// of the weights (E4M3, or E2M1 times its UE4M3 sub-block scale) and of the int8 activations, then one fp32 chain step,
// and the blocks in one fixed order: their own bits, the same at every row count. The f16 tile design (weights as the M
// side, a thread's k positions chosen so its code bytes are contiguous) is adapted from ingine-batch's batch/moe.cu
// (Apache-2.0).

namespace {

// CT: the column tiles of 8 entries a thread holds activations for (a chunk of a group's entries is 8 CT of them)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t nvfp4_frag_s8(const block_nvfp4* b, int iqs, int t) {
    const uint32_t q4 = (uint32_t) get_int_b4(b->qs, 2 * iqs + (t & 1));
    const uint32_t nibs = (t & 2) ? ((q4 >> 4) & 0x0F0F0F0Fu) : (q4 & 0x0F0F0F0Fu);
    const uint32_t sel = (nibs & 0x0Fu) | ((nibs >> 4) & 0xF0u) | ((nibs >> 8) & 0xF00u) | ((nibs >> 12) & 0xF000u);
    const uint32_t lo = __byte_perm(0x03020100u, 0x0c080604u, sel);
    const uint32_t hi = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, sel);
    return __byte_perm(lo, hi, 0x3210u | ((sel & 0x8888u) >> 1));
}

__device__ __forceinline__ void mma_s8_k16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}
#endif

// The stages stream the experts' weights: the work is (group, 16-row tile of its gate and up rows, or of its down
// rows), group-major, and a block takes one contiguous run of it (the runs split it evenly), so it reads its weights as one
// stream. The tiles come into shared memory as bulk copies, NS in flight, the next one past them requested into L2. A
// thread holds the activation words and scales its k steps meet, for up to 8 CT of a group's entries, in registers: read once
// for all the group's tiles of the run. Each warp multiplies the tile's 16 rows against them over its share of K on
// mma.m16n8k16, and the block adds the warps' sums in row_dot_sub8's tree order: the arithmetic and its order are the
// per-tile stage's, on the same bytes, so every NVFP4 gate/up entry's bits are v1's at any row count. MXFP8 and NVFP4 down
// run the same frame with their own block products (mx_tile_mma) and the same tree; an entry's arithmetic never depends on
// the entries beside it.
constexpr int kNvSubRows = 16;
// bytes of a weight row of K NVFP4 values (blocks of 64 values in 36 bytes); a tile's bytes; tiles in flight; a warp's k16
// steps (k = wp + 8 j; MXFP8 rows are K / 128 blocks of 132 bytes, a warp's steps its 32-value blocks u = wp + 8 j)
template<int TY, int K> constexpr int kNvRowBytes = TY == 40 ? K / 64 * 36 : K / 128 * 132;
template<int TY, int K> constexpr int kNvTileBytes = kNvSubRows * kNvRowBytes<TY, K>;
template<bool GU> constexpr int kNvStages = GU ? 2 : 3;
template<int K> constexpr int kNvSteps = K / 128;
template<int K> constexpr int kMxSteps = (K / 32 + 7) / 8;
template<int TY, int K, bool GU, int CT>
constexpr size_t kNvSmemBytes = (size_t) kNvStages<GU> * kNvTileBytes<TY, K> + (size_t) 8 * CT * kNvSubRows * 9 * 4;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ unsigned gu_su(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void gu_mb_wait(unsigned long long* b, unsigned phase) {
    asm volatile("{\n.reg .pred d;\nw_%=:\nmbarrier.try_wait.parity.shared::cta.b64 d, [%0], %1, %2;\n@d bra e_%=;\nbra w_%=;\ne_%=:\n}\n"
                 ::"r"(gu_su(b)), "r"(phase), "r"(0x989680u) : "memory");
}
// thread 0: `bytes` from `src` into `dst`, completing on `bar`
__device__ __forceinline__ void tile_load(uint8_t* dst, unsigned long long* bar, const uint8_t* src, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(gu_su(bar)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(gu_su(dst)), "l"(src), "r"(bytes), "r"(gu_su(bar)) : "memory");
}
// thread 0: `bytes` from `src` into L2
__device__ __forceinline__ void tile_prefetch(const uint8_t* src, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
#endif

// A thread's activations for a chunk of up to 8 CT entries: per column tile ct and step j, the word of entry 8 ct + g it
// gives the B fragment and the scales of entries 8 ct + 2 t and + 1 (0 past the chunk).
template<int K, int CT>
struct NvCols {
    uint32_t b[CT][kNvSteps<K>];
    float d0[CT][kNvSteps<K>], d1[CT][kNvSteps<K>];
};
template<int K, bool GU, int CT>
__device__ __forceinline__ void nv_cols_load(NvCols<K, CT>& x, const block_q8_1* __restrict__ xq, const int32_t* __restrict__ ent, int e,
                                             int ne, int wp, int g, int t) {
    const auto entry = [&](int c) {
        return reinterpret_cast<const uint32_t*>(xq + (GU ? (size_t) ent[e + c] : (size_t) (e + c)) * (K / 32));
    };
#pragma unroll
    for (int ct = 0; ct < CT; ++ct) {
        const int cg = 8 * ct + g, c0 = 8 * ct + 2 * t, c1 = c0 + 1;
        const uint32_t* xg = cg < ne ? entry(cg) : nullptr;
        const uint32_t* x0 = c0 < ne ? entry(c0) : nullptr;
        const uint32_t* x1 = c1 < ne ? entry(c1) : nullptr;
#pragma unroll
        for (int j = 0; j < kNvSteps<K>; ++j) {
            const int kk = wp + 8 * j, yb = kk >> 1, a_off = 1 + 4 * ((kk & 3) & 1) + t;
            x.b[ct][j] = xg ? __ldg(xg + 9 * yb + a_off) : 0u;
            x.d0[ct][j] = x0 ? __low2float(*reinterpret_cast<const half2*>(x0 + 9 * yb)) : 0.0f;
            x.d1[ct][j] = x1 ? __low2float(*reinterpret_cast<const half2*>(x1 + 9 * yb)) : 0.0f;
        }
    }
}

// A warp's sums of a tile's rows g and g + 8 (wra, wrb) against the first CU column tiles of x: step j is k = wp + 8 j, summed in
// acc[.][j % 4] (the per-tile stage's k = wp + 8 s + 32 m).
template<int K, int CU, int CT>
__device__ __forceinline__ void nv_tile_mma(const uint8_t* wra, const uint8_t* wrb, const NvCols<K, CT>& x, int wp, int t,
                                            float (&acc)[CT][4][4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll
    for (int j = 0; j < kNvSteps<K>; ++j) {
        const int s = j & 3, kk = wp + 8 * j;
        const int kbx = kk >> 2, iqs = kk & 3;
        const block_nvfp4* ba = (const block_nvfp4*) wra + kbx;
        const block_nvfp4* bb = (const block_nvfp4*) wrb + kbx;
        const uint32_t a0 = nvfp4_frag_s8(ba, iqs, t), a1 = nvfp4_frag_s8(bb, iqs, t);
        const float d4a = nvfp4_scale_dev(ba->d[iqs]), d4b = nvfp4_scale_dev(bb->d[iqs]);
#pragma unroll
        for (int ct = 0; ct < CU; ++ct) {
            const float d8_0 = x.d0[ct][j], d8_1 = x.d1[ct][j];
            int si[4];
            mma_s8_k16(si, a0, a1, x.b[ct][j]);
            acc[ct][s][0] += d8_0 * d4a * (float) si[0];
            acc[ct][s][1] += d8_1 * d4a * (float) si[1];
            acc[ct][s][2] += d8_0 * d4b * (float) si[2];
            acc[ct][s][3] += d8_1 * d4b * (float) si[3];
        }
    }
#else
    (void) wra; (void) wrb; (void) x; (void) wp; (void) t; (void) acc;
#endif
}

// MXFP8 and NVFP4 down: a thread's activations for a chunk of up to 8 CT entries, as f16 (an int8 is exact in f16), for
// the two k16 MMAs of each 32-value block u = wp + 8 j: b[ct][j] holds entry 8 ct + g's pairs for the first MMA's B
// fragment (its k 2 t, 2 t + 1 and 2 t + 8, 2 t + 9), then the second's, and d0 / d1 the scales of entries 8 ct + 2 t and
// + 1 (0 past the chunk or past K). The MMA k is a permutation of the block's 32 positions that the weights share: NVFP4
// the natural one (a sub-block of 16 a MMA, as its codes pair value j with j + 8), MXFP8 4 t .. 4 t + 3 then
// 16 + 4 t .. 16 + 4 t + 3 (contiguous code bytes).
template<int K, int CT>
struct MxCols {
    uint32_t b[CT][kMxSteps<K>][4];
    float d0[CT][kMxSteps<K>], d1[CT][kMxSteps<K>];
};
__device__ __forceinline__ uint32_t s8x2_half2(uint32_t w) {
    const __half2 h = __halves2half2(__short2half_rn((short) (int8_t) (w & 0xFFu)), __short2half_rn((short) (int8_t) (w >> 8)));
    return *reinterpret_cast<const uint32_t*>(&h);
}
template<int TY, int K, bool GU, int CT>
__device__ __forceinline__ void mx_cols_load(MxCols<K, CT>& x, const block_q8_1* __restrict__ xq, const int32_t* __restrict__ ent, int e,
                                             int ne, int wp, int g, int t) {
    const auto entry = [&](int c) {
        return reinterpret_cast<const uint32_t*>(xq + (GU ? (size_t) ent[e + c] : (size_t) (e + c)) * (K / 32));
    };
#pragma unroll
    for (int ct = 0; ct < CT; ++ct) {
        const int cg = 8 * ct + g, c0 = 8 * ct + 2 * t, c1 = c0 + 1;
        const uint32_t* xg = cg < ne ? entry(cg) : nullptr;
        const uint32_t* x0 = c0 < ne ? entry(c0) : nullptr;
        const uint32_t* x1 = c1 < ne ? entry(c1) : nullptr;
#pragma unroll
        for (int j = 0; j < kMxSteps<K>; ++j) {
            const int u = wp + 8 * j;
            const bool in = u < K / 32;
            if constexpr (TY == 40) {   // bytes 2 t, 2 t + 8, 16 + 2 t, 24 + 2 t and the ones after them
                const int w = 1 + (t >> 1), sh = 16 * (t & 1);
#pragma unroll
                for (int m = 0; m < 4; ++m) x.b[ct][j][m] = s8x2_half2(xg && in ? __ldg(xg + 9 * u + w + 2 * m) >> sh : 0u);
            } else {
                const uint32_t q0 = xg && in ? __ldg(xg + 9 * u + 1 + t) : 0u, q1 = xg && in ? __ldg(xg + 9 * u + 5 + t) : 0u;
                x.b[ct][j][0] = s8x2_half2(q0);
                x.b[ct][j][1] = s8x2_half2(q0 >> 16);
                x.b[ct][j][2] = s8x2_half2(q1);
                x.b[ct][j][3] = s8x2_half2(q1 >> 16);
            }
            x.d0[ct][j] = x0 && in ? __low2float(*reinterpret_cast<const half2*>(x0 + 9 * u)) : 0.0f;
            x.d1[ct][j] = x1 && in ? __low2float(*reinterpret_cast<const half2*>(x1 + 9 * u)) : 0.0f;
        }
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_f16_k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, const float (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// two E4M3 codes (the low 16 bits) as an f16 pair, exactly
__device__ __forceinline__ uint32_t e4m3x2_half2(uint32_t w) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (w & 0xFFFFu), __NV_E4M3);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// two E2M1 codes (the low byte, the first in its low nibble) as an f16 pair, exactly
// (an f16 E2M1 value has a zero low byte: the high byte of magnitude m comes from a table, the sign from bit 3)
__device__ __forceinline__ uint32_t e2m1x2_half2(uint32_t w) {
    const uint32_t hi = __byte_perm(0x3E3C3800u, 0x46444240u, ((w & 7u) << 4) | ((w & 0x70u) << 8));
    return hi | ((w & 8u) << 12) | ((w & 0x80u) << 24);
}
// NVFP4's UE4M3 sub-block scale as nvfp4_scale_dev reads it, but whole (ggml halves it and doubles the codes), in both
// halves of an f16 pair: bit 7 ignored, 0x7F as 0
__device__ __forceinline__ uint32_t ue4m3_half2(uint32_t x) {
    const uint32_t v = x & 0x7Fu;
    const uint32_t h = v == 0x7Fu ? 0u : e4m3x2_half2(v) & 0xFFFFu;
    return h | h << 16;
}
#endif

// A warp's sums of a tile's rows g and g + 8 (wra, wrb) against the first CU column tiles of x: block u = wp + 8 j is two
// k16 MMAs from zero, then acc[.][j % 4] += (d_x * d_w) * that. MXFP8: the E4M3 codes times int8, d_w the block's UE8M0
// scale. NVFP4: the E2M1 codes times their sub-block's UE4M3 scale (exact in f16) times int8, d_w 1 (the expert's FP32
// scale multiplies the sum).
template<int TY, int K, int CU, int CT>
__device__ __forceinline__ void mx_tile_mma(const uint8_t* wra, const uint8_t* wrb, const MxCols<K, CT>& x, int wp, int t,
                                            float (&acc)[CT][4][4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll
    for (int j = 0; j < kMxSteps<K>; ++j) {
        const int u = wp + 8 * j;
        if ((K / 32) % 8 != 0 && u >= K / 32) break;   // warp-uniform: the warps past K's last block
        const int s = j & 3;
        uint32_t a0[4], a1[4];
        float dwa = 1.0f, dwb = 1.0f;
        if constexpr (TY == 40) {   // block u >> 1, sub-blocks 2 (u & 1) and the next: bytes 2 t, 2 t + 1 of each
            const uint8_t* ba = wra + (u >> 1) * 36;
            const uint8_t* bb = wrb + (u >> 1) * 36;
            const int s0 = 2 * (u & 1);
            const auto frag = [&](uint32_t (&a)[4], int sb) {
                const uint32_t ca = *reinterpret_cast<const uint16_t*>(ba + 4 + 8 * sb + 2 * t);
                const uint32_t cb = *reinterpret_cast<const uint16_t*>(bb + 4 + 8 * sb + 2 * t);
                const uint32_t ha = ue4m3_half2(ba[sb]), hb = ue4m3_half2(bb[sb]);
                const auto mul = [](uint32_t v, uint32_t sc) {
                    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&v), *reinterpret_cast<const __half2*>(&sc));
                    return *reinterpret_cast<const uint32_t*>(&r);
                };
                // low nibbles: values 2 t, 2 t + 1; high: 2 t + 8, 2 t + 9
                a[0] = mul(e2m1x2_half2((ca & 0x0Fu) | ((ca >> 4) & 0xF0u)), ha);
                a[1] = mul(e2m1x2_half2((cb & 0x0Fu) | ((cb >> 4) & 0xF0u)), hb);
                a[2] = mul(e2m1x2_half2(((ca >> 4) & 0x0Fu) | ((ca >> 8) & 0xF0u)), ha);
                a[3] = mul(e2m1x2_half2(((cb >> 4) & 0x0Fu) | ((cb >> 8) & 0xF0u)), hb);
            };
            frag(a0, s0);
            frag(a1, s0 + 1);
        } else {
            const uint8_t* ba = wra + (u >> 2) * 132;
            const uint8_t* bb = wrb + (u >> 2) * 132;
            const uint8_t* qa = ba + 4 + 32 * (u & 3) + 4 * t;
            const uint8_t* qb = bb + 4 + 32 * (u & 3) + 4 * t;
            const uint32_t wa0 = *reinterpret_cast<const uint32_t*>(qa), wa1 = *reinterpret_cast<const uint32_t*>(qa + 16);
            const uint32_t wb0 = *reinterpret_cast<const uint32_t*>(qb), wb1 = *reinterpret_cast<const uint32_t*>(qb + 16);
            a0[0] = e4m3x2_half2(wa0); a0[1] = e4m3x2_half2(wb0); a0[2] = e4m3x2_half2(wa0 >> 16); a0[3] = e4m3x2_half2(wb0 >> 16);
            a1[0] = e4m3x2_half2(wa1); a1[1] = e4m3x2_half2(wb1); a1[2] = e4m3x2_half2(wa1 >> 16); a1[3] = e4m3x2_half2(wb1 >> 16);
            dwa = ue8m0_scale_dev(ba[u & 3]);
            dwb = ue8m0_scale_dev(bb[u & 3]);
        }
        constexpr float kZero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ct = 0; ct < CU; ++ct) {
            float c[4];
            mma_f16_k16(c, a0, x.b[ct][j][0], x.b[ct][j][1], kZero);
            mma_f16_k16(c, a1, x.b[ct][j][2], x.b[ct][j][3], c);
            acc[ct][s][0] = __fmaf_rn(__fmul_rn(x.d0[ct][j], dwa), c[0], acc[ct][s][0]);
            acc[ct][s][1] = __fmaf_rn(__fmul_rn(x.d1[ct][j], dwa), c[1], acc[ct][s][1]);
            acc[ct][s][2] = __fmaf_rn(__fmul_rn(x.d0[ct][j], dwb), c[2], acc[ct][s][2]);
            acc[ct][s][3] = __fmaf_rn(__fmul_rn(x.d1[ct][j], dwb), c[3], acc[ct][s][3]);
        }
    }
#else
    (void) wra; (void) wrb; (void) x; (void) wp; (void) t; (void) acc;
#endif
}

// A call's groups at most (the stage's prefix of their chunks sits in shared memory)
constexpr int kNvMaxGroups = 256;

// TY: 40 NVFP4 or 46 MXFP8. GU: gate/up (K = n_embd 2304, rows 2 n_ff, entries' tokens `ent`, out0 gate and out1 up [entry][n_ff]); else down (K =
// n_ff 1152, rows n_embd, entries' destinations `ent`, out0 [dest][n_embd]). DEP: a programmatic dependent, its weights
// requested before the wait (they and the plan are ready before the kernel before it starts). A group of more than 8 CT
// entries runs as one per 8 CT of them (chunk-major), each streaming the group's weights again, so a thread's activations
// change only when its run reaches another chunk.
template<int TY, int K, bool GU, bool DEP, int CT>
__device__ __forceinline__ void expert_stream_stage(int block, int blocks, int tid, uint8_t* smem,
                                                   const unsigned long long* __restrict__ grp_ptr,
                                                   const int32_t* __restrict__ grp_start,
                                                   const int32_t* __restrict__ n_groups,
                                                   const int32_t* __restrict__ ent,
                                                   const block_q8_1* __restrict__ xq,
                                                   NativeExpertLayout L,
                                                   float* __restrict__ out0,
                                                   float* __restrict__ out1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int RB = kNvRowBytes<TY, K>, TB = kNvTileBytes<TY, K>, NS = kNvStages<GU>;
    if constexpr (!DEP) pdl_launch_dependents();
    constexpr int C = 8 * CT;
    __shared__ int s_pre[kNvMaxGroups + 1];   // the work items before each group: nx a chunk of C entries
    __shared__ int s_wsum[8];
    __shared__ __align__(8) unsigned long long bar[NS];
    const int ng = *n_groups;
    const int nx = (GU ? 2 * (int) L.n_ff : (int) L.n_embd) / kNvSubRows;
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    {   // the exclusive prefix of the groups' items, one group a thread
        int v = tid < ng ? (grp_start[tid + 1] - grp_start[tid] + C - 1) / C * nx : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
        }
        if (lane == 31) s_wsum[wp] = v;
        __syncthreads();
        int before = 0;
        for (int w = 0; w < wp; ++w) before += s_wsum[w];
        if (tid < ng) s_pre[tid + 1] = before + v;
        if (tid == 0) s_pre[0] = 0;
        __syncthreads();
    }
    const int total = s_pre[ng];
    const int per = (total + blocks - 1) / blocks;
    const int i0 = block * per, i1 = min(total, i0 + per);
    float* s_part = reinterpret_cast<float*>(smem + NS * TB);
    // item i: its group (searched on from `gi`), the chunk's first entry and the tile's first row
    const auto locate = [&](int i, int& gi, int& e, int& row0) {
        while (i >= s_pre[gi + 1]) ++gi;
        const int u = i - s_pre[gi], c = u / nx;
        e = grp_start[gi] + c * C;
        row0 = (u - c * nx) * kNvSubRows;
    };
    const auto src = [&](int gi, int row0) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[gi];
        if constexpr (GU) {
            const bool is_up = row0 >= (int) L.n_ff;
            return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row0 - (int) L.n_ff : row0) * L.gu_row;
        } else {
            return blob + L.down_off + (size_t) row0 * L.d_row;
        }
    };
    int ahead_g = 0;   // the search position of the items requested ahead
    const auto request = [&](int i, int slot) {   // thread 0: item i's tile into `slot`, the one after it into L2
        int e_a, r_a;
        locate(i, ahead_g, e_a, r_a);
        tile_load(smem + slot * TB, &bar[slot], src(ahead_g, r_a), (unsigned) TB);
        if (i + 1 < i1) {
            int gj = ahead_g;
            locate(i + 1, gj, e_a, r_a);
            tile_prefetch(src(gj, r_a), (unsigned) TB);
        }
    };
    if (tid == 0 && i0 < i1) {
        for (int j = 0; j < NS; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gu_su(&bar[j])) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int j = 0; j < NS && i0 + j < i1; ++j) request(i0 + j, j);
    }
    if constexpr (DEP) {
        pdl_wait();
        pdl_launch_dependents();
    }
    if (i0 >= i1) return;
    __syncthreads();   // the barriers' init seen
    constexpr bool kInt8 = TY == 40 && GU;   // NVFP4 gate/up: v1's exact sub-block dots
    std::conditional_t<kInt8, NvCols<K, CT>, MxCols<K, CT>> x;
    int held_e = -1, gi = 0;
    for (int i = i0, k = 0; i < i1; ++i, ++k) {
        int e, row0;
        locate(i, gi, e, row0);
        const bool is_up = GU && row0 >= (int) L.n_ff;
        const int r0 = is_up ? row0 - (int) L.n_ff : row0;
        float* dst = is_up ? out1 : out0;
        const float sc = TY == 40 ? nvfp4_tail((const void*) grp_ptr[gi], L)[GU ? (is_up ? 1 : 0) : 2] : 1.0f;
        const int ne = min(C, grp_start[gi + 1] - e);
        if (held_e != e) {   // entries are each one group's, so the first entry names the chunk
            if constexpr (kInt8) nv_cols_load<K, GU, CT>(x, xq, ent, e, ne, wp, g, t);
            else mx_cols_load<TY, K, GU, CT>(x, xq, ent, e, ne, wp, g, t);
            held_e = e;
        }
        uint8_t* tile = smem + (k % NS) * TB;
        const uint8_t* wra = tile + (size_t) g * RB;
        const uint8_t* wrb = wra + 8 * RB;
        gu_mb_wait(&bar[k % NS], (unsigned) (k / NS) & 1u);
        float acc[CT][4][4] = {};
        if constexpr (kInt8) {
            if (CT == 1 || ne <= 8) nv_tile_mma<K, 1, CT>(wra, wrb, x, wp, t, acc);
            else nv_tile_mma<K, CT, CT>(wra, wrb, x, wp, t, acc);
        } else {
            if (CT == 1 || ne <= 8) mx_tile_mma<TY, K, 1, CT>(wra, wrb, x, wp, t, acc);
            else mx_tile_mma<TY, K, CT, CT>(wra, wrb, x, wp, t, acc);
        }
#pragma unroll
        for (int ct = 0; ct < CT; ++ct) {
            if (8 * ct >= ne) break;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int col = 8 * ct + 2 * t + (q & 1);
                if (col < ne) {
                    const int row = 8 * (q >> 1) + g;
                    const float u0 = __fadd_rn(acc[ct][0][q], acc[ct][2][q]);
                    const float u1 = __fadd_rn(acc[ct][1][q], acc[ct][3][q]);
                    s_part[(col * kNvSubRows + row) * 9 + wp] = __fadd_rn(u0, u1);
                }
            }
        }
        __syncthreads();   // the sums are in, and every warp is done with the tile
        if (tid == 0 && i + NS < i1) request(i + NS, k % NS);   // the tile's slot takes the item NS on
        for (int idx = tid; idx < ne * kNvSubRows; idx += 256) {
            const int row = idx & (kNvSubRows - 1), col = idx / kNvSubRows;
            const float* p = s_part + (col * kNvSubRows + row) * 9;
            const float r4_0 = __fadd_rn(p[0], p[4]), r4_1 = __fadd_rn(p[1], p[5]);
            const float r4_2 = __fadd_rn(p[2], p[6]), r4_3 = __fadd_rn(p[3], p[7]);
            const float r2_0 = __fadd_rn(r4_0, r4_2), r2_1 = __fadd_rn(r4_1, r4_3);
            const float v = TY == 40 ? __fmul_rn(__fadd_rn(r2_0, r2_1), sc) : __fadd_rn(r2_0, r2_1);
            if constexpr (GU) dst[(size_t) (e + col) * L.n_ff + r0 + row] = v;
            else dst[(size_t) ent[e + col] * L.n_embd + r0 + row] = v;
        }
        __syncthreads();   // the sums are read before the next ones
    }
#else
    (void) block; (void) blocks; (void) tid; (void) smem; (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent;
    (void) xq; (void) L; (void) out0; (void) out1;
    __trap();
#endif
}

}  // namespace
