#pragma once
// Bitwise-identical high-concurrency grouped NVFP4 (type 40) expert stage functions at 2304 x 1152:
//  - Groups/tails with ne <= 2 entries execute register-only row_dot_sub8 (zero shared-memory staging or barriers).
//  - Groups with ne >= 3 entries stage up to 16 columns in shared memory (odd-strided for bank-conflict-free B loads)
//    and accumulate exact int32 sub-block dots on mma.m16n8k32.s32.s8.s8.s32 in row_dot_multi / row_dot_sub8 tree order.

namespace {

constexpr int kFastMoeRows = 32;
constexpr int kFastMoeCols = 16;
constexpr int kFastMoeCTiles = kFastMoeCols / 8;
constexpr int kFastGuWords = 72 * 9;            // 648 words per 2304-dim q8_1 entry
constexpr int kFastGuStride = kFastGuWords + 1; // 649 words (odd -> bank-conflict-free B loads)
constexpr int kFastDnWords = 36 * 9;            // 324 words per 1152-dim q8_1 entry
constexpr int kFastDnStride = kFastDnWords + 1; // 325 words (odd -> bank-conflict-free B loads)

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
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%6}, {%7,%6}, {%6,%6,%6,%6};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(0), "r"(b0));
}
#endif

// The gate/up stage's work items: (group, 32-row tile of its gate and up rows), group-major. A block takes items
// blockIdx.x, + gridDim.x, ... Each item's weight tile comes into shared memory as one bulk copy (a tile's 32 rows are
// contiguous), and the block's next tile is requested into L2 while it multiplies the current one: the weights stream
// in large requests instead of waiting on a warp's loads. The arithmetic and its order are the per-tile stage's, on the
// same bytes. (Two tiles and the activations do not fit a block's shared memory on sm_120.)
constexpr int kGuTileBytes = kFastMoeRows * 36 * 36;   // 32 rows of 2304 NVFP4 values (36 blocks of 36 bytes)
constexpr size_t kGuSmemBytes = (size_t) kGuTileBytes + (size_t) kFastMoeCols * kFastGuStride * 4;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ unsigned gu_su(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void gu_mb_wait(unsigned long long* b, unsigned phase) {
    asm volatile("{\n.reg .pred d;\nw_%=:\nmbarrier.try_wait.parity.shared::cta.b64 d, [%0], %1, %2;\n@d bra e_%=;\nbra w_%=;\ne_%=:\n}\n"
                 ::"r"(gu_su(b)), "r"(phase), "r"(0x989680u) : "memory");
}
// item w's gate/up tile in global memory
__device__ __forceinline__ const uint8_t* gu_tile_src(const unsigned long long* grp_ptr, int w, int nx, const NativeExpertLayout& L) {
    const int gi = w / nx, row0 = (w - gi * nx) * kFastMoeRows;
    const bool is_up = row0 >= (int) L.n_ff;
    return (const uint8_t*) grp_ptr[gi] + (is_up ? L.up_off : 0) + (size_t) (is_up ? row0 - (int) L.n_ff : row0) * L.gu_row;
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

__device__ __forceinline__ void native_gu_nvfp4_mma_stage(int block, int blocks, int tid, uint8_t* smem,
                                                          const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_tok,
                                                          const block_q8_1* __restrict__ xq,
                                                          NativeExpertLayout L,
                                                          float* __restrict__ gate,
                                                          float* __restrict__ up) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    pdl_launch_dependents();
    const int nx = 2 * (int) L.n_ff / kFastMoeRows;
    const int total = *n_groups * nx;
    if (block >= total) return;
    __shared__ __align__(8) unsigned long long bar;
    uint32_t* s_x = reinterpret_cast<uint32_t*>(smem + kGuTileBytes);
    float* s_part = reinterpret_cast<float*>(s_x);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gu_su(&bar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int r_sub_off = tid >> 3, t_sub = tid & 7;
    for (int w = block, k = 0; w < total; w += blocks, ++k) {
        __syncthreads();   // every thread is done with the tile before (and sees the barrier's init)
        if (tid == 0) {
            tile_load(smem, &bar, gu_tile_src(grp_ptr, w, nx, L), (unsigned) kGuTileBytes);
            if (w + blocks < total) tile_prefetch(gu_tile_src(grp_ptr, w + blocks, nx, L), (unsigned) kGuTileBytes);
        }
        const int gi = w / nx, row0 = (w - gi * nx) * kFastMoeRows;
        const bool is_up = row0 >= (int) L.n_ff;
        const int r0 = is_up ? row0 - (int) L.n_ff : row0;
        float* dst = is_up ? up : gate;
        const uint8_t* blob = (const uint8_t*) grp_ptr[gi];
        const float sc = nvfp4_tail((const void*) blob, L)[is_up ? 1 : 0];
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        const uint8_t* tile = smem;   // rows r0 .. r0 + 31 of the half
        const uint8_t* wra0 = tile + (size_t) g * L.gu_row;
        const uint8_t* wrb0 = wra0 + 8 * L.gu_row;
        const uint8_t* wra1 = wra0 + 16 * L.gu_row;
        const uint8_t* wrb1 = wra0 + 24 * L.gu_row;
        bool landed = false;
        for (int e = e0; e < e1; e += kFastMoeCols) {
            const int ne = min(kFastMoeCols, e1 - e);
            if (ne <= 2) {
                if (!landed) { gu_mb_wait(&bar, (unsigned) k & 1u); landed = true; }
                const uint8_t* wr_sub = tile + (size_t) r_sub_off * L.gu_row;
                if (ne == 1) {
                    const int off1[1] = { ent_tok[e] * 72 };
                    float s1[1];
                    row_dot_sub8<40, 1, true, 144>(wr_sub, xq, off1, 1, 144, t_sub, s1);
                    if (t_sub == 0) dst[(size_t) e * L.n_ff + r0 + r_sub_off] = __fmul_rn(s1[0], sc);
                } else {
                    const int off2[2] = { ent_tok[e] * 72, ent_tok[e + 1] * 72 };
                    float s2[2];
                    row_dot_sub8<40, 2, true, 144>(wr_sub, xq, off2, 2, 144, t_sub, s2);
                    if (t_sub < 2) dst[(size_t) (e + t_sub) * L.n_ff + r0 + r_sub_off] = __fmul_rn(t_sub == 0 ? s2[0] : s2[1], sc);
                }
                continue;
            }
            __syncthreads();
            for (int idx = tid; idx < ne * kFastGuWords; idx += 256) {
                const int c = idx / kFastGuWords, rem = idx - c * kFastGuWords;
                s_x[c * kFastGuStride + rem] = reinterpret_cast<const uint32_t*>(xq + (size_t) ent_tok[e + c] * 72)[rem];
            }
            if (!landed) { gu_mb_wait(&bar, (unsigned) k & 1u); landed = true; }
            __syncthreads();
            float acc[2][kFastMoeCTiles][4][4] = {};
            for (int m = 0; m < 5; ++m) {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    if (m == 4 && s >= 2) break;
                    const int k = wp + 8 * s + 32 * m;
                    const int kbx = k >> 2, iqs = k & 3;
                    const block_nvfp4* ba0 = (const block_nvfp4*) wra0 + kbx;
                    const block_nvfp4* bb0 = (const block_nvfp4*) wrb0 + kbx;
                    const block_nvfp4* ba1 = (const block_nvfp4*) wra1 + kbx;
                    const block_nvfp4* bb1 = (const block_nvfp4*) wrb1 + kbx;
                    const uint32_t a0_0 = nvfp4_frag_s8(ba0, iqs, t), a1_0 = nvfp4_frag_s8(bb0, iqs, t);
                    const uint32_t a0_1 = nvfp4_frag_s8(ba1, iqs, t), a1_1 = nvfp4_frag_s8(bb1, iqs, t);
                    const float d4a0 = nvfp4_scale_dev(ba0->d[iqs]), d4b0 = nvfp4_scale_dev(bb0->d[iqs]);
                    const float d4a1 = nvfp4_scale_dev(ba1->d[iqs]), d4b1 = nvfp4_scale_dev(bb1->d[iqs]);
                    const int yb = k >> 1;
                    const int a_off = 1 + 4 * (iqs & 1) + t;
#pragma unroll
                    for (int ct = 0; ct < kFastMoeCTiles; ++ct) {
                        if (8 * ct >= ne) break;
                        const int cg = 8 * ct + g, c0 = 8 * ct + 2 * t, c1 = c0 + 1;
                        const uint32_t b0 = cg < ne ? s_x[cg * kFastGuStride + 9 * yb + a_off] : 0u;
                        const float d8_0 = c0 < ne ? __low2float(*reinterpret_cast<const half2*>(&s_x[c0 * kFastGuStride + 9 * yb])) : 0.0f;
                        const float d8_1 = c1 < ne ? __low2float(*reinterpret_cast<const half2*>(&s_x[c1 * kFastGuStride + 9 * yb])) : 0.0f;
                        int si0[4], si1[4];
                        mma_s8_k16(si0, a0_0, a1_0, b0);
                        mma_s8_k16(si1, a0_1, a1_1, b0);
                        acc[0][ct][s][0] += d8_0 * d4a0 * (float) si0[0];
                        acc[0][ct][s][1] += d8_1 * d4a0 * (float) si0[1];
                        acc[0][ct][s][2] += d8_0 * d4b0 * (float) si0[2];
                        acc[0][ct][s][3] += d8_1 * d4b0 * (float) si0[3];
                        acc[1][ct][s][0] += d8_0 * d4a1 * (float) si1[0];
                        acc[1][ct][s][1] += d8_1 * d4a1 * (float) si1[1];
                        acc[1][ct][s][2] += d8_0 * d4b1 * (float) si1[2];
                        acc[1][ct][s][3] += d8_1 * d4b1 * (float) si1[3];
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int ct = 0; ct < kFastMoeCTiles; ++ct) {
                    if (8 * ct >= ne) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int col = 8 * ct + 2 * t + (q & 1);
                        if (col < ne) {
                            const int row = 16 * mt + 8 * (q >> 1) + g;
                            const float u0 = __fadd_rn(acc[mt][ct][0][q], acc[mt][ct][2][q]);
                            const float u1 = __fadd_rn(acc[mt][ct][1][q], acc[mt][ct][3][q]);
                            s_part[(col * 32 + row) * 9 + wp] = __fadd_rn(u0, u1);
                        }
                    }
                }
            }
            __syncthreads();
            for (int idx = tid; idx < ne * 32; idx += 256) {
                const int row = idx & 31, col = idx >> 5;
                const float* p = s_part + (col * 32 + row) * 9;
                const float r4_0 = __fadd_rn(p[0], p[4]), r4_1 = __fadd_rn(p[1], p[5]);
                const float r4_2 = __fadd_rn(p[2], p[6]), r4_3 = __fadd_rn(p[3], p[7]);
                const float r2_0 = __fadd_rn(r4_0, r4_2), r2_1 = __fadd_rn(r4_1, r4_3);
                dst[(size_t) (e + col) * L.n_ff + r0 + row] = __fmul_rn(__fadd_rn(r2_0, r2_1), sc);
            }
        }
    }
#else
    (void) block; (void) blocks; (void) tid; (void) smem;
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_tok; (void) xq; (void) L; (void) gate; (void) up;
    __trap();
#endif
}

// The down stage the same way: items (group, 32-row tile of its down rows), a tile one bulk copy (its rows contiguous).
constexpr int kDnTileBytes = kFastMoeRows * 18 * 36;   // 32 rows of 1152 NVFP4 values (18 blocks of 36 bytes)
constexpr size_t kDnSmemBytes = (size_t) kDnTileBytes + (size_t) kFastMoeCols * kFastDnStride * 4;

template<bool DEP = false>
__device__ __forceinline__ void native_down_nvfp4_mma_stage(int block, int blocks, int tid, uint8_t* smem,
                                                            const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_dst,
                                                            const block_q8_1* __restrict__ hq,
                                                            NativeExpertLayout L,
                                                            float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (DEP) {
        pdl_wait();
        pdl_launch_dependents();
    }
    const int nx = (int) L.n_embd / kFastMoeRows;
    const int total = *n_groups * nx;
    if (block >= total) return;
    __shared__ __align__(8) unsigned long long bar;
    uint32_t* s_x = reinterpret_cast<uint32_t*>(smem + kDnTileBytes);
    float* s_part = reinterpret_cast<float*>(s_x);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gu_su(&bar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int r_sub_off = tid >> 3, t_sub = tid & 7;
    const auto src = [&](int w) {
        const int gi = w / nx;
        return (const uint8_t*) grp_ptr[gi] + L.down_off + (size_t) ((w - gi * nx) * kFastMoeRows) * L.d_row;
    };
    for (int w = block, k = 0; w < total; w += blocks, ++k) {
        __syncthreads();   // every thread is done with the tile before (and sees the barrier's init)
        if (tid == 0) {
            tile_load(smem, &bar, src(w), (unsigned) kDnTileBytes);
            if (w + blocks < total) tile_prefetch(src(w + blocks), (unsigned) kDnTileBytes);
        }
        const int gi = w / nx, row0 = (w - gi * nx) * kFastMoeRows;
        const uint8_t* blob = (const uint8_t*) grp_ptr[gi];
        const float sc = nvfp4_tail((const void*) blob, L)[2];
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        const uint8_t* tile = smem;   // rows row0 .. row0 + 31
        const uint8_t* wra0 = tile + (size_t) g * L.d_row;
        const uint8_t* wrb0 = wra0 + 8 * L.d_row;
        const uint8_t* wra1 = wra0 + 16 * L.d_row;
        const uint8_t* wrb1 = wra0 + 24 * L.d_row;
        bool landed = false;
        for (int e = e0; e < e1; e += kFastMoeCols) {
            const int ne = min(kFastMoeCols, e1 - e);
            if (ne <= 2) {
                if (!landed) { gu_mb_wait(&bar, (unsigned) k & 1u); landed = true; }
                const uint8_t* wr_sub = tile + (size_t) r_sub_off * L.d_row;
                if (ne == 1) {
                    const int off1[1] = { e * 36 };
                    float s1[1];
                    row_dot_sub8<40, 1, true, 72>(wr_sub, hq, off1, 1, 72, t_sub, s1);
                    if (t_sub == 0) out[(size_t) ent_dst[e] * L.n_embd + row0 + r_sub_off] = __fmul_rn(s1[0], sc);
                } else {
                    const int off2[2] = { e * 36, (e + 1) * 36 };
                    float s2[2];
                    row_dot_sub8<40, 2, true, 72>(wr_sub, hq, off2, 2, 72, t_sub, s2);
                    if (t_sub < 2) out[(size_t) ent_dst[e + t_sub] * L.n_embd + row0 + r_sub_off] = __fmul_rn(t_sub == 0 ? s2[0] : s2[1], sc);
                }
                continue;
            }
            __syncthreads();
            for (int idx = tid; idx < ne * kFastDnWords; idx += 256) {
                const int c = idx / kFastDnWords, rem = idx - c * kFastDnWords;
                s_x[c * kFastDnStride + rem] = reinterpret_cast<const uint32_t*>(hq + (size_t) (e + c) * 36)[rem];
            }
            if (!landed) { gu_mb_wait(&bar, (unsigned) k & 1u); landed = true; }
            __syncthreads();
            float acc[2][kFastMoeCTiles][4][4] = {};
            for (int m = 0; m < 3; ++m) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (m == 2 && i > 0) break;
                    const int k = wp + 8 * i + 32 * m;
                    const int kbx = k >> 2, iqs = k & 3;
                    const block_nvfp4* ba0 = (const block_nvfp4*) wra0 + kbx;
                    const block_nvfp4* bb0 = (const block_nvfp4*) wrb0 + kbx;
                    const block_nvfp4* ba1 = (const block_nvfp4*) wra1 + kbx;
                    const block_nvfp4* bb1 = (const block_nvfp4*) wrb1 + kbx;
                    const uint32_t a0_0 = nvfp4_frag_s8(ba0, iqs, t), a1_0 = nvfp4_frag_s8(bb0, iqs, t);
                    const uint32_t a0_1 = nvfp4_frag_s8(ba1, iqs, t), a1_1 = nvfp4_frag_s8(bb1, iqs, t);
                    const float d4a0 = nvfp4_scale_dev(ba0->d[iqs]), d4b0 = nvfp4_scale_dev(bb0->d[iqs]);
                    const float d4a1 = nvfp4_scale_dev(ba1->d[iqs]), d4b1 = nvfp4_scale_dev(bb1->d[iqs]);
                    const int yb = k >> 1;
                    const int a_off = 1 + 4 * (iqs & 1) + t;
#pragma unroll
                    for (int ct = 0; ct < kFastMoeCTiles; ++ct) {
                        if (8 * ct >= ne) break;
                        const int cg = 8 * ct + g, c0 = 8 * ct + 2 * t, c1 = c0 + 1;
                        const uint32_t b0 = cg < ne ? s_x[cg * kFastDnStride + 9 * yb + a_off] : 0u;
                        const float d8_0 = c0 < ne ? __low2float(*reinterpret_cast<const half2*>(&s_x[c0 * kFastDnStride + 9 * yb])) : 0.0f;
                        const float d8_1 = c1 < ne ? __low2float(*reinterpret_cast<const half2*>(&s_x[c1 * kFastDnStride + 9 * yb])) : 0.0f;
                        int si0[4], si1[4];
                        mma_s8_k16(si0, a0_0, a1_0, b0);
                        mma_s8_k16(si1, a0_1, a1_1, b0);
                        acc[0][ct][i][0] += d8_0 * d4a0 * (float) si0[0];
                        acc[0][ct][i][1] += d8_1 * d4a0 * (float) si0[1];
                        acc[0][ct][i][2] += d8_0 * d4b0 * (float) si0[2];
                        acc[0][ct][i][3] += d8_1 * d4b0 * (float) si0[3];
                        acc[1][ct][i][0] += d8_0 * d4a1 * (float) si1[0];
                        acc[1][ct][i][1] += d8_1 * d4a1 * (float) si1[1];
                        acc[1][ct][i][2] += d8_0 * d4b1 * (float) si1[2];
                        acc[1][ct][i][3] += d8_1 * d4b1 * (float) si1[3];
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int ct = 0; ct < kFastMoeCTiles; ++ct) {
                    if (8 * ct >= ne) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int col = 8 * ct + 2 * t + (q & 1);
                        if (col < ne) {
                            const int row = 16 * mt + 8 * (q >> 1) + g;
                            const float u0 = __fadd_rn(acc[mt][ct][0][q], acc[mt][ct][2][q]);
                            const float u1 = __fadd_rn(acc[mt][ct][1][q], acc[mt][ct][3][q]);
                            s_part[(col * 32 + row) * 9 + wp] = __fadd_rn(u0, u1);
                        }
                    }
                }
            }
            __syncthreads();
            for (int idx = tid; idx < ne * 32; idx += 256) {
                const int row = idx & 31, col = idx >> 5;
                const float* p = s_part + (col * 32 + row) * 9;
                const float r4_0 = __fadd_rn(p[0], p[4]), r4_1 = __fadd_rn(p[1], p[5]);
                const float r4_2 = __fadd_rn(p[2], p[6]), r4_3 = __fadd_rn(p[3], p[7]);
                const float r2_0 = __fadd_rn(r4_0, r4_2), r2_1 = __fadd_rn(r4_1, r4_3);
                out[(size_t) ent_dst[e + col] * L.n_embd + row0 + row] = __fmul_rn(__fadd_rn(r2_0, r2_1), sc);
            }
        }
    }
#else
    (void) block; (void) blocks; (void) tid; (void) smem;
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_dst; (void) hq; (void) L; (void) out;
    __trap();
#endif
}


constexpr int kFastMxfp8Cols = 8;

__device__ __forceinline__ void native_gu_mxfp8_sx8_stage(dim3 block_idx, dim3 grid_dim, int tid,
                                                          const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_tok,
                                                          const block_q8_1* __restrict__ xq,
                                                          NativeExpertLayout L,
                                                          float* __restrict__ gate,
                                                          float* __restrict__ up) {
    const int ng = *n_groups;
    if (ng <= (int) grid_dim.y) pdl_launch_dependents();
    if ((int) block_idx.y >= ng) return;
    __shared__ __align__(16) uint32_t s_x[kFastMxfp8Cols * kFastGuWords];
    const block_q8_1* sx = reinterpret_cast<const block_q8_1*>(s_x);
    const int r_sub_off = tid >> 3, t_sub = tid & 7;
    const int row0 = (int) block_idx.x * kFastMoeRows;
    const int row = row0 + r_sub_off;
    const bool valid = row < 2 * (int) L.n_ff;
    const bool is_up = row >= (int) L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = valid ? (is_up ? L.up_off : 0) + (size_t) r * L.gu_row : 0;
    float* dst = is_up ? up : gate;
    for (int gi = (int) block_idx.y; gi < ng; gi += (int) grid_dim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[gi] + w_off;
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        for (int e = e0; e < e1; e += kFastMxfp8Cols) {
            const int ne = min(kFastMxfp8Cols, e1 - e);
            if (ne <= 2) {
                if (valid) {
                    if (ne == 1) {
                        const int off1[1] = { ent_tok[e] * 72 };
                        float s1[1];
                        row_dot_sub8<46, 1, true, 288>(wr, xq, off1, 1, 288, t_sub, s1);
                        if (t_sub == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
                    } else {
                        const int off2[2] = { ent_tok[e] * 72, ent_tok[e + 1] * 72 };
                        float s2[2];
                        row_dot_sub8<46, 2, true, 288>(wr, xq, off2, 2, 288, t_sub, s2);
                        if (t_sub < 2) dst[(size_t) (e + t_sub) * L.n_ff + r] = s2[t_sub];
                    }
                }
                continue;
            }
            __syncthreads();
            for (int idx = tid; idx < ne * kFastGuWords; idx += 256) {
                const int c = idx / kFastGuWords, rem = idx - c * kFastGuWords;
                s_x[idx] = reinterpret_cast<const uint32_t*>(xq + (size_t) ent_tok[e + c] * 72)[rem];
            }
            __syncthreads();
            if (valid) {
                if (ne <= 4) {
                    int off4[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) off4[c] = min(c, ne - 1) * 72;
                    float s4[4];
                    row_dot_sub8<46, 4, false, 288>(wr, sx, off4, ne, 288, t_sub, s4);
                    if (t_sub < ne) dst[(size_t) (e + t_sub) * L.n_ff + r] = s4[t_sub];
                } else {
                    int off8[8];
#pragma unroll
                    for (int c = 0; c < 8; ++c) off8[c] = min(c, ne - 1) * 72;
                    float s8[8];
                    row_dot_sub8<46, 8, false, 288>(wr, sx, off8, ne, 288, t_sub, s8);
                    if (t_sub < ne) dst[(size_t) (e + t_sub) * L.n_ff + r] = s8[t_sub];
                }
            }
        }
    }
}

template<bool DEP = false>
__device__ __forceinline__ void native_down_mxfp8_sx8_stage(dim3 block_idx, dim3 grid_dim, int tid,
                                                            const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_dst,
                                                            const block_q8_1* __restrict__ hq,
                                                            NativeExpertLayout L,
                                                            float* __restrict__ out) {
    if constexpr (DEP) {
        pdl_wait();
        pdl_launch_dependents();
    }
    const int ng = *n_groups;
    if ((int) block_idx.y >= ng) return;
    __shared__ __align__(16) uint32_t s_x[kFastMxfp8Cols * kFastDnWords];
    const block_q8_1* sx = reinterpret_cast<const block_q8_1*>(s_x);
    const int r_sub_off = tid >> 3, t_sub = tid & 7;
    const int row0 = (int) block_idx.x * kFastMoeRows;
    const int row = row0 + r_sub_off;
    const bool valid = row < (int) L.n_embd;
    const size_t w_off = valid ? L.down_off + (size_t) row * L.d_row : 0;
    for (int gi = (int) block_idx.y; gi < ng; gi += (int) grid_dim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[gi] + w_off;
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        for (int e = e0; e < e1; e += kFastMxfp8Cols) {
            const int ne = min(kFastMxfp8Cols, e1 - e);
            if (ne <= 2) {
                if (valid) {
                    if (ne == 1) {
                        const int off1[1] = { e * 36 };
                        float s1[1];
                        row_dot_sub8<46, 1, true, 144>(wr, hq, off1, 1, 144, t_sub, s1);
                        if (t_sub == 0) out[(size_t) ent_dst[e] * L.n_embd + row] = s1[0];
                    } else {
                        const int off2[2] = { e * 36, (e + 1) * 36 };
                        float s2[2];
                        row_dot_sub8<46, 2, true, 144>(wr, hq, off2, 2, 144, t_sub, s2);
                        if (t_sub < 2) out[(size_t) ent_dst[e + t_sub] * L.n_embd + row] = s2[t_sub];
                    }
                }
                continue;
            }
            __syncthreads();
            for (int idx = tid; idx < ne * kFastDnWords; idx += 256) {
                const int c = idx / kFastDnWords, rem = idx - c * kFastDnWords;
                s_x[idx] = reinterpret_cast<const uint32_t*>(hq + (size_t) (e + c) * 36)[rem];
            }
            __syncthreads();
            if (valid) {
                if (ne <= 4) {
                    int off4[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) off4[c] = min(c, ne - 1) * 36;
                    float s4[4];
                    row_dot_sub8<46, 4, false, 144>(wr, sx, off4, ne, 144, t_sub, s4);
                    if (t_sub < ne) out[(size_t) ent_dst[e + t_sub] * L.n_embd + row] = s4[t_sub];
                } else {
                    int off8[8];
#pragma unroll
                    for (int c = 0; c < 8; ++c) off8[c] = min(c, ne - 1) * 36;
                    float s8[8];
                    row_dot_sub8<46, 8, false, 144>(wr, sx, off8, ne, 144, t_sub, s8);
                    if (t_sub < ne) out[(size_t) ent_dst[e + t_sub] * L.n_embd + row] = s8[t_sub];
                }
            }
        }
    }
}

}  // namespace
