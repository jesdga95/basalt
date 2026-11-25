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

__device__ __forceinline__ void native_gu_nvfp4_mma_stage(dim3 block_idx, dim3 grid_dim, int tid,
                                                          const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_tok,
                                                          const block_q8_1* __restrict__ xq,
                                                          NativeExpertLayout L,
                                                          float* __restrict__ gate,
                                                          float* __restrict__ up) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int ng = *n_groups;
    if (ng <= (int) grid_dim.y) pdl_launch_dependents();
    if ((int) block_idx.y >= ng) return;
    __shared__ __align__(16) uint32_t s_x[kFastMoeCols * kFastGuStride];
    float* s_part = reinterpret_cast<float*>(s_x);
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int r_sub_off = tid >> 3, t_sub = tid & 7;
    const int row0 = (int) block_idx.x * kFastMoeRows;
    const bool is_up = row0 >= (int) L.n_ff;
    const int r0 = is_up ? row0 - (int) L.n_ff : row0;
    const size_t half_off = is_up ? L.up_off : 0;
    const size_t base_off = half_off + (size_t) (r0 + g) * L.gu_row;
    const size_t sub_off = half_off + (size_t) (r0 + r_sub_off) * L.gu_row;
    float* dst = is_up ? up : gate;
    for (int gi = (int) block_idx.y; gi < ng; gi += (int) grid_dim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[gi];
        const float sc = nvfp4_tail((const void*) blob, L)[is_up ? 1 : 0];
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        const uint8_t* wra0 = blob + base_off;
        const uint8_t* wrb0 = wra0 + 8 * L.gu_row;
        const uint8_t* wra1 = wra0 + 16 * L.gu_row;
        const uint8_t* wrb1 = wra0 + 24 * L.gu_row;
        for (int e = e0; e < e1; e += kFastMoeCols) {
            const int ne = min(kFastMoeCols, e1 - e);
            if (ne <= 2) {
                const uint8_t* wr_sub = blob + sub_off;
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
    (void) block_idx; (void) grid_dim; (void) tid;
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_tok; (void) xq; (void) L; (void) gate; (void) up;
    __trap();
#endif
}

template<bool DEP = false>
__device__ __forceinline__ void native_down_nvfp4_mma_stage(dim3 block_idx, dim3 grid_dim, int tid,
                                                            const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_dst,
                                                            const block_q8_1* __restrict__ hq,
                                                            NativeExpertLayout L,
                                                            float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (DEP) {
        pdl_wait();
        pdl_launch_dependents();
    }
    const int ng = *n_groups;
    if ((int) block_idx.y >= ng) return;
    __shared__ __align__(16) uint32_t s_x[kFastMoeCols * kFastDnStride];
    float* s_part = reinterpret_cast<float*>(s_x);
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int r_sub_off = tid >> 3, t_sub = tid & 7;
    const int row0 = (int) block_idx.x * kFastMoeRows;
    const size_t base_off = L.down_off + (size_t) (row0 + g) * L.d_row;
    const size_t sub_off = L.down_off + (size_t) (row0 + r_sub_off) * L.d_row;
    for (int gi = (int) block_idx.y; gi < ng; gi += (int) grid_dim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[gi];
        const float sc = nvfp4_tail((const void*) blob, L)[2];
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        const uint8_t* wra0 = blob + base_off;
        const uint8_t* wrb0 = wra0 + 8 * L.d_row;
        const uint8_t* wra1 = wra0 + 16 * L.d_row;
        const uint8_t* wrb1 = wra0 + 24 * L.d_row;
        for (int e = e0; e < e1; e += kFastMoeCols) {
            const int ne = min(kFastMoeCols, e1 - e);
            if (ne <= 2) {
                const uint8_t* wr_sub = blob + sub_off;
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
    (void) block_idx; (void) grid_dim; (void) tid;
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_dst; (void) hq; (void) L; (void) out;
    __trap();
#endif
}

}  // namespace
