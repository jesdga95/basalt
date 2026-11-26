#pragma once
// Bitwise-identical high-concurrency grouped expert stage functions at 2304 x 1152: NVFP4 (type 40) streams its weights
// through shared memory and sums exact int32 sub-block dots on mma.m16n8k16.s32.s8.s8.s32 in row_dot_sub8's tree order;
// MXFP8 (type 46) stages up to 8 columns and runs row_dot_sub8 itself.

namespace {

constexpr int kFastMoeRows = 32;
constexpr int kFastMoeCols = 16;
constexpr int kFastMoeCTiles = kFastMoeCols / 8;
constexpr int kFastGuWords = 72 * 9;            // 648 words per 2304-dim q8_1 entry
constexpr int kFastDnWords = 36 * 9;            // 324 words per 1152-dim q8_1 entry

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

// The NVFP4 stages stream the experts' weights: the work is (group, 16-row tile of its gate and up rows, or of its down
// rows), group-major, and a block takes one contiguous run of it (the runs split it evenly), so it reads its weights as one
// stream. The tiles come into shared memory as bulk copies, NS in flight, the next one past them requested into L2. A
// thread holds the activation words and scales its k steps meet, for up to 16 of a group's entries, in registers: read once
// for all the group's tiles of the run. Each warp multiplies the tile's 16 rows against them over its share of K on
// mma.m16n8k16, and the block adds the warps' sums in row_dot_sub8's tree order: the arithmetic and its order are the
// per-tile stage's, on the same bytes, so every entry's bits are v1's at any row count.
constexpr int kNvSubRows = 16;
// bytes of a weight row of K NVFP4 values (blocks of 64 values in 36 bytes); a tile's bytes; tiles in flight; a warp's k16
// steps (k = wp + 8 j)
template<int K> constexpr int kNvRowBytes = K / 64 * 36;
template<int K> constexpr int kNvTileBytes = kNvSubRows * kNvRowBytes<K>;
template<bool GU> constexpr int kNvStages = GU ? 2 : 3;
template<int K> constexpr int kNvSteps = K / 128;
template<int K, bool GU> constexpr size_t kNvSmemBytes = (size_t) kNvStages<GU> * kNvTileBytes<K> + (size_t) kFastMoeCols * kNvSubRows * 9 * 4;
constexpr size_t kGuSmemBytes = kNvSmemBytes<2304, true>;
constexpr size_t kDnSmemBytes = kNvSmemBytes<1152, false>;

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

// A thread's activations for a chunk of up to 16 entries: per column tile ct and step j, the word of entry 8 ct + g it
// gives the B fragment and the scales of entries 8 ct + 2 t and + 1 (0 past the chunk).
template<int K>
struct NvCols {
    uint32_t b[kFastMoeCTiles][kNvSteps<K>];
    float d0[kFastMoeCTiles][kNvSteps<K>], d1[kFastMoeCTiles][kNvSteps<K>];
};
template<int K, bool GU>
__device__ __forceinline__ void nv_cols_load(NvCols<K>& x, const block_q8_1* __restrict__ xq, const int32_t* __restrict__ ent, int e,
                                             int ne, int wp, int g, int t) {
    const auto entry = [&](int c) {
        return reinterpret_cast<const uint32_t*>(xq + (GU ? (size_t) ent[e + c] : (size_t) (e + c)) * (K / 32));
    };
#pragma unroll
    for (int ct = 0; ct < kFastMoeCTiles; ++ct) {
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

// A warp's sums of a tile's rows g and g + 8 (wra, wrb) against CT column tiles of x: step j is k = wp + 8 j, summed in
// acc[.][j % 4] (the per-tile stage's k = wp + 8 s + 32 m).
template<int K, int CT>
__device__ __forceinline__ void nv_tile_mma(const uint8_t* wra, const uint8_t* wrb, const NvCols<K>& x, int wp, int t,
                                            float (&acc)[kFastMoeCTiles][4][4]) {
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
        for (int ct = 0; ct < CT; ++ct) {
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

// A call's groups at most (the stage's prefix of their chunks sits in shared memory)
constexpr int kNvMaxGroups = 256;

// GU: gate/up (K = n_embd 2304, rows 2 n_ff, entries' tokens `ent`, out0 gate and out1 up [entry][n_ff]); else down (K =
// n_ff 1152, rows n_embd, entries' destinations `ent`, out0 [dest][n_embd]). DEP: a programmatic dependent, its weights
// requested before the wait (they and the plan are ready before the kernel before it starts). A group of more than 16
// entries runs as one per 16 of them (chunk-major), each streaming the group's weights again, so a thread's activations
// change only when its run reaches another chunk.
template<int K, bool GU, bool DEP>
__device__ __forceinline__ void nvfp4_stream_stage(int block, int blocks, int tid, uint8_t* smem,
                                                   const unsigned long long* __restrict__ grp_ptr,
                                                   const int32_t* __restrict__ grp_start,
                                                   const int32_t* __restrict__ n_groups,
                                                   const int32_t* __restrict__ ent,
                                                   const block_q8_1* __restrict__ xq,
                                                   NativeExpertLayout L,
                                                   float* __restrict__ out0,
                                                   float* __restrict__ out1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int RB = kNvRowBytes<K>, TB = kNvTileBytes<K>, NS = kNvStages<GU>;
    if constexpr (!DEP) pdl_launch_dependents();
    __shared__ int s_pre[kNvMaxGroups + 1];   // the work items before each group: nx a chunk of 16 entries
    __shared__ int s_wsum[8];
    __shared__ __align__(8) unsigned long long bar[NS];
    const int ng = *n_groups;
    const int nx = (GU ? 2 * (int) L.n_ff : (int) L.n_embd) / kNvSubRows;
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    {   // the exclusive prefix of the groups' items, one group a thread
        int v = tid < ng ? (grp_start[tid + 1] - grp_start[tid] + kFastMoeCols - 1) / kFastMoeCols * nx : 0;
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
        e = grp_start[gi] + c * kFastMoeCols;
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
    NvCols<K> x;
    int held_e = -1, gi = 0;
    for (int i = i0, k = 0; i < i1; ++i, ++k) {
        int e, row0;
        locate(i, gi, e, row0);
        const bool is_up = GU && row0 >= (int) L.n_ff;
        const int r0 = is_up ? row0 - (int) L.n_ff : row0;
        float* dst = is_up ? out1 : out0;
        const float sc = nvfp4_tail((const void*) grp_ptr[gi], L)[GU ? (is_up ? 1 : 0) : 2];
        const int ne = min(kFastMoeCols, grp_start[gi + 1] - e);
        if (held_e != e) {   // entries are each one group's, so the first entry names the chunk
            nv_cols_load<K, GU>(x, xq, ent, e, ne, wp, g, t);
            held_e = e;
        }
        uint8_t* tile = smem + (k % NS) * TB;
        const uint8_t* wra = tile + (size_t) g * RB;
        const uint8_t* wrb = wra + 8 * RB;
        gu_mb_wait(&bar[k % NS], (unsigned) (k / NS) & 1u);
        float acc[kFastMoeCTiles][4][4] = {};
        if (ne <= 8) nv_tile_mma<K, 1>(wra, wrb, x, wp, t, acc);
        else nv_tile_mma<K, 2>(wra, wrb, x, wp, t, acc);
#pragma unroll
        for (int ct = 0; ct < kFastMoeCTiles; ++ct) {
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
            const float v = __fmul_rn(__fadd_rn(r2_0, r2_1), sc);
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
