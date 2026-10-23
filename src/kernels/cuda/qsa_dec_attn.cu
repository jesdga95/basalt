// src/kernels/cuda/qsa_dec_attn.cu - see include/basalt/kernels/qsa_dec_attn.hpp.
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "smem_opt_in.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

// `ushort` comes from the POSIX sys/types.h where there is one; redeclaring the same type is harmless.
using ushort = unsigned short;

namespace basalt::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head
constexpr int CHUNK = 64;        // cells per block (the reference's chunk: the partial sums depend on it)
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int NSC = HD / KV_Q8_GROUP;   // int8 scales per row
// The draft kernel's cells per CTA, from the window's width: a narrow window gets more CTAs of one tile, a wide one fewer
// of up to four (the merge reads the same rule).
constexpr int DT = 64;
__device__ __forceinline__ int draft_tpc(int n_ids) {
    const int tiles = (n_ids + DT - 1) / DT;
    return tiles >= 256 ? 4 : tiles >= 96 ? 2 : 1;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// The 12 heads' lane sums (part[12..15] zero) reduce-scattered with warp_sum's pairing (xor 16, 8, 4, 2, 1): every
// output adds the same two operands at every level, so each sum is warp_sum's bit for bit.  Lane l ends holding head
// head_of_lane(l); lanes l and l^1 agree.
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::); }

// shared memory of one chunk block; KV_MODE 0: fp16 rows, 1: int8 rows + fp16 scales
template <int KV_MODE> struct ChunkSmem {
    static constexpr int ROW = KV_MODE == 0 ? HD * 2 : HD;   // bytes of one cell's K (or V) row
    alignas(16) unsigned char k[CHUNK][ROW];
    alignas(16) unsigned char v[CHUNK][ROW];
    alignas(16) uint16_t ks[CHUNK][NSC];
    alignas(16) uint16_t vs[CHUNK][NSC];
    float sp[G][CHUNK];        // scores
    alignas(16) float spt[CHUNK][G];   // probabilities, cell-major
    long long row[CHUNK];      // pool row of each cell, -1 masked or past the width
};

struct AttnArgs {
    const float* q;
    QsaAttnPools p;
    QsaSegTable tab;           // SEG: each query's pools come from its slot
    const int32_t* ids;
    const int32_t* steps;
    int n_kv_heads, page_size, n_chunks, cap;
    int chunk = CHUNK;         // cells per partial: CHUNK, or 0 for the draft kernel's (draft_tpc of the width)
    long long stride;          // scratch floats per query
    float scale;
    float* scratch;
    float* attn;
    const float* q_full;       // non-null: the gated output too
    float* attn_gated;
    void* xq;                  // non-null (with the gated output): the gated rows' q8_1 blocks
};

// k8 of the row's dimensions [d0, d0+8) as the reference's load8 computes them
template <int KV_MODE>
__device__ __forceinline__ void k8_from(const ChunkSmem<KV_MODE>& sm, int c, int d0, float* out) {
    if constexpr (KV_MODE == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(&sm.k[c][d0 * 2]);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else {
        const float sc = __half2float(__ushort_as_half(sm.ks[c][d0 / KV_Q8_GROUP]));
        const uint2 raw = *reinterpret_cast<const uint2*>(&sm.k[c][d0]);
        const int8_t* cc = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) cc[j] * sc;
    }
}
template <int KV_MODE>
__device__ __forceinline__ float v1_from(const ChunkSmem<KV_MODE>& sm, int c, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(*reinterpret_cast<const __half*>(&sm.v[c][d * 2]));
    } else {
        const float sc = __half2float(__ushort_as_half(sm.vs[c][d / KV_Q8_GROUP]));
        return (float) (int8_t) sm.v[c][d] * sc;
    }
}

// One chunk of one query and KV head: partial acc / max / sum for the 12 query heads, as attn_chunk_kernel.
template <int KV_MODE, bool SEG>
__device__ __forceinline__ void attn_chunk_dev(const AttnArgs& a, int chunk, int kvh, int qi, ChunkSmem<KV_MODE>& sm) {
    const QsaAttnPools pools = SEG ? a.tab.slots[a.tab.rows[qi].slot].pools() : a.p;
    constexpr int ROW = ChunkSmem<KV_MODE>::ROW;
    constexpr int PER = ROW / 16;
    const float* q = a.q + (size_t) qi * (size_t) (a.n_kv_heads * G) * HD;
    const int32_t* ids = a.ids + (size_t) qi * (size_t) a.cap;
    const int32_t* step = a.steps + (size_t) qi * kStepCount;
    float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    // this block's cell ids are requested before the width is known (ids past the width are read in bounds and
    // unused), then the query heads' registers (8 of the 256 dimensions per lane)
    const int cell = (t < CHUNK && c0 + t < a.cap) ? ids[c0 + t] : 0;
    const int n_ids = __ldg(step + kStepWidth);
    float qr[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float4 qa = __ldg(reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + lane * 8));
        const float4 qb = __ldg(reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + lane * 8 + 4));
        qr[h][0] = qa.x; qr[h][1] = qa.y; qr[h][2] = qa.z; qr[h][3] = qa.w;
        qr[h][4] = qb.x; qr[h][5] = qb.y; qr[h][6] = qb.z; qr[h][7] = qb.w;
    }
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) return;   // the merge reads only the chunks the width reaches
    const int slot = kvh * a.n_chunks + chunk;
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const long long page = (long long) pools.page_table[cell / a.page_size];
            if (page >= 0) r = (page * a.n_kv_heads + kvh) * a.page_size + (cell % a.page_size);
        }
        sm.row[t] = r;
    }
    __syncthreads();
    {
        const unsigned char* kb = KV_MODE == 0 ? (const unsigned char*) pools.k_pool : (const unsigned char*) pools.k_q;
        const unsigned char* vb = KV_MODE == 0 ? (const unsigned char*) pools.v_pool : (const unsigned char*) pools.v_q;
        for (int i = t; i < CHUNK * PER; i += THREADS) {
            const int c = i / PER, j = i % PER;
            const long long r = sm.row[c];
            if (r < 0) {   // a masked or missing cell's values are zero, so the value loop needs no branch
                *reinterpret_cast<uint4*>(&sm.v[c][j * 16]) = make_uint4(0, 0, 0, 0);
                continue;
            }
            cp_async16(&sm.k[c][j * 16], kb + r * ROW + j * 16);
            cp_async16(&sm.v[c][j * 16], vb + r * ROW + j * 16);
        }
        if constexpr (KV_MODE == 1) {
            if (t < 2 * CHUNK) {
                const int c = t & (CHUNK - 1);
                const long long r = sm.row[c];
                if (r >= 0) {
                    if (t < CHUNK) cp_async8(&sm.ks[c][0], pools.k_scale + r * NSC);
                    else cp_async8(&sm.vs[c][0], pools.v_scale + r * NSC);
                } else if (t >= CHUNK) {
                    *reinterpret_cast<uint2*>(&sm.vs[c][0]) = make_uint2(0, 0);
                }
            }
        }
    }
    cp_async_wait_all();
    __syncthreads();
    // scores: warp w takes cells w, w+8, ...; two cells per pass
    const int hd = head_of_lane(lane);
    constexpr int NC = 2;
#pragma unroll 1
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool ok[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            ok[j] = cc[j] < n_here && sm.row[cc[j]] >= 0;
        }
        float kk[NC][8];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (ok[j]) k8_from<KV_MODE>(sm, cc[j], lane * 8, kk[j]);
            else {
#pragma unroll
                for (int e = 0; e < 8; ++e) kk[j][e] = 0.0f;
            }
        }
        float pp[NC][16];
#pragma unroll
        for (int h = 0; h < G; ++h) {
#pragma unroll
            for (int j = 0; j < NC; ++j)
                pp[j][h] = kk[j][0] * qr[h][0] + kk[j][1] * qr[h][1] + kk[j][2] * qr[h][2] + kk[j][3] * qr[h][3] +
                           kk[j][4] * qr[h][4] + kk[j][5] * qr[h][5] + kk[j][6] * qr[h][6] + kk[j][7] * qr[h][7];
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
            const float sj = reduce12(pp[j], lane);
            if (ok[j]) {
                if ((lane & 1) == 0 && hd < G) sm.sp[hd][cc[j]] = sj * a.scale;
            } else if (lane < G) {
                sm.sp[lane][cc[j]] = -FLT_MAX;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8
    {
        const bool ok0 = lane < n_here && sm.row[lane] >= 0, ok1 = lane + 32 < n_here && sm.row[lane + 32] >= 0;
        const int h0 = warp, h1 = warp + WARPS;   // heads w and w+8 (the second exists for w < 4)
        const bool two = h1 < G;
        const float x0 = sm.sp[h0][lane], x1 = sm.sp[h0][lane + 32];
        const float y0 = two ? sm.sp[h1][lane] : 0.0f, y1 = two ? sm.sp[h1][lane + 32] : 0.0f;
        const float m = warp_max(fmaxf(x0, x1));
        const float n = warp_max(fmaxf(y0, y1));
        const float e0 = ok0 ? __expf(x0 - m) : 0.0f, e1 = ok1 ? __expf(x1 - m) : 0.0f;
        const float f0 = ok0 ? __expf(y0 - n) : 0.0f, f1 = ok1 ? __expf(y1 - n) : 0.0f;
        sm.spt[lane][h0] = e0;
        sm.spt[lane + 32][h0] = e1;
        if (two) {
            sm.spt[lane][h1] = f0;
            sm.spt[lane + 32][h1] = f1;
        }
        const float l = warp_sum(e0 + e1);
        const float l2 = warp_sum(f0 + f1);
        if (lane == 0) {
            part_m[slot * G + h0] = m; part_l[slot * G + h0] = l;
            if (two) { part_m[slot * G + h1] = n; part_l[slot * G + h1] = l2; }
        }
    }
    __syncthreads();
    // values: thread t < 128 owns dimensions 2t, 2t+1 of all 12 heads; cells folded in ascending order.  A masked or
    // missing cell has weight 0 and value 0, which leaves every accumulator unchanged (none is ever -0).
    if (t >= HD / 2) return;
    float acc[G][2];
#pragma unroll
    for (int i = 0; i < G; ++i) acc[i][0] = acc[i][1] = 0.0f;
#pragma unroll 8
    for (int c = 0; c < CHUNK; ++c) {
        float v0, v1;
        if constexpr (KV_MODE == 0) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&sm.v[c][t * 4]));
            v0 = f.x; v1 = f.y;
        } else {
            const float sc = __half2float(__ushort_as_half(sm.vs[c][t / (KV_Q8_GROUP / 2)]));
            const ushort raw = *reinterpret_cast<const ushort*>(&sm.v[c][t * 2]);
            v0 = (float) (int8_t) (raw & 0xff) * sc;
            v1 = (float) (int8_t) (raw >> 8) * sc;
        }
        const float4 w0 = *reinterpret_cast<const float4*>(&sm.spt[c][0]);
        const float4 w1 = *reinterpret_cast<const float4*>(&sm.spt[c][4]);
        const float4 w2 = *reinterpret_cast<const float4*>(&sm.spt[c][8]);
        const float w[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
        for (int i = 0; i < G; ++i) {
            acc[i][0] = fmaf(w[i], v0, acc[i][0]);
            acc[i][1] = fmaf(w[i], v1, acc[i][1]);
        }
    }
#pragma unroll
    for (int i = 0; i < G; ++i)
        *reinterpret_cast<float2*>(&part_acc[((size_t) slot * G + i) * HD + 2 * t]) = make_float2(acc[i][0], acc[i][1]);
}

// sigmoid gate with the fast-math sequence native_qsa_gate_apply compiles to
__device__ __forceinline__ float gate_fast(float x, float raw) {
    float y;
    asm("{\n\t.reg .f32 t, e, d, r;\n\t"
        "mul.ftz.f32 t, %2, 0fBFB8AA3B;\n\t"
        "ex2.approx.ftz.f32 e, t;\n\t"
        "add.ftz.f32 d, e, 0f3F800000;\n\t"
        "rcp.approx.ftz.f32 r, d;\n\t"
        "mul.ftz.f32 %0, %1, r;\n\t}"
        : "=f"(y) : "f"(x), "f"(raw));
    return y;
}

// native_quantize_q8_1_kernel's block (fast-math operations spelled out): lane l holds element l of the block
struct AttnQ81 {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ float fm_max(float a, float b) { float y; asm("max.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fm_abs(float a) { float y; asm("abs.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fm_add(float a, float b) { float y; asm("add.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fm_div(float a, float b) { float y; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ void q8_1_store_fm(float xi, AttnQ81* y, size_t ib, int lane) {
    float amax = fm_abs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fm_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = fm_add(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    float d = fm_div(amax, 127.0f);
    d = q8_1_finite(d);
    int8_t q = 0;
    if (amax != 0.0f) {
        const float r = roundf(fm_div(xi, d));
        q = (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
    }
    y[ib].qs[lane] = q;
    if (lane == 0) y[ib].ds = q8_1_ds(d, sum);
}

// One query head of one query: merge the chunks the width reaches, as attn_merge_kernel (which skips the rest).
__device__ __forceinline__ void attn_merge_dev(const AttnArgs& a, int h, int qi, float* s_w, float* s_l, float* s_red) {
    const float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    const float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    const float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int n_ids = __ldg(a.steps + (size_t) qi * kStepCount + kStepWidth);
    const int chunk_cells = a.chunk > 0 ? a.chunk : DT * draft_tpc(n_ids);
    const int nc = min(a.n_chunks, (n_ids + chunk_cells - 1) / chunk_cells);
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float mx = -FLT_MAX;
    for (int c = d; c < nc; c += THREADS) {
        const int slot = kvh * a.n_chunks + c;
        const float m = part_m[slot * G + hl];
        s_w[c] = m;
        s_l[c] = part_l[slot * G + hl];
        mx = fmaxf(mx, m);
    }
    // the chunks' partial sums are requested before the maxima are known
    constexpr int PRE = 32;
    const float* pa = part_acc + ((size_t) (kvh * a.n_chunks) * G + hl) * HD + d;
    const size_t cs = (size_t) G * HD;
    float xp[PRE];
#pragma unroll
    for (int j = 0; j < PRE; ++j) xp[j] = j < nc ? pa[(size_t) j * cs] : 0.0f;
    mx = warp_max(mx);
    if ((d & 31) == 0) s_red[d >> 5] = mx;
    __syncthreads();
    float M = s_red[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) M = fmaxf(M, s_red[w]);
    for (int c = d; c < nc; c += THREADS) {
        const float m = s_w[c];
        s_w[c] = m == -FLT_MAX ? -1.0f : __expf(m - M);   // -1: a chunk the reference skips
    }
    __syncthreads();
    float L = 0.0f, acc = 0.0f;
    int c = 0;
#pragma unroll
    for (int j = 0; j < PRE; ++j) {
        if (j >= nc) break;
        const float w = s_w[j];
        if (w < 0.0f) continue;
        L = fmaf(s_l[j], w, L);
        acc = fmaf(xp[j], w, acc);
    }
    c = min(nc, PRE);
    for (; c + 8 <= nc; c += 8) {
        float x[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = pa[(size_t) (c + j) * cs];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float w = s_w[c + j];
            if (w < 0.0f) continue;
            L = fmaf(s_l[c + j], w, L);
            acc = fmaf(x[j], w, acc);
        }
    }
    for (; c < nc; ++c) {
        const float w = s_w[c];
        if (w < 0.0f) continue;
        L = fmaf(s_l[c], w, L);
        acc = fmaf(pa[(size_t) c * cs], w, acc);
    }
    const int n_head = a.n_kv_heads * G;
    const float o = L > 0.0f ? acc / L : 0.0f;
    a.attn[((size_t) qi * n_head + h) * HD + d] = o;
    if (a.attn_gated != nullptr) {
        const float raw = a.q_full[((size_t) qi * n_head + h) * 2 * HD + HD + d];
        const float gv = gate_fast(o, raw);
        a.attn_gated[((size_t) qi * n_head + h) * HD + d] = gv;
        if (a.xq != nullptr) q8_1_store_fm(gv, static_cast<AttnQ81*>(a.xq), (((size_t) qi * n_head + h) * HD + d) / 32, d & 31);
    }
}


// ---- the draft layer's attention over a wide window of one row: tensor cores, fp16 products with fp32 accumulation,
// the softmax online over 64-cell tiles, 256 cells per CTA.  The draft only proposes tokens (verify decides every
// one), so this need not be the reference's arithmetic; the partials have the layout the merge reads, with chunks
// of DCH cells.
constexpr int DCH = 256;   // the most cells one CTA takes
constexpr int DTHREADS = 256;
constexpr int DROW = HD + 16;   // smem row bytes: fragment loads from 8 cells land in distinct banks
struct DSmem {
    alignas(16) unsigned char k[2][DT][DROW];
    alignas(16) unsigned char v[2][DT][DROW];
    alignas(16) uint16_t ks[2][DT][NSC];
    alignas(16) uint16_t vs[2][DT][NSC];
    alignas(16) float s[16][DT + 4];
    alignas(16) __half p[16][DT + 8];
    float alpha[16];
    int row[DCH];
};

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void mma_16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__global__ void __launch_bounds__(DTHREADS) draft_attn_chunk(AttnArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    DSmem& sm = *reinterpret_cast<DSmem*>(smem_raw);
    pdl_launch_dependents();
    pdl_wait();
    const int chunk = blockIdx.x, kvh = blockIdx.y, qi = blockIdx.z;
    const QsaAttnPools& pools = a.p;
    const float* q = a.q + (size_t) qi * (size_t) (a.n_kv_heads * G) * HD + (size_t) kvh * G * HD;
    const int32_t* ids = a.ids + (size_t) qi * (size_t) a.cap;
    const int32_t* step = a.steps + (size_t) qi * kStepCount;
    float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, c4 = lane & 3, g8 = lane >> 2;
    const int n_ids = __ldg(step + kStepWidth);
    const int c0 = chunk * (DT * draft_tpc(n_ids));
    const int n_here = min(DT * draft_tpc(n_ids), n_ids - c0);
    if (n_here <= 0) return;
    const int slot = kvh * a.n_chunks + chunk;
    for (int i = t; i < DCH; i += DTHREADS) {
        int r = -1;
        if (i < n_here) {
            const int cell = ids[c0 + i];
            const long long page = (long long) pools.page_table[cell / a.page_size];
            if (page >= 0) r = (int) ((page * a.n_kv_heads + kvh) * a.page_size + (cell % a.page_size));
        }
        sm.row[i] = r;
    }
    for (int i = t; i < 16 * (DT + 8); i += DTHREADS) (&sm.p[0][0])[i] = __float2half(0.0f);
    __syncthreads();
    const unsigned char* kb = (const unsigned char*) pools.k_q;
    const unsigned char* vb = (const unsigned char*) pools.v_q;
    auto load_tile = [&](int ti, int buf) {
        for (int i = t; i < DT * (HD / 16); i += DTHREADS) {
            const int c = i / (HD / 16), j = i % (HD / 16);
            const int r = sm.row[ti * DT + c];
            if (r < 0) {
                *reinterpret_cast<uint4*>(&sm.v[buf][c][j * 16]) = make_uint4(0, 0, 0, 0);
                continue;
            }
            cp_async16(&sm.k[buf][c][j * 16], kb + (size_t) r * HD + j * 16);
            cp_async16(&sm.v[buf][c][j * 16], vb + (size_t) r * HD + j * 16);
        }
        if (t < 2 * DT) {
            const int c = t & (DT - 1);
            const int r = sm.row[ti * DT + c];
            if (r >= 0) {
                if (t < DT) cp_async8(&sm.ks[buf][c][0], pools.k_scale + (size_t) r * NSC);
                else cp_async8(&sm.vs[buf][c][0], pools.v_scale + (size_t) r * NSC);
            } else if (t >= DT) {
                *reinterpret_cast<uint2*>(&sm.vs[buf][c][0]) = make_uint2(0, 0);
            }
        }
    };
    // the query rows as the A operand: 12 heads, rows 12..15 zero
    uint32_t qa[HD / 16][4];
#pragma unroll
    for (int s = 0; s < HD / 16; ++s) {
        const int d0 = 16 * s + 2 * c4;
        const float2 x0 = *reinterpret_cast<const float2*>(q + (size_t) g8 * HD + d0);
        const float2 x2 = *reinterpret_cast<const float2*>(q + (size_t) g8 * HD + d0 + 8);
        float2 x1 = make_float2(0.0f, 0.0f), x3 = x1;
        if (g8 + 8 < G) {
            x1 = *reinterpret_cast<const float2*>(q + (size_t) (g8 + 8) * HD + d0);
            x3 = *reinterpret_cast<const float2*>(q + (size_t) (g8 + 8) * HD + d0 + 8);
        }
        qa[s][0] = pack_h2(x0.x, x0.y); qa[s][1] = pack_h2(x1.x, x1.y);
        qa[s][2] = pack_h2(x2.x, x2.y); qa[s][3] = pack_h2(x3.x, x3.y);
    }
    float o[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m_run[2], l_run[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) { m_run[j] = -FLT_MAX; l_run[j] = 0.0f; }
    const int ntiles = (n_here + DT - 1) / DT;
    load_tile(0, 0);
    cp_async_commit();
    for (int ti = 0; ti < ntiles; ++ti) {
        const int buf = ti & 1;
        if (ti + 1 < ntiles) {
            load_tile(ti + 1, buf ^ 1);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const int base = ti * DT;
        // scores: this warp's 8 cells
        float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        {
            const int cell = 8 * warp + g8;
            const unsigned char* kr = sm.k[buf][cell];
#pragma unroll
            for (int s = 0; s < HD / 16; ++s) {
                const float scl = __half2float(__ushort_as_half(sm.ks[buf][cell][s / 4]));
                const ushort w0 = *reinterpret_cast<const ushort*>(kr + 16 * s + 2 * c4);
                const ushort w1 = *reinterpret_cast<const ushort*>(kr + 16 * s + 2 * c4 + 8);
                const uint32_t b0 = pack_h2((float) (int8_t) (w0 & 0xff) * scl, (float) (int8_t) (w0 >> 8) * scl);
                const uint32_t b1 = pack_h2((float) (int8_t) (w1 & 0xff) * scl, (float) (int8_t) (w1 >> 8) * scl);
                mma_16816(sc, qa[s], b0, b1);
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = g8 + ((e >> 1) << 3), col = 8 * warp + 2 * c4 + (e & 1);
            if (row < G) {
                const bool valid = base + col < n_here && sm.row[base + col] >= 0;
                sm.s[row][col] = valid ? sc[e] * a.scale : -FLT_MAX;
            }
        }
        __syncthreads();
        // online softmax: warp w owns heads w and w + 8 (the second only for w < 4)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int h = warp + 8 * j;
            if (h >= G) continue;
            const float x0 = sm.s[h][lane], x1 = sm.s[h][lane + 32];
            const bool v0 = base + lane < n_here && sm.row[base + lane] >= 0;
            const bool v1 = base + lane + 32 < n_here && sm.row[base + lane + 32] >= 0;
            const float m_new = fmaxf(m_run[j], warp_max(fmaxf(x0, x1)));
            const float p0 = v0 ? __expf(x0 - m_new) : 0.0f, p1 = v1 ? __expf(x1 - m_new) : 0.0f;
            const float alpha = __expf(m_run[j] - m_new);
            l_run[j] = l_run[j] * alpha + warp_sum(p0 + p1);
            m_run[j] = m_new;
            sm.p[h][lane] = __float2half(p0);
            sm.p[h][lane + 32] = __float2half(p1);
            if (lane == 0) sm.alpha[h] = alpha;
        }
        __syncthreads();
        const float alA = sm.alpha[g8], alB = g8 + 8 < G ? sm.alpha[g8 + 8] : 1.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            o[nt][0] *= alA; o[nt][1] *= alA; o[nt][2] *= alB; o[nt][3] *= alB;
        }
        // P . V: this warp's 64 dimensions, 4 k-steps of 16 cells
#pragma unroll
        for (int ks = 0; ks < DT / 16; ++ks) {
            uint32_t pa[4];
            pa[0] = *reinterpret_cast<const uint32_t*>(&sm.p[g8][16 * ks + 2 * c4]);
            pa[1] = *reinterpret_cast<const uint32_t*>(&sm.p[g8 + 8][16 * ks + 2 * c4]);
            pa[2] = *reinterpret_cast<const uint32_t*>(&sm.p[g8][16 * ks + 2 * c4 + 8]);
            pa[3] = *reinterpret_cast<const uint32_t*>(&sm.p[g8 + 8][16 * ks + 2 * c4 + 8]);
            const int ca = 16 * ks + 2 * c4;
            const float vs0 = __half2float(__ushort_as_half(sm.vs[buf][ca][warp >> 1])),
                        vs1 = __half2float(__ushort_as_half(sm.vs[buf][ca + 1][warp >> 1])),
                        vs2 = __half2float(__ushort_as_half(sm.vs[buf][ca + 8][warp >> 1])),
                        vs3 = __half2float(__ushort_as_half(sm.vs[buf][ca + 9][warp >> 1]));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int dim = 32 * warp + 8 * nt + g8;
                const float v0 = (float) (int8_t) sm.v[buf][ca][dim] * vs0, v1 = (float) (int8_t) sm.v[buf][ca + 1][dim] * vs1,
                            v2 = (float) (int8_t) sm.v[buf][ca + 8][dim] * vs2, v3 = (float) (int8_t) sm.v[buf][ca + 9][dim] * vs3;
                mma_16816(o[nt], pa, pack_h2(v0, v1), pack_h2(v2, v3));
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (warp + 8 * j < G) {
                part_m[slot * G + warp + 8 * j] = m_run[j];
                part_l[slot * G + warp + 8 * j] = l_run[j];
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int dim = 32 * warp + 8 * nt + 2 * c4;
        *reinterpret_cast<float2*>(&part_acc[((size_t) slot * G + g8) * HD + dim]) = make_float2(o[nt][0], o[nt][1]);
        if (g8 + 8 < G)
            *reinterpret_cast<float2*>(&part_acc[((size_t) slot * G + g8 + 8) * HD + dim]) = make_float2(o[nt][2], o[nt][3]);
    }
}


// The draft kernel's merge: four chunk groups per head work in parallel (thread: 4 dimensions of one group), then the
// groups are added.  The order differs from attn_merge_dev's, which only the draft layer sees.
__global__ void __launch_bounds__(THREADS) draft_merge(AttnArgs a) {
    extern __shared__ __align__(16) float msm[];
    float* s_w = msm;
    float* s_l = msm + a.n_chunks;
    __shared__ float s_red[WARPS], s_L[4];
    __shared__ __align__(16) float sacc[4][HD];
    pdl_launch_dependents();
    pdl_wait();
    const int h = blockIdx.x, qi = blockIdx.y;
    const float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    const float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    const float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int n_ids = __ldg(a.steps + (size_t) qi * kStepCount + kStepWidth);
    const int chunk_cells = DT * draft_tpc(n_ids);
    const int nc = min(a.n_chunks, (n_ids + chunk_cells - 1) / chunk_cells);
    const int kvh = h / G, hl = h % G;
    const int t = threadIdx.x;
    float mx = -FLT_MAX;
    for (int c = t; c < nc; c += THREADS) {
        const int slot = kvh * a.n_chunks + c;
        const float m = part_m[slot * G + hl];
        s_w[c] = m;
        s_l[c] = part_l[slot * G + hl];
        mx = fmaxf(mx, m);
    }
    mx = warp_max(mx);
    if ((t & 31) == 0) s_red[t >> 5] = mx;
    __syncthreads();
    float M = s_red[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) M = fmaxf(M, s_red[w]);
    for (int c = t; c < nc; c += THREADS) {
        const float m = s_w[c];
        s_w[c] = m == -FLT_MAX ? -1.0f : __expf(m - M);
    }
    __syncthreads();
    const int dg = t & 63, cg = t >> 6;
    const size_t cs = (size_t) G * HD;
    const float* pa = part_acc + ((size_t) (kvh * a.n_chunks) * G + hl) * HD + 4 * dg;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float L = 0.0f;
#pragma unroll 8
    for (int c = cg; c < nc; c += 4) {
        const float w = s_w[c];
        if (w < 0.0f) continue;
        const float4 v = *reinterpret_cast<const float4*>(pa + (size_t) c * cs);
        acc.x = fmaf(v.x, w, acc.x); acc.y = fmaf(v.y, w, acc.y); acc.z = fmaf(v.z, w, acc.z); acc.w = fmaf(v.w, w, acc.w);
        L = fmaf(s_l[c], w, L);
    }
    *reinterpret_cast<float4*>(&sacc[cg][4 * dg]) = acc;
    if (dg == 0) s_L[cg] = L;
    __syncthreads();
    const float Lt = s_L[0] + s_L[1] + s_L[2] + s_L[3];
    const float tot = (sacc[0][t] + sacc[1][t]) + (sacc[2][t] + sacc[3][t]);
    const int n_head = a.n_kv_heads * G;
    const float o = Lt > 0.0f ? tot / Lt : 0.0f;
    a.attn[((size_t) qi * n_head + h) * HD + t] = o;
    if (a.attn_gated != nullptr) {
        const float raw = a.q_full[((size_t) qi * n_head + h) * 2 * HD + HD + t];
        const float gv = gate_fast(o, raw);
        a.attn_gated[((size_t) qi * n_head + h) * HD + t] = gv;
        if (a.xq != nullptr) q8_1_store_fm(gv, static_cast<AttnQ81*>(a.xq), (((size_t) qi * n_head + h) * HD + t) / 32, t & 31);
    }
}

// the int8 rows' block is small enough for two per multiprocessor: a decode window's chunk blocks (3 queries x 2 KV heads x
// 33 chunks) then all run at once
template <int KV_MODE, bool SEG>
__global__ void __launch_bounds__(THREADS, KV_MODE == 1 ? 2 : 1) attn_chunk_v2(AttnArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    pdl_launch_dependents();
    pdl_wait();
    attn_chunk_dev<KV_MODE, SEG>(a, blockIdx.x, blockIdx.y, blockIdx.z, *reinterpret_cast<ChunkSmem<KV_MODE>*>(smem_raw));
}

__global__ void __launch_bounds__(THREADS) attn_merge_v2(AttnArgs a) {
    extern __shared__ __align__(16) float msm[];
    pdl_launch_dependents();
    pdl_wait();
    attn_merge_dev(a, blockIdx.x, blockIdx.y, msm, msm + a.n_chunks, msm + 2 * a.n_chunks);
}

bool a16(const void* p) { return ((uintptr_t) p & 15) == 0; }
int kv_mode_of(const QsaAttnPools& p) {
    if (p.k_q4 != nullptr || p.v_q4 != nullptr) return -1;
    if (p.k_q != nullptr)
        return (p.v_q != nullptr && p.k_scale != nullptr && p.v_scale != nullptr && a16(p.k_q) && a16(p.v_q) &&
                ((uintptr_t) p.k_scale & 7) == 0 && ((uintptr_t) p.v_scale & 7) == 0) ? 1 : -1;
    return (p.k_pool != nullptr && p.v_pool != nullptr && a16(p.k_pool) && a16(p.v_pool)) ? 0 : -1;
}
// the dynamic shared memory opt-in, once per device
template <int MODE> void opt_in(void (*kernel)(AttnArgs), size_t bytes) {
    static std::atomic<bool> done[64];
    const int dev = opt_in_device("qsa_dec_attn");
    if (!done[dev].load(std::memory_order_acquire)) {
        opt_in_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes), "qsa_dec_attn");
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);   // a hint
        done[dev].store(true, std::memory_order_release);
    }
}

}  // namespace

bool qsa_dec_attn_supported(const QsaAttnPools& pools, const QsaShapes& s) {
    return s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv && kv_mode_of(pools) >= 0 &&
           pools.page_table != nullptr;
}

void qsa_dec_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream, const float* q_full,
                  float* attn_gated, void* xq) {
    if (n_q <= 0) return;
    if (!qsa_dec_attn_supported(pools, s) || cap <= 0 || !scratch || !ids || !steps || n_q > 65535) {
        basalt::log::error("basalt", "qsa_dec_attn: unsupported geometry, pools or buffers");
        std::exit(1);
    }
    const int mode = kv_mode_of(pools);
    AttnArgs a;
    a.q = q; a.p = pools; a.ids = ids; a.steps = steps;
    a.n_kv_heads = (int) s.n_head_kv; a.page_size = (int) s.page_size;
    a.n_chunks = (int) ((cap + CHUNK - 1) / CHUNK); a.cap = (int) cap;
    a.stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    a.scale = 1.0f / sqrtf((float) HD);
    a.scratch = scratch; a.attn = attn;
    a.q_full = attn_gated != nullptr ? q_full : nullptr;
    a.attn_gated = q_full != nullptr ? attn_gated : nullptr;
    cudaStream_t st = (cudaStream_t) stream;
    a.xq = a.attn_gated != nullptr ? xq : nullptr;
    const dim3 grid((unsigned) a.n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    if (mode == 1 && n_q == 1 && cap >= 8192) {   // the draft layer's wide window (verify's cap is a few thousand cells)
        a.chunk = 0;
        constexpr size_t bytes = sizeof(DSmem);
        static std::atomic<bool> done[64];
        const int dev = opt_in_device("draft_attn");
        if (!done[dev].load(std::memory_order_acquire)) {
            opt_in_check(cudaFuncSetAttribute(draft_attn_chunk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes),
                         "draft_attn");
            cudaFuncSetAttribute(draft_attn_chunk, cudaFuncAttributePreferredSharedMemoryCarveout, 100);   // a hint
            done[dev].store(true, std::memory_order_release);
        }
        launch_pdl(draft_attn_chunk, dim3((unsigned) a.n_chunks, (unsigned) s.n_head_kv, 1u), dim3(DTHREADS), bytes, st, a);
    } else if (mode == 1) {
        constexpr size_t bytes = sizeof(ChunkSmem<1>);
        opt_in<1>(attn_chunk_v2<1, false>, bytes);
        launch_pdl(attn_chunk_v2<1, false>, grid, dim3(THREADS), bytes, st, a);
    } else {
        constexpr size_t bytes = sizeof(ChunkSmem<0>);
        opt_in<0>(attn_chunk_v2<0, false>, bytes);
        launch_pdl(attn_chunk_v2<0, false>, grid, dim3(THREADS), bytes, st, a);
    }
    const size_t mbytes = (size_t) (2 * a.n_chunks + WARPS) * sizeof(float);
    if (a.chunk == 0) launch_pdl(draft_merge, dim3((unsigned) s.n_head, (unsigned) n_q), dim3(THREADS), mbytes, st, a);
    else launch_pdl(attn_merge_v2, dim3((unsigned) s.n_head, (unsigned) n_q), dim3(THREADS), mbytes, st, a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "qsa_dec_attn: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_dec_attn_seg(const float* q, const QsaSegTable& t, bool int8, const int32_t* ids, const int32_t* steps,
                      int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                      const float* q_full, float* attn_gated, void* xq) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        n_q > 65535 || t.n_rows < n_q) {
        basalt::log::error("basalt", "qsa_dec_attn_seg: unsupported geometry, table or buffers");
        std::exit(1);
    }
    AttnArgs a;
    a.q = q; a.tab = t; a.ids = ids; a.steps = steps;
    a.n_kv_heads = (int) s.n_head_kv; a.page_size = (int) s.page_size;
    a.n_chunks = (int) ((cap + CHUNK - 1) / CHUNK); a.cap = (int) cap;
    a.stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    a.scale = 1.0f / sqrtf((float) HD);
    a.scratch = scratch; a.attn = attn;
    a.q_full = attn_gated != nullptr ? q_full : nullptr;
    a.attn_gated = q_full != nullptr ? attn_gated : nullptr;
    cudaStream_t st = (cudaStream_t) stream;
    a.xq = a.attn_gated != nullptr ? xq : nullptr;
    const dim3 grid((unsigned) a.n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    if (int8) {
        constexpr size_t bytes = sizeof(ChunkSmem<1>);
        opt_in<2>(attn_chunk_v2<1, true>, bytes);
        launch_pdl(attn_chunk_v2<1, true>, grid, dim3(THREADS), bytes, st, a);
    } else {
        constexpr size_t bytes = sizeof(ChunkSmem<0>);
        opt_in<3>(attn_chunk_v2<0, true>, bytes);
        launch_pdl(attn_chunk_v2<0, true>, grid, dim3(THREADS), bytes, st, a);
    }
    const size_t mbytes = (size_t) (2 * a.n_chunks + WARPS) * sizeof(float);
    launch_pdl(attn_merge_v2, dim3((unsigned) s.n_head, (unsigned) n_q), dim3(THREADS), mbytes, st, a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "qsa_dec_attn_seg: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
