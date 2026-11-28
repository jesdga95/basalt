// src/models/gemma4/kernels/attn_mma.cu - the attention on the fp16 tensor cores, flash style: the prompt attention
// (basalt/models/gemma4/kernels.hpp) and the decode attention's chunks (attn_split.cuh). The m16n8k16 fragment use (the
// scores' accumulators reused as the probabilities' A fragments, keys and values through ldmatrix) follows
// FlashAttention-2 as llama.cpp's fattn-mma-f16 lays it out (MIT).
//
// A prompt block takes the query heads of one key/value head over a run of positions: its rows are (head, position)
// pairs, head-major, in groups of 16. A decode block takes one chunk of one row's keys for the query heads of one key/value
// head: one group of 16 rows, the heads first (the rest padding), its result the chunk's unnormalized output with its
// max and sum. A row group's head width is split in slices (of 128 dims, or the whole 256 for a prompt), a warp a
// (group, slice): each
// warp holds its rows' queries and outputs for its slice only, so the scores of a key tile are the sum of the slices'
// partial products, exchanged through shared memory and added in slice order (every slice of a group gets the same
// bits). Keys stream in tiles of kKeys on a fixed grid of positions: the raw cache rows of the next tile load while the
// current one is computed, and each tile is converted to fp16 key and value tiles first (a full layer whose keys are its
// values derives K = rope(V * k_norm) there). The softmax is online, in fp32, base 2.
#include "attn_split.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/gemma4/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::gemma4 {
namespace {

namespace bk = basalt::kernels;

// What a launch reads: the prompt's rows at pos0.. of one conversation (pos null), or the decode rows at their own
// positions in their segments' caches (`out` then the scratch). span_end > 0: the prompt's rows are one picture's, and a
// window layer's row sees every key up to span_end - 1.
struct MmaArgs {
    const float* q = nullptr;
    const int32_t* pos = nullptr;
    int64_t pos0 = 0;
    int64_t span_end = 0;
    const float* k_norm = nullptr;
    RopeTab rope;
    LayerShape sh;
    KvType kv_type = KvType::F16;
    KvSegs kv;
    float* out = nullptr;
};

void check(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(e));
}

constexpr int kKeys = 16;                  // keys a tile
constexpr float kLog2e = 1.4426950408889634f;

template <int HD, int RG, int NS = HD / 128>
struct Cfg {
    static constexpr int DS = NS;                    // slices of a head row
    static constexpr int SW = HD / DS;               // dims a slice
    static constexpr int kThreads = RG * DS * 32;
    static constexpr int kRows = 16 * RG;            // rows a block
    static constexpr int kLd = HD + 8;               // a tile row in halves, padded: ldmatrix without bank conflicts
    static constexpr int kSLd = kKeys + 4;           // a row of exchanged scores in floats
    // DIRECT: two stages of fp16 key and value tiles; else two stages of raw rows and one of converted tiles (and, `lo`,
    // the derived keys' second fp16 term)
    static size_t smem(int raw_stage, bool direct, bool lo) {
        return (direct ? 0 : 2 * (size_t) raw_stage) + (direct ? 4 : lo ? 3 : 2) * (size_t) kKeys * kLd * 2 +
               (size_t) RG * DS * 16 * kSLd * 4;
    }
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void bar_sync(int id, int threads) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads)); }
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
// value i of a raw cache row (fp16 or E4M3)
__device__ __forceinline__ float raw_at(const uint8_t* row, int i, bool f8) {
    return f8 ? float(reinterpret_cast<const __nv_fp8_e4m3*>(row)[i]) : __half2float(reinterpret_cast<const __half*>(row)[i]);
}

// DIRECT (an fp16 cache whose keys are stored): the cache rows go straight into the tiles, no conversion.
template <int HD, int RG, int NS, bool DEC, bool DIRECT>
__global__ void __launch_bounds__(Cfg<HD, RG, NS>::kThreads, 1) attn_mma_kernel(MmaArgs a, int rows) {
    using C = Cfg<HD, RG, NS>;
    constexpr int KS = C::SW / 16, NT = C::SW / 8;   // a slice's 16-dim steps and 8-dim output tiles
    extern __shared__ __align__(16) uint8_t sm[];
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const LayerShape sh = a.sh;
    const int G = sh.n_head / sh.n_kv;
    const bool f8 = a.kv_type == KvType::F8, one = sh.one;
    const int eb = f8 ? 1 : 2;
    const int raw_row = HD * eb;                              // bytes of a key's head row
    const int raw_stage = kKeys * raw_row * (one ? 1 : 2);    // a tile's V rows, then its K rows
    uint8_t* raw = sm;
    // a decode row's derived keys as two fp16 terms (the rounding and what it leaves), so its scores keep fp32 keys
    const bool klo = DEC && !DIRECT && one;
    __half* Kt = reinterpret_cast<__half*>(sm + (DIRECT ? 0 : 2 * raw_stage));   // DIRECT: [2] stages, K then V each
    __half* Vt = Kt + kKeys * C::kLd;
    __half* Kl = Vt + kKeys * C::kLd;                                            // klo: the keys' second terms
    float* Sx = reinterpret_cast<float*>(Kt + (DIRECT ? 4 : klo ? 3 : 2) * kKeys * C::kLd);   // [RG][DS][16][kSLd]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int rg = warp / C::DS, ds = warp % C::DS;
    const int g = lane >> 2, t = lane & 3;
    const int W = sh.window;
    // the block's keys [kmin, kmax], its cache, and this thread's rows g and g + 8 of its group: (query head, row of the
    // call) and each one's keys [rlo, rhi]; a padding row reads a real one and is not stored
    int h, kmin, kmax, ch = 0;
    SegKv cache;
    int rlo[2], rhi[2], rrow[2], rhead[2];
    bool rok[2];
    if constexpr (DEC) {
        const int r = blockIdx.y / sh.n_kv;
        h = blockIdx.y % sh.n_kv;
        if (r >= rows) return;
        const int pos = a.pos[r];
        const Split sp = split_of(pos, W);
        ch = blockIdx.x;
        if (ch >= sp.chunks) return;
        kmin = (int) (sp.lo + ch * sp.len);
        kmax = (int) min(sp.lo + (ch + 1) * sp.len, sp.lo + sp.n) - 1;
        cache = a.kv.c[a.kv.at.n > 0 ? bk::seg_of(a.kv.at, r) : 0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int rr = g + 8 * e;
            rok[e] = rr < G;
            rrow[e] = r;
            rhead[e] = h * G + (rok[e] ? rr : 0);
            rlo[e] = kmin, rhi[e] = kmax;
        }
    } else {
        const int P = C::kRows / G;
        h = blockIdx.y;
        const int first = (int) (gridDim.x - 1 - blockIdx.x) * P;   // the longest key runs start first
        const int last = min(first + P, rows) - 1;
        const bool span = W > 0 && a.span_end > 0;
        kmax = span ? (int) a.span_end - 1 : (int) a.pos0 + last;
        kmin = W > 0 ? max(0, (int) a.pos0 + first - W + 1) : 0;
        cache = a.kv.c[0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int rr = rg * 16 + g + 8 * e;
            const int row = first + rr % P;
            rok[e] = row <= last;
            rrow[e] = rok[e] ? row : last;
            rhead[e] = h * G + rr / P;
            const int p = (int) a.pos0 + rrow[e];
            rhi[e] = span ? kmax : p;
            rlo[e] = W > 0 ? p - W + 1 : 0;
        }
    }
    const int t0 = kmin / kKeys, t1 = kmax / kKeys;
    // the queries of this warp's slice as A fragments: 8 steps of 16 dims; a decode row's as two fp16 terms (the rounding
    // and what it leaves), so its scores keep the fp32 queries' precision
    unsigned qf[KS][4], ql[DEC ? KS : 1][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int e = r & 1, d = ds * C::SW + ks * 16 + 2 * t + (r >> 1) * 8;
            const float2 v = *reinterpret_cast<const float2*>(a.q + ((size_t) rrow[e] * sh.n_head + rhead[e]) * HD + d);
            const __half2 hi = __floats2half2_rn(v.x, v.y);
            qf[ks][r] = rok[e] || !DEC ? *reinterpret_cast<const unsigned*>(&hi) : 0u;
            if constexpr (DEC) {
                const float2 h = __half22float2(hi);
                ql[ks][r] = rok[e] ? pack_h2(v.x - h.x, v.y - h.y) : 0u;
            }
        }

    const auto stage_raw = [&](int tile, int buf) {
        const int per_row = raw_row / 16, per_mat = kKeys * per_row, n = per_mat * (one ? 1 : 2);
        for (int i = tid; i < n; i += C::kThreads) {
            const int which = i / per_mat, rem = i - which * per_mat, j = rem / per_row, c = rem - j * per_row;
            const int key = tile * kKeys + j;
            const bool ok = key >= kmin && key <= kmax;
            const void* base = which == 0 && !one ? cache.v : cache.k;
            const uint8_t* src = static_cast<const uint8_t*>(base) + (((ok ? (int64_t) (key % (int) cache.cap) : 0) * sh.n_kv + h) * HD) * eb + 16 * c;
            cp16(raw + buf * raw_stage + which * per_mat * 16 + j * raw_row + 16 * c, src, ok);
        }
    };
    const auto stage_tiles = [&](int tile, int buf) {   // DIRECT
        constexpr int per_row = HD / 8, n = 2 * kKeys * per_row;
        __half* kt = reinterpret_cast<__half*>(sm) + buf * 2 * kKeys * C::kLd;
        for (int i = tid; i < n; i += C::kThreads) {
            const int isv = i >= kKeys * per_row, rem = isv ? i - kKeys * per_row : i, j = rem / per_row, c = rem - j * per_row;
            const int key = tile * kKeys + j;
            const bool ok = key >= kmin && key <= kmax;
            const __half* src = static_cast<const __half*>(isv ? cache.v : cache.k) +
                                ((ok ? (int64_t) (key % (int) cache.cap) : 0) * sh.n_kv + h) * HD + 8 * c;
            cp16(kt + (isv ? kKeys * C::kLd : 0) + j * C::kLd + 8 * c, src, ok);
        }
    };
    const auto convert = [&](int tile, int buf) {
        const uint8_t* rv = raw + buf * raw_stage;
        if (one) {   // V as stored; K = rope(V * k_norm): two pairs (i, i + HD / 2) an item
            constexpr int H = HD / 2;
            for (int it = tid; it < kKeys * HD / 4; it += C::kThreads) {
                const int j = it / (HD / 4), i = (it - j * (HD / 4)) * 2;
                const uint8_t* row = rv + j * raw_row;
                float x[2], y[2];
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    x[c] = raw_at(row, i + c, f8);
                    y[c] = raw_at(row, i + H + c, f8);
                }
                *reinterpret_cast<__half2*>(Vt + j * C::kLd + i) = __floats2half2_rn(x[0], x[1]);
                *reinterpret_cast<__half2*>(Vt + j * C::kLd + i + H) = __floats2half2_rn(y[0], y[1]);
                const int pos = (int) (tile * kKeys + j);
                const float2* lo = reinterpret_cast<const float2*>(a.rope.lo) + (size_t) (pos & 255) * H;
                const float2* hi = reinterpret_cast<const float2*>(a.rope.hi) + (size_t) (pos >> 8) * H;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const float kx = x[c] * a.k_norm[i + c], ky = y[c] * a.k_norm[i + H + c];
                    if (i + c < sh.rot) {
                        const float2 p = hi[i + c], q = lo[i + c];
                        const float co = p.x * q.x - p.y * q.y, si = p.y * q.x + p.x * q.y;
                        x[c] = kx * co - ky * si;
                        y[c] = ky * co + kx * si;
                    } else {
                        x[c] = kx;
                        y[c] = ky;
                    }
                }
                const __half2 kx2 = __floats2half2_rn(x[0], x[1]), ky2 = __floats2half2_rn(y[0], y[1]);
                *reinterpret_cast<__half2*>(Kt + j * C::kLd + i) = kx2;
                *reinterpret_cast<__half2*>(Kt + j * C::kLd + i + H) = ky2;
                if (klo) {
                    const float2 hx = __half22float2(kx2), hy = __half22float2(ky2);
                    *reinterpret_cast<__half2*>(Kl + j * C::kLd + i) = __floats2half2_rn(x[0] - hx.x, x[1] - hx.y);
                    *reinterpret_cast<__half2*>(Kl + j * C::kLd + i + H) = __floats2half2_rn(y[0] - hy.x, y[1] - hy.y);
                }
            }
        } else {     // V and K as stored, 8 values an item
            const uint8_t* rk = rv + kKeys * raw_row;
            for (int it = tid; it < 2 * kKeys * HD / 8; it += C::kThreads) {
                const int isk = it >= kKeys * HD / 8, r = isk ? it - kKeys * HD / 8 : it;
                const int j = r / (HD / 8), d = (r - j * (HD / 8)) * 8;
                const uint8_t* row = (isk ? rk : rv) + j * raw_row;
                __half* dst = (isk ? Kt : Vt) + j * C::kLd + d;
                if (f8) {
                    __half2 o[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) o[c] = __floats2half2_rn(raw_at(row, d + 2 * c, true), raw_at(row, d + 2 * c + 1, true));
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(o);
                } else {
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(row + 2 * d);
                }
            }
        }
    };

    float o[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float* sx_own = Sx + (size_t) (rg * C::DS + ds) * 16 * C::kSLd;
    const float* sx_grp = Sx + (size_t) rg * C::DS * 16 * C::kSLd;

    if constexpr (DIRECT) stage_tiles(t0, 0);
    else stage_raw(t0, 0);
    cp_commit();
    for (int tile = t0; tile <= t1; ++tile) {
        const int buf = (int) ((tile - t0) & 1);
        cp_wait_all();
        __syncthreads();   // the tile's rows are in; every warp is done with the last tile's K, V and scores
        if constexpr (DIRECT) {
            if (tile < t1) stage_tiles(tile + 1, buf ^ 1);
            cp_commit();
            Kt = reinterpret_cast<__half*>(sm) + buf * 2 * kKeys * C::kLd;
            Vt = Kt + kKeys * C::kLd;
        } else {
            convert(tile, buf);
            if (tile < t1) stage_raw(tile + 1, buf ^ 1);
            cp_commit();
            __syncthreads();
        }
        // this slice's partial scores of the group's 16 rows against the tile's keys
        float s[2][4] = {};
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            unsigned b0, b1, b2, b3;
            ldsm_x4(b0, b1, b2, b3, Kt + ((lane & 7) + ((lane >> 4) << 3)) * C::kLd + ds * C::SW + ks * 16 + ((lane >> 3) & 1) * 8);
            mma16816(s[0], qf[ks], b0, b1);
            mma16816(s[1], qf[ks], b2, b3);
            if constexpr (DEC) {
                mma16816(s[0], ql[ks], b0, b1);
                mma16816(s[1], ql[ks], b2, b3);
                if (klo) {
                    ldsm_x4(b0, b1, b2, b3, Kl + ((lane & 7) + ((lane >> 4) << 3)) * C::kLd + ds * C::SW + ks * 16 + ((lane >> 3) & 1) * 8);
                    mma16816(s[0], qf[ks], b0, b1);
                    mma16816(s[1], qf[ks], b2, b3);
                }
            }
        }
        if constexpr (C::DS > 1) {   // the slices' partial scores to the group
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                *reinterpret_cast<float2*>(sx_own + g * C::kSLd + nt * 8 + 2 * t) = make_float2(s[nt][0], s[nt][1]);
                *reinterpret_cast<float2*>(sx_own + (g + 8) * C::kSLd + nt * 8 + 2 * t) = make_float2(s[nt][2], s[nt][3]);
            }
            bar_sync(1 + rg, C::DS * 32);
        }
        float p[2][2][2];   // [row e][n-tile][key]
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float v[2][2], mx = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float x = s[nt][2 * e + c];
                    if constexpr (C::DS > 1) {
                        const int off = (g + 8 * e) * C::kSLd + nt * 8 + 2 * t + c;
                        x = sx_grp[off];
#pragma unroll
                        for (int d = 1; d < C::DS; ++d) x += sx_grp[d * 16 * C::kSLd + off];
                    }
                    const int key = tile * kKeys + nt * 8 + 2 * t + c;
                    const bool ok = key >= rlo[e] && key <= rhi[e];
                    v[nt][c] = ok ? x * kLog2e : -INFINITY;
                    mx = fmaxf(mx, v[nt][c]);
                }
            mx = quad_max(mx);
            const float mn = fmaxf(m[e], mx);
            const float corr = mn == -INFINITY ? 1.f : exp2f(m[e] - mn);
            float sum = 0.f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    p[e][nt][c] = v[nt][c] == -INFINITY ? 0.f : exp2f(v[nt][c] - mn);
                    sum += p[e][nt][c];
                }
            l[e] = l[e] * corr + sum;
            m[e] = mn;
#pragma unroll
            for (int dt = 0; dt < NT; ++dt) {
                o[dt][2 * e] *= corr;
                o[dt][2 * e + 1] *= corr;
            }
        }
        const unsigned pa[4] = {pack_h2(p[0][0][0], p[0][0][1]), pack_h2(p[1][0][0], p[1][0][1]), pack_h2(p[0][1][0], p[0][1][1]),
                                pack_h2(p[1][1][0], p[1][1][1])};
#pragma unroll
        for (int dt = 0; dt < NT; dt += 2) {
            unsigned b0, b1, b2, b3;
            ldsm_x4_t(b0, b1, b2, b3, Vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * C::kLd + ds * C::SW + dt * 8 + (lane >> 4) * 8);
            mma16816(o[dt], pa, b0, b1);
            mma16816(o[dt + 1], pa, b2, b3);
        }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const float L = quad_sum(l[e]);
        if (!rok[e]) continue;
        if constexpr (DEC) {   // the chunk's unnormalized output, its max (natural log units) and sum
            float* out = a.out + (((size_t) rrow[e] * sh.n_head + rhead[e]) * kMaxChunks + ch) * (HD + 2);
            if (ds == 0 && t == 0) {
                out[0] = m[e] * 0.6931471805599453f;
                out[1] = L;
            }
#pragma unroll
            for (int dt = 0; dt < NT; ++dt) {
                out[2 + ds * C::SW + dt * 8 + 2 * t] = o[dt][2 * e];
                out[3 + ds * C::SW + dt * 8 + 2 * t] = o[dt][2 * e + 1];
            }
        } else {
            const float inv = 1.f / L;
            float* out = a.out + ((size_t) rrow[e] * sh.n_head + rhead[e]) * HD + ds * C::SW + 2 * t;
#pragma unroll
            for (int dt = 0; dt < NT; ++dt)
                *reinterpret_cast<float2*>(out + dt * 8) = make_float2(o[dt][2 * e] * inv, o[dt][2 * e + 1] * inv);
        }
    }
}

template <int HD, int RG, int NS, bool DEC>
void launch(const MmaArgs& a, int rows, dim3 grid, cudaStream_t st) {
    using C = Cfg<HD, RG, NS>;
    const int raw_stage = kKeys * HD * kv_elem_bytes(a.kv_type) * (a.sh.one ? 1 : 2);
    const bool direct = a.kv_type == KvType::F16 && !a.sh.one;
    const size_t smem = C::smem(raw_stage, direct, DEC && !direct && a.sh.one);
    const auto run = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        bk::launch_pdl(kernel, grid, dim3(C::kThreads), smem, st, a, rows);
    };
    if (direct) run(attn_mma_kernel<HD, RG, NS, DEC, true>);
    else run(attn_mma_kernel<HD, RG, NS, DEC, false>);
}

void check_shape(const LayerShape& sh) {
    const int G = sh.n_kv > 0 ? sh.n_head / sh.n_kv : 0;
    if ((sh.hd != 256 && sh.hd != 512) || G < 1 || sh.n_head % sh.n_kv || G > kMaxGroup || (G & (G - 1)) || sh.rot < 0 ||
        sh.rot > sh.hd / 2)
        throw std::invalid_argument("gemma4 attention: unsupported shape");
}

}  // namespace

void attn_prompt(const AttnPrompt& p, int rows, void* stream) {
    if (rows <= 0) return;
    check_shape(p.sh);
    MmaArgs a;
    a.q = p.q, a.pos0 = p.pos0, a.k_norm = p.k_norm, a.rope = p.rope, a.sh = p.sh, a.kv_type = p.kv_type, a.out = p.out;
    a.span_end = p.span ? p.pos0 + rows : 0;
    a.kv.c[0] = p.kv;
    const int G = p.sh.n_head / p.sh.n_kv;
    const auto grid = [&](int block_rows) { return dim3((unsigned) ((rows + block_rows / G - 1) / (block_rows / G)), (unsigned) p.sh.n_kv); };
    if (p.sh.hd == 256) launch<256, 8, 1, false>(a, rows, grid(Cfg<256, 8, 1>::kRows), (cudaStream_t) stream);
    else launch<512, 4, 4, false>(a, rows, grid(Cfg<512, 4, 4>::kRows), (cudaStream_t) stream);
    check("attn_prompt");
}

void attn_chunks(const Attn& d, int rows, cudaStream_t st) {
    check_shape(d.sh);
    MmaArgs a;
    a.q = d.q, a.pos = d.pos, a.k_norm = d.k_norm, a.rope = d.rope, a.sh = d.sh, a.kv_type = d.kv_type, a.kv = d.kv;
    a.out = d.scratch;
    const int64_t keys = d.sh.window > 0 && d.max_keys > d.sh.window ? d.sh.window : d.max_keys;
    const dim3 grid((unsigned) chunk_bound(keys), (unsigned) (rows * d.sh.n_kv));
    if (d.sh.hd == 256) launch<256, 1, 2, true>(a, rows, grid, st);
    else launch<512, 1, 4, true>(a, rows, grid, st);
    check("attn_chunks");
}

}  // namespace basalt::models::gemma4
