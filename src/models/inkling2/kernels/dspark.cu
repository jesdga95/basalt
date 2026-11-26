// src/models/inkling2/kernels/dspark.cu - the DSpark drafter's own kernels (basalt/models/inkling2/dspark_kernels.hpp).
#include "basalt/models/inkling2/dspark_kernels.hpp"

#include "basalt/kernels/head_proxy.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace px = basalt::kernels::px;
using basalt::kernels::launch_pdl;
using basalt::kernels::pdl_launch_dependents;
using basalt::kernels::pdl_wait;

constexpr int kThreads = 256;
constexpr int kPer = kWidth / kThreads;   // 9
constexpr int kQ8Block = 34;              // ggml's block_q8_0: an fp16 scale, then 32 codes
constexpr int kQ8Row = kWidth / 32 * kQ8Block;
static_assert(kWidth % kThreads == 0, "the width splits evenly over a block");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 dspark ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(uint16_t u) { return __uint_as_float((uint32_t) u << 16); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// The sum of one value per thread over a block of kThreads, in a fixed order; every thread gets it.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < kThreads / 32; ++j) s += red[j];
    __syncthreads();
    return s;
}

// ---- the mixes of the target's features -------------------------------------------------------------------------------

// Row r's mix j.
__device__ __forceinline__ void mix_stage(int r, int j, const float* __restrict__ tap, int64_t tap_stride, const DsMixArgs& a,
                                          float* __restrict__ out, int64_t out_stride, const int32_t* __restrict__ pos,
                                          float* __restrict__ fr, int fr_cap, int32_t* __restrict__ pos3, float* red) {
    float al[kDsTargets];
#pragma unroll
    for (int k = 0; k < kDsTargets; ++k) al[k] = a.alpha[j * kDsTargets + k];
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        float m = 0.f;
#pragma unroll
        for (int k = 0; k < kDsTargets; ++k) m += al[k] * tap[(int64_t) k * tap_stride + (int64_t) r * kWidth + c];
        v[i] = m;
        ss += m * m;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) kWidth + a.eps);
    const int32_t p = pos[r];
    float* dst = j < kDsMixes - 1 ? out + j * out_stride + (int64_t) r * kWidth : fr + (int64_t) (p % fr_cap) * kWidth;
    const float* w = a.norm[j];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        dst[c] = v[i] * inv * w[c];
    }
    if (pos3 && j == 0 && threadIdx.x < 3) pos3[r * 3 + threadIdx.x] = p;
}

__global__ void __launch_bounds__(kThreads) mix_kernel(const float* __restrict__ tap, int64_t tap_stride, DsMixArgs a,
                                                       float* __restrict__ out, int64_t out_stride,
                                                       const int32_t* __restrict__ pos, float* __restrict__ fr, int fr_cap,
                                                       int32_t* __restrict__ pos3) {
    if ((int) blockIdx.y == kDsMixes - 1 && (int) blockIdx.x + fr_cap < (int) gridDim.x) return;
    __shared__ float red[kThreads / 32];
    mix_stage((int) blockIdx.x, (int) blockIdx.y, tap, tap_stride, a, out, out_stride, pos, fr, fr_cap, pos3, red);
}

// ---- the block's inputs -------------------------------------------------------------------------------------------------

// Block row t: the anchor's normed embedding (row 0) or the mask row. Row 2's block also copies the Markov head's context
// of position pos0 - 1 out of its ring.
__device__ __forceinline__ void block_inputs_stage(int t, const uint16_t* __restrict__ emb, const int32_t* __restrict__ meta,
                                                   const float* __restrict__ norm, float eps, const float* __restrict__ mask,
                                                   float* __restrict__ x, int32_t* __restrict__ pos3, int32_t* __restrict__ prev,
                                                   float* __restrict__ state, const float* __restrict__ fr, int fr_cap,
                                                   float* __restrict__ fr_out, const uint16_t* __restrict__ w1,
                                                   uint16_t* __restrict__ w1_row, float* red) {
    const int32_t anchor = meta[0], pos0 = meta[1];
    float* row = x + (int64_t) t * kWidth;
    if (pos3 && threadIdx.x < 3) pos3[t * 3 + threadIdx.x] = pos0 + t;
    if (t == 1) {   // the Markov head's start
        if (threadIdx.x == 0) *prev = anchor;
        for (int i = threadIdx.x; i < kDsRank; i += kThreads) state[i] = 0.f;
        if (threadIdx.x < kDsRank / 8)
            reinterpret_cast<uint4*>(w1_row)[threadIdx.x] = reinterpret_cast<const uint4*>(w1 + (int64_t) anchor * kDsRank)[threadIdx.x];
    }
    if (t == 2) {
        const float* src = fr + (int64_t) ((pos0 - 1) % fr_cap) * kWidth;
#pragma unroll
        for (int i = 0; i < kPer; ++i) fr_out[threadIdx.x + i * kThreads] = src[threadIdx.x + i * kThreads];
    }
    if (t > 0) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) row[threadIdx.x + i * kThreads] = mask[threadIdx.x + i * kThreads];
        return;
    }
    const uint16_t* e = emb + (int64_t) anchor * kWidth;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        v[i] = bf16f(e[threadIdx.x + i * kThreads]);
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) kWidth + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        row[c] = v[i] * inv * norm[c];
    }
}

__global__ void __launch_bounds__(kThreads) block_inputs_kernel(const uint16_t* __restrict__ emb, const int32_t* __restrict__ meta,
                                                                const float* __restrict__ norm, float eps,
                                                                const float* __restrict__ mask, float* __restrict__ x,
                                                                int32_t* __restrict__ pos3, int32_t* __restrict__ prev,
                                                                float* __restrict__ state, const float* __restrict__ fr, int fr_cap,
                                                                float* __restrict__ fr_out, const uint16_t* __restrict__ w1,
                                                                uint16_t* __restrict__ w1_row) {
    __shared__ float red[kThreads / 32];
    block_inputs_stage((int) blockIdx.x, emb, meta, norm, eps, mask, x, pos3, prev, state, fr, fr_cap, fr_out, w1, w1_row, red);
}

// ---- the Markov head's recurrence ---------------------------------------------------------------------------------------

constexpr int kStepWarps = 32;                     // a warp per recurrence index
constexpr int kStepBlocks = kDsRank / kStepWarps;  // 8

// Block b: recurrence indices 32 b .. 32 b + 31, each with its gate, candidate and latent rows.
__device__ __forceinline__ void markov_step_stage(int b, const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
                                                  const float* __restrict__ bias, const uint16_t* __restrict__ w_sp,
                                                  const uint16_t* __restrict__ w1_row, const float* __restrict__ state_in,
                                                  float* __restrict__ state_out, float* __restrict__ latent, float* xs) {
    for (int i = threadIdx.x; i < 2 * kDsRank; i += kStepWarps * 32)
        xs[i] = i < kDsRank ? state_in[i] : bf16f(w1_row[i - kDsRank]);
    __syncthreads();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = b * kStepWarps + w;
    float raw[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int o = g * kDsRank + j;
        const uint32_t* row = reinterpret_cast<const uint32_t*>(w_sp + (int64_t) o * 2 * kDsRank);   // pairs of BF16
        float acc = 0.f;
#pragma unroll
        for (int k = lane; k < kDsRank; k += 32) {
            const uint32_t u = row[k];
            acc += __uint_as_float(u << 16) * xs[2 * k] + __uint_as_float(u & 0xffff0000u) * xs[2 * k + 1];
        }
        raw[g] = warp_sum(acc) + ctx_h[o] + ctx_f[o] + bias[o];
    }
    if (lane == 0) {
        const float gate = 1.f / (1.f + expf(-raw[0]));
        state_out[j] = gate * xs[j] + (1.f - gate) * tanhf(raw[1]);
        latent[j] = tanhf(raw[2]);
    }
}

__global__ void __launch_bounds__(kStepWarps * 32) markov_step_kernel(const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
                                                                      const float* __restrict__ bias, const uint16_t* __restrict__ w_sp,
                                                                      const uint16_t* __restrict__ w1_row,
                                                                      const float* __restrict__ state_in, float* __restrict__ state_out,
                                                                      float* __restrict__ latent) {
    __shared__ float xs[2 * kDsRank];
    markov_step_stage((int) blockIdx.x, ctx_h, ctx_f, bias, w_sp, w1_row, state_in, state_out, latent, xs);
}

// ---- the head, screened ------------------------------------------------------------------------------------------------
//
// Draft i's logits are l[v] = cap tanh((head_v . h + fold_v . lat_i + fold_b[v]) / cap), h = block row i + 1 and lat_i
// the recurrence's latent, both as q8_1. The screen (basalt/kernels/head_proxy.cuh) scores every row approximately from
// 2-bit copies of the head and the fold: the head term once for all the block's rows (one pass over the copy), the fold
// term per draft; then the rows near the best are scored exactly (exact_logit) and the exact best is the pick. The
// reference (dspark_head_exact) scores every row with the same exact_logit, so a pick is the full head's whenever the full
// head's best is among the rescored rows.
//
// The copies hold the tokens transposed: copy row r (tile r / 32, lane r % 32) is token (r % 32) T + r / 32, T the tiles.
// A tile keeps only its best kPer rows, and a tokenizer's frequent tokens sit together (punctuation among the first
// ids): in token order a tile of them can hold the best five, and the fifth is lost. Transposed, a tile's tokens are T
// apart.

constexpr int kHeadNb = kWidth / 32;    // 72 q8_1 blocks of a row
constexpr int kFoldNb = kDsRank / 32;   // 8
constexpr int kFoldRow = kFoldNb * kQ8Block;
constexpr int kTilesPerBlock = 4;       // the fold pass: a warp per tile of 32 rows

struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "ggml's block_q8_1");

__device__ __forceinline__ float capf(float x, float cap) { return cap * tanhf(x / cap); }

// The token of copy row r over `tiles` tiles.
__host__ __device__ __forceinline__ int64_t token_of(int64_t r, int64_t tiles) { return (r & 31) * tiles + (r >> 5); }

// A Q8_0 word of a block's codes (the blocks are 2-byte aligned).
__device__ __forceinline__ int q80_word(const uint8_t* qs, int i32) {
    const auto* x = reinterpret_cast<const uint16_t*>(qs);
    return (int) x[2 * i32] | ((int) x[2 * i32 + 1] << 16);
}

// One Q8_0 row (nb blocks) by a q8_1 row on a warp: lane L takes words 2 (L % 4), +1 of blocks L / 4 + 8 j, in j order.
__device__ __forceinline__ float q80_dot(const uint8_t* __restrict__ w, const Q81* x, int nb, int lane) {
    const int k = 2 * (lane & 3);
    float s = 0.f;
    for (int b = lane >> 2; b < nb; b += 8) {
        const uint8_t* blk = w + (size_t) b * kQ8Block;
        const int* xq = reinterpret_cast<const int*>(x[b].qs);
        int sumi = __dp4a(q80_word(blk + 2, k), xq[k], 0);
        sumi = __dp4a(q80_word(blk + 2, k + 1), xq[k + 1], sumi);
        s += __half2float(*reinterpret_cast<const __half*>(blk)) * __low2float(x[b].ds) * (float) sumi;
    }
    return warp_sum(s);
}

// One q8_planes row (n_in = nb * 32) by a q8_1 row on a warp (16-byte aligned int4 codes + fp16 scales).
__device__ __forceinline__ float q8p_dot(const void* __restrict__ planes, int64_t n_vocab, int64_t v, const Q81* x, int nb, int lane) {
    const int n_in = nb * 32;
    const int8_t* codes = static_cast<const int8_t*>(planes) + v * n_in;
    const __half* scales = reinterpret_cast<const __half*>(static_cast<const int8_t*>(planes) + n_vocab * n_in) + v * nb;
    const int half_b = lane & 1;
    float s = 0.f;
    for (int b = lane >> 1; b < nb; b += 16) {
        const int4 q = __ldg(reinterpret_cast<const int4*>(codes + b * 32 + half_b * 16));
        const int* xq = reinterpret_cast<const int*>(x[b].qs) + half_b * 4;
        int sumi = __dp4a(q.x, xq[0], 0);
        sumi = __dp4a(q.y, xq[1], sumi);
        sumi = __dp4a(q.z, xq[2], sumi);
        sumi = __dp4a(q.w, xq[3], sumi);
        s += __half2float(__ldg(scales + b)) * __low2float(x[b].ds) * (float) sumi;
    }
    return warp_sum(s);
}

// Row v's exact logit (every lane gets it).
__device__ __forceinline__ float exact_logit(const DsHead& h, int64_t v, const Q81* xh, const Q81* xl, int lane) {
    const float a = h.head_planes
                        ? q8p_dot(h.head, h.n_vocab, v, xh, kHeadNb, lane)
                        : q80_dot(static_cast<const uint8_t*>(h.head) + v * kQ8Row, xh, kHeadNb, lane);
    const float b = q80_dot(static_cast<const uint8_t*>(h.fold) + v * kFoldRow, xl, kFoldNb, lane);
    return capf(a + b + h.fold_b[v], h.cap);
}

// One thread per 32-weight group of a copy row: its token's group into the 2-bit copy (q80_screen.cu's packing).
template <int N_IN>
__global__ void screen_pack_kernel(const uint8_t* __restrict__ q8, int64_t n_rows, uint4* __restrict__ wq, uint16_t* __restrict__ sc) {
    constexpr int NB = N_IN / 32;
    const int64_t tiles = px::tiles(n_rows);
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 32 * NB) return;
    const int64_t row = i / NB, tok = token_of(row, tiles);
    const int g = (int) (i % NB);
    if (tok >= n_rows) return;   // past the vocabulary: zeros, masked when scored
    const uint8_t* b = q8 + (tok * NB + g) * kQ8Block;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    float w[32];
    for (int l = 0; l < 32; ++l) w[l] = d * (float) (int8_t) b[2 + l];
    px::pack_group<N_IN, true>(w, row, g, wq, sc);
}

// The copy's approximate score of a row (a lane each) over this thread's uint4 words j0 .. j0 + NW - 1: the group sums of
// head_proxy.cuh's score_tile.
template <int NW>
__device__ __forceinline__ float screen_acc(const uint4 (&w)[NW], const uint32_t (&s)[NW], int j0, const int* xi, const float* xs,
                                            const float* xdd) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        const int w4 = j0 + j;
        const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int g = 2 * w4 + hh;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v = ww[2 * hh + c];
                const int4 x0 = *reinterpret_cast<const int4*>(&xi[g * 8 + c * 4]);
                P = __dp4a((int) (v & 0x03030303u), x0.x, P);
                P = __dp4a((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = __dp4a((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = __dp4a((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (hh ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    return acc;
}

// q8_1 blocks into the scoring layout: the codes as ints, each block's scale and scaled code sum.
__device__ __forceinline__ void screen_x(const Q81* __restrict__ x, int nb, int* xi, float* xs, float* xdd, int tid, int nthreads) {
    for (int i = tid; i < nb * 8; i += nthreads) xi[i] = reinterpret_cast<const int*>(x[i / 8].qs)[i % 8];
    for (int b = tid; b < nb; b += nthreads) {
        const int* q = reinterpret_cast<const int*>(x[b].qs);
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum = __dp4a(0x01010101, q[i], sum);
        const float d8 = __low2float(x[b].ds);
        xdd[b] = d8;
        xs[b] = d8 * (float) sum;
    }
}

// Tile `tile`: approx[r][row] = the head copy's score of q8_1 row r (r < rows) for its 32 copy rows. Four warps split the row
// length, lane = row; the copy is read once for all the rows.
__global__ void __launch_bounds__(128) head_base_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                        const Q81* __restrict__ x, int rows, int64_t n_pad,
                                                        float* __restrict__ approx) {
    constexpr int kNw4 = kWidth / 64, NW = kNw4 / 4;
    __shared__ __align__(16) int xi[kDsDrafts][kWidth / 4];
    __shared__ float xs[kDsDrafts][kHeadNb], xdd[kDsDrafts][kHeadNb];
    __shared__ float red[kDsDrafts][4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t tile = blockIdx.x;
    const size_t base = ((size_t) tile * kNw4 + warp * NW) * 32 + lane;
    uint4 w[NW];
    uint32_t s[NW];
#pragma unroll
    for (int j = 0; j < NW; ++j) {   // constant data: read before the wait
        w[j] = __ldcs(wq + base + (size_t) j * 32);
        s[j] = __ldcs(sc + base + (size_t) j * 32);
    }
    pdl_wait();
    for (int r = 0; r < rows; ++r) screen_x(x + (size_t) r * kHeadNb, kHeadNb, xi[r], xs[r], xdd[r], tid, 128);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kDsDrafts; ++r)
        if (r < rows) red[r][warp][lane] = screen_acc<NW>(w, s, warp * NW, xi[r], xs[r], xdd[r]);
    __syncthreads();
    const int64_t row = tile * 32 + lane;
    for (int r = warp; r < rows; r += 4)
        approx[r * n_pad + row] = red[r][0][lane] + red[r][1][lane] + red[r][2][lane] + red[r][3][lane];
}

// A warp per tile of 32 copy rows (lane = row): the fold copy's score of the latent plus the head term and the bias (in
// copy order), capped;
// then the tile's softmax partial and its best kPer rows (head_proxy.cuh's score_tile tail).
__global__ void __launch_bounds__(32 * kTilesPerBlock) head_step_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                                        const Q81* __restrict__ lat, const float* __restrict__ approx,
                                                                        const float* __restrict__ fold_bp, float cap, int64_t n_rows,
                                                                        int n_tiles, float2* __restrict__ cand, float2* __restrict__ part) {
    constexpr int kNw4 = kDsRank / 64;
    __shared__ __align__(16) int xi[kDsRank / 4];
    __shared__ float xs[kFoldNb], xdd[kFoldNb];
    const int tid = threadIdx.x, lane = tid & 31;
    const int tile = (int) blockIdx.x * kTilesPerBlock + (tid >> 5);
    const int64_t row = (int64_t) tile * 32 + lane;
    const bool live = tile < n_tiles, real = live && token_of(row, n_tiles) < n_rows;
    uint4 w[kNw4];
    uint32_t s[kNw4];
    float fb = 0.f;
#pragma unroll
    for (int j = 0; j < kNw4; ++j) {   // constant data: read before the wait
        const size_t at = ((size_t) (live ? tile : 0) * kNw4 + j) * 32 + lane;
        w[j] = __ldcs(wq + at);
        s[j] = __ldcs(sc + at);
    }
    if (live) fb = __ldg(fold_bp + row);
    pdl_wait();
    screen_x(lat, kFoldNb, xi, xs, xdd, tid, 32 * kTilesPerBlock);
    __syncthreads();
    if (!live) return;
    const float ninf = __int_as_float(0xff800000);
    float v = ninf;
    if (real) v = capf(approx[row] + screen_acc<kNw4>(w, s, 0, xi, xs, xdd) + fb, cap);
    float m = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float e = v > ninf ? __expf(v - m) : 0.0f;
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) part[tile] = make_float2(m, e);
    for (int k = 0; k < px::kPer; ++k) {
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * px::kPer + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = ninf;
    }
}

// The activations of draft i's exact logits into shared memory: block row i + 1's q8_1 row and the latent's.
__device__ __forceinline__ void exact_x(const Q81* __restrict__ xh_g, const Q81* __restrict__ xl_g, Q81* xh, Q81* xl, int tid,
                                        int nthreads) {
    for (int k = tid; k < (kHeadNb + kFoldNb) * 9; k += nthreads) {   // nine words a block
        const int b = k / 9, j = k % 9;
        const int* src = reinterpret_cast<const int*>(b < kHeadNb ? xh_g + b : xl_g + (b - kHeadNb));
        int* dst = reinterpret_cast<int*>(b < kHeadNb ? xh + b : xl + (b - kHeadNb));
        dst[j] = src[j];
    }
}

// px::collect<false, true> from the tiles' maxima: a tile's softmax partial holds its best score (part[t].x, the score
// of its first candidate), so the best is the tiles' best and only a tile whose best is within kDelta of it can hold a
// listed candidate; the list is the same, read from a quarter of the scores. More than kMaxCand: px::collect's radix
// select over every candidate.
__device__ __forceinline__ int collect_tiles(px::Final& f, int n_tiles, const float2* __restrict__ cand,
                                             const float2* __restrict__ part, int* __restrict__ out_id,
                                             float* __restrict__ out_prob) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float ninf = __int_as_float(0xff800000);
    float m = ninf;
    for (int t = tid; t < n_tiles; t += 512) m = fmaxf(m, __ldcg(&part[t].x));
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) f.wmax[warp] = m;
    __syncthreads();
    m = f.wmax[0];
    for (int w = 1; w < 16; ++w) m = fmaxf(m, f.wmax[w]);
    if (m == ninf) {   // every row is masked
        if (tid == 0) {
            *out_id = 0;
            *out_prob = 1.0f;
        }
        return 0;
    }
    for (int t = tid; t < n_tiles; t += 512) {
        const float tm = __ldcg(&part[t].x);
        if (!(tm > ninf && tm >= m - px::kDelta)) continue;
        for (int k = 0; k < px::kPer; ++k) {
            const float2 c = __ldcg(&cand[t * px::kPer + k]);
            if (c.x > ninf && c.x >= m - px::kDelta) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < px::kMaxCand) { f.list[p] = __float_as_int(c.y); f.lp[p] = c.x; }
            }
        }
    }
    __syncthreads();
    const int total = f.cnt;
    if (total > px::kMaxCand) {
        const int nc = n_tiles * px::kPer;
        const unsigned long long kth = px::kth_key(f, nc, cand, m);
        if (tid == 0) f.cnt = 0;
        __syncthreads();
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (c.x > ninf && c.x >= m - px::kDelta && px::cand_key(c) >= kth) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < px::kMaxCand) { f.list[p] = __float_as_int(c.y); f.lp[p] = c.x; }
            }
        }
        __syncthreads();
    }
    const int n = total < px::kMaxCand ? total : px::kMaxCand;
    int row = 0, rank = 0;
    float lp = 0.0f;
    if (tid < n) {
        row = f.list[tid];
        lp = f.lp[tid];
        for (int j = 0; j < n; ++j) rank += f.list[j] < row;
    }
    __syncthreads();
    if (tid < n) { f.list[rank] = row; f.lp[rank] = lp; }
    __syncthreads();
    return n;
}

// One block: the rows near the best approximate logit scored exactly (a warp per row); the exact best is draft i:
// *prev = ids[i] = it, conf[i] = its probability (the tiles' softmax sums with the rescored rows made exact), w1_row = its
// row of w1: each rescored row's is read (host memory) while the row is scored, the pick's then copied out.
constexpr int kW1Words = kDsRank * 2 / 16;   // uint4 of a w1 row: a lane each
static_assert(kW1Words == 32, "a w1 row is a warp's uint4");
__global__ void __launch_bounds__(512) head_final_kernel(DsHead h, const Q81* __restrict__ xh_g, const Q81* __restrict__ xl_g,
                                                         int n_tiles, const float2* __restrict__ cand, const float2* __restrict__ part,
                                                         int32_t* __restrict__ prev, int32_t* __restrict__ ids,
                                                         float* __restrict__ conf, int i, const uint16_t* __restrict__ w1,
                                                         uint16_t* __restrict__ w1_row) {
    __shared__ Q81 xh[kHeadNb], xl[kFoldNb];
    __shared__ px::Final f;
    __shared__ uint4 wr[px::kMaxCand][kW1Words];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    pdl_wait();
    if (tid == 0) f.cnt = 0;
    exact_x(xh_g, xl_g, xh, xl, tid, 512);
    const int n = collect_tiles(f, n_tiles, cand, part, prev, conf + i);
    if (n == 0) {   // never: the vocabulary has rows
        if (tid == 0) ids[i] = *prev;
        if (warp == 0) reinterpret_cast<uint4*>(w1_row)[lane] = reinterpret_cast<const uint4*>(w1 + (int64_t) *prev * kDsRank)[lane];
        return;
    }
    for (int c = warp; c < n; c += 16) {   // by token from here: the pick's tie goes to the lower token
        const int64_t tok = token_of(f.list[c], n_tiles);
        wr[c][lane] = reinterpret_cast<const uint4*>(w1 + tok * kDsRank)[lane];
        const float y = exact_logit(h, tok, xh, xl, lane);
        if (lane == 0) {
            f.ex[c] = y;
            f.list[c] = (int) tok;
        }
    }
    __syncthreads();
    px::pick_block<false>(f, n, n_tiles, nullptr, part, prev, conf + i);
    __syncthreads();
    const int pick = *prev;
    if (tid == 0) ids[i] = pick;
    for (int c = warp; c < n; c += 16)
        if (f.list[c] == pick) reinterpret_cast<uint4*>(w1_row)[lane] = wr[c][lane];
}

// by_row[r] = by_token[token_of(r)] (0 past the vocabulary).
__global__ void screen_rows_kernel(const float* __restrict__ by_token, int64_t n, int64_t tiles, float* __restrict__ by_row) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= tiles * 32) return;
    const int64_t tok = token_of(r, tiles);
    by_row[r] = tok < n ? by_token[tok] : 0.f;
}

// The reference: every row's exact logit, a warp per row.
__global__ void __launch_bounds__(256) head_exact_kernel(DsHead h, const Q81* __restrict__ xh_g, const Q81* __restrict__ xl_g,
                                                         float* __restrict__ logits) {
    __shared__ Q81 xh[kHeadNb], xl[kFoldNb];
    exact_x(xh_g, xl_g, xh, xl, threadIdx.x, 256);
    __syncthreads();
    const int64_t v = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (v >= h.n_vocab) return;
    const float y = exact_logit(h, v, xh, xl, threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) logits[v] = y;
}

// ---- the folds, once at load ---------------------------------------------------------------------------------------------

constexpr int kFoldRows = 32;   // head rows of a block
constexpr int kFoldK = 64;      // the width a step stages

// Block b: head rows 32 b .. 32 b + 31 times w; thread t is column t of the rank. Each warp's 32 columns of a row are one
// Q8_0 block of the output.
__device__ __forceinline__ void fold_stage(int b, const uint8_t* __restrict__ head, int64_t n_rows, const uint16_t* __restrict__ w,
                                           uint8_t* __restrict__ fold, float (*hs)[kFoldK]) {
    const int64_t r0 = (int64_t) b * kFoldRows;
    const int t = threadIdx.x;
    float acc[kFoldRows];
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) acc[r] = 0.f;
    for (int e0 = 0; e0 < kWidth; e0 += kFoldK) {
        for (int idx = t; idx < kFoldRows * kFoldK; idx += kDsRank) {
            const int rr = idx / kFoldK, e = e0 + idx % kFoldK;
            const int64_t row = r0 + rr;
            float val = 0.f;
            if (row < n_rows) {
                const uint8_t* blk = head + row * kQ8Row + (int64_t) (e / 32) * kQ8Block;
                val = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) (int8_t) blk[2 + e % 32];
            }
            hs[rr][idx % kFoldK] = val;
        }
        __syncthreads();
        float wv[kFoldK];
#pragma unroll
        for (int e = 0; e < kFoldK; ++e) wv[e] = bf16f(w[(int64_t) (e0 + e) * kDsRank + t]);
#pragma unroll
        for (int r = 0; r < kFoldRows; ++r) {
            float a = acc[r];
#pragma unroll
            for (int e = 0; e < kFoldK; ++e) a += hs[r][e] * wv[e];
            acc[r] = a;
        }
        __syncthreads();
    }
    const int lane = t & 31, wb = t >> 5;
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) {
        const int64_t row = r0 + r;
        if (row >= n_rows) break;
        const float x = acc[r];
        const float d = warp_max(fabsf(x)) / 127.f;   // ggml's quantize_row_q8_0_ref
        const float id = d != 0.f ? 1.f / d : 0.f;
        uint8_t* blk = fold + row * (kDsRank / 32 * kQ8Block) + (int64_t) wb * kQ8Block;
        blk[2 + lane] = (uint8_t) (int8_t) roundf(x * id);
        if (lane == 0) *reinterpret_cast<__half*>(blk) = __float2half(d);
    }
}

__global__ void __launch_bounds__(kDsRank) fold_kernel(const uint8_t* __restrict__ head, int64_t n_rows,
                                                       const uint16_t* __restrict__ w, uint8_t* __restrict__ fold) {
    __shared__ float hs[kFoldRows][kFoldK];
    fold_stage((int) blockIdx.x, head, n_rows, w, fold, hs);
}

// A warp per head row: fold_b[row] = head[row] . b.
__global__ void __launch_bounds__(256) fold_b_kernel(const uint8_t* __restrict__ head, int64_t n_rows, const float* __restrict__ b,
                                                     float* __restrict__ out) {
    const int64_t row = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_rows) return;
    const uint8_t* r = head + row * kQ8Row;
    float acc = 0.f;
    for (int k = 0; k < kWidth / 32; ++k) {
        const uint8_t* blk = r + (int64_t) k * kQ8Block;
        acc += __half2float(*reinterpret_cast<const __half*>(blk)) * (float) (int8_t) blk[2 + lane] * b[k * 32 + lane];
    }
    acc = warp_sum(acc);
    if (lane == 0) out[row] = acc;
}

__global__ void __launch_bounds__(256) split_rnn_kernel(const uint16_t* __restrict__ w, uint16_t* __restrict__ w_sp,
                                                        uint16_t* __restrict__ w_h, uint16_t* __restrict__ w_f) {
    constexpr int kIn = 2 * kDsRank + 2 * kWidth;
    const int64_t o = blockIdx.x;
    const uint16_t* src = w + o * kIn;
    for (int c = threadIdx.x; c < kIn; c += 256) {
        const uint16_t v = src[c];
        if (c < 2 * kDsRank) w_sp[o * 2 * kDsRank + c] = v;
        else if (c < 2 * kDsRank + kWidth) w_h[o * kWidth + c - 2 * kDsRank] = v;
        else w_f[o * kWidth + c - 2 * kDsRank - kWidth] = v;
    }
}

}  // namespace

void dspark_mix(const float* tap, int64_t tap_stride, int rows, const DsMixArgs& a, float* out, int64_t out_stride,
                const int32_t* pos, float* fr, int fr_cap, int32_t* pos3, void* stream) {
    if (rows <= 0) return;
    mix_kernel<<<dim3((unsigned) rows, kDsMixes), kThreads, 0, (cudaStream_t) stream>>>(tap, tap_stride, a, out, out_stride, pos, fr,
                                                                                         fr_cap, pos3);
    check("mix");
}

void dspark_block_inputs(const uint16_t* embedding, const int32_t* meta, const float* embed_norm, float eps, const float* mask,
                         int block, float* x, int32_t* pos3, int32_t* prev, float* state, const float* fr, int fr_cap, float* fr_out,
                         const uint16_t* w1, uint16_t* w1_row, void* stream) {
    if (block < 3) throw std::runtime_error("inkling2 dspark block inputs: a block of at least three rows");
    block_inputs_kernel<<<(unsigned) block, kThreads, 0, (cudaStream_t) stream>>>(embedding, meta, embed_norm, eps, mask, x, pos3, prev,
                                                                                    state, fr, fr_cap, fr_out, w1, w1_row);
    check("block inputs");
}

void dspark_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const uint16_t* w_sp, const uint16_t* w1_row,
                        const float* state_in, float* state_out, float* latent, void* stream) {
    markov_step_kernel<<<kStepBlocks, kStepWarps * 32, 0, (cudaStream_t) stream>>>(ctx_h, ctx_f, bias, w_sp, w1_row, state_in,
                                                                                    state_out, latent);
    check("Markov step");
}

size_t dspark_screen_bytes(int n_in, int64_t n_rows) { return px::bytes(n_rows, n_in); }

void dspark_screen_pack(const void* q8_0, int n_in, int64_t n_rows, void* screen, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    cudaMemsetAsync(screen, 0, px::bytes(n_rows, n_in), s);   // the last tile's rows past n_rows score as zeros
    const int64_t total = px::tiles(n_rows) * 32 * (n_in / 32);
    const unsigned grid = (unsigned) ((total + 127) / 128);
    uint16_t* sc = px::scales(screen, n_rows, n_in);
    if (n_in == kWidth)
        screen_pack_kernel<kWidth><<<grid, 128, 0, s>>>(static_cast<const uint8_t*>(q8_0), n_rows, (uint4*) screen, sc);
    else if (n_in == kDsRank)
        screen_pack_kernel<kDsRank><<<grid, 128, 0, s>>>(static_cast<const uint8_t*>(q8_0), n_rows, (uint4*) screen, sc);
    else
        throw std::runtime_error("inkling2 dspark screen: rows of " + std::to_string(n_in) + " values");
    check("screen pack");
}

int64_t dspark_screen_rows(int64_t n_vocab) { return px::tiles(n_vocab) * 32; }

void dspark_screen_order(const float* by_token, int64_t n_vocab, float* by_row, void* stream) {
    const int64_t n = dspark_screen_rows(n_vocab);
    screen_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(by_token, n_vocab, px::tiles(n_vocab), by_row);
    check("screen order");
}

size_t dspark_head_scratch_bytes(int64_t n_vocab) { return px::scratch_bytes(n_vocab); }

void dspark_head_base(const DsHead& h, const void* hq, int rows, float* approx, void* stream) {
    if (rows < 1 || rows > kDsDrafts) throw std::runtime_error("inkling2 dspark head: 1 to 7 rows");
    launch_pdl(head_base_kernel, dim3((unsigned) px::tiles(h.n_vocab)), dim3(128), 0, (cudaStream_t) stream, (const uint4*) h.head_sc,
               (const uint32_t*) px::scales(const_cast<void*>(h.head_sc), h.n_vocab, kWidth), (const Q81*) hq, rows,
               dspark_screen_rows(h.n_vocab), approx);
    check("head base");
}

void dspark_head_pick(const DsHead& h, const void* hq, const void* latq, const float* approx, int i, void* scratch, int32_t* prev,
                      int32_t* ids, float* conf, const uint16_t* w1, uint16_t* w1_row, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const int tiles = (int) px::tiles(h.n_vocab);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) tiles * px::kPer;
    launch_pdl(head_step_kernel, dim3((unsigned) ((tiles + kTilesPerBlock - 1) / kTilesPerBlock)), dim3(32 * kTilesPerBlock), 0, s,
               (const uint4*) h.fold_sc, (const uint32_t*) px::scales(const_cast<void*>(h.fold_sc), h.n_vocab, kDsRank),
               (const Q81*) latq, approx + (size_t) i * dspark_screen_rows(h.n_vocab), h.fold_bp, h.cap, h.n_vocab, tiles, cand,
               part);
    launch_pdl(head_final_kernel, dim3(1), dim3(512), 0, s, h, (const Q81*) hq + (size_t) i * kHeadNb, (const Q81*) latq, tiles,
               (const float2*) cand, (const float2*) part, prev, ids, conf, i, w1, w1_row);
    check("head pick");
}

void dspark_head_exact(const DsHead& h, const void* hq, const void* latq, int i, float* logits, void* stream) {
    head_exact_kernel<<<(unsigned) ((h.n_vocab + 7) / 8), 256, 0, (cudaStream_t) stream>>>(
        h, (const Q81*) hq + (size_t) i * kHeadNb, (const Q81*) latq, logits);
    check("head exact");
}

void dspark_fold(const void* head_q8_0, int64_t n_rows, const uint16_t* w, const float* b, void* fold_q8_0, float* fold_b,
                 void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    fold_kernel<<<(unsigned) ((n_rows + kFoldRows - 1) / kFoldRows), kDsRank, 0, s>>>(
        static_cast<const uint8_t*>(head_q8_0), n_rows, w, static_cast<uint8_t*>(fold_q8_0));
    check("fold");
    fold_b_kernel<<<(unsigned) ((n_rows + 7) / 8), 256, 0, s>>>(static_cast<const uint8_t*>(head_q8_0), n_rows, b, fold_b);
    check("fold bias");
}

void dspark_split_rnn(const uint16_t* w, uint16_t* w_sp, uint16_t* w_h, uint16_t* w_f, void* stream) {
    split_rnn_kernel<<<kDsRnn, 256, 0, (cudaStream_t) stream>>>(w, w_sp, w_h, w_f);
    check("split");
}

// ---- batched multi-session commit & draft kernels -----------------------------------------------------------------------

namespace {

struct DsRopeFreq {
    float inv[kHeadDim / 2];
};

DsRopeFreq make_ds_rope(float base) {
    DsRopeFreq f{};
    for (int i = 0; i < kHeadDim / 2; ++i) f.inv[i] = (float) std::pow((double) base, -2.0 * (double) i / (double) kHeadDim);
    return f;
}

__device__ __forceinline__ float block_sum_128(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[w] = v;
    __syncthreads();
    const float s = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return s;
}

__global__ void __launch_bounds__(kThreads) mix_batch_kernel(const float* tap, int64_t tap_stride, DsMixArgs a,
                                                             float* __restrict__ out, int64_t out_stride,
                                                             const int32_t* pos, DsCommitSegs segs, int fr_cap) {
    pdl_wait();
    __shared__ float red[kThreads / 32];
    const int r = (int) blockIdx.x, j = (int) blockIdx.y;
    const int sk = segs.at_d ? basalt::kernels::seg_of(*segs.at_d, r) : 0;
    mix_stage(r, j, tap, tap_stride, a, out, out_stride, pos, segs.fr[sk], fr_cap, nullptr, red);
}

__global__ void __launch_bounds__(kHeadDim) kv_commit_batch_kernel(const float* kb, const float* vb,
                                                                   const float* __restrict__ k_norm, float eps,
                                                                   DsRopeFreq rope, const int32_t* pos, DsCommitSegs segs,
                                                                   int64_t cap) {
    pdl_wait();
    __shared__ float red[kHeadDim / 32];
    __shared__ float xs[kHeadDim];
    const int t = (int) blockIdx.x, head = (int) blockIdx.y, d = (int) threadIdx.x;
    const float x = kb[((int64_t) t * kKvHeads + head) * kHeadDim + d];
    const float inv = rsqrtf(block_sum_128(x * x, red) / (float) kHeadDim + eps);
    float y = x * inv * k_norm[d];
    xs[d] = y;
    __syncthreads();
    constexpr int half = kHeadDim / 2;
    const int i = d < half ? d : d - half;
    const int32_t p = pos[t];
    float sn, cs;
    sincosf((float) p * rope.inv[i], &sn, &cs);
    y = d < half ? y * cs - xs[d + half] * sn : y * cs + xs[d - half] * sn;
    const int sk = segs.at_d ? basalt::kernels::seg_of(*segs.at_d, t) : 0;
    const int64_t cell = cap > 0 ? (int64_t) p % cap : (int64_t) p;
    const int64_t at = (cell * kKvHeads + head) * kHeadDim + d;
    reinterpret_cast<__half*>(segs.k[sk])[at] = __float2half_rn(y);
    reinterpret_cast<__half*>(segs.v[sk])[at] = __float2half_rn(vb[((int64_t) t * kKvHeads + head) * kHeadDim + d]);
}

__global__ void __launch_bounds__(kThreads) block_inputs_batch_kernel(const uint16_t* __restrict__ emb,
                                                                      const int32_t* meta,
                                                                      const float* __restrict__ norm, float eps,
                                                                      const float* __restrict__ mask, int block,
                                                                      float* __restrict__ x, int32_t* __restrict__ prev,
                                                                      float* __restrict__ state, DsBatchFr frs, int fr_cap,
                                                                      float* __restrict__ fr_out,
                                                                      const uint16_t* __restrict__ w1,
                                                                      uint16_t* __restrict__ w1_row) {
    __shared__ float red[kThreads / 32];
    const int t = (int) blockIdx.x, b = (int) blockIdx.y;
    block_inputs_stage(t, emb, meta + 2 * b, norm, eps, mask, x + (int64_t) b * block * kWidth, nullptr, prev + b,
                       state + (int64_t) b * 2 * kDsRank, frs.fr[b], fr_cap, fr_out + (int64_t) b * kWidth, w1,
                       w1_row + (int64_t) b * kDsRank, red);
}

__global__ void __launch_bounds__(kHeadDim) block_qk_prep_batch_kernel(const float* q, const float* kb, const float* vb,
                                                                       const float* __restrict__ q_norm,
                                                                       const float* __restrict__ k_norm, float eps,
                                                                       DsRopeFreq rope, const int32_t* meta,
                                                                       int block, float* q_out, DsBatchKv kv, int64_t cap) {
    pdl_wait();
    __shared__ float red[kHeadDim / 32];
    __shared__ float xs[kHeadDim];
    const int row = (int) blockIdx.x, j = (int) blockIdx.y, d = (int) threadIdx.x;
    const int b = row / block, t = row - b * block;
    const bool is_q = j < kHeads;
    const int head = is_q ? j : j - kHeads;
    const float x = is_q ? q[((int64_t) row * kHeads + head) * kHeadDim + d]
                         : kb[((int64_t) row * kKvHeads + head) * kHeadDim + d];
    const float inv = rsqrtf(block_sum_128(x * x, red) / (float) kHeadDim + eps);
    float y = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    xs[d] = y;
    __syncthreads();
    constexpr int half = kHeadDim / 2;
    const int i = d < half ? d : d - half;
    const int32_t p = meta[2 * b + 1] + t;
    float sn, cs;
    sincosf((float) p * rope.inv[i], &sn, &cs);
    y = d < half ? y * cs - xs[d + half] * sn : y * cs + xs[d - half] * sn;
    if (is_q) {
        q_out[((int64_t) row * kHeads + head) * kHeadDim + d] = y;
        return;
    }
    const int64_t cell = cap > 0 ? (int64_t) p % cap : (int64_t) p;
    const int64_t at = (cell * kKvHeads + head) * kHeadDim + d;
    reinterpret_cast<__half*>(kv.k[b])[at] = __float2half_rn(y);
    reinterpret_cast<__half*>(kv.v[b])[at] = __float2half_rn(vb[((int64_t) row * kKvHeads + head) * kHeadDim + d]);
}

__global__ void __launch_bounds__(256) pack_draft_rows_kernel(const float* h_chunk, int block, int slot0,
                                                              int n_max, float* __restrict__ hd_all) {
    pdl_wait();
    const int row = (int) blockIdx.x;
    const int c = row / n_max, i = row - c * n_max;
    const float4* src = reinterpret_cast<const float4*>(h_chunk + ((int64_t) c * block + 1 + i) * kWidth);
    float4* dst = reinterpret_cast<float4*>(hd_all + ((int64_t) (slot0 + c) * n_max + i) * kWidth);
    for (int j = threadIdx.x; j < kWidth / 4; j += 256) dst[j] = src[j];
}

// Reads the 2-bit head screen once into registers and scores all `count` sessions' `n_max` draft rows.
__global__ void __launch_bounds__(128) head_base_batch_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                              const Q81* x, int count, int n_max,
                                                              int64_t n_pad, float* __restrict__ approx) {
    constexpr int kNw4 = kWidth / 64, NW = kNw4 / 4;
    __shared__ __align__(16) int xi[kDsDrafts][kWidth / 4];
    __shared__ float xs[kDsDrafts][kHeadNb], xdd[kDsDrafts][kHeadNb];
    __shared__ float red[kDsDrafts][4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t tile = blockIdx.x;
    const size_t base = ((size_t) tile * kNw4 + warp * NW) * 32 + lane;
    uint4 w[NW];
    uint32_t s[NW];
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        w[j] = __ldcs(wq + base + (size_t) j * 32);
        s[j] = __ldcs(sc + base + (size_t) j * 32);
    }
    pdl_wait();
    const int64_t row = tile * 32 + lane;
    for (int b = 0; b < count; ++b) {
        if (b > 0) __syncthreads();
        const Q81* xb = x + (size_t) b * n_max * kHeadNb;
        for (int r = 0; r < n_max; ++r) screen_x(xb + (size_t) r * kHeadNb, kHeadNb, xi[r], xs[r], xdd[r], tid, 128);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kDsDrafts; ++r)
            if (r < n_max) red[r][warp][lane] = screen_acc<NW>(w, s, warp * NW, xi[r], xs[r], xdd[r]);
        __syncthreads();
        float* apb = approx + (size_t) b * n_max * n_pad;
        for (int r = warp; r < n_max; r += 4)
            apb[r * n_pad + row] = red[r][0][lane] + red[r][1][lane] + red[r][2][lane] + red[r][3][lane];
    }
}

__global__ void __launch_bounds__(kStepWarps * 32) markov_step_batch_kernel(const float* ctx_h,
                                                                            const float* ctx_f,
                                                                            const float* __restrict__ bias,
                                                                            const uint16_t* __restrict__ w_sp,
                                                                            const uint16_t* w1_row,
                                                                            float* state, int step_i,
                                                                            int n_max, float* __restrict__ latent) {
    pdl_wait();
    __shared__ float xs[2 * kDsRank];
    const int blk = (int) blockIdx.x, b = (int) blockIdx.y;
    const float* st_in = state + ((size_t) b * 2 + (step_i & 1)) * kDsRank;
    float* st_out = state + ((size_t) b * 2 + ((step_i + 1) & 1)) * kDsRank;
    markov_step_stage(blk, ctx_h + ((size_t) b * n_max + step_i) * kDsRnn, ctx_f + (size_t) b * kDsRnn, bias, w_sp,
                      w1_row + (size_t) b * kDsRank, st_in, st_out, latent + (size_t) b * kDsRank, xs);
}

__global__ void __launch_bounds__(32 * kTilesPerBlock) head_step_batch_kernel(const uint4* __restrict__ wq,
                                                                              const uint32_t* __restrict__ sc,
                                                                              const Q81* lat_step,
                                                                              const float* approx,
                                                                              const float* __restrict__ fold_bp, float cap,
                                                                              int64_t n_rows, int n_tiles, int step_i,
                                                                              int n_max, int64_t n_pad,
                                                                              uint8_t* __restrict__ scratch,
                                                                              size_t scratch_stride) {
    constexpr int kNw4 = kDsRank / 64;
    __shared__ __align__(16) int xi[kDsRank / 4];
    __shared__ float xs[kFoldNb], xdd[kFoldNb];
    const int tid = threadIdx.x, lane = tid & 31, b = (int) blockIdx.y;
    const int tile = (int) blockIdx.x * kTilesPerBlock + (tid >> 5);
    const int64_t row = (int64_t) tile * 32 + lane;
    const bool live = tile < n_tiles, real = live && token_of(row, n_tiles) < n_rows;
    uint4 w[kNw4];
    uint32_t s[kNw4];
    float fb = 0.f;
#pragma unroll
    for (int j = 0; j < kNw4; ++j) {
        const size_t at = ((size_t) (live ? tile : 0) * kNw4 + j) * 32 + lane;
        w[j] = __ldcs(wq + at);
        s[j] = __ldcs(sc + at);
    }
    if (live) fb = __ldg(fold_bp + row);
    pdl_wait();
    pdl_launch_dependents();
    screen_x(lat_step + (size_t) b * kFoldNb, kFoldNb, xi, xs, xdd, tid, 32 * kTilesPerBlock);
    __syncthreads();
    if (!live) return;
    float2* cand = reinterpret_cast<float2*>(scratch + (size_t) b * scratch_stride);
    float2* part = cand + (size_t) n_tiles * px::kPer;
    const float* ap = approx + ((size_t) b * n_max + step_i) * n_pad;
    const float ninf = __int_as_float(0xff800000);
    float v = ninf;
    if (real) v = capf(ap[row] + screen_acc<kNw4>(w, s, 0, xi, xs, xdd) + fb, cap);
    float m = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float e = v > ninf ? __expf(v - m) : 0.0f;
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) part[tile] = make_float2(m, e);
    for (int k = 0; k < px::kPer; ++k) {
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * px::kPer + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = ninf;
    }
}

__global__ void __launch_bounds__(512) head_final_batch_kernel(DsHead h, const Q81* __restrict__ hq,
                                                               const Q81* __restrict__ lat_step, int n_tiles, int step_i,
                                                               int n_max, const uint8_t* __restrict__ scratch,
                                                               size_t scratch_stride, int32_t* __restrict__ prev,
                                                               int32_t* __restrict__ ids, float* __restrict__ conf,
                                                               const uint16_t* __restrict__ w1,
                                                               uint16_t* __restrict__ w1_row) {
    __shared__ Q81 xh[kHeadNb], xl[kFoldNb];
    __shared__ px::Final f;
    __shared__ uint4 wr[px::kMaxCand][kW1Words];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = (int) blockIdx.x;
    pdl_wait();
    pdl_launch_dependents();
    if (tid == 0) f.cnt = 0;
    const Q81* xh_g = hq + ((size_t) b * n_max + step_i) * kHeadNb;
    const Q81* xl_g = lat_step + (size_t) b * kFoldNb;
    exact_x(xh_g, xl_g, xh, xl, tid, 512);
    const float2* cand = reinterpret_cast<const float2*>(scratch + (size_t) b * scratch_stride);
    const float2* part = cand + (size_t) n_tiles * px::kPer;
    int32_t* prev_b = prev + b;
    int32_t* ids_b = ids + (size_t) b * kDsBlock;
    float* conf_b = conf + (size_t) b * kDsBlock;
    uint16_t* w1_b = w1_row + (size_t) b * kDsRank;
    const int n = collect_tiles(f, n_tiles, cand, part, prev_b, conf_b + step_i);
    if (n == 0) {
        if (tid == 0) ids_b[step_i] = *prev_b;
        if (warp == 0) reinterpret_cast<uint4*>(w1_b)[lane] = reinterpret_cast<const uint4*>(w1 + (int64_t) *prev_b * kDsRank)[lane];
        return;
    }
    for (int c = warp; c < n; c += 16) {
        const int64_t tok = token_of(f.list[c], n_tiles);
        wr[c][lane] = reinterpret_cast<const uint4*>(w1 + tok * kDsRank)[lane];
        const float y = exact_logit(h, tok, xh, xl, lane);
        if (lane == 0) {
            f.ex[c] = y;
            f.list[c] = (int) tok;
        }
    }
    __syncthreads();
    px::pick_block<false>(f, n, n_tiles, nullptr, part, prev_b, conf_b + step_i);
    __syncthreads();
    const int pick = *prev_b;
    if (tid == 0) ids_b[step_i] = pick;
    for (int c = warp; c < n; c += 16)
        if (f.list[c] == pick) reinterpret_cast<uint4*>(w1_b)[lane] = wr[c][lane];
}

}  // namespace

void dspark_mix_batch(const float* tap, int64_t tap_stride, int rows, const DsMixArgs& a, float* out, int64_t out_stride,
                      const int32_t* pos, const DsCommitSegs& segs, int fr_cap, void* stream) {
    if (rows <= 0) return;
    launch_pdl(mix_batch_kernel, dim3((unsigned) rows, kDsMixes), dim3(kThreads), 0, (cudaStream_t) stream, tap, tap_stride, a,
               out, out_stride, pos, segs, fr_cap);
    check("mix batch");
}

void dspark_kv_commit_batch(const float* kb, const float* vb, const float* k_norm, float eps, float rope_base,
                            const int32_t* pos, int rows, const DsCommitSegs& segs, int64_t cap, void* stream) {
    if (rows <= 0) return;
    launch_pdl(kv_commit_batch_kernel, dim3((unsigned) rows, kKvHeads), dim3(kHeadDim), 0, (cudaStream_t) stream, kb, vb,
               k_norm, eps, make_ds_rope(rope_base), pos, segs, cap);
    check("kv commit batch");
}

void dspark_block_inputs_batch(const uint16_t* embedding, const int32_t* meta, const float* embed_norm, float eps,
                               const float* mask, int block, int count, float* x, int32_t* prev, float* state,
                               const DsBatchFr& frs, int fr_cap, float* fr_out, const uint16_t* w1, uint16_t* w1_row,
                               void* stream) {
    if (count <= 0) return;
    block_inputs_batch_kernel<<<dim3((unsigned) block, (unsigned) count), kThreads, 0, (cudaStream_t) stream>>>(
        embedding, meta, embed_norm, eps, mask, block, x, prev, state, frs, fr_cap, fr_out, w1, w1_row);
    check("block inputs batch");
}

void dspark_block_qk_prep_batch(const float* q, const float* kb, const float* vb, const float* q_norm, const float* k_norm,
                                float eps, float rope_base, const int32_t* meta, int block, int count, float* q_out,
                                const DsBatchKv& kv, int64_t cap, void* stream) {
    if (count <= 0) return;
    launch_pdl(block_qk_prep_batch_kernel, dim3((unsigned) (count * block), kHeads + kKvHeads), dim3(kHeadDim), 0,
               (cudaStream_t) stream, q, kb, vb, q_norm, k_norm, eps, make_ds_rope(rope_base), meta, block, q_out, kv, cap);
    check("block qk prep batch");
}

void dspark_pack_draft_rows(const float* h_chunk, int block, int count_chunk, int slot0, int n_max, float* hd_all,
                            void* stream) {
    if (count_chunk <= 0 || n_max <= 0) return;
    launch_pdl(pack_draft_rows_kernel, dim3((unsigned) (count_chunk * n_max)), dim3(256), 0, (cudaStream_t) stream, h_chunk,
               block, slot0, n_max, hd_all);
    check("pack draft rows");
}

void dspark_head_base_batch(const DsHead& h, const void* hq, int count, int n_max, float* approx, void* stream) {
    if (count <= 0 || n_max < 1 || n_max > kDsDrafts) return;
    launch_pdl(head_base_batch_kernel, dim3((unsigned) px::tiles(h.n_vocab)), dim3(128), 0, (cudaStream_t) stream,
               (const uint4*) h.head_sc, (const uint32_t*) px::scales(const_cast<void*>(h.head_sc), h.n_vocab, kWidth),
               (const Q81*) hq, count, n_max, dspark_screen_rows(h.n_vocab), approx);
    check("head base batch");
}

void dspark_markov_step_batch(const float* ctx_h, const float* ctx_f, const float* bias, const uint16_t* w_sp,
                              const uint16_t* w1_row, float* state, int step_i, int n_max, int count, float* latent,
                              void* stream) {
    if (count <= 0) return;
    launch_pdl(markov_step_batch_kernel, dim3(kStepBlocks, (unsigned) count), dim3(kStepWarps * 32), 0, (cudaStream_t) stream,
               ctx_h, ctx_f, bias, w_sp, w1_row, state, step_i, n_max, latent);
    check("Markov step batch");
}

void dspark_head_pick_batch(const DsHead& h, const void* hq, const void* latq_step, const float* approx, int step_i,
                            int n_max, int count, void* scratch, size_t scratch_stride, int32_t* prev, int32_t* ids,
                            float* conf, const uint16_t* w1, uint16_t* w1_row, void* stream) {
    if (count <= 0) return;
    const cudaStream_t s = (cudaStream_t) stream;
    const int tiles = (int) px::tiles(h.n_vocab);
    const int64_t n_pad = dspark_screen_rows(h.n_vocab);
    launch_pdl(head_step_batch_kernel, dim3((unsigned) ((tiles + kTilesPerBlock - 1) / kTilesPerBlock), (unsigned) count),
               dim3(32 * kTilesPerBlock), 0, s, (const uint4*) h.fold_sc,
               (const uint32_t*) px::scales(const_cast<void*>(h.fold_sc), h.n_vocab, kDsRank), (const Q81*) latq_step,
               approx, h.fold_bp, h.cap, h.n_vocab, tiles, step_i, n_max, n_pad, (uint8_t*) scratch, scratch_stride);
    launch_pdl(head_final_batch_kernel, dim3((unsigned) count), dim3(512), 0, s, h, (const Q81*) hq, (const Q81*) latq_step,
               tiles, step_i, n_max, (const uint8_t*) scratch, scratch_stride, prev, ids, conf, w1, w1_row);
    check("head pick batch");
}

}  // namespace basalt::models::inkling2
