// src/kernels/cuda/sampler.cu - the sampler chain, in llama.cpp's order.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose default
// is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } - ONE penalties stage, first,
// and TEMPERATURE AFTER THE TRUNCATION FILTERS.  Every order produces a valid token, so only a comparison at the
// distribution level can tell them apart; the parity test does that against an independently computed distribution.
//
// `sampler_greedy_kernel` is the plain argmax, one block per token over the vocabulary (with thread-block clusters and without penalties,
// `sampler_greedy_cluster_kernel`: the same token from a cluster of 8 CTAs per row).  The sampled chain has these
// implementations, which pick the same token, bit for bit:
//   - the SORTED SPLIT (top_k <= 32, the engine's): `sampler_sort_kernel` cuts each row into 4,096-logit blocks over
//     the whole GPU, each finds its first k with bitonic networks, and the row's last block merges the lists and runs
//     the tail (`sampled_tail_pair`), all in one launch;
//   - the ROUNDS SPLIT (a wider top_k): `sampler_fused_kernel`, the same blocks with k warp-argmax rounds each, also in
//     one launch, and `sampled_tail_warp`;
//   - the two-launch split (the parity test's reference): `sampler_split_part_kernel` keeps each block's top_k and
//     `sampler_split_merge_kernel` merges the lists and runs `sampled_tail_warp`;
//   - `sampler_one_block_kernel` (the fallback when no split can run): one block per token, `top_k` block-argmax
//     rounds, each over the logits after the previous pick.
// `sampled_tail_warp` and `sampled_tail_pair` run top_p / min_p / temperature / draw with the same arithmetic.
#include "basalt/kernels/sampler.hpp"
#include "basalt/core/diag.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

// Philox 4x32-10, a counter-based generator.  Counter-based matters because it makes the stream a function of
// (seed, position) rather than of how many draws came before - so a batch can be sampled in any order and a run is
// reproducible.
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

__device__ __forceinline__ float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

/// /// **THE GREEDY ARGMAX, ONE BLOCK PER TOKEN, COVERING THE VOCABULARY.**
/// ///
/// /// **WHY THIS IS A SEPARATE KERNEL AND NOT A BRANCH.**  The decode path has `n_tokens == 1`, so a grid over TOKENS
/// /// with 64 threads would leave 63 threads exiting at once and ONE THREAD walking the whole vocabulary in a
/// /// dependent loop on one SM.  Parallelising the scan inside such a kernel is WRONG: with one thread per token, a
/// /// block reduction over the vocabulary has nothing to reduce, and the threads that returned early are not there for
/// /// `__syncthreads` or `__shfl_down_sync`.  The grid has to be over tokens with the BLOCK over the vocabulary, which
/// /// is a different launch configuration and therefore a different kernel.
/// ///
/// /// **THE TIE RULE IS THE WHOLE CORRECTNESS ARGUMENT.**  A serial scan walks `v` ascending with `if (s > bv)`, so
/// /// the LOWEST index wins a tie.  Each thread keeps that rule over its own strided subset and the reduction resolves
/// /// two candidates by taking the larger value and, on equality, the SMALLER index - the same total order.
__global__ void sampler_greedy_kernel(const float* __restrict__ logits, int n_vocab,
                                      const int* __restrict__ history, int history_len, const SamplerParams p,
                                      int pmin, int plen, int* __restrict__ out) {
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    (void) pmin;
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // PENALTY MEMBERSHIP AS A BITMAP.  The history touches at most `hlen` tokens of a quarter-million vocabulary, but
    // a `history_count` per candidate per argmax round costs O(k x n_vocab x hlen) integer compares.  A shared bitmap
    // gives an O(1) membership test, and only the (at most hlen) hits pay the count scan; the counts - and therefore
    // every sampled value - are exactly what the per-candidate scan produces.
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // The gate needs a NON-EMPTY WINDOW (`hlen > 0`): the launch sizes the shared bitmap only when penalties
    // are on, so a caller handing over a history buffer with `penalty_last_n == 0` must not touch it.
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        const float s = apply_penalties(p.logit_bias ? l[v] + p.logit_bias[v] : l[v], hit_count(v), p);
        if (s > bv) { bv = s; best = v; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) ((blockDim.x + 31) >> 5);
        float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        int wi = lane < nw ? si[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
        }
        // A tie between two `-inf` candidates leaves `wi == n_vocab`, and the serial version answered 0.
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

/// /// **THE SAME ARGMAX ON A THREAD-BLOCK CLUSTER, FOR THE CALLS WITHOUT PENALTIES.**  One block per token
/// /// reading a long row on one SM with one load in flight per thread is latency-bound, for every verify window and
/// /// every draft step.  Here a cluster of `kAmCtas` CTAs shares the row, each thread keeps four loads in flight, and
/// /// the CTAs' results meet in CTA 0's shared memory (distributed shared memory).
/// ///
/// /// The answer is a function of the row alone, so it is the one-block kernel's bit for bit: the LOWEST index whose value
/// /// is the largest non-NaN value above -inf (each thread walks its elements in ascending order with a strict `>`, and
/// /// every merge takes the larger value or, on equality, the smaller index - an order-free rule), and 0 when there is
/// /// none (all -inf / NaN), as there.  NaN never wins a `>`.
/// ///
/// /// Barriers: a relaxed cluster arrive at entry, waited before the remote store (CTA 0 must be running); each CTA's
/// /// result goes to slot `rank` of CTA 0, released by the next arrive; CTAs 1.. then exit (a cluster wait counts the
/// /// threads that have not exited) and CTA 0 waits, reads its own slots, and writes the token.
constexpr int kAmCtas = 8;      // CTAs per token (the portable cluster size)
constexpr int kAmThreads = 1024;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define AM_CLUSTER 1
#else
#define AM_CLUSTER 0     // older targets: a trap, never launched (sample_greedy_cluster checks)
#endif
#if AM_CLUSTER
__device__ __forceinline__ void am_take(float s, int v, float& bv, int& best) {
    if (s > bv) { bv = s; best = v; }
}
__device__ __forceinline__ void am_merge(float ov, int oi, float& bv, int& best) {
    if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
}
#endif
// grid (kAmCtas, n_tokens), cluster (kAmCtas, 1, 1), kAmThreads threads
__global__ void __launch_bounds__(kAmThreads) sampler_greedy_cluster_kernel(const float* __restrict__ logits,
                                                                            int n_vocab, int* __restrict__ out) {
#if AM_CLUSTER
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float cv[kAmCtas];    // on CTA 0: each CTA's result
    __shared__ int ci[kAmCtas];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    const float* l = logits + (size_t) blockIdx.y * n_vocab;
    float bv = __int_as_float(0xff800000);   // -inf; `n_vocab` is "no candidate", as in sampler_greedy_kernel
    int best = n_vocab;
    constexpr int S = kAmCtas * kAmThreads;
    int v = (int) rank * kAmThreads + (int) threadIdx.x;
    for (; v + 3 * S < n_vocab; v += 4 * S) {   // four independent loads, then taken in ascending order
        const float x0 = l[v], x1 = l[v + S], x2 = l[v + 2 * S], x3 = l[v + 3 * S];
        am_take(x0, v, bv, best);
        am_take(x1, v + S, bv, best);
        am_take(x2, v + 2 * S, bv, best);
        am_take(x3, v + 3 * S, bv, best);
    }
    for (; v < n_vocab; v += S) am_take(l[v], v, bv, best);
    for (int off = 16; off > 0; off >>= 1)
        am_merge(__shfl_down_sync(0xFFFFFFFFu, bv, off), __shfl_down_sync(0xFFFFFFFFu, best, off), bv, best);
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        bv = sv[lane];
        best = si[lane];
        for (int off = 16; off > 0; off >>= 1)
            am_merge(__shfl_down_sync(0xFFFFFFFFu, bv, off), __shfl_down_sync(0xFFFFFFFFu, best, off), bv, best);
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");   // CTA 0 runs
    if (threadIdx.x == 0) {
        uint64_t a;
        asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(a) : "l"((uint64_t) &cv[rank]), "r"(0u));
        *reinterpret_cast<float*>(a) = bv;
        asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(a) : "l"((uint64_t) &ci[rank]), "r"(0u));
        *reinterpret_cast<int*>(a) = best;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    if (rank != 0) return;
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    if (warp == 0) {
        bv = lane < kAmCtas ? cv[lane] : __int_as_float(0xff800000);
        best = lane < kAmCtas ? ci[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1)
            am_merge(__shfl_down_sync(0xFFFFFFFFu, bv, off), __shfl_down_sync(0xFFFFFFFFu, best, off), bv, best);
        if (lane == 0) out[blockIdx.y] = (best < n_vocab) ? best : 0;
    }
#else
    (void) logits; (void) n_vocab; (void) out;
    __trap();
#endif
}

/// The cluster argmax above, then on CTA 0 the pick's probability with row_top_prob_kernel's arithmetic in its order
/// (kAmThreads threads, the same strided sums, warp tree and serial finish), then the subset map: one launch for what
/// was three, every output bitwise the same.  grid (kAmCtas, n_tokens), cluster (kAmCtas, 1, 1), kAmThreads threads.
__global__ void __launch_bounds__(kAmThreads) sampler_greedy_top_prob_cluster_kernel(const float* __restrict__ logits,
                                                                                     int n_vocab,
                                                                                     const int32_t* __restrict__ map,
                                                                                     int* __restrict__ out,
                                                                                     float* __restrict__ probs) {
#if AM_CLUSTER
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float cv[kAmCtas];
    __shared__ int ci[kAmCtas];
    __shared__ float part[32];
    __shared__ int pick;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    const float* l = logits + (size_t) blockIdx.y * n_vocab;
    float bv = __int_as_float(0xff800000);
    int best = n_vocab;
    constexpr int S = kAmCtas * kAmThreads;
    int v = (int) rank * kAmThreads + (int) threadIdx.x;
    for (; v + 3 * S < n_vocab; v += 4 * S) {
        const float x0 = l[v], x1 = l[v + S], x2 = l[v + 2 * S], x3 = l[v + 3 * S];
        am_take(x0, v, bv, best);
        am_take(x1, v + S, bv, best);
        am_take(x2, v + 2 * S, bv, best);
        am_take(x3, v + 3 * S, bv, best);
    }
    for (; v < n_vocab; v += S) am_take(l[v], v, bv, best);
    for (int off = 16; off > 0; off >>= 1)
        am_merge(__shfl_down_sync(0xFFFFFFFFu, bv, off), __shfl_down_sync(0xFFFFFFFFu, best, off), bv, best);
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        bv = sv[lane];
        best = si[lane];
        for (int off = 16; off > 0; off >>= 1)
            am_merge(__shfl_down_sync(0xFFFFFFFFu, bv, off), __shfl_down_sync(0xFFFFFFFFu, best, off), bv, best);
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    if (threadIdx.x == 0) {
        uint64_t a;
        asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(a) : "l"((uint64_t) &cv[rank]), "r"(0u));
        *reinterpret_cast<float*>(a) = bv;
        asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(a) : "l"((uint64_t) &ci[rank]), "r"(0u));
        *reinterpret_cast<int*>(a) = best;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    if (rank != 0) return;
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    if (warp == 0) {
        bv = lane < kAmCtas ? cv[lane] : __int_as_float(0xff800000);
        best = lane < kAmCtas ? ci[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1)
            am_merge(__shfl_down_sync(0xFFFFFFFFu, bv, off), __shfl_down_sync(0xFFFFFFFFu, best, off), bv, best);
        if (lane == 0) pick = (best < n_vocab) ? best : 0;
    }
    __syncthreads();
    const int id = pick;
    // row_top_prob_kernel's loop and reductions, unchanged
    const float m = l[id];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[blockIdx.y] = 1.0f / tot;
        out[blockIdx.y] = map != nullptr ? map[id] : id;
    }
#else
    (void) logits; (void) n_vocab; (void) map; (void) out; (void) probs;
    __trap();
#endif
}

// ---- the sampled list, in selection order ----
//
// THE SELECTION ORDER.  The rounds rank a candidate (value, id) before (value', id') when value > value', or value ==
// value' and id < id' - each thread's ascending scan keeps the first maximum it meets (strict `>`), and the reductions
// resolve a tie to the lower id.  That is a strict total order (-0 and +0 compare equal and fall to the id), so round
// i picks the i-th logit of the order.  NaN and -inf are never picked (`s > bv` from -inf fails); a round that finds
// nothing yields the SENTINEL (-inf, n_vocab), stored as id 0.

constexpr int kSelMax = 64;   // the widest top_k list
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
__host__ __device__ __forceinline__ int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// (bv, bi) <- the first of (bv, bi) and (ov, oi) in the selection order
__device__ __forceinline__ void take_first(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// The first of the warp's 32 candidates, left in EVERY lane.  An XOR butterfly is exact here: "the first of two" is
// associative and commutative in a strict total order (two lanes never hold different candidates that compare
// equal - an id is in one lane at most, and two sentinels are the same pair), so every lane ends with the same pair.
__device__ __forceinline__ void warp_first(float& bv, int& bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(kFullMask, bv, off);
        const int oi = __shfl_xor_sync(kFullMask, bi, off);
        take_first(bv, bi, ov, oi);
    }
}

/// /// **THE TAIL ON ONE WARP.**  `sel_ids` / `sel_logit` (shared, `k` entries) are the top_k list in selection order;
/// /// the 32 lanes of one warp call this and lane 0 writes `out[t]`.  The tail's doubles run on one warp (FP64 issues
/// /// at a small fraction of FP32): the lanes share the `exp`s (one per entry, into `ex`) and then the quotients, and
/// /// lane 0 alone runs the two ORDERED sums and the two cumulative scans: the same `exp` of the same argument, the
/// /// sums in the same order, `cum += e / sum` with the same correctly rounded quotient.  (`n_keep == 0`, reachable
/// /// only with min_p > 1, which the callers clamp, reads `sel_ids[0]`.)
/// /// `qprob` (optional): for t < nq, qprob[t] <- the final probability of token query[t] (0 outside the kept list),
/// /// written after the pick, which it does not touch.
template <bool kProb = false>
__device__ void sampled_tail_warp(const int* sel_ids, const float* sel_logit, int k, const SamplerParams& p, int t,
                                  int* __restrict__ out, double* ex, float* prob_out = nullptr,
                                  const int* query = nullptr, int nq = 0, float* qprob = nullptr) {
    const int lane = (int) (threadIdx.x & 31);
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = lane; i < k; i += 32) ex[i] = exp((double) sel_logit[i] - (double) mx);
        __syncwarp();
        double sum = 0.0;
        if (lane == 0)
            for (int i = 0; i < k; ++i) sum += ex[i];
        sum = __shfl_sync(kFullMask, sum, 0);
        __syncwarp();                            // lane 0 has read every `ex` before it is overwritten
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
        __syncwarp();
        int cut = k;
        if (lane == 0) {
            double cum = 0.0;
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
        }
        cut = __shfl_sync(kFullMask, cut, 0);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        __syncwarp();                            // `ex` is written again below
    }
   // min_p on top_p's survivors (every lane, the same float arithmetic)
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature, then one Philox draw
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, sel_logit[i] * inv_t);
    for (int i = lane; i < n_keep; i += 32) ex[i] = exp((double) (sel_logit[i] * inv_t) - (double) smx);
    __syncwarp();
    double sum = 0.0;
    if (lane == 0)
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
    sum = __shfl_sync(kFullMask, sum, 0);
    __syncwarp();
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    __syncwarp();
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        double cum = 0.0;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((double) u < cum) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
        if constexpr (kProb) *prob_out = n_keep > 0 ? (float) ex[pi] : 1.0f;
        if (qprob != nullptr && t < nq) {
            const int q = query[t];
            float pq = 0.0f;
            for (int i = 0; i < n_keep; ++i)
                if (sel_ids[i] == q) { pq = (float) ex[i]; break; }
            if (n_keep == 0 && sel_ids[0] == q) pq = 1.0f;
            qprob[t] = pq;
        }
    }
}

/// /// **THE ONE-BLOCK SAMPLED PATH.**  Round i's candidates are the logits strictly AFTER round i-1's pick in the
/// /// selection order, `s < prev_v || (s == prev_v && v > prev_i)`: the picks so far are the first i of the order, so
/// /// what comes after the last one is what has not been picked.  The round is a block argmax with the same tie rule.
/// /// An empty round leaves (-inf, id 0), and every round after it is empty (nothing after -inf beats -inf).
/// /// O(k x n_vocab) per row, then warp 0 runs the tail.  The fallback when the split path cannot run.
__global__ void __launch_bounds__(1024)
sampler_one_block_kernel(const float* __restrict__ logits, int n_vocab, const int* __restrict__ history,
                         int history_len, const SamplerParams p, int* __restrict__ out,
                         const int* __restrict__ query, int nq, float* __restrict__ qprob) {
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;

   // the penalty window and its membership bitmap
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    const int k = sampled_k(p.top_k, n_vocab);
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    for (int i = 0; i < k; ++i) {
        float bv = __int_as_float(0xff800000);   // -inf
        int best = n_vocab;
        for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
            const float s = apply_penalties(p.logit_bias ? l[v] + p.logit_bias[v] : l[v], hit_count(v), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        __syncthreads();
        if (warp == 0) {
            const int nw = (int) ((blockDim.x + 31) >> 5);
            float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
                const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        __syncthreads();
        // An empty round leaves (-inf, 0): nothing comes after it, as nothing was left untaken.
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
    }
    if (warp != 0) return;
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex, nullptr, query, nq, qprob);
}

// ---- THE SPLIT top_k.  The one-block kernel keeps a row on ONE SM and walks its logits k times.  Here a warp keeps the
// top_k of 1,024 logits held in registers, four warps make a block of 4,096 logits that merges their lists, and a row
// is 61 such blocks over the whole GPU; one warp per row then merges the 61 lists and runs the tail.  Exact, because
// the first k of a union are within the first k of each part and a merge of ordered lists is ordered - the selection
// order being a strict total order (above).
constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per warp
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per block
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                               // rows per split launch (the scratch's bound)

/// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in the selection order and
/// padded with sentinels - into their first `k`: `sink(i, value, id)` runs in every lane for i = 0..k-1 with the
/// same pair.  Lane owns lists `lane` and `lane + 32`; a round takes the first of all heads and advances the list
/// it came from.  An id is in one list at most (the lists cover disjoint logits), so exactly one head matches.
template <typename Sink>
__device__ __forceinline__ void warp_merge_lists(const int2* lists, int nl, int stride, int k, int n_vocab,
                                                 Sink&& sink) {
    const int lane = (int) (threadIdx.x & 31);
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = __int_as_float(0xff800000);
        hi[m] = n_vocab;
        if (L < nl) {
            const int2 c = lists[(size_t) L * stride];
            hi[m] = c.x;
            hv[m] = __int_as_float(c.y);
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        warp_first(bv, bi);
        sink(i, bv, bi);
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const int2 c = lists[(size_t) (lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.x;
                    hv[m] = __int_as_float(c.y);
                } else {
                    hi[m] = n_vocab;
                    hv[m] = __int_as_float(0xff800000);
                }
            }
        }
    }
}

/// **SPLIT STAGE 1: THE top_k OF EACH 4,096-LOGIT BLOCK.**  Grid (blocks per row, rows), 128 threads.  Each warp
/// loads its 1,024 penalised logits once into registers (lane + 32 j: every load is one coalesced 128-byte line)
/// and runs `k` warp-argmax rounds over them with the threshold of `sampler_one_block_kernel` - no shared memory
/// and no block barrier per round.  Warp 0 then merges the four warp lists into the block's list in `cand`
/// (row-major: row t, block b, entry i at `(t * n_blocks + b) * k + i`, as (id, value bits)).
__global__ void __launch_bounds__(kSplitWarps * 32)
sampler_split_part_kernel(const float* __restrict__ logits, int n_vocab, const int* __restrict__ history,
                          int history_len, const SamplerParams p, int k, int n_blocks, int2* __restrict__ cand) {
    const int t = blockIdx.y;
    const float* l = logits + (size_t) t * n_vocab;
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int blo = (int) blockIdx.x * kSplitBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    // The membership bitmap of THIS BLOCK'S 4,096 logits (512 bytes, not the vocabulary's 31 KB): the same test,
    // and a hit pays the same exact count over the whole window.
    __shared__ unsigned int bits[kSplitBlockSpan / 32];
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < kSplitBlockSpan / 32; w += blockDim.x) bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan)
                atomicOr(&bits[(h - blo) >> 5], 1u << ((h - blo) & 31));
        }
        __syncthreads();
    }

    // This warp's logits, penalised: `apply_penalties` with a zero count returns the logit unchanged, so only the
    // bitmap's hits go through it.  Past the vocabulary: -inf, which no round picks.
    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
#pragma unroll
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? (p.logit_bias ? l[v] + p.logit_bias[v] : l[v]) : __int_as_float(0xff800000);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }

    __shared__ int2 wl[kSplitWarps][kSelMax];
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        // two chains (even and odd j), each walked in ascending id with a strict `>`, so each keeps its first in
        // the order; `take_first` then orders the two
        float b0 = __int_as_float(0xff800000), b1 = __int_as_float(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(b0, i0);
        if (i0 >= n_vocab) break;                // the same in every lane: nothing left in these 1,024 logits
        if (lane == 0) wl[warp][i] = make_int2(i0, __float_as_int(b0));
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + lane; r < k; r += 32) wl[warp][r] = make_int2(n_vocab, (int) 0xff800000u);   // sentinels
    __syncthreads();
    if (warp == 0) {
        int2* dst = cand + ((size_t) t * n_blocks + blockIdx.x) * k;
        warp_merge_lists(&wl[0][0], kSplitWarps, kSelMax, k, n_vocab, [&](int r, float v, int id) {
            if (lane == 0) dst[r] = make_int2(id, __float_as_int(v));
        });
    }
}

/// **SPLIT STAGE 2: ONE WARP PER ROW MERGES THE BLOCK LISTS, THEN RUNS THE TAIL.**  The row's lists (at most 64 x
/// 64 entries, 32 KB) are copied to shared memory, merged into the row's top_k list, and `sampled_tail_warp` picks.
__global__ void __launch_bounds__(32)
sampler_split_merge_kernel(const int2* __restrict__ cand, int n_blocks, int n_vocab, const SamplerParams p, int k,
                           int* __restrict__ out, const int* __restrict__ query, int nq, float* __restrict__ qprob) {
    const int t = blockIdx.x;
    const int lane = (int) threadIdx.x;
    __shared__ int2 lists[kSplitMaxBlocks * kSelMax];
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    const int2* src = cand + (size_t) t * n_blocks * k;
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
    __syncwarp();
    warp_merge_lists(lists, n_blocks, k, k, n_vocab, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
    });
    __syncwarp();
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex, nullptr, query, nq, qprob);
}

/// **THE SPLIT IN ONE LAUNCH.**  Grid (blocks per row, rows), `kWarps` warps of `kPer` logits per lane: each block
/// keeps the top_k of its `kWarps * 32 * kPer` logits exactly as `sampler_split_part_kernel` does (the selection
/// order is a strict total order, so a block's list does not depend on how its logits are cut among warps), writes it
/// to `cand`, and counts itself done in `done[t]`.  The row's last block to finish resets the counter, merges the
/// row's lists in block order and runs the tail, as `sampler_split_merge_kernel` does: the same lists, the same merge,
/// the same tail, so the same token and the same probabilities, bit for bit.
template <int kPer, int kWarps>
__global__ void __launch_bounds__(kWarps * 32)
sampler_fused_kernel(const float* __restrict__ logits, int n_vocab, const int* __restrict__ history, int history_len,
                     const SamplerParams p, int k, int n_blocks, int2* __restrict__ cand, unsigned* __restrict__ done,
                     int* __restrict__ out, const int* __restrict__ query, int nq, float* __restrict__ qprob) {
    constexpr int kWarpSpan = 32 * kPer;
    constexpr int kBlockSpan = kWarps * kWarpSpan;
    const int t = blockIdx.y;
    const float* l = logits + (size_t) t * n_vocab;
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int blo = (int) blockIdx.x * kBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    __shared__ unsigned int bits[kBlockSpan / 32];
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < kBlockSpan / 32; w += blockDim.x) bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kBlockSpan)
                atomicOr(&bits[(h - blo) >> 5], 1u << ((h - blo) & 31));
        }
        __syncthreads();
    }

    const int lo = blo + warp * kWarpSpan;
    float s[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? (p.logit_bias ? l[v] + p.logit_bias[v] : l[v]) : __int_as_float(0xff800000);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }

    __shared__ int2 wl[kWarps][kSelMax];
    __shared__ int2 lists[kSplitMaxBlocks * kSelMax];
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    __shared__ int last;
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        float b0 = __int_as_float(0xff800000), b1 = __int_as_float(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kPer; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(b0, i0);
        if (i0 >= n_vocab) break;
        if (lane == 0) wl[warp][i] = make_int2(i0, __float_as_int(b0));
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + lane; r < k; r += 32) wl[warp][r] = make_int2(n_vocab, (int) 0xff800000u);   // sentinels
    __syncthreads();
    if (warp == 0) {
        int2* dst = cand + ((size_t) t * n_blocks + blockIdx.x) * k;
        warp_merge_lists(&wl[0][0], kWarps, kSelMax, k, n_vocab, [&](int r, float v, int id) {
            if (lane == 0) dst[r] = make_int2(id, __float_as_int(v));
        });
        __threadfence();                     // the list is visible device-wide before the count that publishes it
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prior = atomicAdd(&done[t], 1u);
        last = prior == (unsigned) (n_blocks - 1);
        if (last) done[t] = 0u;              // every block of the row has counted: ready for the next launch
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    // the whole block stages the row's lists, so the L2 reads overlap instead of queueing on one warp
    const int2* src = cand + (size_t) t * n_blocks * k;
    for (int e = threadIdx.x; e < n_blocks * k; e += blockDim.x) lists[e] = __ldcg(src + e);
    __syncthreads();
    if (warp != 0) return;
    warp_merge_lists(lists, n_blocks, k, k, n_vocab, [&](int r, float v, int id) {
        if (lane == 0) { sel_ids[r] = id < n_vocab ? id : 0; sel_logit[r] = v; }
    });
    __syncwarp();
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex, nullptr, query, nq, qprob);
}

// ---- THE SORTED SPLIT (top_k <= 32).  The same first k of the selection order, found with bitonic networks instead of
// k argmax rounds: a pair (value, id) that no round could pick (-inf, NaN) becomes the sentinel (-inf, n_vocab), which
// is what a round that finds nothing yields; every other pair sorts by the selection order.  The order is strict on
// the selectable pairs and every sentinel is the same pair, so each network's output is exactly the ordered list.
constexpr int kSortSpan = 32;                         // entries per sorted list (one per lane)
constexpr int kSortPer = 8;                           // logits per lane
constexpr int kSortWarps = 16;
constexpr int kSortBlockSpan = kSortWarps * 32 * kSortPer;   // 4,096: the split's block span

struct Cand {
    float v;
    int id;
};

// branch-free: a short-circuit `||` compiles to a divergent branch around every compare-exchange
__device__ __forceinline__ bool cand_first(const Cand& a, const Cand& b) {
    return (a.v > b.v) | ((a.v == b.v) & (a.id < b.id));
}

__device__ __forceinline__ Cand cand_shfl_xor(const Cand& c, int m) {
    return {__shfl_xor_sync(kFullMask, c.v, m), __shfl_xor_sync(kFullMask, c.id, m)};
}

// one compare-exchange step against the lane `stride` away: the lane keeps the first of the pair when `keep_first`
__device__ __forceinline__ Cand cand_exchange(const Cand& c, int stride, bool keep_first) {
    const Cand o = cand_shfl_xor(c, stride);
    return (cand_first(o, c) == keep_first) ? o : c;
}

// the warp's 32 pairs (one per lane) sorted in the selection order: lane i holds the i-th
__device__ __forceinline__ Cand warp_sort32(Cand c) {
    const int lane = (int) (threadIdx.x & 31);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const bool lower = (lane & stride) == 0;
            const bool up = size == 32 || (lane & size) == 0;
            c = cand_exchange(c, stride, lower == up);
        }
    }
    return c;
}

// the first 32 of two sorted lists, sorted: the first of a[i] and b[31 - i] keeps the union's first 32 as a bitonic
// sequence, which the half-cleaners then sort
__device__ __forceinline__ Cand warp_merge32(const Cand& a, const Cand& b) {
    const int lane = (int) (threadIdx.x & 31);
    const Cand rb = {__shfl_sync(kFullMask, b.v, 31 - lane), __shfl_sync(kFullMask, b.id, 31 - lane)};
    Cand c = cand_first(rb, a) ? rb : a;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) c = cand_exchange(c, stride, (lane & stride) == 0);
    return c;
}

// the first 32 of `n` (a power of two, <= 16) sorted lists, by a tree of merges
template <int n>
__device__ __forceinline__ Cand warp_merge_tree(Cand (&c)[n]) {
#pragma unroll
    for (int w = 1; w < n; w <<= 1) {
#pragma unroll
        for (int i = 0; i + w < n; i += 2 * w) c[i] = warp_merge32(c[i], c[i + w]);
    }
    return c[0];
}

// The two warps of `sampled_tail_pair`, and the 64 threads of its named barrier.
constexpr int kTailBarrier = 1;
__device__ __forceinline__ void tail_pair_sync() { asm volatile("bar.sync %0, 64;" ::"n"(kTailBarrier) : "memory"); }

struct TailShared {
    double e1[kSortSpan];                    // top_p's terms, then its quotients
    double e2[kSortSpan];                    // the temperature pass's terms, then its quotients
    double pre2[kSortSpan];                  // the ordered prefix sums of e2
    float x[kSortSpan];
    int id[kSortSpan];
};

// the warp's 32 values sorted, largest first (values only: no NaN reaches here)
__device__ __forceinline__ float warp_sort32_values(float v) {
    const int lane = (int) (threadIdx.x & 31);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const float o = __shfl_xor_sync(kFullMask, v, stride);
            const bool keep_max = ((lane & stride) == 0) == (size == 32 || (lane & size) == 0);
            v = keep_max ? fmaxf(v, o) : fminf(v, o);
        }
    }
    return v;
}

__device__ __forceinline__ bool cand_live(const Cand& c, float tau) {
    return (c.v >= tau) & (c.v > __int_as_float(0xff800000));
}

/// The warp's first k (k <= 32) of its 8 x 32 pairs, sorted, one per lane; the entries past k are sorted too but may
/// leave pairs out, which no caller reads (the first k of a union are within the first k of each part).  With the
/// lane maxima sorted, the smallest j whose maximum has at least k live pairs at or above it bounds the warp's k-th
/// value from below, so the first k are among the pairs at or above it: when at most 32 are, one sort of them is the
/// answer; otherwise each set of 32 is sorted and the sets are merged.
__device__ __forceinline__ Cand warp_first_k(Cand (&c)[kSortPer], int k, int n_vocab, Cand* buf) {
    const int lane = (int) (threadIdx.x & 31);
    float m = c[0].v;
#pragma unroll
    for (int j = 1; j < kSortPer; ++j) m = fmaxf(m, c[j].v);
    const float sm = warp_sort32_values(m);
    auto live_count = [&](float tau) {
        int n = 0;
#pragma unroll
        for (int j = 0; j < kSortPer; ++j) n += __popc(__ballot_sync(kFullMask, cand_live(c[j], tau)));
        return n;
    };
    int lo = 0, hi = k - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (live_count(__shfl_sync(kFullMask, sm, mid)) >= k) hi = mid;
        else lo = mid + 1;
    }
    const float tau = __shfl_sync(kFullMask, sm, lo);
    unsigned bal[kSortPer];
    int total = 0;
#pragma unroll
    for (int j = 0; j < kSortPer; ++j) {
        bal[j] = __ballot_sync(kFullMask, cand_live(c[j], tau));
        total += __popc(bal[j]);
    }
    if (total <= kSortSpan) {
        const unsigned below = (1u << lane) - 1u;
        if (lane >= total) buf[lane] = Cand{__int_as_float(0xff800000), n_vocab};
        int base = 0;
#pragma unroll
        for (int j = 0; j < kSortPer; ++j) {
            if ((bal[j] >> lane) & 1u) buf[base + __popc(bal[j] & below)] = c[j];
            base += __popc(bal[j]);
        }
        __syncwarp();
        const Cand r = buf[lane];
        __syncwarp();
        return warp_sort32(r);
    }
#pragma unroll
    for (int j = 0; j < kSortPer; ++j) c[j] = warp_sort32(c[j]);
    return warp_merge_tree(c);
}

// The ordered running sum acc_i = (..((0 + a[0]) + a[1]) ..) + a[i] over i < n (n <= 32), each term added in turn:
// `at(i, acc_i)` sees every value, and a true return stops the scan after it.  The terms are read eight at a time
// ahead of the adds, so the adds wait only on each other.
template <typename At>
__device__ __forceinline__ void ordered_scan(const double* a, int n, At&& at) {
    double acc = 0.0;
#pragma unroll
    for (int c = 0; c < kSortSpan; c += 8) {
        if (c >= n) break;
        double v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = a[c + j];
        bool stop = false;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (c + j < n) {
                acc += v[j];
                stop |= at(c + j, acc);
            }
        if (stop) break;
    }
}

/// **THE TAIL ON TWO WARPS** (k <= 32): the sorted list's first 32 entries are in `ts.x` / `ts.id` (the ids mapped as
/// `sampled_tail_warp` reads them), and warps 0 and 1 of the block call this.  The arithmetic is
/// `sampled_tail_warp`'s, value for value: the same `exp` arguments, every ordered sum and cumulative scan in the same
/// order (on one lane), the same correctly rounded quotients.  Only where the work waits changes:
/// - the list is sorted, so its first entry is its maximum (`fmaxf` over it gives that value up to the sign of a zero,
///   and the `exp` of a difference that is a zero of either sign is 1), and the temperature pass's maximum is the first
///   entry's too: warp 1 runs that pass's `exp`s and its ordered prefix sums while warp 0 runs top_p, and the sum over
///   the first n_keep is the n_keep-th prefix sum;
/// - a scan reads its terms ahead of its adds and records where it first crosses, so no compare waits in the chain
///   of adds (the few sums past the crossing are never read).
__device__ void sampled_tail_pair(TailShared& ts, int k, const SamplerParams& p, int t, int* __restrict__ out,
                                  const int* query, int nq, float* qprob) {
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    const float x0 = ts.x[0];
    const bool mine = lane < k;
    if (warp == 1) {
        const float smx = x0 * inv_t;
        if (mine) ts.e2[lane] = exp((double) (ts.x[lane] * inv_t) - (double) smx);
        __syncwarp();
        if (lane == 0) ordered_scan(ts.e2, k, [&](int i, double acc) { ts.pre2[i] = acc; return false; });
        tail_pair_sync();
        return;
    }
    int n_keep = k;
    if (p.top_p < 1.0f) {
        if (mine) ts.e1[lane] = exp((double) ts.x[lane] - (double) x0);
        __syncwarp();
        double sum = 0.0;
        if (lane == 0) ordered_scan(ts.e1, k, [&](int, double acc) { sum = acc; return false; });
        sum = __shfl_sync(kFullMask, sum, 0);
        if (mine) ts.e1[lane] = ts.e1[lane] / sum;
        __syncwarp();
        int cut = k;
        if (lane == 0)
            ordered_scan(ts.e1, k, [&](int i, double cum) {
                if (cut == k && cum >= (double) p.top_p) cut = i + 1;
                return cut != k;
            });
        cut = __shfl_sync(kFullMask, cut, 0);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = x0 + logf(p.min_p);
        const unsigned below = __ballot_sync(kFullMask, mine && ts.x[lane] < thresh);
        const int first = below ? __ffs(below) - 1 : k;
        if (first < n_keep) n_keep = first;
    }
    tail_pair_sync();                        // warp 1's terms and prefix sums
    const double sum2 = n_keep > 0 ? ts.pre2[n_keep - 1] : 0.0;
    if (lane < n_keep) ts.e2[lane] = ts.e2[lane] / sum2;
    __syncwarp();
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        bool found = false;
        ordered_scan(ts.e2, n_keep, [&](int i, double cum) {
            if (!found && (double) u < cum) { pi = i; found = true; }
            return found;
        });
        out[t] = ts.id[pi];
        if (qprob != nullptr && t < nq) {
            const int q = query[t];
            float pq = 0.0f;
            for (int i = 0; i < n_keep; ++i)
                if (ts.id[i] == q) { pq = (float) ts.e2[i]; break; }
            if (n_keep == 0 && ts.id[0] == q) pq = 1.0f;
            qprob[t] = pq;
        }
    }
}

/// **THE SORTED SPLIT IN ONE LAUNCH.**  Grid (blocks per row, rows), 16 warps.  Each warp finds the first k of its
/// 256 penalised logits (`warp_first_k`), warp 0 merges the 16 warp lists into the block's and writes it to `cand`
/// (32 entries per block).  The row's last block (counted in `done[t]`, reset by it) merges the row's lists, 16 warps
/// four lists each and then warp 0 the 16 results, and runs the tail on two warps: the first k that
/// `sampler_split_merge_kernel` finds, so the same token and probabilities, bit for bit.
__global__ void __launch_bounds__(kSortWarps * 32)
sampler_sort_kernel(const float* __restrict__ logits, int n_vocab, const int* __restrict__ history, int history_len,
                    const SamplerParams p, int k, int n_blocks, int2* __restrict__ cand, unsigned* __restrict__ done,
                    int* __restrict__ out, const int* __restrict__ query, int nq, float* __restrict__ qprob) {
    const int t = blockIdx.y;
    const float* l = logits + (size_t) t * n_vocab;
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int blo = (int) blockIdx.x * kSortBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    __shared__ unsigned int bits[kSortBlockSpan / 32];
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < kSortBlockSpan / 32; w += blockDim.x) bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSortBlockSpan)
                atomicOr(&bits[(h - blo) >> 5], 1u << ((h - blo) & 31));
        }
        __syncthreads();
    }

    const int lo = blo + warp * 32 * kSortPer;
    Cand c[kSortPer];
#pragma unroll
    for (int j = 0; j < kSortPer; ++j) {
        const int v = lo + 32 * j + lane;
        float s = v < n_vocab ? (p.logit_bias ? l[v] + p.logit_bias[v] : l[v]) : __int_as_float(0xff800000);
        if (use_bits) {
            const int b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31)))) s = apply_penalties(s, history_count(hrow, hlen, v), p);
        }
        c[j] = s > __int_as_float(0xff800000) ? Cand{s, v} : Cand{__int_as_float(0xff800000), n_vocab};
    }
    __shared__ Cand wl[kSortWarps][kSortSpan];
    __shared__ Cand lists[kSplitMaxBlocks * kSortSpan];
    __shared__ TailShared ts;
    __shared__ int last;
    wl[warp][lane] = warp_first_k(c, k, n_vocab, wl[warp]);
    __syncthreads();
    if (warp == 0) {
        Cand w[kSortWarps];
#pragma unroll
        for (int i = 0; i < kSortWarps; ++i) w[i] = wl[i][lane];
        const Cand b = warp_merge_tree(w);
        cand[((size_t) t * n_blocks + blockIdx.x) * kSortSpan + lane] = make_int2(b.id, __float_as_int(b.v));
        __threadfence();                     // the list is visible device-wide before the count that publishes it
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prior = atomicAdd(&done[t], 1u);
        last = prior == (unsigned) (n_blocks - 1);
        if (last) done[t] = 0u;              // every block of the row has counted: ready for the next launch
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int2* src = cand + (size_t) t * n_blocks * kSortSpan;
    for (int e = threadIdx.x; e < kSplitMaxBlocks * kSortSpan; e += blockDim.x) {
        const int2 x = e < n_blocks * kSortSpan ? __ldcg(src + e) : make_int2(n_vocab, (int) 0xff800000u);
        lists[e] = Cand{__int_as_float(x.y), x.x};
    }
    __syncthreads();
    {   // 64 lists: warp w merges lists 4w .. 4w + 3 into wl[w]
        Cand m[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) m[i] = lists[(warp * 4 + i) * kSortSpan + lane];
        wl[warp][lane] = warp_merge_tree(m);
    }
    __syncthreads();
    if (warp > 1) return;
    if (warp == 0) {
        Cand w[kSortWarps];
#pragma unroll
        for (int i = 0; i < kSortWarps; ++i) w[i] = wl[i][lane];
        const Cand r = warp_merge_tree(w);
        ts.x[lane] = r.v;
        ts.id[lane] = r.id < n_vocab ? r.id : 0;
    }
    tail_pair_sync();
    sampled_tail_pair(ts, k, p, t, out, query, nq, qprob);
}

// A stream being captured into a graph must not reach `split_scratch` (cudaMalloc): it
// gets the one-block kernel, which needs no memory of its own.  The legacy stream cannot be captured.
bool stream_capturing(void* stream) {
    if (stream == nullptr) return false;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &st) != cudaSuccess) {
        (void) cudaGetLastError();
        return true;
    }
    return st != cudaStreamCaptureStatusNone;
}

// The split's block lists, one buffer per (device, stream): launches on one stream run in order, so a stream reuses
// its buffer with no sync, and two streams never share one.  Grown on demand (at least doubling, up to the size of
// `kSplitMaxRows` rows at the widest vocabulary: 2 MB), never shrunk.  A grown-out buffer is retired, not freed: a
// pointer handed out earlier (to another host thread on the same stream, say) may still be waiting for its launch,
// and freeing it would need a sync that proves nothing about that thread.  Doubling keeps a slot's retired buffers
// smaller than its live one, so a slot holds about 4 MB at most (0.5 MB for the engine's <= 16 rows).  nullptr
// when the memory cannot be had, and the caller falls back to the one-block kernel; a failed grow is remembered, so
// later calls of that size do not retry cudaMalloc each time.
int2* split_scratch(void* stream, size_t entries, unsigned** done) {
    struct Slot {
        int device;
        void* stream;
        int2* ptr;
        size_t entries;
        size_t failed;                       // the smallest size cudaMalloc refused (0: none)
        unsigned* done;                      // the one-launch split's per-row block counters, zero between launches
    };
    static std::mutex mu;
    static std::vector<Slot> slots;
    static std::vector<int2*> retired;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mu);
    Slot* slot = nullptr;
    for (Slot& s : slots)
        if (s.device == device && s.stream == stream) slot = &s;
    if (slot == nullptr) {
        slots.push_back({device, stream, nullptr, 0, 0, nullptr});
        slot = &slots.back();
    }
    if (slot->done == nullptr) {
        unsigned* d = nullptr;
        // zeroed in the stream's own order: a non-blocking stream does not wait for the legacy stream
        if (cudaMalloc(&d, kSplitMaxRows * sizeof(unsigned)) != cudaSuccess ||
            cudaMemsetAsync(d, 0, kSplitMaxRows * sizeof(unsigned), (cudaStream_t) stream) != cudaSuccess) {
            (void) cudaGetLastError();
            if (d != nullptr) cudaFree(d);
            return nullptr;
        }
        slot->done = d;
    }
    *done = slot->done;
    if (slot->entries >= entries) return slot->ptr;
    if (slot->failed != 0 && entries >= slot->failed) return nullptr;
    constexpr size_t kCap = (size_t) kSplitMaxRows * kSplitMaxBlocks * kSelMax;
    size_t want = 2 * slot->entries < kCap ? 2 * slot->entries : kCap;
    if (want < entries) want = entries;
    int2* ptr = nullptr;
    if (cudaMalloc(&ptr, want * sizeof(int2)) != cudaSuccess) {
        (void) cudaGetLastError();
        want = entries;
        if (cudaMalloc(&ptr, want * sizeof(int2)) != cudaSuccess) {
            (void) cudaGetLastError();
            slot->failed = entries;
            return nullptr;
        }
    }
    if (slot->ptr != nullptr) retired.push_back(slot->ptr);
    slot->ptr = ptr;
    slot->entries = want;
    return ptr;
}

}  // namespace

bool sample_greedy_cluster(const float* logits, int n_tokens, int n_vocab, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return true;
    if (n_tokens > 65535) return false;
    // Per device (a layer split runs on several): 1 the cluster kernel runs here, 2 it does not - thread-block clusters, code built
    // for it (an older build's PTX holds a trap: the PTX version says which), and room for one cluster of kAmCtas CTAs.
    static int ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = kAmCtas;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(kAmCtas, 1, 1);
    cfg.blockDim = dim3(kAmThreads, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = (cudaStream_t) stream;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (ok[dev] == 0) {
        int major = 0, clusters = 0;
        cudaFuncAttributes fa{};
        const bool runs = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          major >= 9 &&
                          cudaFuncGetAttributes(&fa, sampler_greedy_cluster_kernel) == cudaSuccess &&
                          fa.ptxVersion >= 90 && fa.binaryVersion >= 90 &&
                          cudaOccupancyMaxActiveClusters(&clusters, sampler_greedy_cluster_kernel, &cfg) ==
                              cudaSuccess &&
                          clusters >= 1;
        cudaGetLastError();
        ok[dev] = runs ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    cfg.gridDim = dim3(kAmCtas, (unsigned) n_tokens, 1);
    const cudaError_t e = cudaLaunchKernelEx(&cfg, sampler_greedy_cluster_kernel, logits, n_vocab, out);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "sample_tokens cluster launch: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

bool sample_greedy_top_prob(const float* logits, int n_tokens, int n_vocab, const int32_t* map, int* out, float* probs,
                            void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return true;
    if (n_tokens > 65535) return false;
    // per device, as in sample_greedy_cluster: 1 runs here, 2 does not
    static int ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = kAmCtas;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(kAmCtas, 1, 1);
    cfg.blockDim = dim3(kAmThreads, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = (cudaStream_t) stream;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (ok[dev] == 0) {
        int major = 0, clusters = 0;
        cudaFuncAttributes fa{};
        const bool runs = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          major >= 9 &&
                          cudaFuncGetAttributes(&fa, sampler_greedy_top_prob_cluster_kernel) == cudaSuccess &&
                          fa.ptxVersion >= 90 && fa.binaryVersion >= 90 &&
                          cudaOccupancyMaxActiveClusters(&clusters, sampler_greedy_top_prob_cluster_kernel, &cfg) ==
                              cudaSuccess &&
                          clusters >= 1;
        cudaGetLastError();
        ok[dev] = runs ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    cfg.gridDim = dim3(kAmCtas, (unsigned) n_tokens, 1);
    const cudaError_t e = cudaLaunchKernelEx(&cfg, sampler_greedy_top_prob_cluster_kernel, logits, n_vocab, map, out,
                                             probs);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "sample_greedy_top_prob launch: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    sample_tokens_probe(logits, n_tokens, n_vocab, history, history_len, p, out, nullptr, 0, nullptr, stream);
}

namespace {

// The sampled path's split: 0 the two launches (block lists, then one warp per row), 1 the one launch with k argmax
// rounds per warp (`sampler_fused_kernel`), 2 the one launch with bitonic networks (`sampler_sort_kernel`, top_k <= 32;
// a wider top_k takes 1).  Every one picks the same token.
constexpr int kEngineSplit = 2;

void sample_tokens_impl(int split, const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                        const SamplerParams& p, int* out, const int* query, int n_query, float* qprob, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (qprob == nullptr) n_query = 0;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        basalt::log::error("basalt", "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    const unsigned shmem = (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
                               ? (unsigned) ((n_vocab + 31) / 32) * sizeof(unsigned)   // the penalty bitmap
                               : 0;
    const cudaStream_t cs = (cudaStream_t) stream;
    if (p.greedy || p.temperature <= 0.0f) {
        // Without penalties (shmem == 0: no window) with cluster support, a cluster of CTAs per token - the same token; see
        // `sampler_greedy_cluster_kernel`.
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        const int gthreads = 1024;
        if (!(shmem == 0 && p.logit_bias == nullptr && sample_greedy_cluster(logits, n_tokens, n_vocab, out, stream)))
            sampler_greedy_kernel<<<(unsigned) n_tokens, gthreads, shmem, cs>>>(
                logits, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n, out);
    } else {
        // The split top_k over 4,096-logit blocks (61 per row for 248,320 logits).  The one-block kernel when the
        // split cannot run: a wider vocabulary than the merge holds, more than `kSplitMaxRows` rows (the engine
        // samples at most a verify window), a stream under capture, no scratch.
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        int2* scratch = nullptr;
        unsigned* done = nullptr;
        if (n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows &&
            !stream_capturing(stream))
            // sized for 16 rows and 64 entries at least, so a verify window or a wider top_k does not regrow it
            scratch = split_scratch(stream, (size_t) (n_tokens > 16 ? n_tokens : 16) * n_blocks * kSelMax, &done);
        const dim3 grid((unsigned) n_blocks, (unsigned) n_tokens);
        if (scratch == nullptr) {
            sampler_one_block_kernel<<<(unsigned) n_tokens, 1024, shmem, cs>>>(
                logits, n_vocab, history, history_len, p, out, query, n_query, qprob);
        } else if (split == 2 && k <= kSortSpan) {
            sampler_sort_kernel<<<grid, kSortWarps * 32, 0, cs>>>(logits, n_vocab, history, history_len, p, k, n_blocks,
                                                                  scratch, done, out, query, n_query, qprob);
        } else if (split >= 1) {
            sampler_fused_kernel<8, 16><<<grid, 512, 0, cs>>>(logits, n_vocab, history, history_len, p, k, n_blocks,
                                                              scratch, done, out, query, n_query, qprob);
        } else {
            sampler_split_part_kernel<<<grid, kSplitWarps * 32, 0, cs>>>(logits, n_vocab, history, history_len, p, k,
                                                                          n_blocks, scratch);
            sampler_split_merge_kernel<<<(unsigned) n_tokens, 32, 0, cs>>>(scratch, n_blocks, n_vocab, p, k, out,
                                                                          query, n_query, qprob);
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "sample_tokens launch: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace

void sample_tokens_probe(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                         const SamplerParams& p, int* out, const int* query, int n_query, float* qprob, void* stream) {
    sample_tokens_impl(kEngineSplit, logits, n_tokens, n_vocab, history, history_len, p, out, query, n_query, qprob,
                       stream);
}

void sample_tokens_split(int split, const float* logits, int n_tokens, int n_vocab, const int* history,
                         int history_len, const SamplerParams& p, int* out, const int* query, int n_query, float* qprob,
                         void* stream) {
    sample_tokens_impl(split, logits, n_tokens, n_vocab, history, history_len, p, out, query, n_query, qprob, stream);
}

}  // namespace basalt::kernels
