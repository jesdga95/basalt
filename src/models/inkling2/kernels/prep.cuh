// src/models/inkling2/kernels/prep.cuh - the decode attention preparation's stages, shared by glue.cu's prepare kernel and
// attn.cu's fused attention so both make the same bits: a key/value head's convolved, clipped keys and normed values of a
// row (warps 0 and 1), and its three query heads' clipped queries, relative states and sinks (warps 2 and 3).
#pragma once

#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace basalt::models::inkling2::prep {

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
// Two scales per head row: each half-warp covers 64 adjacent channels.
__device__ __forceinline__ void int8_scales(const float (&v)[4], float f, int lane, float (&scale)[2]) {
    float a = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) a = fmaxf(a, fabsf(v[u] * f));
#pragma unroll
    for (int m = 8; m; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    scale[lane >> 4] = __half2float(__float2half_rn(a / 127.f));
}

__device__ __forceinline__ float tau_of(int p, float floor_, float alpha) {
    return 1.f + alpha * logf(fmaxf((float) (p + 1) / floor_, 1.f));
}

// The convolution input of channel c at position p from the ring, zero before position 0.
__device__ __forceinline__ float conv_ring(const float* __restrict__ ring, int ring_w, int p, int c) {
    if (p < 0) return 0.f;
    return ring[(size_t) (p % kConvRing) * ring_w + c];
}

// The convolution input of channel c at position pos[r] - j: a row of the window (stride ld), else the ring.
__device__ __forceinline__ float conv_in(const float* __restrict__ y, int64_t ld, const float* __restrict__ ring, int ring_w,
                                         const int32_t* __restrict__ pos, int r, int j, int c) {
    if (r - j >= 0) return y[(size_t) (r - j) * ld + c];
    return conv_ring(ring, ring_w, pos[r] - j, c);
}
// conv_in with row r's position pr in hand (read before the kernel's wait: the positions are staged before the window);
// the window's rows of r's conversation start at row `first` (a batch window's segment, else 0).
__device__ __forceinline__ float conv_in_at(const float* __restrict__ y, int64_t ld, const float* __restrict__ ring, int ring_w,
                                            int pr, int r, int j, int c, int first = 0) {
    if (r - j >= first) return y[(size_t) (r - j) * ld + c];
    return conv_ring(ring, ring_w, pr - j, c);
}

// The weights warp w reads before its kernel's pdl_wait: warps 0 and 1 their channels' taps (tap kConv-1 first), warps 2
// and 3 their heads' clip scales.
__device__ __forceinline__ void load_weights(const AttnPrep& a, int g, int w, int lane, float (&wt)[4][kConv]) {
    if (w < 2) {
        const float* taps = w ? a.v_conv : a.k_conv;
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int j = 0; j < kConv; ++j) wt[u][j] = taps[(g * kHeadDim + lane * 4 + u) * kConv + (kConv - 1 - j)];
    } else {
        const int h0 = g * 3 + (w == 2 ? 0 : 2);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            wt[u][0] = a.q_scale[h0 * kHeadDim + lane * 4 + u];
            wt[u][1] = w == 2 ? a.q_scale[(h0 + 1) * kHeadDim + lane * 4 + u] : 0.f;
        }
        if (w == 3)
#pragma unroll
            for (int k = 0; k < 3; ++k) wt[k][2] = lane < kRel ? a.r_scale[(g * 3 + k) * kRel + lane] : 0.f;
    }
}

// Warp 0 (keys) or 1 (values) of head g for row r at position pr: channels g * kHeadDim + lane * 4.. of the convolved
// row, `raw` their convolution inputs at the row's own position, and the factor the cache takes them scaled by (the key
// clip, the value norm). `ring`: the warp's ring of r's conversation; its rows of the window start at `first`.
__device__ __forceinline__ void kv_row(const AttnPrep& a, int r, int pr, int g, int w, int lane, const float (&wt)[4][kConv],
                                       float (&val)[4], float (&raw)[4], float& f, const float* ring, int first, float (&scale)[2]) {
    const float* yall = w ? a.yv : a.yk;
    const int64_t ld = w ? a.ldv : a.ldk;
    const float* y = yall + (size_t) r * ld;
    const float inv_k = a.full ? (1.f / 3.f) : rsqrtf((float) kConv + 1.f);
    float ss = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int c = g * kHeadDim + lane * 4 + u;
        float acc = 0.f;
        raw[u] = 0.f;
#pragma unroll
        for (int j = 0; j < kConv; ++j) {
            const float x = conv_in_at(yall, ld, ring, kKvHeads * kHeadDim, pr, r, j, c, first);
            if (j == 0) raw[u] = x;
            float t = wt[u][j];
            if (a.full) t += y[kOffTaps + g * kConv + j];
            acc = fmaf(t, x, acc);
        }
        val[u] = (acc + raw[u]) * inv_k;
        ss = fmaf(val[u], val[u], ss);
    }
    ss = warp_sum(ss);
    f = w == 0 ? fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) : rsqrtf(ss / (float) kHeadDim + a.eps);
    if (a.int8) int8_scales(val, f, lane, scale);
}

// kv_row for the n (1..N) consecutive rows r0.. at positions pr0.., every input loaded before the first sum (a row's older
// inputs are the rows' before it): each row's val, raw and f are kv_row's bits.
template <int N>
__device__ __forceinline__ void kv_rows(const AttnPrep& a, int r0, int n, int pr0, int g, int w, int lane,
                                        const float (&wt)[4][kConv], float (&val)[N][4], float (&raw)[N][4], float (&f)[N], float (&scale)[N][2]) {
    const float* yall = w ? a.yv : a.yk;
    const int64_t ld = w ? a.ldv : a.ldk;
    const float* ring = w ? a.v_ring : a.k_ring;
    const float inv_k = a.full ? (1.f / 3.f) : rsqrtf((float) kConv + 1.f);
    // x[i]: the convolution input at position pr0 + N - 1 - i (row r0 + N - 1 - i of the window, else the ring), zero for
    // the rows past n. Row r0 + k takes x[N - 1 - k + j] as kv_row's tap j.
    float x[kConv + N - 1][4], tp[N][kConv];
#pragma unroll
    for (int i = 0; i < kConv + N - 1; ++i) {
        const int rq = r0 + N - 1 - i;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = g * kHeadDim + lane * 4 + u;
            x[i][u] = N - 1 - i >= n ? 0.f : rq >= 0 ? yall[(size_t) rq * ld + c] : conv_ring(ring, kKvHeads * kHeadDim, pr0 + N - 1 - i, c);
        }
    }
#pragma unroll
    for (int k = 0; k < N; ++k)
#pragma unroll
        for (int j = 0; j < kConv; ++j)
            tp[k][j] = a.full && k < n ? yall[(size_t) (r0 + k) * ld + kOffTaps + g * kConv + j] : 0.f;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        if (k >= n) break;
        float ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            float acc = 0.f;
            raw[k][u] = x[N - 1 - k][u];
#pragma unroll
            for (int j = 0; j < kConv; ++j) {
                float t = wt[u][j];
                if (a.full) t += tp[k][j];
                acc = fmaf(t, x[N - 1 - k + j][u], acc);
            }
            val[k][u] = (acc + raw[k][u]) * inv_k;
            ss = fmaf(val[k][u], val[k][u], ss);
        }
        ss = warp_sum(ss);
        f[k] = w == 0 ? fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) : rsqrtf(ss / (float) kHeadDim + a.eps);
        if (a.int8) int8_scales(val[k], f[k], lane, scale[k]);
    }
}

__device__ __forceinline__ void store_kv(void* row, bool fp8, bool int8, size_t idx, float v, const float (&scale)[2], int lane) {
    if (int8) {
        const int group = (int) idx / 64;
        const float s = scale[group];
        const int q = s > 0.f ? max(-127, min(127, __float2int_rn(v / s))) : 0;
        static_cast<int8_t*>(row)[idx] = (int8_t) q;
        if (lane == group * 16)
            reinterpret_cast<__half*>(static_cast<uint8_t*>(row) + kHeadDim)[group] = __float2half_rn(s);
        if (idx == kHeadDim - 1)   // the last channel writes the row padding as zeros
            for (int b = kHeadDim + 4; b < kKvInt8RowBytes; b += 4) *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(row) + b) = 0u;
    } else if (fp8) {
        __nv_fp8_e4m3 q(v);
        static_cast<uint8_t*>(row)[idx] = *reinterpret_cast<uint8_t*>(&q);
    } else {
        static_cast<__half*>(row)[idx] = __float2half_rn(v);
    }
}

// Warps 2 and 3 of head g for row r at position p: the clipped (and scaled) queries of heads 3g..3g+2 into q3
// [3][kHeadDim], warp 3 also their relative states into r3 [3][kRel] and their sinks into s3 [3]. Every projection value a
// warp takes is loaded before its first sum, so the warp waits for one memory trip, not one a head.
__device__ __forceinline__ void q_heads(const AttnPrep& a, int r, int p, int g, int w, int lane, const float (&wt)[4][kConv],
                                        float* q3, float* r3, float* s3) {
    const float* yq = a.yq + (size_t) r * a.ldq;
    const float tau = a.full ? tau_of(p, a.log_floor, a.log_alpha) : 1.f;
    // warp 2: heads 3g and 3g + 1; warp 3: head 3g + 2, then the three relative states and sinks
    const int h0 = g * 3 + (w == 2 ? 0 : 2), nh = w == 2 ? 2 : 1;
    float y[2][4], yr[3], ys[3];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int u = 0; u < 4; ++u) y[k][u] = k < nh ? yq[kOffQ + (h0 + k) * kHeadDim + lane * 4 + u] : 0.f;
    if (w == 3)
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            yr[k] = lane < kRel ? yq[kOffR + (g * 3 + k) * kRel + lane] : 0.f;
            ys[k] = yq[kOffSink + g * 3 + k];
        }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        if (k >= nh) break;
        const int h = h0 + k;
        float q[4], ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            q[u] = y[k][u] * (k == 0 ? wt[u][0] : wt[u][1]);
            ss = fmaf(q[u], q[u], ss);
        }
        ss = warp_sum(ss);
        const float f = fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) * tau;
        *reinterpret_cast<float4*>(q3 + (h - g * 3) * kHeadDim + lane * 4) = make_float4(q[0] * f, q[1] * f, q[2] * f, q[3] * f);
    }
    if (w == 3) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const float rv = lane < kRel ? yr[k] * (k == 0 ? wt[0][2] : k == 1 ? wt[1][2] : wt[2][2]) : 0.f;
            const float rs = warp_sum(rv * rv);
            const float fr = fminf(a.rel_clip / (sqrtf(rs) + 1e-6f), 1.f) * tau;
            if (lane < kRel) r3[k * kRel + lane] = rv * fr;
            if (lane == 0) s3[k] = ys[k];
        }
    }
}

}  // namespace basalt::models::inkling2::prep
