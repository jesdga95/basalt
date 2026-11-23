// tests/models/qwen4exp/kernels/kv_k4v8_parity.cpp - hybrid K4V8 KV against host references (GPU, no model).
//
// K4V8 is K8V4 with the planes swapped: K is stored as Q4_0 blocks after the Walsh-Hadamard rotation, V as INT8
// codes with an fp16 scale per 64 values, unrotated.  The queries and K rotate; the output does NOT.
//
//   1. the composed appends (kv_append_q4_step for rotated K, kv_append_q8_step for V), bitwise vs the host
//      references of kv_q4.hpp and kv_q8.hpp;
//   2. qsa_decode_attn_step (KV_MODE 4: q4_0 K + INT8 V) vs fp32 host attention over the SAME dequantized K/V;
//   3. qsa_prompt_attn_batch (KV_MODE 5) on the same pools, which must be taken (not refused into the slow
//      decode fallback) and must land inside the tensor-core path's accuracy;
//   4. the output stays in the rotated domain: a run whose output is un-rotated anyway must NOT match, which is
//      what proves the engine's "no un-rotation for K4V8" rule is the right one.

#include "basalt/kernels/f16_bits.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// kv_q4_parity.cpp's / kv_hybrid_parity.cpp's exact CPU references
void fwht256_host_reference(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < n; i += 2 * s)
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j], v = x[i + j + s];
                x[i + j] = u + v;
                x[i + j + s] = u - v;
            }
    for (int i = 0; i < n; ++i) x[i] *= 1.0f / 16.0f;
}

void quantize_block_q4_0_host(const float* x, k::block_q4_0& blk) {
    float amax = 0.0f, max_val = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float v = x[i];
        if (amax < std::fabs(v)) { amax = std::fabs(v); max_val = v; }
    }
    const float d = max_val / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    blk.d = k::f16_from_f32(d);
    for (int i = 0; i < 16; ++i) {
        const float x0 = x[i] * id;
        const float x1 = x[i + 16] * id;
        int q0 = (int) std::nearbyint(x0 + 8.0f);
        int q1 = (int) std::nearbyint(x1 + 8.0f);
        q0 = std::max(0, std::min(15, q0));
        q1 = std::max(0, std::min(15, q1));
        blk.qs[i] = (uint8_t) (q0 | (q1 << 4));
    }
}

}  // namespace

int main() {
    std::printf("=== Running kv_k4v8_parity test ===\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int QH = (int) s.n_head, PER = QH / H;
    const int pages = 24, cells = pages * P;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 7 + 1) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    std::vector<float> K((size_t) cells * H * D), V((size_t) cells * H * D);
    for (auto& x : K) x = nd(rng) * 0.5f;
    for (auto& x : V) x = nd(rng) * 0.5f;

    // ---- host reference: rotated K in Q4_0 (kv_q4.hpp), unrotated V in INT8 (kv_q8.hpp)
    std::vector<float> Krot((size_t) cells * H * D);
    std::memcpy(Krot.data(), K.data(), K.size() * 4);
    for (int c = 0; c < cells * H; ++c) fwht256_host_reference(Krot.data() + (size_t) c * D);
    std::vector<k::block_q4_0> rk4((size_t) cells * H * (D / 32));
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int b = 0; b < D / 32; ++b)
                quantize_block_q4_0_host(&Krot[((size_t) c * H + h) * D + b * 32], rk4[((size_t) c * H + h) * (D / 32) + b]);

    std::vector<int8_t> rv((size_t) cells * H * D);
    std::vector<uint16_t> rv_s((size_t) cells * H * G);
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int g = 0; g < G; ++g) {
                float amax = 0.0f;
                for (int t = 0; t < k::KV_Q8_GROUP; ++t)
                    amax = std::max(amax, std::fabs(V[((size_t) c * H + h) * D + g * k::KV_Q8_GROUP + t]));
                const uint16_t sb = k::f16_from_f32(amax / 127.0f);
                const float sf = k::f32_from_f16(sb);
                for (int t = 0; t < k::KV_Q8_GROUP; ++t) {
                    int q = 0;
                    if (sf > 0.0f) {
                        q = (int) std::nearbyint(V[((size_t) c * H + h) * D + g * k::KV_Q8_GROUP + t] / sf);
                        q = std::max(-127, std::min(127, q));
                    }
                    rv[((size_t) c * H + h) * D + g * k::KV_Q8_GROUP + t] = (int8_t) q;
                }
                rv_s[((size_t) c * H + h) * G + g] = sb;
            }

    // ---- device: every cell appended through the K4V8 composition (qsa_layer's branch), then compared bitwise
    const int64_t q4_row = (int64_t) k::kv_q4_bytes_per_head(D);
    uint8_t* d_k4 = dalloc<uint8_t>((size_t) cells * H * q4_row);
    int8_t* d_vq = dalloc<int8_t>((size_t) cells * H * D);
    uint16_t* d_vs = dalloc<uint16_t>((size_t) cells * H * G);
    float *d_kcur = dalloc<float>((size_t) H * D), *d_vcur = dalloc<float>((size_t) H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);
    int append_bad = 0;
    for (int c = 0; c < cells; ++c) {
        ck(cudaMemcpy(d_kcur, &K[(size_t) c * H * D], (size_t) H * D * 4, cudaMemcpyHostToDevice), "kcur");
        ck(cudaMemcpy(d_vcur, &V[(size_t) c * H * D], (size_t) H * D * 4, cudaMemcpyHostToDevice), "vcur");
        int32_t step[k::kStepCount];
        k::qsa_step_fill(step, c, s);
        ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
        // exactly what record_qsa.cpp's K4V8 branch does: rotate K, store K in q4_0, V in INT8
        k::fwht256_inplace_cuda(d_kcur, H, nullptr);
        k::kv_append_q4_step(d_k4, d_k4, d_table, d_step, d_kcur, d_kcur, s, nullptr, nullptr);
        k::kv_append_q8_step(d_vq, d_vq, d_vs, d_vs, d_table, d_step, d_vcur, d_vcur, s, nullptr, nullptr);
    }
    ck(cudaDeviceSynchronize(), "append sync");
    std::vector<uint8_t> hk4((size_t) cells * H * q4_row);
    std::vector<int8_t> hv((size_t) cells * H * D);
    std::vector<uint16_t> hvs((size_t) cells * H * G);
    ck(cudaMemcpy(hk4.data(), d_k4, hk4.size(), cudaMemcpyDeviceToHost), "k4");
    ck(cudaMemcpy(hv.data(), d_vq, hv.size(), cudaMemcpyDeviceToHost), "vq");
    ck(cudaMemcpy(hvs.data(), d_vs, hvs.size() * 2, cudaMemcpyDeviceToHost), "vs");
    const k::block_q4_0* hk4b = reinterpret_cast<const k::block_q4_0*>(hk4.data());
    for (int c = 0; c < cells; ++c) {
        const long long row = (long long) table[c / P] * H;
        for (int h = 0; h < H; ++h) {
            const long long r = (row + h) * P + c % P;
            for (int d = 0; d < D; ++d)
                if (hv[(size_t) r * D + d] != rv[((size_t) c * H + h) * D + d]) ++append_bad;
            for (int g = 0; g < G; ++g)
                if (hvs[(size_t) r * G + g] != rv_s[((size_t) c * H + h) * G + g]) ++append_bad;
            for (int b = 0; b < D / 32; ++b) {
                const k::block_q4_0& got = hk4b[(size_t) r * (D / 32) + b];
                if (got.d != rk4[((size_t) c * H + h) * (D / 32) + b].d ||
                    std::memcmp(got.qs, rk4[((size_t) c * H + h) * (D / 32) + b].qs, 16) != 0)
                    ++append_bad;
            }
        }
    }
    std::printf("[1/4] composed appends (rotated K4_0 + INT8 V): %s (%d bad of %zu K / %zu V)\n",
                append_bad == 0 ? "PASS" : "FAIL", append_bad, hk4.size(), hv.size());
    if (append_bad != 0) g_fail = 1;

    // ---- attention: the queries rotate too (K is stored rotated); the output stays rotated for K4V8
    std::vector<float> q((size_t) QH * D);
    for (auto& x : q) x = nd(rng);
    for (int h = 0; h < QH; ++h) fwht256_host_reference(q.data() + (size_t) h * D);
    float* d_q = dalloc<float>(q.size());
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");

    std::vector<int32_t> ids(cells);
    for (int i = 0; i < cells; ++i) ids[i] = i;
    int32_t* d_ids = dalloc<int32_t>(cells);
    ck(cudaMemcpy(d_ids, ids.data(), cells * 4, cudaMemcpyHostToDevice), "ids");
    int32_t step[k::kStepCount];
    k::qsa_step_fill(step, cells - 1, s);
    step[k::kStepWidth] = cells;
    ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step2");

    k::QsaAttnPools pools;
    pools.k_q4 = d_k4; pools.v_q = d_vq; pools.v_scale = d_vs; pools.page_table = d_table;

    std::vector<float> Kdq((size_t) cells * H * D), Vdq((size_t) cells * H * D);
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                const k::block_q4_0& b = rk4[((size_t) c * H + h) * (D / 32) + d / 32];
                const int nib = d % 32 < 16 ? (b.qs[d % 32] & 0x0F) - 8 : ((b.qs[d % 32 - 16] >> 4) - 8);
                Kdq[((size_t) c * H + h) * D + d] = (float) nib * k::f32_from_f16(b.d);
                Vdq[((size_t) c * H + h) * D + d] =
                    (float) rv[((size_t) c * H + h) * D + d] * k::f32_from_f16(rv_s[((size_t) c * H + h) * G + d / k::KV_Q8_GROUP]);
            }
    // host attention in the ROTATED domain: q rotated, K rotated, V unrotated -> the output needs no transform
    auto host_attn = [&](std::vector<float>& out) {
        out.assign((size_t) QH * D, 0.0f);
        for (int h = 0; h < QH; ++h) {
            const int kvh = h / PER;
            std::vector<float> sc(cells);
            float m = -1e30f;
            for (int c = 0; c < cells; ++c) {
                float acc = 0.0f;
                for (int t = 0; t < D; ++t) acc += q[(size_t) h * D + t] * Kdq[((size_t) c * H + kvh) * D + t];
                sc[c] = acc / std::sqrt((float) D);
                m = std::max(m, sc[c]);
            }
            float l = 0.0f;
            for (int c = 0; c < cells; ++c) { sc[c] = std::exp(sc[c] - m); l += sc[c]; }
            for (int c = 0; c < cells; ++c)
                for (int t = 0; t < D; ++t)
                    out[(size_t) h * D + t] += sc[c] / l * Vdq[((size_t) c * H + kvh) * D + t];
        }
    };
    std::vector<float> ref;
    host_attn(ref);

    float* d_scr = dalloc<float>(k::qsa_decode_attn_scratch_floats(cells, s) + 16);
    float* d_attn = dalloc<float>((size_t) QH * D);
    k::qsa_decode_attn_step(d_q, pools, d_ids, d_step, cells, s, d_scr, d_attn, nullptr);
    ck(cudaDeviceSynchronize(), "attn sync");
    std::vector<float> h_attn((size_t) QH * D);
    ck(cudaMemcpy(h_attn.data(), d_attn, h_attn.size() * 4, cudaMemcpyDeviceToHost), "attn");
    double max_d = 0.0;
    for (size_t i = 0; i < h_attn.size(); ++i) max_d = std::max(max_d, (double) std::fabs(h_attn[i] - ref[i]));
    const bool d_ok = max_d < 5e-3;
    std::printf("[2/4] decode attention (KV_MODE 4, no output rotation): %s (vs dequant ref %.2e)\n",
                d_ok ? "PASS" : "FAIL", max_d);
    if (!d_ok) g_fail = 1;

    // ---- prompt attention: mode 5 must be TAKEN (a false return means the engine fell back to 32-query decode)
    {
        float* d_at5 = dalloc<float>((size_t) QH * D);
        const bool took = k::qsa_prompt_attn_batch(d_q, pools, d_ids, d_step, cells, s, d_at5, 1, nullptr);
        if (!took) {
            std::printf("[3/4] qsa_prompt_attn KV_MODE 5: FAIL (refused the K4V8 pools)\n");
            g_fail = 1;
        } else {
            ck(cudaDeviceSynchronize(), "at5 sync");
            std::vector<float> h_at5((size_t) QH * D);
            ck(cudaMemcpy(h_at5.data(), d_at5, h_at5.size() * 4, cudaMemcpyDeviceToHost), "at5");
            double max_p = 0.0;
            for (size_t i = 0; i < h_at5.size(); ++i) max_p = std::max(max_p, (double) std::fabs(h_at5[i] - ref[i]));
            const bool p_ok = max_p < 1e-2;
            std::printf("[3/4] prompt attention (KV_MODE 5, tensor cores, no output rotation): %s (vs dequant ref %.2e)\n",
                        p_ok ? "PASS" : "FAIL", max_p);
            if (!p_ok) g_fail = 1;
        }
    }

    // ---- a run that un-rotates K4V8's output anyway must NOT match: that is the rule the engine relies on
    {
        float* d_attn2 = dalloc<float>((size_t) QH * D);
        ck(cudaMemcpy(d_attn2, d_attn, (size_t) QH * D * 4, cudaMemcpyDeviceToDevice), "copy");
        k::fwht256_inplace_cuda(d_attn2, QH, nullptr);
        ck(cudaDeviceSynchronize(), "rot sync");
        std::vector<float> h2((size_t) QH * D);
        ck(cudaMemcpy(h2.data(), d_attn2, h2.size() * 4, cudaMemcpyDeviceToHost), "a2");
        double max_w = 0.0;
        for (size_t i = 0; i < h2.size(); ++i) max_w = std::max(max_w, (double) std::fabs(h2[i] - ref[i]));
        const bool w_ok = max_w > 1e-2;   // it has to be visibly wrong
        std::printf("[4/4] an un-rotated K4V8 output is rejected: %s (mismatch %.2e)\n", w_ok ? "PASS" : "FAIL", max_w);
        if (!w_ok) g_fail = 1;
    }

    const bool ok = g_fail == 0;
    std::printf("=== kv_k4v8_parity: %s ===\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
