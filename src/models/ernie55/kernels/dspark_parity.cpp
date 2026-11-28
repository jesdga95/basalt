// src/models/ernie55/kernels/dspark_parity.cpp - GPU vs FP64 CPU parity tests for ERNIE-5.5 DSpark kernels.
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace basalt::models::ernie55;

namespace {

int g_fail = 0;

void check_err(const char* name, double err, double tol) {
    if (err <= tol) {
        std::printf("[PASS] %-36s max_err=%.3e (tol=%.1e)\n", name, err, tol);
    } else {
        std::printf("[FAIL] %-36s max_err=%.3e > tol=%.1e\n", name, err, tol);
        ++g_fail;
    }
}

void test_mhc_stream_mean() {
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    const int T = 7, H = 512, n_slots = 3, stride = n_slots * H;
    std::vector<float> res((size_t) T * 4 * H), out((size_t) T * stride, 0.f);
    for (float& v : res) v = dist(rng);

    float *d_res = nullptr, *d_out = nullptr;
    cudaMalloc(&d_res, res.size() * sizeof(float));
    cudaMalloc(&d_out, out.size() * sizeof(float));
    cudaMemset(d_out, 0, out.size() * sizeof(float));
    cudaMemcpy(d_res, res.data(), res.size() * sizeof(float), cudaMemcpyHostToDevice);

    mhc_stream_mean(d_res, T, H, d_out + H, stride, nullptr);   // write into slot 1
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < H; ++c) {
            double ref = 0.0;
            for (int s = 0; s < 4; ++s) ref += res[(size_t) (t * 4 + s) * H + c];
            ref *= 0.25;
            max_err = std::max(max_err, std::abs((double) out[(size_t) t * stride + H + c] - ref));
        }
    }
    cudaFree(d_res);
    cudaFree(d_out);
    check_err("mhc_stream_mean", max_err, 1e-6);
}

void test_dspark_ctx_and_swa_attn() {
    std::mt19937 rng(202);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);
    const int n_ctx = 140, n_block = 7, n_heads = 8, head_dim = 512, win_size = 128, win_cap = 256;
    const float rms_eps = 1e-6f;
    const RopeTable rt = rope_table_plain(8000000.f);

    std::vector<float> kv_in((size_t) n_ctx * head_dim), norm_w(head_dim);
    std::vector<float> q((size_t) n_block * n_heads * head_dim), bkv((size_t) n_block * head_dim);
    std::vector<float> sink(n_heads);
    for (float& v : kv_in) v = dist(rng);
    for (float& v : norm_w) v = 0.5f + 0.5f * dist(rng);
    for (float& v : q) v = dist(rng);
    for (float& v : bkv) v = dist(rng);
    for (float& v : sink) v = dist(rng);

    float *d_kvin = nullptr, *d_nw = nullptr, *d_q = nullptr, *d_bkv = nullptr, *d_sink = nullptr, *d_out = nullptr;
    uint16_t* d_ring = nullptr;
    cudaMalloc(&d_kvin, kv_in.size() * sizeof(float));
    cudaMalloc(&d_nw, norm_w.size() * sizeof(float));
    cudaMalloc(&d_ring, (size_t) win_cap * head_dim * sizeof(uint16_t));
    cudaMalloc(&d_q, q.size() * sizeof(float));
    cudaMalloc(&d_bkv, bkv.size() * sizeof(float));
    cudaMalloc(&d_sink, sink.size() * sizeof(float));
    cudaMalloc(&d_out, q.size() * sizeof(float));

    cudaMemcpy(d_kvin, kv_in.data(), kv_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nw, norm_w.data(), norm_w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bkv, bkv.data(), bkv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sink, sink.data(), sink.size() * sizeof(float), cudaMemcpyHostToDevice);

    dspark_kv_ctx_store(d_kvin, d_nw, 0, n_ctx, head_dim, win_cap, rms_eps, rt, d_ring, nullptr);
    dspark_swa_block_attn(d_q, d_ring, d_bkv, d_sink, n_ctx, 0, n_block, n_heads, head_dim, win_size, win_cap, d_out,
                          nullptr);

    std::vector<uint16_t> h_ring((size_t) win_cap * head_dim);
    std::vector<float> h_out(q.size());
    cudaMemcpy(h_ring.data(), d_ring, h_ring.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // FP64 reference for ctx_store (verify last row)
    double ring_err = 0.0;
    std::vector<std::vector<double>> ctx_ref((size_t) n_ctx, std::vector<double>(head_dim));
    for (int t = 0; t < n_ctx; ++t) {
        double ss = 0.0;
        for (int c = 0; c < head_dim; ++c) ss += (double) kv_in[(size_t) t * head_dim + c] * kv_in[(size_t) t * head_dim + c];
        double inv = 1.0 / std::sqrt(ss / head_dim + rms_eps);
        for (int c = 0; c < head_dim; ++c) ctx_ref[(size_t) t][c] = (double) kv_in[(size_t) t * head_dim + c] * inv * norm_w[c];
        for (int pr = 0; pr < 32; ++pr) {
            int idx = head_dim - 64 + 2 * pr;
            double ang = (double) t * (double) rt.step[pr];
            double c0 = std::cos(ang), s0 = std::sin(ang);
            double x0 = ctx_ref[(size_t) t][idx], x1 = ctx_ref[(size_t) t][idx + 1];
            ctx_ref[(size_t) t][idx] = x0 * c0 - x1 * s0;
            ctx_ref[(size_t) t][idx + 1] = x0 * s0 + x1 * c0;
        }
        int slot = t % win_cap;
        for (int c = 0; c < head_dim; ++c) {
            __half hf;
            std::memcpy(&hf, &h_ring[(size_t) slot * head_dim + c], 2);
            ring_err = std::max(ring_err, std::abs((double) __half2float(hf) - ctx_ref[(size_t) t][c]));
        }
    }
    check_err("dspark_kv_ctx_store (FP16)", ring_err, 5e-3);

    // FP64 reference for bidirectional block + SWA attention
    double attn_err = 0.0;
    const double scale = 1.0 / std::sqrt((double) head_dim);
    const int lo = std::max(0, n_ctx - win_size);
    for (int r = 0; r < n_block; ++r) {
        for (int h = 0; h < n_heads; ++h) {
            const float* qrow = q.data() + ((size_t) r * n_heads + h) * head_dim;
            std::vector<double> scores;
            std::vector<std::vector<double>> vals;
            for (int jp = lo; jp < n_ctx; ++jp) {
                double d = 0.0;
                for (int c = 0; c < head_dim; ++c) d += (double) qrow[c] * ctx_ref[(size_t) jp][c];
                scores.push_back(d * scale);
                vals.push_back(ctx_ref[(size_t) jp]);
            }
            for (int j = 0; j < n_block; ++j) {
                double d = 0.0;
                std::vector<double> v(head_dim);
                for (int c = 0; c < head_dim; ++c) {
                    v[c] = bkv[(size_t) j * head_dim + c];
                    d += (double) qrow[c] * v[c];
                }
                scores.push_back(d * scale);
                vals.push_back(std::move(v));
            }
            double m = sink[h];
            for (double s : scores) m = std::max(m, s);
            double den = std::exp((double) sink[h] - m);
            std::vector<double> acc(head_dim, 0.0);
            for (size_t i = 0; i < scores.size(); ++i) {
                double w = std::exp(scores[i] - m);
                den += w;
                for (int c = 0; c < head_dim; ++c) acc[c] += w * vals[i][c];
            }
            const float* got = h_out.data() + ((size_t) r * n_heads + h) * head_dim;
            for (int c = 0; c < head_dim; ++c) attn_err = std::max(attn_err, std::abs((double) got[c] - acc[c] / den));
        }
    }
    check_err("dspark_swa_block_attn", attn_err, 5e-3);

    cudaFree(d_kvin); cudaFree(d_nw); cudaFree(d_ring); cudaFree(d_q); cudaFree(d_bkv); cudaFree(d_sink); cudaFree(d_out);
}

void test_markov_multimax_argmax_and_conf() {
    std::mt19937 rng(303);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    const int V = 201216, H = 4096, R = 256;
    std::vector<float> base(V), bias(V), mmr = {-2.f, 2.f, -4.f, 4.f}, mmt = {0.1f, -0.05f, 0.02f, -0.01f};
    for (int i = 0; i < V; ++i) {
        base[i] = dist(rng);
        bias[i] = dist(rng) * 0.5f;
    }
    base[123456] = 25.0f;   // clear winner
    base[777] = -INFINITY;  // masked token stays -inf

    double best_v = -1e300;
    int32_t best_i = 0;
    for (int i = 0; i < V; ++i) {
        double x = (double) base[i] + (double) bias[i];
        if (!std::isinf(x)) {
            double d0 = std::max((double) mmr[0] - x, 0.0);
            double d1 = std::max(x - (double) mmr[1], 0.0);
            double d2 = std::max((double) mmr[2] - x, 0.0);
            double d3 = std::max(x - (double) mmr[3], 0.0);
            x = x + mmt[0] * d0 + mmt[1] * d1 + mmt[2] * d2 * d2 + mmt[3] * d3 * d3;
        }
        if (x > best_v) { best_v = x; best_i = i; }
    }

    float *d_base = nullptr, *d_bias = nullptr, *d_mmr = nullptr, *d_mmt = nullptr, *d_sv = nullptr;
    int32_t *d_si = nullptr, *d_tok = nullptr;
    cudaMalloc(&d_base, V * sizeof(float));
    cudaMalloc(&d_bias, V * sizeof(float));
    cudaMalloc(&d_mmr, 4 * sizeof(float));
    cudaMalloc(&d_mmt, 4 * sizeof(float));
    cudaMalloc(&d_sv, 256 * sizeof(float));
    cudaMalloc(&d_si, 256 * sizeof(int32_t));
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMemcpy(d_base, base.data(), V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mmr, mmr.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mmt, mmt.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

    dspark_markov_multimax_argmax(d_base, d_bias, d_mmr, d_mmt, V, d_sv, d_si, d_tok, nullptr);
    int32_t got_tok = -1;
    cudaMemcpy(&got_tok, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost);
    check_err("dspark_markov_multimax_argmax", std::abs(got_tok - best_i), 0.0);

    cudaFree(d_base); cudaFree(d_bias); cudaFree(d_mmr); cudaFree(d_mmt); cudaFree(d_sv); cudaFree(d_si); cudaFree(d_tok);

    // Confidence head test
    std::vector<float> cw(H + R), hc(H), me(R);
    for (float& v : cw) v = dist(rng) * 0.02f;
    for (float& v : hc) v = dist(rng) * 0.2f;
    for (float& v : me) v = dist(rng) * 0.2f;
    double dot = 0.0;
    for (int i = 0; i < H; ++i) dot += (double) cw[i] * hc[i];
    for (int j = 0; j < R; ++j) dot += (double) cw[H + j] * me[j];
    const double ref_conf = 1.0 / (1.0 + std::exp(-dot));

    float *d_cw = nullptr, *d_hc = nullptr, *d_me = nullptr, *d_cf = nullptr;
    cudaMalloc(&d_cw, cw.size() * sizeof(float));
    cudaMalloc(&d_hc, hc.size() * sizeof(float));
    cudaMalloc(&d_me, me.size() * sizeof(float));
    cudaMalloc(&d_cf, sizeof(float));
    cudaMemcpy(d_cw, cw.data(), cw.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hc, hc.data(), hc.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_me, me.data(), me.size() * sizeof(float), cudaMemcpyHostToDevice);

    dspark_confidence(d_cw, d_hc, d_me, H, R, d_cf, nullptr);
    float got_conf = -1.f;
    cudaMemcpy(&got_conf, d_cf, sizeof(float), cudaMemcpyDeviceToHost);
    check_err("dspark_confidence", std::abs((double) got_conf - ref_conf), 1e-6);

    cudaFree(d_cw); cudaFree(d_hc); cudaFree(d_me); cudaFree(d_cf);
}

}  // namespace

int main() {
    test_mhc_stream_mean();
    test_dspark_ctx_and_swa_attn();
    test_markov_multimax_argmax_and_conf();
    if (g_fail) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("ALL DSpark parity checks PASSED.\n");
    return 0;
}
