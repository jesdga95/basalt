// src/models/inkling2/kernels/dflash2_parity.cpp - CPU vs GPU parity & solo/batched bit-identity tests for Inkling 2's
// DFlash 2 block-diffusion drafter kernels (dflash2_kernels.hpp), both plain and inside PdlScope. Exit 77 without a GPU.
#include "basalt/core/device_copy.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/models/inkling2/dflash2_kernels.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::inkling2;

namespace {

constexpr int E = kWidth;
constexpr int kVocab = 200064;
std::mt19937 rng(42);
int failures = 0;

std::vector<float> randn(size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<uint16_t> bf16s(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = to_bf16(v[i]);
    return o;
}

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(h.size() * sizeof(T), 16));
    basalt::core::device_copy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    const size_t b = std::max<size_t>(n * sizeof(T), 16);
    cudaMalloc(&p, b);
    cudaMemset(p, 0, b);
    return p;
}

template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    basalt::core::device_copy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

void check_close(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs((double) got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("  %-44s max_abs=%.3e rel=%.3e %s\n", what, max_abs, rel, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

void test_pack_tap_and_split_fc() {
    const int rows = 5, tap_rows = 16;
    auto tap_h = randn((size_t) kDf2Targets * tap_rows * E);
    std::vector<float> cat_h((size_t) rows * kDf2Targets * E);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < kDf2Targets; ++k)
            std::memcpy(&cat_h[((size_t) r * kDf2Targets + k) * E], &tap_h[((size_t) k * tap_rows + r) * E], E * 4);

    float* tap_d = dev(tap_h);
    float* cat_d = dev(cat_h);
    const size_t q81_b = basalt::kernels::native_q8_1_bytes(kDf2Targets * E, rows);
    uint8_t *got_d = zeros<uint8_t>(q81_b), *ref_d = zeros<uint8_t>(q81_b);
    uint16_t *bf_got_d = zeros<uint16_t>((size_t) rows * kDf2Targets * E);
    uint16_t *bf_ref_d = zeros<uint16_t>((size_t) rows * kDf2Targets * E);
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);
    dflash2_pack_tap_bf16(tap_d, (int64_t) tap_rows * E, rows, bf_got_d, st);
    basalt::kernels::wgemm::pack(cat_d, kDf2Targets * E, kDf2Targets * E, rows, bf_ref_d, st);
    dflash2_pack_tap_q8_1(tap_d, (int64_t) tap_rows * E, rows, got_d, st);
    basalt::kernels::native_quantize_q8_1(cat_d, ref_d, kDf2Targets * E, rows, st);
    cudaStreamSynchronize(st);

    const bool ok_bf = (host(bf_got_d, (size_t) rows * kDf2Targets * E) == host(bf_ref_d, (size_t) rows * kDf2Targets * E));
    std::printf("  %-44s %s\n", "pack_tap_bf16 bitwise vs wgemm::pack", ok_bf ? "PASS" : "FAIL");
    if (!ok_bf) ++failures;

    const auto got = host(got_d, q81_b), ref = host(ref_d, q81_b);
    const bool ok_tap = (got == ref);
    std::printf("  %-44s %s\n", "pack_tap_q8_1 bitwise vs native_quantize", ok_tap ? "PASS" : "FAIL");
    if (!ok_tap) ++failures;

    // Test split_fc_q8 layout
    const size_t full_b = (size_t) E * (kDf2Targets * E / 32) * 34;
    std::vector<uint8_t> fc_h(full_b);
    for (size_t i = 0; i < full_b; ++i) fc_h[i] = (uint8_t) (rng() & 0xff);
    uint8_t* fc_d = dev(fc_h);
    uint8_t* sp_d = zeros<uint8_t>(full_b);
    dflash2_split_fc_q8(fc_d, sp_d, st);
    const auto sp_h = host(sp_d, full_b);
    bool ok_sp = true;
    const size_t row_blk = (size_t) kDf2Targets * (E / 32);
    const size_t part_blk = E / 32;
    for (int p = 0; p < kDf2Targets && ok_sp; ++p)
        for (int r = 0; r < E && ok_sp; ++r) {
            const uint8_t* s0 = &fc_h[(r * row_blk + p * part_blk) * 34];
            const uint8_t* d0 = &sp_h[((p * E + r) * part_blk) * 34];
            if (std::memcmp(s0, d0, part_blk * 34) != 0) ok_sp = false;
        }
    std::printf("  %-44s %s\n", "split_fc_q8 exact block slice", ok_sp ? "PASS" : "FAIL");
    if (!ok_sp) ++failures;

    cudaStreamDestroy(st);
    cudaFree(tap_d);
    cudaFree(cat_d);
    cudaFree(got_d);
    cudaFree(ref_d);
    cudaFree(bf_got_d);
    cudaFree(bf_ref_d);
    cudaFree(fc_d);
    cudaFree(sp_d);
}

void test_conv2_and_qk_prep() {
    const int count = 3, B = kDf2Block, R = count * B;
    auto z_h = randn((size_t) R * E);
    auto dyn_h = randn((size_t) R * kDf2Dyn, 0.2f);
    auto base_h = randn(4 * E, 0.5f);
    auto x_h = randn((size_t) R * E);
    auto nw_h = randn(E, 0.4f);

    std::vector<double> ref_pre((size_t) R * E), ref_add((size_t) R * E);
    std::vector<uint16_t> ref_bf16((size_t) R * E);
    for (int r = 0; r < R; ++r) {
        const int t = r % B;
        for (int c = 0; c < E; ++c) {
            const int g = c / kDf2ConvGroup;
            double a0 = ((double) base_h[c] + (double) dyn_h[(size_t) r * kDf2Dyn + g]) * (double) z_h[(size_t) r * E + c];
            if (t > 0)
                a0 += ((double) base_h[E + c] + (double) dyn_h[(size_t) r * kDf2Dyn + kDf2ConvGroups + g]) *
                      (double) z_h[(size_t) (r - 1) * E + c];
            ref_pre[(size_t) r * E + c] = a0;
            ref_bf16[(size_t) r * E + c] = to_bf16((float) a0);

            double a1 = ((double) base_h[2 * E + c] + (double) dyn_h[(size_t) r * kDf2Dyn + 2 * kDf2ConvGroups + g]) *
                        (double) z_h[(size_t) r * E + c];
            if (t > 0)
                a1 += ((double) base_h[3 * E + c] + (double) dyn_h[(size_t) r * kDf2Dyn + 3 * kDf2ConvGroups + g]) *
                      (double) z_h[(size_t) (r - 1) * E + c];
            ref_add[(size_t) r * E + c] = (double) x_h[(size_t) r * E + c] + a1;
        }
    }

    float *z_d = dev(z_h), *dyn_d = dev(dyn_h), *base_d = dev(base_h), *x_d = dev(x_h), *x2_d = dev(x_h);
    float *nw_d = dev(nw_h), *y_d = zeros<float>((size_t) R * E);
    float *xn_sep_d = zeros<float>((size_t) R * E), *xn_fus_d = zeros<float>((size_t) R * E);
    uint16_t *act_d = zeros<uint16_t>((size_t) R * E);
    uint16_t *act_sep_d = zeros<uint16_t>((size_t) R * E), *act_fus_d = zeros<uint16_t>((size_t) R * E);
    {
        basalt::kernels::PdlScope pdl(true);
        dflash2_conv2_pack(z_d, dyn_d, kDf2Dyn, base_d, B, count, act_d, y_d, nullptr);
        dflash2_conv2_add(z_d, dyn_d, kDf2Dyn, base_d, B, count, x_d, nullptr);
        basalt::kernels::wgemm::norm_pack(x_d, nw_d, 1e-6f, E, R, act_sep_d, xn_sep_d, nullptr);
        dflash2_conv2_add_norm_pack(z_d, dyn_d, kDf2Dyn, base_d, B, count, x2_d, nw_d, 1e-6f, act_fus_d, xn_fus_d,
                                    nullptr);
    }
    check_close("conv2_pack fp32", host(y_d, (size_t) R * E), ref_pre, 1e-6);
    check_close("conv2_add residual", host(x_d, (size_t) R * E), ref_add, 1e-6);
    const bool fus_ok = (host(x_d, (size_t) R * E) == host(x2_d, (size_t) R * E)) &&
                        (host(xn_sep_d, (size_t) R * E) == host(xn_fus_d, (size_t) R * E)) &&
                        (host(act_sep_d, (size_t) R * E) == host(act_fus_d, (size_t) R * E));
    std::printf("  %-44s %s\n", "conv2_add_norm_pack bitwise vs unfused", fus_ok ? "PASS" : "FAIL");
    if (!fus_ok) ++failures;

    const auto got_bf16 = host(act_d, (size_t) R * E);
    bool bf_ok = true;
    for (size_t i = 0; i < got_bf16.size(); ++i)
        if (std::abs((int) got_bf16[i] - (int) ref_bf16[i]) > 1) bf_ok = false;
    std::printf("  %-44s %s\n", "conv2_pack bf16 <= 1 ULP", bf_ok ? "PASS" : "FAIL");
    if (!bf_ok) ++failures;

    // Solo vs batched bit-identity on block_qk_prep_batch + norm_draft_pack
    const int cap = 2112;
    auto q_h = randn((size_t) R * kDf2Q), kb_h = randn((size_t) R * kDf2KV), vb_h = randn((size_t) R * kDf2KV);
    auto qn_h = randn(kDf2HeadDim, 0.3f), kn_h = randn(kDf2HeadDim, 0.3f);
    std::vector<int32_t> meta_h = {100, 17, 200, 2045, 300, 512};
    float *q_d = dev(q_h), *kb_d = dev(kb_h), *vb_d = dev(vb_h), *qn_d = dev(qn_h), *kn_d = dev(kn_h);
    int32_t* meta_d = dev(meta_h);
    float *qout_b = zeros<float>((size_t) R * kDf2Q), *qout_s = zeros<float>((size_t) R * kDf2Q);
    DsBatchKv kv_b{}, kv_s{};
    for (int s = 0; s < count; ++s) {
        kv_b.k[s] = zeros<uint16_t>((size_t) cap * kDf2KV);
        kv_b.v[s] = zeros<uint16_t>((size_t) cap * kDf2KV);
        kv_s.k[s] = zeros<uint16_t>((size_t) cap * kDf2KV);
        kv_s.v[s] = zeros<uint16_t>((size_t) cap * kDf2KV);
    }
    dflash2_block_qk_prep_batch(q_d, kb_d, vb_d, qn_d, kn_d, 1e-6f, 1e4f, meta_d, B, count, qout_b, kv_b, cap, nullptr);
    for (int s = 0; s < count; ++s) {
        DsBatchKv one{};
        one.k[0] = kv_s.k[s];
        one.v[0] = kv_s.v[s];
        dflash2_block_qk_prep_batch(q_d + (size_t) s * B * kDf2Q, kb_d + (size_t) s * B * kDf2KV,
                                    vb_d + (size_t) s * B * kDf2KV, qn_d, kn_d, 1e-6f, 1e4f, meta_d + 2 * s, B, 1,
                                    qout_s + (size_t) s * B * kDf2Q, one, cap, nullptr);
    }
    bool qk_id = (host(qout_b, (size_t) R * kDf2Q) == host(qout_s, (size_t) R * kDf2Q));
    for (int s = 0; s < count; ++s) {
        if (host(kv_b.k[s], (size_t) cap * kDf2KV) != host(kv_s.k[s], (size_t) cap * kDf2KV)) qk_id = false;
        if (host(kv_b.v[s], (size_t) cap * kDf2KV) != host(kv_s.v[s], (size_t) cap * kDf2KV)) qk_id = false;
    }
    std::printf("  %-44s %s\n", "block_qk_prep solo vs batch bitwise", qk_id ? "PASS" : "FAIL");
    if (!qk_id) ++failures;

    const int D = count * kDf2Drafts;
    const size_t hq_b = basalt::kernels::native_q8_1_bytes(E, D);
    uint16_t *nd_act_b = zeros<uint16_t>((size_t) D * E), *nd_act_s = zeros<uint16_t>((size_t) D * E);
    uint8_t *nd_hq_b = zeros<uint8_t>(hq_b), *nd_hq_s = zeros<uint8_t>(hq_b);
    float* nd_f_b = zeros<float>((size_t) D * E);
    dflash2_norm_draft_pack(z_d, nw_d, 1e-6f, B, count, kDf2Drafts, nd_act_b, nd_hq_b, nd_f_b, nullptr);
    const size_t hq_one = basalt::kernels::native_q8_1_bytes(E, kDf2Drafts);
    for (int s = 0; s < count; ++s) {
        dflash2_norm_draft_pack(z_d + (size_t) s * B * E, nw_d, 1e-6f, B, 1, kDf2Drafts,
                                nd_act_s + (size_t) s * kDf2Drafts * E, nd_hq_s + s * hq_one, nullptr, nullptr);
    }
    const bool nd_id = (host(nd_act_b, (size_t) D * E) == host(nd_act_s, (size_t) D * E)) &&
                       (host(nd_hq_b, hq_b) == host(nd_hq_s, hq_b));
    std::printf("  %-44s %s\n", "norm_draft_pack solo vs batch bitwise", nd_id ? "PASS" : "FAIL");
    if (!nd_id) ++failures;

    cudaFree(z_d); cudaFree(dyn_d); cudaFree(base_d); cudaFree(x_d); cudaFree(x2_d); cudaFree(y_d); cudaFree(act_d);
    cudaFree(xn_sep_d); cudaFree(xn_fus_d); cudaFree(act_sep_d); cudaFree(act_fus_d);
    cudaFree(q_d); cudaFree(kb_d); cudaFree(vb_d); cudaFree(qn_d); cudaFree(kn_d); cudaFree(nw_d); cudaFree(meta_d);
    cudaFree(qout_b); cudaFree(qout_s);
    for (int s = 0; s < count; ++s) {
        cudaFree(kv_b.k[s]); cudaFree(kv_b.v[s]); cudaFree(kv_s.k[s]); cudaFree(kv_s.v[s]);
    }
    cudaFree(nd_act_b); cudaFree(nd_act_s); cudaFree(nd_hq_b); cudaFree(nd_hq_s); cudaFree(nd_f_b);
}

void test_top16_and_selector() {
    const int count = 2, n_dr = kDf2Drafts, rows = count * n_dr;
    const float cap = 30.0f;
    auto logits_h = randn((size_t) rows * kVocab, 3.0f);
    // Inject an exact tie to verify deterministic lower-id tie-breaking.
    logits_h[123] = 25.5f;
    logits_h[45] = 25.5f;

    float* logits_d = dev(logits_h);
    uint8_t* scr_d = zeros<uint8_t>(dflash2_top16_scratch_bytes(rows, kVocab));
    int32_t* cand_d = zeros<int32_t>((size_t) rows * 2 * kDf2TopK);
    float* tail_lse_d = zeros<float>((size_t) rows * 2);
    dflash2_top16(logits_d, nullptr, false, nullptr, rows, kVocab, cap, scr_d, cand_d, tail_lse_d, nullptr);
    const auto cand_h = host(cand_d, (size_t) rows * 2 * kDf2TopK);
    const auto tail_h = host(tail_lse_d, (size_t) rows * 2);

    bool top_ok = true;
    std::vector<double> ref_full_p;
    std::vector<float> got_full_p;
    for (int r = 0; r < rows && top_ok; ++r) {
        std::vector<int> ord(kVocab);
        for (int i = 0; i < kVocab; ++i) ord[i] = i;
        const float* lr = &logits_h[(size_t) r * kVocab];
        std::partial_sort(ord.begin(), ord.begin() + kDf2TopK, ord.end(), [&](int a, int b) {
            return lr[a] > lr[b] || (lr[a] == lr[b] && a < b);
        });
        for (int k = 0; k < kDf2TopK; ++k) {
            float gv;
            std::memcpy(&gv, &cand_h[(size_t) r * 32 + 16 + k], 4);
            if (cand_h[(size_t) r * 32 + k] != ord[k] || gv != lr[ord[k]]) top_ok = false;
        }
        // Check full-200,064-vocabulary softmax probability recovered from Top-16 + tail_lse
        const double top_c = cap * std::tanh((double) lr[ord[0]] / cap);
        double z_exact = 0.0;
        for (int i = 0; i < kVocab; ++i) z_exact += std::exp(cap * std::tanh((double) lr[i] / cap) - top_c);
        ref_full_p.push_back(1.0 / z_exact);

        const double m_t = tail_h[2 * r], l_t = tail_h[2 * r + 1];
        const double m_ref = std::max(top_c, m_t);
        double z16 = 0.0;
        for (int k = 0; k < kDf2TopK; ++k)
            z16 += std::exp(cap * std::tanh((double) lr[ord[k]] / cap) - m_ref);
        got_full_p.push_back((float) (std::exp(top_c - m_ref) / (z16 + l_t * std::exp(m_t - m_ref))));
    }
    std::printf("  %-44s %s\n", "top16 exact ids & tie-break over 200,064", top_ok ? "PASS" : "FAIL");
    if (!top_ok) ++failures;
    check_close("top16 + tail_lse full-200k softmax prob", got_full_p, ref_full_p, 1e-4);

    // Test 16-warp selector path walk with after_correction softcap = 30.0 + tail_lse calibration
    const int small_v = 512;
    auto pred_f = randn((size_t) small_v * kDf2Rank, 0.25f);
    auto succ_f = randn((size_t) small_v * kDf2Rank, 0.25f);
    auto pred_h = bf16s(pred_f), succ_h = bf16s(succ_f);
    auto g_h = randn((size_t) rows * kDf2Rank, 0.5f);
    std::vector<int32_t> sc_cand_h((size_t) rows * 32);
    std::vector<float> sc_tail_h((size_t) rows * 2);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < kDf2TopK; ++c) {
            sc_cand_h[(size_t) r * 32 + c] = (r * 19 + c * 7 + 3) % small_v;
            const float lv = -1.5f * (float) c + 4.0f;
            std::memcpy(&sc_cand_h[(size_t) r * 32 + 16 + c], &lv, 4);
        }
        sc_tail_h[2 * r] = 2.5f;
        sc_tail_h[2 * r + 1] = 18.0f + (float) r;
    }
    std::vector<int32_t> meta_h = {11, 0, 77, 0};

    uint16_t *pred_d = dev(pred_h), *succ_d = dev(succ_h);
    float *g_d = dev(g_h), *sc_tail_d = dev(sc_tail_h);
    int32_t *sc_cand_d = dev(sc_cand_h), *meta_d = dev(meta_h);
    int32_t *ids_d = zeros<int32_t>(count * kDf2Block), *ids_s = zeros<int32_t>(count * kDf2Block);
    float *conf_d = zeros<float>(count * kDf2Block), *conf_s = zeros<float>(count * kDf2Block);

    dflash2_select(meta_d, 2, sc_cand_d, sc_tail_d, g_d, pred_d, succ_d, count, n_dr, cap, ids_d, kDf2Block, conf_d,
                   kDf2Block, nullptr);
    for (int s = 0; s < count; ++s) {
        dflash2_select(meta_d + 2 * s, 2, sc_cand_d + (size_t) s * n_dr * 32, sc_tail_d + (size_t) s * n_dr * 2,
                       g_d + (size_t) s * n_dr * kDf2Rank, pred_d, succ_d, 1, n_dr, cap, ids_s + s * kDf2Block,
                       kDf2Block, conf_s + s * kDf2Block, kDf2Block, nullptr);
    }

    const auto got_ids = host(ids_d, count * kDf2Block), solo_ids = host(ids_s, count * kDf2Block);
    const auto got_conf = host(conf_d, count * kDf2Block), solo_conf = host(conf_s, count * kDf2Block);
    bool sel_id_ok = (got_ids == solo_ids && got_conf == solo_conf);
    std::vector<double> ref_conf;
    std::vector<float> got_conf_vec;
    for (int s = 0; s < count; ++s) {
        int32_t prev = meta_h[2 * s];
        for (int d = 0; d < n_dr; ++d) {
            const int r = s * n_dr + d;
            double sc[kDf2TopK];
            int best = 0;
            int32_t best_id = sc_cand_h[(size_t) r * 32];
            for (int c = 0; c < kDf2TopK; ++c) {
                const int32_t cid = sc_cand_h[(size_t) r * 32 + c];
                float lv;
                std::memcpy(&lv, &sc_cand_h[(size_t) r * 32 + 16 + c], 4);
                double dot = 0.0;
                for (int i = 0; i < kDf2Rank; ++i) {
                    const double pg = (double) from_bf16(pred_h[(size_t) prev * kDf2Rank + i]) *
                                      (double) g_h[(size_t) r * kDf2Rank + i];
                    dot += pg * (double) from_bf16(succ_h[(size_t) cid * kDf2Rank + i]);
                }
                sc[c] = (double) lv + dot;
                if (c > 0 && (sc[c] > sc[best] || (sc[c] == sc[best] && cid < best_id))) {
                    best = c;
                    best_id = cid;
                }
            }
            if (got_ids[s * kDf2Block + d] != best_id) sel_id_ok = false;
            const double top = cap * std::tanh(sc[best] / cap);
            const double m_t = sc_tail_h[2 * r], l_t = sc_tail_h[2 * r + 1];
            const double m_ref = std::max(top, m_t);
            double z = l_t * std::exp(m_t - m_ref);
            for (int c = 0; c < kDf2TopK; ++c) z += std::exp(cap * std::tanh(sc[c] / cap) - m_ref);
            ref_conf.push_back(std::exp(top - m_ref) / z);
            got_conf_vec.push_back(got_conf[s * kDf2Block + d]);
            prev = best_id;
        }
    }
    std::printf("  %-44s %s\n", "select 16-warp greedy path & solo==batch", sel_id_ok ? "PASS" : "FAIL");
    if (!sel_id_ok) ++failures;
    check_close("select calibrated full-vocab conf", got_conf_vec, ref_conf, 1e-4);

    cudaFree(logits_d); cudaFree(scr_d); cudaFree(cand_d); cudaFree(tail_lse_d);
    cudaFree(pred_d); cudaFree(succ_d); cudaFree(g_d); cudaFree(sc_tail_d); cudaFree(sc_cand_d); cudaFree(meta_d);
    cudaFree(ids_d); cudaFree(ids_s); cudaFree(conf_d); cudaFree(conf_s);
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("inkling2_dflash2_parity: no CUDA device, skipping\n");
        return 77;
    }
    std::printf("inkling2_dflash2_parity:\n");
    test_pack_tap_and_split_fc();
    test_conv2_and_qk_prep();
    test_top16_and_selector();
    return failures ? 1 : 0;
}
