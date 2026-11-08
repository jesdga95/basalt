// src/models/deepseek41/dspark_parity.cpp - parity of the DSpark drafter's kernels against plain CPU references, at the
// drafter's geometry (5120 -> 2304 -> 5120 MXFP4 experts, 512-wide latents, a 129280-row rank-256 Markov table), with
// a microbench line for the expert kernels; and a drafter that keeps some experts computes what the whole one does on them.
#include "basalt/models/deepseek41/kernels.hpp"
#include "models/deepseek41/dspark.hpp"
#include "models/deepseek41/dspark_kernels.hpp"
#include "models/deepseek41/kernels/test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

using namespace basalt::models::deepseek41;
using dstest::check;
using dstest::Dev;

namespace {

const float kMx[16] = {0, 1, 2, 3, 4, 6, 8, 12, -0.f, -1, -2, -3, -4, -6, -8, -12};

float scale_of(unsigned e) {
    uint32_t bits = e < 2 ? (0x00200000u << e) : ((e - 1) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// One MXFP4 row dot x, in double.
double row_dot(const uint8_t* row, const float* x, int n) {
    double s = 0;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 17;
        const double sc = scale_of(blk[0]);
        for (int j = 0; j < 16; ++j) {
            s += sc * kMx[blk[1 + j] & 15] * x[b * 32 + j];
            s += sc * kMx[blk[1 + j] >> 4] * x[b * 32 + j + 16];
        }
    }
    return s;
}

std::vector<uint8_t> random_mxfp4(std::mt19937& rng, size_t rows, int n_in) {
    std::vector<uint8_t> w(rows * mxfp4_row_bytes(n_in) + kMxfp4Pad);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (rng() & 0xff);
    for (size_t r = 0; r < rows; ++r)
        for (int b = 0; b < n_in / 32; ++b) w[r * mxfp4_row_bytes(n_in) + (size_t) b * 17] = (uint8_t) (118 + rng() % 6);
    return w;
}

int failures = 0;

void verdict(const char* what, const dstest::Err& e, double tol) {
    const bool ok = e.rms_rel < tol;
    std::printf("%-22s rms_rel %.2e max_abs %.2e  %s\n", what, e.rms_rel, e.max_abs, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int kT = 5, kK = 3, kE = 16, kIn = 5120, kFF = 2304, kOut = 5120;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ---- routed experts
    const auto gate = random_mxfp4(rng, (size_t) kE * kFF, kIn), up = random_mxfp4(rng, (size_t) kE * kFF, kIn);
    const auto down = random_mxfp4(rng, (size_t) kE * kOut, kFF);
    std::vector<float> x((size_t) kT * kIn);
    for (float& v : x) v = nd(rng);
    std::vector<int32_t> ids(kT * kK);
    for (int p = 0; p < kT * kK; ++p) ids[p] = (int32_t) (rng() % kE);
    Dev<uint8_t> dg(gate), du(up), dd(down);
    Dev<float> dx(x), dh((size_t) kT * kK * kFF), dout((size_t) kT * kK * kOut);
    Dev<int32_t> did(ids);
    dspark_moe_gate_up(dg.p, du.p, kIn, kFF, dx.p, did.p, kT, kK, 10.f, dh.p, nullptr);
    dspark_moe_down(dd.p, kFF, kOut, dh.p, did.p, kT, kK, dout.p, nullptr);
    check(cudaDeviceSynchronize(), "experts");
    std::vector<float> h_ref((size_t) kT * kK * kFF), o_ref((size_t) kT * kK * kOut);
    for (int p = 0; p < kT * kK; ++p) {
        const int t = p / kK, e = ids[p];
        for (int r = 0; r < kFF; ++r) {
            const double g = std::fmin(row_dot(&gate[((size_t) e * kFF + r) * mxfp4_row_bytes(kIn)], &x[(size_t) t * kIn], kIn), 10.0);
            const double u = std::fmin(std::fmax(row_dot(&up[((size_t) e * kFF + r) * mxfp4_row_bytes(kIn)], &x[(size_t) t * kIn], kIn), -10.0), 10.0);
            h_ref[(size_t) p * kFF + r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
    }
    verdict("moe gate_up", dstest::compare(dh.down(), h_ref, 1e-2), 1e-4);
    const std::vector<float> h_got = dh.down();   // the down check starts from the kernel's own hidden state
    for (int p = 0; p < kT * kK; ++p)
        for (int r = 0; r < kOut; ++r)
            o_ref[(size_t) p * kOut + r] = (float) row_dot(&down[((size_t) ids[p] * kOut + r) * mxfp4_row_bytes(kFF)], &h_got[(size_t) p * kFF], kFF);
    verdict("moe down", dstest::compare(dout.down(), o_ref, 1e-2), 1e-4);
    {   // frozen: the hidden state and the outputs hash to the bits of the kernels that read one row per warp
        constexpr uint64_t kFrozen = 0x5970ce9413f8b809ull;
        uint64_t hh = 1469598103934665603ull;
        for (const auto& v : {h_got, dout.down()}) {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(v.data());
            for (size_t i = 0; i < v.size() * sizeof(float); ++i) hh = (hh ^ b[i]) * 1099511628211ull;
        }
        const bool ok = hh == kFrozen;
        std::printf("moe frozen             %016llx (want %016llx)  %s\n", (unsigned long long) hh, (unsigned long long) kFrozen, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }

    // ---- combine
    std::vector<float> w(kT * kK), sh((size_t) kT * kOut);
    for (float& v : w) v = nd(rng);
    for (float& v : sh) v = nd(rng);
    Dev<float> dw(w), dsh(sh), dc((size_t) kT * kOut);
    dspark_moe_combine(dout.p, dw.p, dsh.p, kT, kK, kOut, dc.p, nullptr);
    check(cudaDeviceSynchronize(), "combine");
    std::vector<float> c_ref((size_t) kT * kOut);
    for (int t = 0; t < kT; ++t)
        for (int i = 0; i < kOut; ++i) {
            double s = sh[(size_t) t * kOut + i];
            for (int k = 0; k < kK; ++k) s += (double) w[t * kK + k] * o_ref[((size_t) t * kK + k) * kOut + i];
            c_ref[(size_t) t * kOut + i] = (float) s;
        }
    verdict("moe combine", dstest::compare(dc.down(), c_ref, 1e-2), 1e-4);

    // ---- latent rotation
    std::vector<float> raw((size_t) kT * 512), nw(512), freq(32);
    for (float& v : raw) v = nd(rng);
    for (float& v : nw) v = 0.5f + std::fabs(nd(rng));
    for (int i = 0; i < 32; ++i) freq[i] = std::pow(10000.f, -(float) i / 32.f);
    std::vector<int32_t> pos = {1000, 1001, 1002, 1003, 1004};
    Dev<float> draw(raw), dnw(nw), dfreq(freq), dkv((size_t) kT * 512);
    Dev<int32_t> dpos(pos);
    Dev<uint16_t> dring((size_t) 132 * 512);
    dspark_kv_rope(draw.p, dnw.p, dpos.p, kT, 1e-20f, dfreq.p, dkv.p, dring.p, 132, nullptr);
    check(cudaDeviceSynchronize(), "kv_rope");
    std::vector<float> kv_ref((size_t) kT * 512);
    for (int t = 0; t < kT; ++t) {
        double ss = 0;
        for (int c = 0; c < 512; ++c) ss += (double) raw[(size_t) t * 512 + c] * raw[(size_t) t * 512 + c];
        const double r = 1.0 / std::sqrt(ss / 512 + 1e-20);
        std::vector<double> y(512);
        for (int c = 0; c < 512; ++c) y[c] = raw[(size_t) t * 512 + c] * r * nw[c];
        for (int c = 0; c < 448; ++c) kv_ref[(size_t) t * 512 + c] = (float) y[c];
        for (int p = 0; p < 32; ++p) {
            const double a = pos[t] * (double) freq[p], co = std::cos(a), s = std::sin(a);
            kv_ref[(size_t) t * 512 + 448 + 2 * p] = (float) (y[448 + 2 * p] * co - y[449 + 2 * p] * s);
            kv_ref[(size_t) t * 512 + 449 + 2 * p] = (float) (y[448 + 2 * p] * s + y[449 + 2 * p] * co);
        }
    }
    verdict("kv rope", dstest::compare(dkv.down(), kv_ref, 1e-2), 1e-5);

    // ---- stream mean, per row and independent of how many rows share a launch
    {
        constexpr int kR = 3, kStride = 3 * kOut;
        std::vector<float> streams((size_t) kR * 4 * kOut);
        for (float& v : streams) v = nd(rng);
        Dev<float> ds(streams), dm((size_t) kR * kStride), dm1((size_t) kStride);
        dspark_hc_mean(ds.p, kR, kOut, dm.p, kStride, nullptr);
        dspark_hc_mean(ds.p + (size_t) 4 * kOut, 1, kOut, dm1.p, kStride, nullptr);
        check(cudaDeviceSynchronize(), "hc_mean");
        std::vector<float> m_ref((size_t) kR * kStride, 0.f);
        const std::vector<float> got = dm.down();
        for (int r = 0; r < kR; ++r)
            for (int i = 0; i < kOut; ++i) {
                double acc = 0;
                for (int c = 0; c < 4; ++c) acc += streams[((size_t) r * 4 + c) * kOut + i];
                m_ref[(size_t) r * kStride + i] = (float) (acc * 0.25);
            }
        std::vector<float> got_rows, ref_rows;
        for (int r = 0; r < kR; ++r) {
            got_rows.insert(got_rows.end(), got.begin() + (size_t) r * kStride, got.begin() + (size_t) r * kStride + kOut);
            ref_rows.insert(ref_rows.end(), m_ref.begin() + (size_t) r * kStride, m_ref.begin() + (size_t) r * kStride + kOut);
        }
        verdict("stream mean", dstest::compare(got_rows, ref_rows, 1e-2), 1e-6);
        const std::vector<float> one = dm1.down();
        const bool same = std::memcmp(one.data(), got.data() + (size_t) kStride, (size_t) kOut * sizeof(float)) == 0;
        std::printf("%-22s %s\n", "stream mean width", same ? "ok" : "FAIL");
        if (!same) ++failures;
    }

    // ---- Markov bias and confidence
    constexpr int kV = 129280;
    std::vector<uint16_t> w2((size_t) kV * 256), w1((size_t) 8 * 256);
    auto bf16 = [&](float f) { uint32_t b; std::memcpy(&b, &f, 4); return (uint16_t) (b >> 16); };
    auto unbf = [&](uint16_t h) { uint32_t b = (uint32_t) h << 16; float f; std::memcpy(&f, &b, 4); return f; };
    for (uint16_t& v : w2) v = bf16(0.1f * nd(rng));
    for (uint16_t& v : w1) v = bf16(nd(rng));
    std::vector<float> logits(kV);
    for (float& v : logits) v = nd(rng);
    std::vector<int32_t> tok = {5};
    Dev<uint16_t> dw2(w2), dw1(w1);
    Dev<float> dlog(logits), dcol(kV), dp(256);
    Dev<int32_t> dtok(tok);
    dspark_markov_gather(dw1.p, dtok.p, dp.p, nullptr);
    dspark_markov_bias(dw2.p, dp.p, dlog.p, kV, dcol.p, nullptr);
    check(cudaDeviceSynchronize(), "markov");
    std::vector<float> p_ref(256), col_ref(kV);
    for (int r = 0; r < 256; ++r) p_ref[r] = unbf(w1[5 * 256 + r]);
    verdict("markov gather", dstest::compare(dp.down(), p_ref, 1e-3), 1e-7);
    for (int v = 0; v < kV; ++v) {
        double s = logits[v];
        for (int r = 0; r < 256; ++r) s += (double) unbf(w2[(size_t) v * 256 + r]) * p_ref[r];
        col_ref[v] = (float) s;
    }
    verdict("markov bias", dstest::compare(dcol.down(), col_ref, 1e-2), 1e-5);
    std::vector<float> cw(5120 + 256), hh(5120);
    for (float& v : cw) v = 0.05f * nd(rng);
    for (float& v : hh) v = nd(rng);
    Dev<float> dcw(cw), dhh(hh), dcf(1);
    dspark_conf(dcw.p, dhh.p, dp.p, 5120, dcf.p, nullptr);
    check(cudaDeviceSynchronize(), "conf");
    double s = 0;
    for (int i = 0; i < 5120; ++i) s += (double) cw[i] * hh[i];
    for (int i = 0; i < 256; ++i) s += (double) cw[5120 + i] * p_ref[i];
    verdict("confidence", dstest::compare(dcf.down(), std::vector<float>{(float) (1.0 / (1.0 + std::exp(-s)))}, 1e-3), 1e-5);

    // ---- kept experts: the ranking's first per stage, topped up in index order, ascending
    {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int e = 127; e >= 0; --e) ranked.push_back({0, e});                  // stage 0: most routed is the highest index
        for (int e = 0; e < 20; ++e) ranked.push_back({1, (e * 37) % 128});      // stage 1: names fewer than it keeps
        ranked.push_back({7, 3});                                               // out of range: ignored
        const auto kept = dspark_kept_experts(ranked, 3, 128, 32);
        bool ok = kept.size() == 3;
        for (const auto& k : kept) ok = ok && k.size() == 32 && std::is_sorted(k.begin(), k.end());
        ok = ok && kept[0].front() == 96 && kept[0].back() == 127;               // 96..127
        for (int e = 0; e < 20 && ok; ++e) ok = std::binary_search(kept[1].begin(), kept[1].end(), (e * 37) % 128);
        ok = ok && kept[2].front() == 0 && kept[2].back() == 31;                 // no ranking: the first 32
        std::printf("%-22s %s\n", "kept experts", ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }

    // ---- a kept subset against the whole: the experts by remapped ids, the router with the rest's bias at -inf, bit for bit
    {
        const std::vector<int> keep = {1, 4, 5, 9, 12, 13, 14, 15};
        const size_t gu_one = (size_t) kFF * mxfp4_row_bytes(kIn), d_one = (size_t) kOut * mxfp4_row_bytes(kFF);
        std::vector<uint8_t> kg(keep.size() * gu_one + kMxfp4Pad), ku(keep.size() * gu_one + kMxfp4Pad), kd(keep.size() * d_one + kMxfp4Pad);
        for (size_t i = 0; i < keep.size(); ++i) {
            std::memcpy(&kg[i * gu_one], &gate[(size_t) keep[i] * gu_one], gu_one);
            std::memcpy(&ku[i * gu_one], &up[(size_t) keep[i] * gu_one], gu_one);
            std::memcpy(&kd[i * d_one], &down[(size_t) keep[i] * d_one], d_one);
        }
        std::vector<int32_t> cid(kT * kK), fid(kT * kK);
        for (int p = 0; p < kT * kK; ++p) {
            cid[p] = (int32_t) (rng() % keep.size());
            fid[p] = keep[(size_t) cid[p]];
        }
        Dev<uint8_t> dkg(kg), dku(ku), dkd(kd);
        Dev<int32_t> dcid(cid), dfid(fid);
        Dev<float> h1((size_t) kT * kK * kFF), h2((size_t) kT * kK * kFF), o1((size_t) kT * kK * kOut), o2((size_t) kT * kK * kOut);
        dspark_moe_gate_up(dg.p, du.p, kIn, kFF, dx.p, dfid.p, kT, kK, 10.f, h1.p, nullptr);
        dspark_moe_down(dd.p, kFF, kOut, h1.p, dfid.p, kT, kK, o1.p, nullptr);
        dspark_moe_gate_up(dkg.p, dku.p, kIn, kFF, dx.p, dcid.p, kT, kK, 10.f, h2.p, nullptr);
        dspark_moe_down(dkd.p, kFF, kOut, h2.p, dcid.p, kT, kK, o2.p, nullptr);
        check(cudaDeviceSynchronize(), "kept experts");
        const bool same_e = o1.down() == o2.down() && h1.down() == h2.down();

        constexpr int kAll = 128, kKeep = 32;
        std::vector<float> rw((size_t) kAll * kIn), rb(kAll);
        for (float& v : rw) v = 0.02f * nd(rng);
        for (float& v : rb) v = 0.1f * nd(rng);
        std::vector<int> rk;
        for (int e = 0; e < kAll; ++e)
            if (rng() % 4 == 0 && (int) rk.size() < kKeep) rk.push_back(e);
        for (int e = 0; (int) rk.size() < kKeep; ++e)
            if (!std::binary_search(rk.begin(), rk.end(), e)) { rk.push_back(e); std::sort(rk.begin(), rk.end()); }
        std::vector<float> masked = rb, krw((size_t) kKeep * kIn), krb(kKeep);
        for (int e = 0; e < kAll; ++e)
            if (!std::binary_search(rk.begin(), rk.end(), e)) masked[(size_t) e] = -INFINITY;
        for (int i = 0; i < kKeep; ++i) {
            std::memcpy(&krw[(size_t) i * kIn], &rw[(size_t) rk[(size_t) i] * kIn], (size_t) kIn * 4);
            krb[(size_t) i] = rb[(size_t) rk[(size_t) i]];
        }
        Dev<float> drw(rw), dmb(masked), dkrw(krw), dkrb(krb), l1((size_t) kT * kAll), l2((size_t) kT * kKeep), w1(kT * kK), w2(kT * kK);
        Dev<int32_t> i1(kT * kK), i2(kT * kK);
        RouterArgs ra;
        ra.n_embd = kIn;
        ra.k = kK;
        ra.n_expert = kAll;
        router(dx.p, drw.p, dmb.p, kT, ra, l1.p, i1.p, w1.p, nullptr);
        ra.n_expert = kKeep;
        router(dx.p, dkrw.p, dkrb.p, kT, ra, l2.p, i2.p, w2.p, nullptr);
        check(cudaDeviceSynchronize(), "kept router");
        const auto gi1 = i1.down(), gi2 = i2.down();
        bool same_r = w1.down() == w2.down();
        for (int p = 0; p < kT * kK; ++p) same_r = same_r && gi2[p] >= 0 && gi2[p] < kKeep && gi1[p] == rk[(size_t) gi2[p]];
        std::printf("%-22s experts %s, router %s\n", "kept against whole", same_e ? "bit for bit" : "DIFFER", same_r ? "bit for bit" : "DIFFER");
        if (!same_e || !same_r) ++failures;
    }

    // ---- microbench: one layer's expert work for a 5-row block (bytes are what the kernels touch)
    float ms = 0;
    cudaEventRecord(e0);
    constexpr int kReps = 20;
    for (int i = 0; i < kReps; ++i) {
        dspark_moe_gate_up(dg.p, du.p, kIn, kFF, dx.p, did.p, kT, kK, 10.f, dh.p, nullptr);
        dspark_moe_down(dd.p, kFF, kOut, dh.p, did.p, kT, kK, dout.p, nullptr);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    const double bytes = (double) kT * kK * (2.0 * kFF * mxfp4_row_bytes(kIn) + (double) kOut * mxfp4_row_bytes(kFF));
    std::printf("bench: 5 rows x top-3 expert layer %.1f us (%.0f GB/s of pair-reads)\n", 1000.0 * ms / kReps,
                bytes / (ms / kReps * 1e-3) / 1e9);

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
