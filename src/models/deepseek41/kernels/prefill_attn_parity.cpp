// Tensor-core sparse MQA parity: sparse_mqa_tc over the packed cache rows against a double-precision CPU reference of
// their values and against the scalar sparse_mqa at the real geometry (64 heads x 512, up to 128 window rows and 512
// selected rows, with the sink and the inverse RoPE), and the token's independence of the call's width. Prints
// microbench lines.
#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "basalt/models/deepseek41/prefill_attn.hpp"
#include "test_util.hpp"

using namespace basalt::models::deepseek41;
using dstest::Dev;
using dstest::Round;

namespace {

struct Case {
    int T, n_sel, win_cap, n_comp;
    bool masked_ids;
    std::vector<int> win_n;
    bool fp16 = false;   ///< the caches' fp16 rows instead of the packed ones
};

std::vector<double> reference(const Case& c, const std::vector<float>& q, const std::vector<float>& win,
                              const std::vector<float>& comp, const std::vector<int32_t>& sel,
                              const std::vector<int32_t>& limit, const std::vector<int32_t>& wstart,
                              const std::vector<float>& sinks, const std::vector<int32_t>& pos, const RopeTable& rope) {
    std::vector<double> out((size_t) c.T * 64 * 512);
    for (int t = 0; t < c.T; ++t) {
        std::vector<const float*> keys;
        for (int j = 0; j < c.win_n[t]; ++j) keys.push_back(&win[(size_t) ((wstart[t] + j) % c.win_cap) * 512]);
        for (int i = 0; i < c.n_sel; ++i) {
            const int id = sel[(size_t) t * c.n_sel + i];
            if (id >= 0 && id < limit[t]) keys.push_back(&comp[(size_t) id * 512]);
        }
        for (int h = 0; h < 64; ++h) {
            const float* qh = &q[((size_t) t * 64 + h) * 512];
            std::vector<double> s(keys.size());
            double m = sinks[h];
            for (size_t k = 0; k < keys.size(); ++k) {
                double d = 0;
                for (int e = 0; e < 512; ++e) d += (double) qh[e] * keys[k][e];
                s[k] = d * 0.044194174;
                m = std::max(m, s[k]);
            }
            double den = std::exp(sinks[h] - m);
            std::vector<double> o(512, 0.0);
            for (size_t k = 0; k < keys.size(); ++k) {
                const double w = std::exp(s[k] - m);
                den += w;
                for (int e = 0; e < 512; ++e) o[e] += w * keys[k][e];
            }
            for (int e = 0; e < 512; ++e) o[e] /= den;
            for (int i = 0; i < 32; ++i) {   // inverse RoPE
                const double ang = (double) ((float) pos[t] * rope.step[i]);
                const double cs = std::cos(ang) * rope.mscale, sn = -std::sin(ang) * rope.mscale;
                const double x0 = o[448 + 2 * i], x1 = o[449 + 2 * i];
                o[448 + 2 * i] = x0 * cs - x1 * sn;
                o[449 + 2 * i] = x0 * sn + x1 * cs;
            }
            std::copy(o.begin(), o.end(), out.begin() + ((size_t) t * 64 + h) * 512);
        }
    }
    return out;
}

// The cache rows (window: Fp8, compressed: Fp4; or fp16 both) of values v[rows][512]; v becomes the values the rows hold.
std::vector<uint8_t> packed(std::vector<float>& v, bool window, bool fp16) {
    if (fp16) {
        std::vector<uint8_t> b(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            const __half h = __float2half(v[i]);
            v[i] = __half2float(h);
            std::memcpy(&b[2 * i], &h, 2);
        }
        return b;
    }
    const Round k = window ? Round::Fp8Pow2 : Round::Fp4E4m3;
    const int rb = window ? kWinRowBytes : kCompRowBytes;
    std::vector<uint8_t> b;
    for (size_t r = 0; r < v.size() / 512; ++r) {
        const auto row = dstest::ref_pack(&v[r * 512], 512, k, rb);
        const auto x = dstest::ref_unpack(row.data(), 512, k);
        std::copy(x.begin(), x.end(), v.begin() + (ptrdiff_t) (r * 512));
        b.insert(b.end(), row.begin(), row.end());
    }
    return b;
}

struct Result {
    double max_abs, rms_rel;
    bool token_independent;
};

Result run(const Case& c, uint32_t seed, bool bench, const char* name) {
    auto q = dstest::randn((size_t) c.T * 64 * 512, seed, 0.6f);
    auto win = dstest::randn((size_t) c.win_cap * 512, seed + 1, 1.0f);
    auto comp = dstest::randn((size_t) c.n_comp * 512, seed + 2, 1.0f);
    auto sinks = dstest::randn(64, seed + 3, 1.0f);
    const auto wrows = packed(win, true, c.fp16), crows = packed(comp, false, c.fp16);
    std::mt19937 g(seed + 4);
    std::vector<int32_t> sel((size_t) c.T * std::max(c.n_sel, 1)), limit(c.T), wstart(c.T), pos(c.T), wn(c.win_n);
    for (int t = 0; t < c.T; ++t) {
        limit[t] = c.masked_ids ? c.n_comp * 3 / 4 : c.n_comp;
        wstart[t] = (int) (g() % 1000);
        pos[t] = 3000 + (int) (g() % 50000);
        for (int i = 0; i < c.n_sel; ++i) {
            int id = (int) (g() % c.n_comp);
            if (c.masked_ids && g() % 7 == 0) id = -1;
            sel[(size_t) t * c.n_sel + i] = id;
        }
    }
    const RopeTable rope = rope_table_compressed();
    const auto ref = reference(c, q, win, comp, sel, limit, wstart, sinks, pos, rope);

    Dev<uint8_t> wdev(wrows), cdev(crows);
    Dev<float> dq(q), dsink(sinks), dout((size_t) c.T * 64 * 512), dws(sparse_mqa_workspace_floats(c.T, c.n_sel));
    Dev<int32_t> dsel(sel), dlim(limit), dws0(wstart), dwn(wn), dpos(pos);

    MqaArgs a;
    a.n_tokens = c.T; a.q = dq.p; a.win = wdev.p; a.win_type = c.fp16 ? LatentType::F16 : LatentType::Fp8; a.win_cap = c.win_cap;
    a.win_start = dws0.p; a.win_n = dwn.p; a.comp = cdev.p; a.comp_type = c.fp16 ? LatentType::F16 : LatentType::Fp4;
    a.sel = c.n_sel ? dsel.p : nullptr; a.n_sel = c.n_sel; a.comp_limit = dlim.p; a.sinks = dsink.p;
    a.pos = dpos.p; a.rope = rope; a.out = dout.p;
    cudaStream_t s;
    cudaStreamCreate(&s);
    sparse_mqa_tc(a, dws.p, s);
    dstest::check(cudaStreamSynchronize(s), "run");
    const auto got = dout.down();
    const auto e = dstest::compare(got, ref, 1e-2);
    Dev<float> dscalar((size_t) c.T * 64 * 512);
    {
        MqaArgs b = a;
        b.out = dscalar.p;
        sparse_mqa(b, dws.p, s);
        dstest::check(cudaStreamSynchronize(s), "scalar");
    }
    const auto es = dstest::compare(got, dscalar.down(), 1e-2);

    // token t alone must give the same bits as inside the batch
    bool same = true;
    for (int t = 0; t < c.T && same; ++t) {
        MqaArgs b = a;
        b.n_tokens = 1; b.q = dq.p + (size_t) t * 64 * 512; b.win_start = dws0.p + t; b.win_n = dwn.p + t;
        b.sel = c.n_sel ? dsel.p + (size_t) t * c.n_sel : nullptr; b.comp_limit = dlim.p + t; b.pos = dpos.p + t;
        Dev<float> o1(64 * 512);
        b.out = o1.p;
        sparse_mqa_tc(b, dws.p, s);
        dstest::check(cudaStreamSynchronize(s), "run1");
        const auto v = o1.down();
        same = std::memcmp(v.data(), got.data() + (size_t) t * 64 * 512, v.size() * sizeof(float)) == 0;
    }
    if (bench) {
        const double us = dstest::bench_us([&] { sparse_mqa_tc(a, dws.p, nullptr); });
        const double us0 = dstest::bench_us([&] { sparse_mqa(a, dws.p, nullptr); });
        std::printf("scalar sparse_mqa %.1f us, tensor-core %.1f us\n", us0, us);
        dstest::bench_line(name, us, ((double) c.win_n[0] * kWinRowBytes + (double) c.n_sel * kCompRowBytes) * c.T +
                                         (double) c.T * 64 * 512 * 8);
    }
    std::printf("%-26s T=%d sel=%3d win=%3d  max_abs %.2e  rms_rel %.2e (vs scalar %.2e)  token-independent %s\n", name, c.T,
                c.n_sel, c.win_n[0], e.max_abs, e.rms_rel, es.rms_rel, same ? "yes" : "NO");
    cudaStreamDestroy(s);
    return {e.max_abs, e.rms_rel, same};
}

}  // namespace

int main() {
    dstest::Verdict v;
    struct Row { const char* name; Case c; };
    std::vector<Row> rows = {
        {"full T1", {1, 512, 130, 3000, true, {128}}},
        {"full T4", {4, 512, 132, 3000, true, {128, 100, 128, 7}}},
        {"full T3 masked", {3, 512, 135, 4096, true, {128, 128, 64}}},
        {"full fp16 T4", {4, 512, 132, 3000, true, {128, 100, 128, 7}, true}},
        {"window only fp16 T4", {4, 0, 128, 10, false, {128, 5, 77, 1}, true}},
        {"short sel T2", {2, 200, 128, 300, false, {128, 20}}},
        {"window only T4", {4, 0, 128, 10, false, {128, 5, 77, 1}}},
        {"nothing visible T2", {2, 64, 128, 100, true, {0, 3}}},
    };
    for (auto& r : rows) {
        const auto res = run(r.c, 100 + (uint32_t) (&r - &rows[0]), false, r.name);
        v.expect(res.rms_rel < 3e-3, r.name);
        v.expect(res.token_independent, "token independence");
    }
    run(rows[0].c, 7, true, "mqa_tc T1 sel512 win128");
    run(rows[1].c, 8, true, "mqa_tc T4 sel512 win128");
    std::printf(v.failures ? "FAILED\n" : "PASSED\n");
    return v.failures ? 1 : 0;
}
