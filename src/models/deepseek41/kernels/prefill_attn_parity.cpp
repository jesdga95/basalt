// Tensor-core sparse MQA parity: sparse_mqa_tc against a double-precision CPU reference and against the scalar
// sparse_mqa at the real geometry (64 heads x 512, up to 128 window rows and 512 selected rows, with the sink and the
// inverse RoPE), and the token's independence of the call's width. Prints microbench lines.
#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/prefill_attn.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

using namespace basalt::models::deepseek41;
using dstest::Dev;

namespace {

float half_round(float x) { return __half2float(__float2half(x)); }

struct Case {
    int T, n_sel, win_cap, n_comp;
    LatentType wt, ct;
    bool masked_ids;
    std::vector<int> win_n;
};

std::vector<double> reference(const Case& c, const std::vector<float>& q, const std::vector<float>& win,
                              const std::vector<float>& comp, int64_t comp_stride, const std::vector<int32_t>& sel,
                              const std::vector<int32_t>& limit, const std::vector<int32_t>& wstart,
                              const std::vector<float>& sinks, const std::vector<int32_t>& pos, const RopeTable& rope) {
    std::vector<double> out((size_t) c.T * 64 * 512);
    for (int t = 0; t < c.T; ++t) {
        std::vector<const float*> keys;
        for (int j = 0; j < c.win_n[t]; ++j) keys.push_back(&win[(size_t) ((wstart[t] + j) % c.win_cap) * 512]);
        for (int i = 0; i < c.n_sel; ++i) {
            const int id = sel[(size_t) t * c.n_sel + i];
            if (id >= 0 && id < limit[t]) keys.push_back(&comp[(size_t) id * comp_stride]);
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

template <class T>
std::vector<T> convert(const std::vector<float>& v) {
    std::vector<T> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = (T) v[i];
    return r;
}
template <>
std::vector<__half> convert<__half>(const std::vector<float>& v) {
    std::vector<__half> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = __float2half(v[i]);
    return r;
}

struct Result {
    double max_abs, rms_rel;
    bool token_independent;
};

Result run(const Case& c, uint32_t seed, bool bench, const char* name) {
    const int64_t comp_stride = 640;   // row = 512 latent + 128 indexer key, as the compressed cache stores it
    auto q = dstest::randn((size_t) c.T * 64 * 512, seed, 0.6f);
    auto win = dstest::randn((size_t) c.win_cap * 512, seed + 1, 1.0f);
    auto comp = dstest::randn((size_t) c.n_comp * comp_stride, seed + 2, 1.0f);
    auto sinks = dstest::randn(64, seed + 3, 1.0f);
    if (c.wt == LatentType::F16) for (auto& x : win) x = half_round(x);
    if (c.ct == LatentType::F16) for (auto& x : comp) x = half_round(x);
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
    const auto ref = reference(c, q, win, comp, comp_stride, sel, limit, wstart, sinks, pos, rope);

    void* wdev = nullptr; void* cdev = nullptr;
    size_t wb = win.size() * (c.wt == LatentType::F16 ? 2 : 4), cb = comp.size() * (c.ct == LatentType::F16 ? 2 : 4);
    dstest::check(cudaMalloc(&wdev, wb), "m");
    dstest::check(cudaMalloc(&cdev, cb), "m");
    if (c.wt == LatentType::F16) { auto h = convert<__half>(win); cudaMemcpy(wdev, h.data(), wb, cudaMemcpyHostToDevice); }
    else cudaMemcpy(wdev, win.data(), wb, cudaMemcpyHostToDevice);
    if (c.ct == LatentType::F16) { auto h = convert<__half>(comp); cudaMemcpy(cdev, h.data(), cb, cudaMemcpyHostToDevice); }
    else cudaMemcpy(cdev, comp.data(), cb, cudaMemcpyHostToDevice);
    Dev<float> dq(q), dsink(sinks), dout((size_t) c.T * 64 * 512), dws(sparse_mqa_workspace_floats(c.T, c.n_sel));
    Dev<int32_t> dsel(sel), dlim(limit), dws0(wstart), dwn(wn), dpos(pos);

    MqaArgs a;
    a.n_tokens = c.T; a.q = dq.p; a.win = wdev; a.win_type = c.wt; a.win_cap = c.win_cap;
    a.win_start = dws0.p; a.win_n = dwn.p; a.comp = cdev; a.comp_type = c.ct; a.comp_stride = comp_stride;
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
        const double elem = (c.wt == LatentType::F16 ? 2.0 : 4.0);
        const int keys = c.win_n[0] + c.n_sel;
        dstest::bench_line(name, us, (double) keys * 512 * elem * c.T + (double) c.T * 64 * 512 * 8);
    }
    std::printf("%-26s T=%d sel=%3d win=%3d  max_abs %.2e  rms_rel %.2e (vs scalar %.2e)  token-independent %s\n", name, c.T,
                c.n_sel, c.win_n[0], e.max_abs, e.rms_rel, es.rms_rel, same ? "yes" : "NO");
    cudaFree(wdev); cudaFree(cdev); cudaStreamDestroy(s);
    return {e.max_abs, e.rms_rel, same};
}

}  // namespace

int main() {
    dstest::Verdict v;
    struct Row { const char* name; Case c; };
    const LatentType F = LatentType::F16;
    std::vector<Row> rows = {
        {"full T1", {1, 512, 130, 3000, F, F, true, {128}}},
        {"full T4", {4, 512, 132, 3000, F, F, true, {128, 100, 128, 7}}},
        {"full T3 masked", {3, 512, 135, 4096, F, F, true, {128, 128, 64}}},
        {"short sel T2", {2, 200, 128, 300, F, F, false, {128, 20}}},
        {"window only T4", {4, 0, 128, 10, F, F, false, {128, 5, 77, 1}}},
        {"nothing visible T2", {2, 64, 128, 100, F, F, true, {0, 3}}},
        {"f32 falls back T2", {2, 128, 128, 300, LatentType::F32, LatentType::F32, false, {128, 20}}},
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
