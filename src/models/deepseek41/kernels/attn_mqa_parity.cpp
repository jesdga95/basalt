// Sparse MQA parity: sparse_mqa against a double-precision CPU reference at the real geometry (64 heads x 512, up to
// 128 window rows and 512 selected rows, with the sink and the inverse RoPE), the packed cache rows bit for bit against
// fp32 rows of their values, and the token's independence of the call's width. Prints microbench lines.
#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

using namespace basalt::models::deepseek41;
using dstest::Dev;
using dstest::Round;

namespace {

struct Case {
    int T, n_sel, win_cap, n_comp;
    LatentType wt, ct;
    bool masked_ids;
    std::vector<int> win_n;
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

// The device rows of `type` for values v[rows][512]; v becomes the values the rows hold.
std::vector<uint8_t> device_rows(std::vector<float>& v, LatentType type) {
    const size_t n = v.size() / 512;
    std::vector<uint8_t> b;
    if (type == LatentType::F32) {
        b.resize(v.size() * 4);
        std::memcpy(b.data(), v.data(), b.size());
    } else if (type == LatentType::F16) {
        b.resize(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            const __half h = __float2half(v[i]);
            v[i] = __half2float(h);
            std::memcpy(&b[2 * i], &h, 2);
        }
    } else {
        const Round k = type == LatentType::Fp8 ? Round::Fp8Pow2 : Round::Fp4E4m3;
        const int rb = type == LatentType::Fp8 ? kWinRowBytes : kCompRowBytes;
        for (size_t r = 0; r < n; ++r) {
            const auto row = dstest::ref_pack(&v[r * 512], 512, k, rb);
            const auto x = dstest::ref_unpack(row.data(), 512, k);
            std::copy(x.begin(), x.end(), v.begin() + (ptrdiff_t) (r * 512));
            b.insert(b.end(), row.begin(), row.end());
        }
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
    const auto wrows = device_rows(win, c.wt), crows = device_rows(comp, c.ct);
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

    Dev<uint8_t> wdev(wrows), cdev(crows), wf((size_t) win.size() * 4), cf((size_t) comp.size() * 4);
    dstest::check(cudaMemcpy(wf.p, win.data(), win.size() * 4, cudaMemcpyHostToDevice), "up");
    dstest::check(cudaMemcpy(cf.p, comp.data(), comp.size() * 4, cudaMemcpyHostToDevice), "up");
    Dev<float> dq(q), dsink(sinks), dout((size_t) c.T * 64 * 512), dws(sparse_mqa_workspace_floats(c.T, c.n_sel));
    Dev<int32_t> dsel(sel), dlim(limit), dws0(wstart), dwn(wn), dpos(pos);

    MqaArgs a;
    a.n_tokens = c.T; a.q = dq.p; a.win = wdev.p; a.win_type = c.wt; a.win_cap = c.win_cap;
    a.win_start = dws0.p; a.win_n = dwn.p; a.comp = cdev.p; a.comp_type = c.ct;
    a.sel = c.n_sel ? dsel.p : nullptr; a.n_sel = c.n_sel; a.comp_limit = dlim.p; a.sinks = dsink.p;
    a.pos = dpos.p; a.rope = rope; a.out = dout.p;
    cudaStream_t s;
    cudaStreamCreate(&s);
    sparse_mqa(a, dws.p, s);
    dstest::check(cudaStreamSynchronize(s), "run");
    const auto got = dout.down();
    const auto e = dstest::compare(got, ref, 1e-2);
    // the same values as fp32 rows give the same bits
    bool same_f32 = true;
    if (c.wt != LatentType::F32 || c.ct != LatentType::F32) {
        MqaArgs b = a;
        b.win = wf.p; b.win_type = LatentType::F32; b.comp = cf.p; b.comp_type = LatentType::F32;
        Dev<float> o2((size_t) c.T * 64 * 512);
        b.out = o2.p;
        sparse_mqa(b, dws.p, s);
        dstest::check(cudaStreamSynchronize(s), "f32 rows");
        const auto v2 = o2.down();
        same_f32 = std::memcmp(v2.data(), got.data(), got.size() * 4) == 0;
    }

    // token t alone must give the same bits as inside the batch
    bool same = true;
    for (int t = 0; t < c.T && same; ++t) {
        MqaArgs b = a;
        b.n_tokens = 1; b.q = dq.p + (size_t) t * 64 * 512; b.win_start = dws0.p + t; b.win_n = dwn.p + t;
        b.sel = c.n_sel ? dsel.p + (size_t) t * c.n_sel : nullptr; b.comp_limit = dlim.p + t; b.pos = dpos.p + t;
        Dev<float> o1(64 * 512);
        b.out = o1.p;
        sparse_mqa(b, dws.p, s);
        dstest::check(cudaStreamSynchronize(s), "run1");
        const auto v = o1.down();
        same = std::memcmp(v.data(), got.data() + (size_t) t * 64 * 512, v.size() * sizeof(float)) == 0;
    }
    if (bench) {
        const double us = dstest::bench_us([&] { sparse_mqa(a, dws.p, nullptr); });
        const auto row_bytes = [](LatentType t) {
            return t == LatentType::F32 ? 2048.0 : t == LatentType::F16 ? 1024.0 : t == LatentType::Fp8 ? kWinRowBytes : kCompRowBytes;
        };
        dstest::bench_line(name, us, ((double) c.win_n[0] * row_bytes(c.wt) + (double) c.n_sel * row_bytes(c.ct)) * c.T +
                                         (double) c.T * 64 * 512 * 8);
    }
    uint64_t hash = 1469598103934665603ull;   // FNV-1a of the output bits: equal across kernel versions = bitwise
    for (float f : got) { uint32_t u; std::memcpy(&u, &f, 4); hash = (hash ^ u) * 1099511628211ull; }
    std::printf("%-26s T=%d sel=%3d win=%3d  max_abs %.2e  rms_rel %.2e  token-independent %s  = fp32 rows %s  bits %016llx\n",
                name, c.T, c.n_sel, c.win_n[0], e.max_abs, e.rms_rel, same ? "yes" : "NO", same_f32 ? "yes" : "NO",
                (unsigned long long) hash);
    cudaStreamDestroy(s);
    return {e.max_abs, e.rms_rel, same && same_f32};
}

}  // namespace

int main() {
    dstest::Verdict v;
    struct Row { const char* name; Case c; };
    std::vector<Row> rows = {
        {"full f32 T1", {1, 512, 130, 3000, LatentType::F32, LatentType::F32, true, {128}}},
        {"full f32 T4", {4, 512, 130, 3000, LatentType::F32, LatentType::F32, true, {128, 100, 128, 7}}},
        {"full fp8/fp4 T3", {3, 512, 136, 4096, LatentType::Fp8, LatentType::Fp4, true, {128, 128, 64}}},
        {"full fp8/fp4 T8", {8, 512, 136, 4096, LatentType::Fp8, LatentType::Fp4, true, {128, 128, 64, 1, 128, 100, 9, 128}}},
        {"window only fp8 T2", {2, 0, 136, 10, LatentType::Fp8, LatentType::Fp4, false, {128, 33}}},
        {"full fp16 T3", {3, 512, 136, 4096, LatentType::F16, LatentType::F16, true, {128, 128, 64}}},
        {"window only fp16 T2", {2, 0, 136, 10, LatentType::F16, LatentType::F16, false, {128, 33}}},
        {"drafter f16/f32 T4", {4, 4, 136, 300, LatentType::F16, LatentType::F32, false, {128, 128, 2, 77}}},
        {"short sel f32 T2", {2, 200, 128, 300, LatentType::F32, LatentType::F32, false, {128, 20}}},
        {"window only T4", {4, 0, 128, 10, LatentType::F32, LatentType::F32, false, {128, 5, 77, 1}}},
        {"nothing visible T2", {2, 64, 128, 100, LatentType::F32, LatentType::F32, false, {0, 3}}},
    };
    for (auto& r : rows) {
        if (std::string(r.name) == "nothing visible T2") {
            r.c.masked_ids = true;
        }
        const auto res = run(r.c, 100 + (uint32_t) (&r - &rows[0]), false, r.name);
        v.expect(res.rms_rel < 2e-6, r.name);
        v.expect(res.token_independent, "token independence, packed rows equal fp32 rows");
    }
    run(rows[0].c, 7, true, "mqa T1 sel512 win128 f32");
    run(rows[1].c, 8, true, "mqa T4 sel512 win128 f32");
    Case h = rows[2].c; h.T = 1; h.win_n = {128};
    run(h, 9, true, "mqa T1 sel512 win128 fp8/fp4");
    h.T = 4; h.win_n = {128, 128, 128, 128};
    run(h, 10, true, "mqa T4 sel512 win128 fp8/fp4");
    // RoPE: forward then inverse restores the rows; forward against the CPU rotation
    {
        const RopeTable rope = rope_table_compressed();
        const int n = 4 * 64;
        auto x = dstest::randn((size_t) n * 512, 5);
        std::vector<int32_t> pos = {0, 17, 4000, 70000};
        Dev<float> dx(x);
        Dev<int32_t> dp(pos);
        cudaStream_t s; cudaStreamCreate(&s);
        rope_rows(dx.p, n, 512, 64, dp.p, rope, false, s);
        cudaStreamSynchronize(s);
        const auto fw = dx.down();
        std::vector<double> ref(x.begin(), x.end());
        for (int r = 0; r < n; ++r)
            for (int i = 0; i < 32; ++i) {
                const double ang = (double) ((float) pos[r / 64] * rope.step[i]);
                const double c = std::cos(ang) * rope.mscale, sn = std::sin(ang) * rope.mscale;
                const double x0 = x[(size_t) r * 512 + 448 + 2 * i], x1 = x[(size_t) r * 512 + 449 + 2 * i];
                ref[(size_t) r * 512 + 448 + 2 * i] = x0 * c - x1 * sn;
                ref[(size_t) r * 512 + 449 + 2 * i] = x0 * sn + x1 * c;
            }
        const auto e = dstest::compare(fw, ref, 1e-3);
        rope_rows(dx.p, n, 512, 64, dp.p, rope, true, s);
        cudaStreamSynchronize(s);
        const auto back = dx.down();
        const auto e2 = dstest::compare(back, x, 1e-3);
        std::printf("rope fwd max_abs %.2e, fwd+inverse max_abs %.2e\n", e.max_abs, e2.max_abs);
        v.expect(e.max_abs < 1e-5 && e2.max_abs < 1e-5, "rope");
        const RopeTable w = rope_table_window();
        std::printf("rope steps: window[0]=%g [31]=%g, compressed[0]=%g [31]=%g mscale %g\n", w.step[0], w.step[31],
                    rope.step[0], rope.step[31], rope.mscale);
    }
    std::printf(v.failures ? "FAILED\n" : "PASSED\n");
    return v.failures ? 1 : 0;
}
