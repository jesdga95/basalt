// Parity test of the batched prompt index side (gemm_bf16, kv_window_stage, compress_batch, indexer_select_batch)
// against the decode-width kernels at the real geometry, plus microbench lines.
#include "basalt/models/deepseek41/prefill_index.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <set>

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

constexpr int D = 5120;

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float from_f16(uint16_t h) { return __half2float(*reinterpret_cast<const __half*>(&h)); }

std::vector<uint16_t> rand_bf16(size_t n, uint32_t seed, float sigma) {
    const auto f = randn(n, seed, sigma);
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = to_bf16(f[i]);
    return v;
}

std::vector<float> make_table() {
    RopeParams p{160000.f, 1.f / 16.f, 1.f, 32.f, 1.f, 65536};
    std::vector<float> t(kRopePairs);
    rope_freq_table(p, t.data());
    return t;
}

std::vector<float> halves(const std::vector<uint16_t>& h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) f[i] = from_f16(h[i]);
    return f;
}

void test_gemm(Verdict& v) {
    struct Shape { int in, out, T; };
    for (const Shape s : {Shape{D, 512, 1}, Shape{D, 512, 70}, Shape{1280, 4096, 133}, Shape{D, 32, 64}, Shape{512, 128, 1000}}) {
        const auto w = rand_bf16((size_t) s.out * s.in, 11, 0.03f);
        const auto x = randn((size_t) s.T * s.in, 12, 1.f);
        Dev<uint16_t> dw(w);
        Dev<float> dx(x), dy((size_t) s.T * s.out);
        gemm_bf16(dw.p, dx.p, s.T, s.in, s.out, dy.p, nullptr);
        check(cudaDeviceSynchronize(), "gemm");
        const auto got = dy.down();
        std::vector<double> ref((size_t) s.T * s.out);
        for (int t = 0; t < s.T; ++t)
            for (int o = 0; o < s.out; ++o) {
                double a = 0;
                for (int i = 0; i < s.in; ++i) a += (double) from_bf16(w[(size_t) o * s.in + i]) * x[(size_t) t * s.in + i];
                ref[(size_t) t * s.out + o] = a;
            }
        const Err e = compare(got, ref);
        std::printf("gemm_bf16 in=%d out=%d T=%d: rms_rel %.2e\n", s.in, s.out, s.T, e.rms_rel);
        v.expect(e.rms_rel < 1e-5, "gemm_bf16 vs double reference");
    }
}

void test_window(Verdict& v) {
    const int N = 300, P0 = 40;
    const auto freq = make_table();
    const auto raw = randn((size_t) N * kLatent, 31, 1.f);
    const auto nw = randn(kLatent, 32, 1.f, 1.f);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = P0 + i;
    Dev<float> draw(raw), dnw(nw), dfreq(freq);
    Dev<int32_t> dpos(pos);
    std::vector<uint16_t> zero((size_t) kWindowSlots * kLatent, 0);
    Dev<uint16_t> ring_a(zero), ring_b(zero);
    for (int off = 0; off < N; off += 4)
        kv_window(draw.p + (size_t) off * kLatent, dnw.p, dpos.p + off, std::min(4, N - off), 1e-20f, dfreq.p, nullptr, ring_a.p, nullptr);
    const int R = 1152;
    Dev<uint16_t> stage((size_t) R * kLatent);
    kv_window_stage(draw.p, dnw.p, dpos.p, N, 1e-20f, dfreq.p, stage.p, R, ring_b.p, std::max(0, N - kWindowSlots), nullptr);
    check(cudaDeviceSynchronize(), "window");
    v.expect(ring_a.down() == ring_b.down(), "kv_window_stage leaves the decode ring as the token-by-token run does");
    // staged copy of the rows before a later sub-chunk
    Dev<uint16_t> stage2((size_t) R * kLatent);
    const int64_t p_lo = P0 + N - 127;
    window_stage_copy(ring_b.p, stage2.p, R, p_lo, 127, nullptr);
    check(cudaDeviceSynchronize(), "copy");
    const auto sa = stage.down(), sb = stage2.down();
    bool same = true;
    for (int64_t p = p_lo; p < P0 + N; ++p)
        same = same && std::equal(sa.begin() + (p % R) * kLatent, sa.begin() + (p % R + 1) * kLatent, sb.begin() + (p % R) * kLatent);
    v.expect(same, "window_stage_copy rows equal the staged rows");
}

void test_compress(Verdict& v) {
    const int N = 41, P0 = 5;
    const auto freq = make_table();
    Dev<uint16_t> dwkv(rand_bf16((size_t) kLatent * D, 21, 0.02f)), dwg(rand_bf16((size_t) kLatent * D, 22, 0.02f));
    Dev<uint16_t> dwik(rand_bf16((size_t) kIdxDim * kLatent, 23, 0.05f));
    Dev<float> dnw(randn(kLatent, 24, 1.f, 1.f)), dnk(randn(kIdxDim, 25, 1.f, 1.f)), dfreq(freq), dx(randn((size_t) N * D, 26, 1.f));
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = P0 + i;
    Dev<int32_t> dpos(pos);
    Dev<char> ws(compress_workspace_bytes()), wsb(compress_batch_workspace_bytes(N));
    for (int ratio = 1; ratio <= 2; ++ratio) {
        const int cap = 48;
        struct St {
            Dev<float> skv, ssc;
            Dev<uint16_t> ckv, cik;
            St(int r, int c) : skv(r * kLatent), ssc(r * kLatent), ckv((size_t) c * kLatent), cik((size_t) c * kIdxDim) {
                skv.up(randn(r * kLatent, 70, 1.f));
                ssc.up(randn(r * kLatent, 71, 1.f));
                ckv.up(std::vector<uint16_t>(ckv.n, 0));
                cik.up(std::vector<uint16_t>(cik.n, 0));
            }
        } a(ratio, cap), b(ratio, cap);
        auto args = [&](St& s) {
            CompressArgs c;
            c.ratio = ratio; c.w_kv = dwkv.p; c.w_gate = ratio > 1 ? dwg.p : nullptr; c.norm_w = dnw.p;
            c.w_idx_k = dwik.p; c.idx_k_norm_w = dnk.p; c.rope_freq = dfreq.p;
            c.state_kv = s.skv.p; c.state_score = s.ssc.p; c.comp_kv = s.ckv.p; c.comp_ik = s.cik.p;
            return c;
        };
        const CompressArgs ca = args(a), cb = args(b);
        for (int off = 0; off < N; off += 4) compress(ca, dx.p + (size_t) off * D, dpos.p + off, std::min(4, N - off), ws.p, nullptr);
        const int split = 17;   // two calls: the second starts mid-group for odd splits
        compress_batch(cb, dx.p, dpos.p, split, wsb.p, nullptr);
        compress_batch(cb, dx.p + (size_t) split * D, dpos.p + split, N - split, wsb.p, nullptr);
        check(cudaDeviceSynchronize(), "compress");
        const Err e1 = compare(halves(b.ckv.down()), halves(a.ckv.down()), 1e-2);
        const Err e2 = compare(halves(b.cik.down()), halves(a.cik.down()), 1e-2);
        const Err e3 = compare(b.skv.down(), a.skv.down()), e4 = compare(b.ssc.down(), a.ssc.down());
        std::printf("compress_batch ratio=%d: comp_kv rms_rel %.2e, comp_ik %.2e, state %.2e / %.2e\n", ratio, e1.rms_rel,
                    e2.rms_rel, e3.rms_rel, e4.rms_rel);
        v.expect(e1.rms_rel < 1e-3 && e2.rms_rel < 1e-3, "compress_batch rows vs decode-width compress");
        v.expect(e3.rms_rel < 3e-5 && e4.rms_rel < 3e-5, "compress_batch carried state vs decode-width compress");
    }
}

void test_indexer(Verdict& v) {
    const int n_comp = 900, N = 48;
    const auto freq = make_table();
    Dev<uint16_t> wq(rand_bf16((size_t) kIdxHeads * kIdxDim * 1280, 41, 0.03f)), wp(rand_bf16((size_t) kIdxHeads * D, 42, 0.03f));
    const auto ikf = randn((size_t) n_comp * kIdxDim, 43, 1.f);
    std::vector<uint16_t> ik(ikf.size());
    for (size_t i = 0; i < ik.size(); ++i) { const __half h = __float2half(ikf[i]); ik[i] = *reinterpret_cast<const uint16_t*>(&h); }
    Dev<uint16_t> dik(ik);
    Dev<float> qr(randn((size_t) N * 1280, 44, 1.f)), x(randn((size_t) N * D, 45, 1.f)), dfreq(freq);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = (n_comp - N + i) * 2 + 1;   // ratio 2: row counts n_comp-N+i+1
    Dev<int32_t> dpos(pos);
    for (IndexMode mode : {IndexMode::Full, IndexMode::Level1, IndexMode::Level2}) {
        Dev<int32_t> ia((size_t) N * kTopK), na(N), ib((size_t) N * kTopK), nb(N), ca((size_t) N * kMaxCandBlocks), cna(N),
            cb((size_t) N * kMaxCandBlocks), cnb(N);
        if (mode == IndexMode::Level2) {   // candidates from a Level1 pass of the decode-width kernel
            Dev<char> w1(indexer_workspace_bytes(n_comp));
            IndexerArgs a;
            a.mode = IndexMode::Level1; a.ratio = 2; a.n_comp = n_comp; a.w_q = wq.p; a.w_proj = wp.p; a.rope_freq = dfreq.p;
            a.comp_ik = dik.p; a.cand_ids = ca.p; a.cand_n = cna.p; a.out_ids = ia.p; a.out_n = na.p; a.cand_blocks = 40;
            for (int off = 0; off < N; off += 4) {
                IndexerArgs g = a;
                g.cand_ids += (size_t) off * kMaxCandBlocks; g.cand_n += off; g.out_ids += (size_t) off * kTopK; g.out_n += off;
                indexer_select(g, qr.p + (size_t) off * 1280, x.p + (size_t) off * D, dpos.p + off, 4, w1.p, nullptr);
            }
            cb.up(ca.down());
            cnb.up(cna.down());
        }
        Dev<char> wa(indexer_workspace_bytes(n_comp)), wb(indexer_batch_workspace_bytes(N, n_comp));
        IndexerArgs a;
        a.mode = mode; a.ratio = 2; a.n_comp = n_comp; a.w_q = wq.p; a.w_proj = wp.p; a.rope_freq = dfreq.p; a.comp_ik = dik.p;
        a.cand_blocks = 40;
        IndexerArgs b = a;
        a.cand_ids = ca.p; a.cand_n = cna.p; a.out_ids = ia.p; a.out_n = na.p;
        b.cand_ids = cb.p; b.cand_n = cnb.p; b.out_ids = ib.p; b.out_n = nb.p;
        for (int off = 0; off < N; off += 4) {
            IndexerArgs g = a;
            g.cand_ids += (size_t) off * kMaxCandBlocks; g.cand_n += off; g.out_ids += (size_t) off * kTopK; g.out_n += off;
            indexer_select(g, qr.p + (size_t) off * 1280, x.p + (size_t) off * D, dpos.p + off, 4, wa.p, nullptr);
        }
        indexer_select_batch(b, qr.p, x.p, dpos.p, N, wb.p, nullptr);
        check(cudaDeviceSynchronize(), "indexer");
        const auto ga = ia.down(), gb = ib.down();
        const auto ra = na.down(), rb = nb.down();
        std::printf("  n[0..3]: decode %d %d %d %d, batch %d %d %d %d\n", ra[0], ra[1], ra[2], ra[3], rb[0], rb[1], rb[2], rb[3]);
        size_t diff = 0, total = 0;
        for (int t = 0; t < N; ++t) {
            std::set<int> s(ga.begin() + (size_t) t * kTopK, ga.begin() + (size_t) t * kTopK + ra[t]);
            for (int i = 0; i < rb[t]; ++i) diff += s.count(gb[(size_t) t * kTopK + i]) ? 0 : 1;
            diff += (size_t) std::abs(ra[t] - rb[t]);
            total += (size_t) ra[t];
        }
        std::printf("indexer_select_batch mode=%d: %zu of %zu selected ids differ\n", (int) mode, diff, total);
        v.expect(diff * 200 <= total, "indexer_select_batch ids vs decode-width indexer");
    }
}

void bench() {
    const int T = 1024, n_comp = 3000;
    Dev<uint16_t> wq(rand_bf16((size_t) kIdxHeads * kIdxDim * 1280, 51, 0.03f)), wkv(rand_bf16((size_t) kLatent * D, 52, 0.02f));
    Dev<float> qr(randn((size_t) T * 1280, 53, 1.f)), x(randn((size_t) T * D, 54, 1.f)), y((size_t) T * kIdxHeads * kIdxDim);
    bench_stream_us([&](void* s) { gemm_bf16(wq.p, qr.p, T, 1280, kIdxHeads * kIdxDim, y.p, s); }, 20);
    bench_line("gemm_bf16 1280->4096 T=1024",
               bench_stream_us([&](void* s) { gemm_bf16(wq.p, qr.p, T, 1280, kIdxHeads * kIdxDim, y.p, s); }, 50),
               (double) T * 1280 * 4 + 1280.0 * 4096 * 2);
    bench_line("gemm_bf16 5120->512 T=1024",
               bench_stream_us([&](void* s) { gemm_bf16(wkv.p, x.p, T, D, kLatent, y.p, s); }, 50),
               (double) T * D * 4 + 512.0 * D * 2);
    Dev<uint16_t> dik(std::vector<uint16_t>((size_t) n_comp * kIdxDim, 0x3c00)), wp(rand_bf16((size_t) kIdxHeads * D, 55, 0.03f));
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = 2 * n_comp - T + i;
    Dev<int32_t> dpos(pos), ids((size_t) T * kTopK), nn(T), cand((size_t) T * kMaxCandBlocks), cn(T);
    Dev<float> dfreq(make_table());
    Dev<char> ws(indexer_batch_workspace_bytes(T, n_comp));
    IndexerArgs a;
    a.mode = IndexMode::Level1; a.ratio = 2; a.n_comp = n_comp; a.w_q = wq.p; a.w_proj = wp.p; a.rope_freq = dfreq.p;
    a.comp_ik = dik.p; a.cand_ids = cand.p; a.cand_n = cn.p; a.out_ids = ids.p; a.out_n = nn.p;
    bench_line("indexer_select_batch Level1 T=1024",
               bench_stream_us([&](void* s) { indexer_select_batch(a, qr.p, x.p, dpos.p, T, ws.p, s); }, 10),
               (double) n_comp * kIdxDim * 2);
}

}  // namespace

int main() {
    Verdict v;
    test_gemm(v);
    test_window(v);
    test_compress(v);
    test_indexer(v);
    bench();
    std::printf(v.failures ? "FAILED\n" : "PASS\n");
    return v.failures ? 1 : 0;
}
