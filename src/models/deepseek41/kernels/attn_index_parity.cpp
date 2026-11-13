// Parity test of the DeepSeek-V4.1 attention index side (window ring, compressor, indexer) at the real geometry:
// random inputs against a double-precision host reference with the cache roundings, the packed window rows against the
// CPU packing byte for byte, independence from the call width T and the position, and a microbench.
#include "basalt/models/deepseek41/attn_index.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>
#include <random>

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

std::vector<uint16_t> rand_bf16(size_t n, uint32_t seed, float sigma) {
    const auto f = randn(n, seed, sigma);
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = to_bf16(f[i]);
    return v;
}

std::vector<float> make_table(bool compressed) {
    RopeParams p;
    if (compressed) p = RopeParams{160000.f, 1.f / 16.f, 1.f, 32.f, 1.f, 65536};
    std::vector<float> t(kRopePairs);
    rope_freq_table(p, t.data());
    return t;
}

// y[r] = rms_norm(v) * w, then the last 64 dims rotated at pos
void norm_rope(std::vector<double>& v, const float* w, double eps, double pos, const std::vector<float>& freq) {
    double ss = 0;
    for (double x : v) ss += x * x;
    const double r = 1.0 / std::sqrt(ss / v.size() + eps);
    for (size_t i = 0; i < v.size(); ++i) v[i] *= r * w[i];
    const size_t off = v.size() - kRopeDim;
    for (int p = 0; p < kRopePairs; ++p) {
        const double th = pos * (double) freq[p], c = std::cos(th), s = std::sin(th);
        const double x0 = v[off + 2 * p], x1 = v[off + 2 * p + 1];
        v[off + 2 * p] = x0 * c - x1 * s;
        v[off + 2 * p + 1] = x0 * s + x1 * c;
    }
}

// The cache format the tests run on (both, in turn).
CacheFormat g_fmt = CacheFormat::Fp8;
bool packed() { return g_fmt == CacheFormat::Fp8; }

// The values of cache rows `ids` of d values (row r at r * rb): packed as `k`, or fp16.
std::vector<float> unpack_rows(const std::vector<uint8_t>& b, const std::vector<int>& ids, int d, int rb, Round k) {
    std::vector<float> v;
    for (int r : ids) {
        if (packed()) {
            const auto x = ref_unpack(&b[(size_t) r * rb], d, k);
            v.insert(v.end(), x.begin(), x.end());
        } else {
            for (int c = 0; c < d; ++c) {
                __half h;
                std::memcpy(&h, &b[(size_t) r * rb + 2 * (size_t) c], 2);
                v.push_back(__half2float(h));
            }
        }
    }
    return v;
}

// n index-key rows of random values in the cache format.
std::vector<uint8_t> key_rows(int n, uint32_t seed) {
    if (packed()) return packed_rows(n, kIdxDim, kIdxRowBytes, Round::Fp4Pow2, seed);
    const auto f = randn((size_t) n * kIdxDim, seed, 1.f);
    std::vector<uint8_t> b(f.size() * 2);
    for (size_t i = 0; i < f.size(); ++i) { const __half h = __float2half(f[i]); std::memcpy(&b[2 * i], &h, 2); }
    return b;
}
int WRB() { return win_row_bytes(g_fmt); }
int CRB() { return comp_row_bytes(g_fmt); }
int IRB() { return idx_row_bytes(g_fmt); }

// ---------------------------------------------------------------------------------------------------------------

void test_kv_window(Verdict& v) {
    for (int comp = 0; comp < 2; ++comp) {
        const auto freq = make_table(comp);
        const int N = 8;
        const auto raw = randn((size_t) N * kLatent, 11, 3.f);
        const auto w = randn(kLatent, 12, 1.f, 1.f);
        std::vector<int32_t> pos(N);
        for (int i = 0; i < N; ++i) pos[i] = 1000 + 3 * i;
        Dev<float> draw(raw), dw(w), dfreq(freq), dout((size_t) N * kLatent), dout1((size_t) N * kLatent);
        Dev<int32_t> dpos(pos);
        Dev<uint8_t> ring((size_t) kWindowSlots * WRB()), ring1((size_t) kWindowSlots * WRB());
        for (int b = 0; b < N; b += 4)
            kv_window(draw.p + (size_t) b * kLatent, dw.p, dpos.p + b, 4, 1e-20f, dfreq.p, dout.p + (size_t) b * kLatent, ring.p, g_fmt, nullptr);
        for (int i = 0; i < N; ++i)
            kv_window(draw.p + (size_t) i * kLatent, dw.p, dpos.p + i, 1, 1e-20f, dfreq.p, dout1.p + (size_t) i * kLatent, ring1.p, g_fmt, nullptr);
        check(cudaDeviceSynchronize(), "kv_window");
        const auto got = dout.down(), got1 = dout1.down();
        std::vector<double> ref;
        for (int i = 0; i < N; ++i) {
            std::vector<double> r(raw.begin() + (size_t) i * kLatent, raw.begin() + (size_t) (i + 1) * kLatent);
            norm_rope(r, w.data(), 1e-20, pos[i], freq);
            ref.insert(ref.end(), r.begin(), r.end());
        }
        const Err e = compare(got, ref);
        v.expect(e.rms_rel < 1e-5, "kv_window vs reference");
        v.expect(got == got1, "kv_window independent of T");
        const auto rg = ring.down(), rg1 = ring1.down();
        v.expect(rg == rg1, "kv_window ring independent of T");
        // each ring row is the CPU packing of the token's fp32 row; its values the reference rounding of the double row
        size_t bad = 0;
        std::vector<int> slots;
        for (int i = 0; i < N; ++i) {
            const int slot = pos[i] % kWindowSlots;
            slots.push_back(slot);
            if (!packed()) continue;
            const auto want = ref_pack(&got[(size_t) i * kLatent], kLatent, Round::Fp8Pow2, kWinRowBytes);
            bad += std::memcmp(want.data(), &rg[(size_t) slot * kWinRowBytes], kWinRowBytes) != 0;
        }
        auto rounded = ref;
        for (int i = 0; i < N && packed(); ++i) ref_round(&rounded[(size_t) i * kLatent], kLatent, Round::Fp8Pow2);
        const auto dec = unpack_rows(rg, slots, kLatent, WRB(), Round::Fp8Pow2);
        const Err er = compare(dec, rounded);
        const double fl = flips(dec, rounded);
        std::printf("[%s] kv_window comp=%d: rms_rel %.2e max_abs %.2e; ring rows differing from the CPU packing %zu, FP8 roundings "
                    "on the other side of the double reference's %.4f (rms_rel %.2e)\n", packed() ? "fp8" : "fp16", comp, e.rms_rel, e.max_abs, bad, fl, er.rms_rel);
        v.expect(bad == 0, "kv_window ring rows equal the CPU packing");
        v.expect(packed() ? er.rms_rel < 1e-2 && fl < 0.01 : er.rms_rel < 1e-3, "kv_window ring vs (rounded) reference");
    }
}

// ---------------------------------------------------------------------------------------------------------------

struct CompState {
    Dev<float> skv, ssc;
    Dev<uint8_t> ckv, cik;
    CompState(int ratio, int cap) : skv(ratio * kLatent), ssc(ratio * kLatent), ckv((size_t) cap * CRB()), cik((size_t) cap * IRB()) {
        std::vector<float> z(ratio * kLatent, 0.f);
        skv.up(z);
        ssc.up(z);
        std::vector<uint8_t> zz(ckv.n, 0), zi(cik.n, 0);
        ckv.up(zz);
        cik.up(zi);
    }
};

// ---------------------------------------------------------------------------------------------------------------

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// The bf16 projections and the compressor's rows, index keys and state at the real geometry hash to the bits of the
// kernels they replaced (frozen): their sums are bitwise-neutral.
void test_frozen(Verdict& v) {
    constexpr uint64_t kGemv = 0xbcf5bf9e8c0478ccull, kComp = 0xf66492ac94ff28d9ull;
    uint64_t hg = 1469598103934665603ull, hc = hg;
    const auto x = randn((size_t) kMaxT * D, 61, 1.f);
    Dev<float> dx(x);
    const int shapes[][2] = {{D, kLatent}, {D, kIdxHeads}, {1280, kIdxHeads * kIdxDim}};
    for (const auto& sh : shapes) {
        Dev<uint16_t> w(rand_bf16((size_t) sh[0] * sh[1], 62, 0.02f));
        Dev<float> y((size_t) kMaxT * sh[1]);
        for (int T : {1, 2, 3, 5, 8}) {
            gemv_bf16(w.p, dx.p, T, sh[0], sh[1], y.p, nullptr);
            check(cudaDeviceSynchronize(), "gemv_bf16");
            const auto got = y.down();
            hg = fnv(hg, got.data(), (size_t) T * sh[1] * sizeof(float));
        }
    }
    const auto freq = make_table(true);
    Dev<uint16_t> wkv(rand_bf16((size_t) kLatent * D, 63, 0.02f)), wg(rand_bf16((size_t) kLatent * D, 64, 0.02f)),
        wik(rand_bf16((size_t) kIdxDim * kLatent, 65, 0.05f));
    Dev<float> nw(randn(kLatent, 66, 1.f, 1.f)), nk(randn(kIdxDim, 67, 1.f, 1.f)), dfreq(freq);
    Dev<float> dxs(randn((size_t) 24 * D, 68, 1.f));
    std::vector<int32_t> pos(24);
    for (int i = 0; i < 24; ++i) pos[i] = 7 + i;
    Dev<int32_t> dpos(pos);
    Dev<char> ws(compress_workspace_bytes());
    for (CacheFormat f : {CacheFormat::Fp8, CacheFormat::Fp16}) {
        g_fmt = f;
        for (int ratio = 1; ratio <= 2; ++ratio) {
            CompState st(ratio, 32);
            CompressArgs a;
            a.ratio = ratio; a.w_kv = wkv.p; a.w_gate = ratio > 1 ? wg.p : nullptr; a.norm_w = nw.p;
            a.w_idx_k = wik.p; a.idx_k_norm_w = nk.p; a.rope_freq = dfreq.p;
            a.state_kv = st.skv.p; a.state_score = st.ssc.p; a.comp_kv = st.ckv.p; a.comp_ik = st.cik.p; a.format = f;
            const int width[] = {3, 1, 8, 3, 2, 5}, kept[] = {2, 1, 4, 3, 1, 5};
            int off = 0;
            for (int i = 0; off < 24; i = (i + 1) % 6) {
                const int T = std::min(width[i], 24 - off), q = std::min(kept[i], T);
                if (T == 1) {
                    compress(a, dxs.p + (size_t) off * D, dpos.p + off, 1, ws.p, nullptr);
                    off += 1;
                } else {
                    compress_window(a, dxs.p + (size_t) off * D, dpos.p + off, T, ws.p, nullptr);
                    compress_window_commit(a, dpos.p + off, q, ws.p, nullptr);
                    off += q;
                }
            }
            check(cudaDeviceSynchronize(), "compress frozen");
            const auto rk = st.ckv.down(), ri = st.cik.down();
            const auto sk = st.skv.down(), ss = st.ssc.down();
            hc = fnv(hc, rk.data(), rk.size());
            hc = fnv(hc, ri.data(), ri.size());
            hc = fnv(hc, sk.data(), sk.size() * sizeof(float));
            hc = fnv(hc, ss.data(), ss.size() * sizeof(float));
        }
    }
    g_fmt = CacheFormat::Fp8;
    std::printf("frozen: gemv_bf16 %016llx (want %016llx), compressor rows and keys %016llx (want %016llx)\n",
                (unsigned long long) hg, (unsigned long long) kGemv, (unsigned long long) hc, (unsigned long long) kComp);
    v.expect(hg == kGemv, "gemv_bf16 bits equal the frozen kernel");
    v.expect(hc == kComp, "compressor rows and index keys equal the frozen kernels");
}

void test_compress(Verdict& v) {
    const int N = 12, P0 = 5;
    const auto freq = make_table(true);
    const auto wkv = rand_bf16((size_t) kLatent * D, 21, 0.02f), wg = rand_bf16((size_t) kLatent * D, 22, 0.02f);
    const auto wik = rand_bf16((size_t) kIdxDim * kLatent, 23, 0.05f);
    const auto nw = randn(kLatent, 24, 1.f, 1.f), nk = randn(kIdxDim, 25, 1.f, 1.f);
    const auto x = randn((size_t) N * D, 26, 1.f);
    Dev<uint16_t> dwkv(wkv), dwg(wg), dwik(wik);
    Dev<float> dnw(nw), dnk(nk), dfreq(freq), dx(x);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = P0 + i;
    Dev<int32_t> dpos(pos);
    Dev<char> ws(compress_workspace_bytes());

    for (int ratio = 1; ratio <= 2; ++ratio) {
        const int cap = 32;
        std::vector<std::vector<int>> parts = {{4, 4, 4}, std::vector<int>(12, 1), {3, 2, 4, 1, 2}};
        std::vector<std::vector<uint8_t>> outs, outk;
        for (const auto& part : parts) {
            CompState st(ratio, cap);
            CompressArgs a;
            a.ratio = ratio; a.w_kv = dwkv.p; a.w_gate = ratio > 1 ? dwg.p : nullptr; a.norm_w = dnw.p;
            a.w_idx_k = dwik.p; a.idx_k_norm_w = dnk.p; a.rope_freq = dfreq.p;
            a.state_kv = st.skv.p; a.state_score = st.ssc.p; a.comp_kv = st.ckv.p; a.comp_ik = st.cik.p; a.format = g_fmt;
            int off = 0;
            for (int n : part) { compress(a, dx.p + (size_t) off * D, dpos.p + off, n, ws.p, nullptr); off += n; }
            check(cudaDeviceSynchronize(), "compress");
            outs.push_back(st.ckv.down());
            outk.push_back(st.cik.down());
        }
        v.expect(outs[0] == outs[1] && outs[0] == outs[2], "compress independent of call split (latents)");
        v.expect(outk[0] == outk[1] && outk[0] == outk[2], "compress independent of call split (index keys)");
        {
            // speculative windows of several widths: the rows of a rejected tail leave no trace
            CompState st(ratio, cap);
            CompressArgs a;
            a.ratio = ratio; a.w_kv = dwkv.p; a.w_gate = ratio > 1 ? dwg.p : nullptr; a.norm_w = dnw.p;
            a.w_idx_k = dwik.p; a.idx_k_norm_w = dnk.p; a.rope_freq = dfreq.p;
            a.state_kv = st.skv.p; a.state_score = st.ssc.p; a.comp_kv = st.ckv.p; a.comp_ik = st.cik.p; a.format = g_fmt;
            const int width[] = {4, 2, 5, 8, 3, 1}, kept[] = {2, 2, 1, 3, 3, 1};
            int off = 0;
            for (int i = 0; off < N; i = (i + 1) % 6) {
                const int T = std::min(width[i], N - off), q = std::min(kept[i], T);
                compress_window(a, dx.p + (size_t) off * D, dpos.p + off, T, ws.p, nullptr);
                compress_window_commit(a, dpos.p + off, q, ws.p, nullptr);
                off += q;
            }
            check(cudaDeviceSynchronize(), "compress_window");
            v.expect(st.ckv.down() == outs[0], "compress_window with rejected tails equals compress (latents)");
            v.expect(st.cik.down() == outk[0], "compress_window with rejected tails equals compress (index keys)");
        }

        // reference: state carried through positions P0..P0+N-1 starting from zero (rows before P0 are not part of it)
        std::vector<std::vector<double>> kvs(N, std::vector<double>(kLatent)), scs(N, std::vector<double>(kLatent));
        for (int t = 0; t < N; ++t)
            for (int o = 0; o < kLatent; ++o) {
                double a = 0, b = 0;
                for (int d = 0; d < D; ++d) {
                    a += (double) from_bf16(wkv[(size_t) o * D + d]) * x[(size_t) t * D + d];
                    if (ratio > 1) b += (double) from_bf16(wg[(size_t) o * D + d]) * x[(size_t) t * D + d];
                }
                kvs[t][o] = a; scs[t][o] = b;
            }
        std::vector<double> rows, iks;
        std::vector<int> row_ids;
        for (int t = 0; t < N; ++t) {
            const int p = pos[t];
            if ((p + 1) % ratio) continue;
            std::vector<double> comp(kLatent);
            for (int c = 0; c < kLatent; ++c) {
                double m = -1e300, den = 0, num = 0;
                const auto sc = [&](int j) { const int tt = t - (ratio - 1) + j; return tt < 0 ? 0.0 : scs[tt][c]; };
                const auto kk = [&](int j) { const int tt = t - (ratio - 1) + j; return tt < 0 ? 0.0 : kvs[tt][c]; };
                for (int j = 0; j < ratio; ++j) m = std::max(m, sc(j));
                for (int j = 0; j < ratio; ++j) { const double e = std::exp(sc(j) - m); den += e; num += kk(j) * e; }
                comp[c] = num / den;
            }
            std::vector<double> pre = comp;
            norm_rope(pre, nw.data(), 1e-20, p + 1 - ratio, freq);  // rotated copy for the cache
            // un-rotated normed latent for the index key
            std::vector<double> nrm = comp;
            { double ss = 0; for (double q : nrm) ss += q * q; const double r = 1 / std::sqrt(ss / kLatent + 1e-20); for (int c = 0; c < kLatent; ++c) nrm[c] *= r * nw[c]; }
            std::vector<double> ik(kIdxDim);
            for (int o = 0; o < kIdxDim; ++o) { double a = 0; for (int c = 0; c < kLatent; ++c) a += (double) from_bf16(wik[(size_t) o * kLatent + c]) * nrm[c]; ik[o] = a; }
            norm_rope(ik, nk.data(), 1e-20, p + 1 - ratio, freq);
            if (packed()) {
                ref_round(pre.data(), pre.size(), Round::Fp4E4m3);
                ref_round(ik.data(), ik.size(), Round::Fp4Pow2);
            }
            row_ids.push_back(p / ratio);
            rows.insert(rows.end(), pre.begin(), pre.end());
            iks.insert(iks.end(), ik.begin(), ik.end());
        }
        const auto g = unpack_rows(outs[0], row_ids, kLatent, CRB(), Round::Fp4E4m3);
        const auto gk = unpack_rows(outk[0], row_ids, kIdxDim, IRB(), Round::Fp4Pow2);
        // the first group of a call sequence that starts mid-group sees a zero partner row in both
        const Err e = compare(g, rows), ek = compare(gk, iks);
        const double fl = flips(g, rows), flk = flips(gk, iks);
        std::printf("compress ratio %d: %zu rows, latent rms_rel %.2e (FP4 roundings on the other side %.4f), index key "
                    "rms_rel %.2e (%.4f)\n", ratio, row_ids.size(), e.rms_rel, fl, ek.rms_rel, flk);
        v.expect(packed() ? e.rms_rel < 3e-2 && fl < 0.01 : e.rms_rel < 2e-3, "compress latents vs (rounded) reference");
        v.expect(packed() ? ek.rms_rel < 3e-2 && flk < 0.01 : ek.rms_rel < 3e-3, "compress index keys vs (rounded) reference");
    }
}

// ---------------------------------------------------------------------------------------------------------------

struct Ref {
    std::vector<double> scores;  // [T][n_comp]
};

Ref ref_scores(const std::vector<float>& qr, const std::vector<float>& x, const std::vector<uint16_t>& wq,
               const std::vector<uint16_t>& wp, const std::vector<uint8_t>& ikb, const std::vector<int32_t>& pos,
               int n_comp, int ratio, const std::vector<float>& freq, const std::vector<uint8_t>* ok, int bs, int n_blk) {
    const int T = (int) pos.size();
    Ref r;
    r.scores.assign((size_t) T * n_comp, -INFINITY);
    std::vector<int> all(n_comp);
    std::iota(all.begin(), all.end(), 0);
    const auto ik = unpack_rows(ikb, all, kIdxDim, IRB(), Round::Fp4Pow2);
    for (int t = 0; t < T; ++t) {
        std::vector<double> iq((size_t) kIdxHeads * kIdxDim), iw(kIdxHeads);
        for (int o = 0; o < kIdxHeads * kIdxDim; ++o) {
            double a = 0;
            for (int d = 0; d < 1280; ++d) a += (double) from_bf16(wq[(size_t) o * 1280 + d]) * qr[(size_t) t * 1280 + d];
            iq[o] = a;
        }
        for (int h = 0; h < kIdxHeads; ++h) {
            for (int p = 0; p < kRopePairs; ++p) {
                double* b = &iq[(size_t) h * kIdxDim + kIdxDim - kRopeDim + 2 * p];
                const double th = (double) pos[t] * freq[p], c = std::cos(th), s = std::sin(th);
                const double x0 = b[0], x1 = b[1];
                b[0] = x0 * c - x1 * s; b[1] = x0 * s + x1 * c;
            }
            if (packed()) ref_round(&iq[(size_t) h * kIdxDim], kIdxDim, Round::Fp4Pow2);
            double a = 0;
            for (int d = 0; d < D; ++d) a += (double) from_bf16(wp[(size_t) h * D + d]) * x[(size_t) t * D + d];
            iw[h] = a / std::sqrt((double) kIdxHeads * kIdxDim);
        }
        const int nv = (pos[t] + 1) / ratio;
        for (int j = 0; j < std::min(nv, n_comp); ++j) {
            if (ok && !(*ok)[(size_t) t * n_blk + j / bs]) continue;
            double s = 0;
            for (int h = 0; h < kIdxHeads; ++h) {
                double a = 0;
                for (int d = 0; d < kIdxDim; ++d) a += iq[(size_t) h * kIdxDim + d] * ik[(size_t) j * kIdxDim + d];
                s += std::max(a, 0.0) * iw[h];
            }
            r.scores[(size_t) t * n_comp + j] = s;
        }
    }
    return r;
}

// Compares a selected id set with the reference top-k of `s`; differing ids must be near-ties of the k-th score.
// Returns the overlap fraction.
double check_topk(Verdict& v, const char* what, const std::vector<double>& s, const int32_t* got, int got_n, int K, int n_valid) {
    std::vector<int> idx;
    for (size_t i = 0; i < s.size(); ++i) if (s[i] > -INFINITY) idx.push_back((int) i);
    const int want_n = std::min<int>(K, (int) idx.size());
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b]; });
    const double kth = want_n ? s[idx[want_n - 1]] : 0.0;
    std::vector<int> ref(idx.begin(), idx.begin() + want_n);
    std::sort(ref.begin(), ref.end());
    std::vector<int> g(got, got + got_n);
    bool sorted = std::is_sorted(g.begin(), g.end());
    int common = 0, bad = 0;
    for (int id : g) {
        if (std::binary_search(ref.begin(), ref.end(), id)) ++common;
        else if (!(std::fabs(s[id] - kth) <= 1e-4 * std::max(1.0, std::fabs(kth)))) ++bad;
    }
    const double ov = want_n ? (double) common / want_n : 1.0;
    std::printf("  %-28s n=%d (ref %d) overlap %.4f far-misses %d\n", what, got_n, want_n, ov, bad);
    v.expect(got_n == want_n && bad == 0 && sorted, what);
    (void) n_valid;
    return ov;
}

void test_indexer(Verdict& v) {
    const int ratio = 1, n_comp = 6000, T = 4, bs = 8, n_blk = (n_comp + bs - 1) / bs, cand_blocks = 300;
    const auto freq = make_table(true);
    const auto wq = rand_bf16((size_t) kIdxHeads * kIdxDim * 1280, 31, 0.03f), wp = rand_bf16((size_t) kIdxHeads * D, 32, 0.03f);
    const auto wq2 = rand_bf16((size_t) kIdxHeads * kIdxDim * 1280, 33, 0.03f), wp2 = rand_bf16((size_t) kIdxHeads * D, 34, 0.03f);
    const auto ik = key_rows(n_comp, 35);
    const auto qr = randn((size_t) T * 1280, 36, 1.f), x = randn((size_t) T * D, 37, 1.f);
    // positions: the last token sees n_comp rows; the others fewer (and one far below the top-k count)
    std::vector<int32_t> pos = {300, 4500, n_comp - 2, n_comp - 1};

    Dev<uint16_t> dwq(wq), dwp(wp), dwq2(wq2), dwp2(wp2);
    Dev<uint8_t> dik(ik);
    Dev<float> dqr(qr), dx(x), dfreq(freq);
    Dev<int32_t> dpos(pos), ids((size_t) T * kTopK), nvld(T), cand((size_t) T * kMaxCandBlocks), cn(T), ids1((size_t) T * kTopK), n1(T), c1((size_t) T * kMaxCandBlocks), cn1(T);
    Dev<char> ws(indexer_workspace_bytes(n_comp));

    for (int mode = 0; mode < 3; ++mode) {
        IndexerArgs a;
        a.mode = mode == 0 ? IndexMode::Full : mode == 1 ? IndexMode::Level1 : IndexMode::Level2;
        a.ratio = ratio; a.n_comp = n_comp; a.block_size = bs; a.cand_blocks = cand_blocks;
        a.w_q = mode == 2 ? dwq2.p : dwq.p; a.w_proj = mode == 2 ? dwp2.p : dwp.p; a.rope_freq = dfreq.p; a.comp_ik = dik.p;
        a.cand_ids = cand.p; a.cand_n = cn.p; a.out_ids = ids.p; a.out_n = nvld.p; a.format = g_fmt;
        // Level2 consumes the candidates the Level1 run (mode 1) left in `cand`
        indexer_select(a, dqr.p, dx.p, dpos.p, T, ws.p, nullptr);
        // T = 1 calls, one token each
        IndexerArgs b = a;
        b.cand_ids = c1.p; b.cand_n = cn1.p; b.out_ids = ids1.p; b.out_n = n1.p;
        for (int t = 0; t < T; ++t) {
            b.cand_ids = c1.p + (size_t) t * kMaxCandBlocks; b.cand_n = cn1.p + t;
            b.out_ids = ids1.p + (size_t) t * kTopK; b.out_n = n1.p + t;
            if (mode == 2) { b.cand_ids = cand.p + (size_t) t * kMaxCandBlocks; b.cand_n = cn.p + t; }
            indexer_select(b, dqr.p + (size_t) t * 1280, dx.p + (size_t) t * D, dpos.p + t, 1, ws.p, nullptr);
        }
        check(cudaDeviceSynchronize(), "indexer");
        const auto gid = ids.down(), gn = nvld.down(), gid1 = ids1.down(), gn1 = n1.down(), gc = cand.down(), gcn = cn.down(), gc1 = c1.down(), gcn1 = cn1.down();
        std::printf("[%s] indexer mode %s:\n", packed() ? "fp8" : "fp16", mode == 0 ? "Full" : mode == 1 ? "Level1" : "Level2");
        v.expect(gid == gid1 && gn == gn1, "indexer ids independent of T");
        if (mode == 1) {
            for (int t = 0; t < T; ++t) {
                v.expect(std::equal(gc.begin() + (size_t) t * kMaxCandBlocks, gc.begin() + (size_t) (t + 1) * kMaxCandBlocks, gc1.begin() + (size_t) t * kMaxCandBlocks) && gcn[t] == gcn1[t], "cand blocks independent of T");
            }
        }
        std::vector<uint8_t> ok;
        const std::vector<uint8_t>* okp = nullptr;
        if (mode == 2) {
            ok.assign((size_t) T * n_blk, 0);
            for (int t = 0; t < T; ++t) for (int i = 0; i < gcn[t]; ++i) ok[(size_t) t * n_blk + gc[(size_t) t * kMaxCandBlocks + i]] = 1;
            okp = &ok;
        }
        const Ref r = ref_scores(qr, x, wq, wp, ik, pos, n_comp, ratio, freq, nullptr, bs, n_blk);
        const Ref r2 = mode == 2 ? ref_scores(qr, x, wq2, wp2, ik, pos, n_comp, ratio, freq, okp, bs, n_blk) : r;
        for (int t = 0; t < T; ++t) {
            char name[64];
            std::snprintf(name, sizeof name, "top-k token %d (%d visible)", t, pos[t] + 1);
            std::vector<double> s(r2.scores.begin() + (size_t) t * n_comp, r2.scores.begin() + (size_t) (t + 1) * n_comp);
            check_topk(v, name, s, gid.data() + (size_t) t * kTopK, gn[t], kTopK, 0);
            if (mode == 1) {
                // candidate blocks: block max of the reference scores, newest block pinned
                std::vector<double> bm(n_blk, -INFINITY);
                for (int j = 0; j < n_comp; ++j) bm[j / bs] = std::max(bm[j / bs], s[j]);
                const int nv = pos[t] + 1;
                bm[(nv - 1) / bs] = 1e30;
                std::snprintf(name, sizeof name, "cand blocks token %d", t);
                check_topk(v, name, bm, gc.data() + (size_t) t * kMaxCandBlocks, gcn[t], cand_blocks, 0);
                v.expect(std::binary_search(gc.begin() + (size_t) t * kMaxCandBlocks, gc.begin() + (size_t) t * kMaxCandBlocks + gcn[t], (nv - 1) / bs), "newest block pinned");
            }
        }
    }
}

// The many-block top-k against the one-block kernel, bit for bit: random scores, ties across the cut, rows that are not
// valid, fewer rows than K, and a capacity above the visible rows.
void test_top_rows(Verdict& v) {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    int cases = 0;
    for (int n : {1, 3, 255, 256, 257, 1000, 4096, 8192, 12288, 12289, 16384, 32768, 65536, 200000, 600000, 1048576}) {
        for (int K : {1, 16, kTopK, 2048}) {
            for (int kind = 0; kind < 5; ++kind) {
                std::vector<float> s((size_t) n);
                for (int i = 0; i < n; ++i) {
                    const float x = nd(rng);
                    switch (kind) {
                        case 0: s[(size_t) i] = x; break;
                        case 1: s[(size_t) i] = std::floor(x * 8.f) / 8.f; break;                  // many ties
                        case 2: s[(size_t) i] = (i % 16) ? -INFINITY : x; break;                   // sparse rows
                        case 3: s[(size_t) i] = 0.f; break;                                        // all equal
                        default: s[(size_t) i] = (x > 0.f ? 1e30f : 1.f) * std::fabs(x) * (i & 1 ? -1.f : 1.f); break;   // a wide range, both signs
                    }
                }
                const int cap = kind == 0 ? 1048577 : n + 300;   // a full context's capacity, whatever is visible
                std::vector<float> full((size_t) cap, -INFINITY);
                std::copy(s.begin(), s.end(), full.begin());
                Dev<float> ds(full);
                Dev<int32_t> a((size_t) std::max(K, 2048)), b((size_t) std::max(K, 2048)), na(1), nb(1);
                Dev<int32_t> pos(std::vector<int32_t>{n - 1});
                Dev<char> ws(top_rows_workspace_bytes(cap));
                top_rows(ds.p, cap, K, a.p, std::max(K, 2048), na.p, pos.p, 1, 0, nullptr, nullptr);
                top_rows(ds.p, cap, K, b.p, std::max(K, 2048), nb.p, pos.p, 1, 0, ws.p, nullptr);
                check(cudaDeviceSynchronize(), "top_rows");
                const auto ga = a.down(), gb = b.down(), gna = na.down(), gnb = nb.down();
                if (ga != gb || gna != gnb) std::printf("  top_rows differs: n=%d K=%d kind=%d (%d vs %d)\n", n, K, kind, gna[0], gnb[0]);
                v.expect(ga == gb && gna == gnb, "top_rows many blocks = one block");
                ++cases;
            }
        }
    }
    std::printf("top_rows: %d cases, ids bit for bit against the one-block kernel\n", cases);
}

void bench_top_rows() {
    std::mt19937 rng(78);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {64, 512, 2048, 4096, 8192, 12288, 16384, 32768, 65536, 131072, 524288, 1048576}) {
        std::vector<float> s((size_t) n);
        for (float& x : s) x = nd(rng);
        Dev<float> ds(s);
        Dev<int32_t> a(kTopK), na(1), pos(std::vector<int32_t>{n - 1});
        Dev<char> ws(top_rows_workspace_bytes(n));
        char name[64];
        std::snprintf(name, sizeof name, "top_rows one block n=%d K=%d", n, kTopK);
        bench_line(name, bench_us([&] { top_rows(ds.p, n, kTopK, a.p, kTopK, na.p, pos.p, 1, 0, nullptr, nullptr); }, 50), (double) n * 4);
        std::snprintf(name, sizeof name, "top_rows many blocks n=%d K=%d", n, kTopK);
        bench_line(name, bench_us([&] { top_rows(ds.p, n, kTopK, a.p, kTopK, na.p, pos.p, 1, 0, ws.p, nullptr); }, 50), (double) n * 4 * 5);
    }
    // the rows a long conversation sees, in a full context's capacity
    const int cap = 1048577, seen = 131072;
    std::vector<float> s((size_t) cap, -INFINITY);
    for (int i = 0; i < seen; ++i) s[(size_t) i] = nd(rng);
    Dev<float> ds(s);
    Dev<int32_t> a(kTopK), na(1), pos(std::vector<int32_t>{seen - 1});
    Dev<char> ws(top_rows_workspace_bytes(cap));
    bench_line("top_rows one block n=131072 cap=1048577", bench_us([&] { top_rows(ds.p, cap, kTopK, a.p, kTopK, na.p, pos.p, 1, 0, nullptr, nullptr); }, 50), (double) seen * 4);
    bench_line("top_rows many blocks n=131072 cap=1048577", bench_us([&] { top_rows(ds.p, cap, kTopK, a.p, kTopK, na.p, pos.p, 1, 0, ws.p, nullptr); }, 50), (double) seen * 4 * 5);
}

void bench() {
    const int n_comp = 16384;
    const auto freq = make_table(true);
    Dev<uint16_t> wq(rand_bf16((size_t) kIdxHeads * kIdxDim * 1280, 41, 0.03f)), wp(rand_bf16((size_t) kIdxHeads * D, 42, 0.03f));
    Dev<uint8_t> dik(packed_rows(n_comp, kIdxDim, kIdxRowBytes, Round::Fp4Pow2, 43));
    Dev<float> qr(randn(4 * 1280, 44, 1.f)), x(randn(4 * D, 45, 1.f)), dfreq(freq);
    Dev<int32_t> pos(std::vector<int32_t>{n_comp - 4, n_comp - 3, n_comp - 2, n_comp - 1});
    Dev<int32_t> ids(4 * kTopK), nn(4), cand(4 * kMaxCandBlocks), cn(4);
    Dev<char> ws(indexer_workspace_bytes(n_comp));
    for (int T : {1, 4}) {
        IndexerArgs a;
        a.mode = IndexMode::Level1; a.ratio = 1; a.n_comp = n_comp; a.w_q = wq.p; a.w_proj = wp.p; a.rope_freq = dfreq.p;
        a.comp_ik = dik.p; a.cand_ids = cand.p; a.cand_n = cn.p; a.out_ids = ids.p; a.out_n = nn.p;
        char name[64];
        const double ikb = (double) n_comp * kIdxRowBytes * T;
        std::snprintf(name, sizeof name, "indexer Level1 T=%d n=%d", T, n_comp);
        bench_line(name, bench_us([&] { indexer_select(a, qr.p, x.p, pos.p + (4 - T), T, ws.p, nullptr); }, 100), (double) kIdxHeads * kIdxDim * 1280 * 2 + ikb);
        a.mode = IndexMode::Full;
        std::snprintf(name, sizeof name, "indexer Full T=%d n=%d", T, n_comp);
        bench_line(name, bench_us([&] { indexer_select(a, qr.p, x.p, pos.p + (4 - T), T, ws.p, nullptr); }, 100), (double) kIdxHeads * kIdxDim * 1280 * 2 + ikb);
    }
    {
        // buffers for a long context, few rows visible: what the idle blocks of the capped grid cost
        Dev<char> big(indexer_workspace_bytes(524288));
        IndexerArgs a;
        a.mode = IndexMode::Full; a.ratio = 1; a.n_comp = 524288; a.w_q = wq.p; a.w_proj = wp.p; a.rope_freq = dfreq.p;
        a.comp_ik = dik.p; a.cand_ids = cand.p; a.cand_n = cn.p; a.out_ids = ids.p; a.out_n = nn.p;
        bench_line("indexer Full T=1 n=16384, buffers for 524288", bench_us([&] { indexer_select(a, qr.p, x.p, pos.p + 3, 1, big.p, nullptr); }, 100),
                   (double) kIdxHeads * kIdxDim * 1280 * 2 + (double) n_comp * kIdxRowBytes);
    }
    Dev<uint16_t> wkv(rand_bf16((size_t) kLatent * D, 46, 0.02f)), wg(rand_bf16((size_t) kLatent * D, 47, 0.02f)), wik(rand_bf16((size_t) kIdxDim * kLatent, 48, 0.05f));
    Dev<float> nw(randn(kLatent, 49, 1.f, 1.f)), nk(randn(kIdxDim, 50, 1.f, 1.f));
    CompState st(2, 64);
    CompressArgs c;
    c.ratio = 2; c.w_kv = wkv.p; c.w_gate = wg.p; c.norm_w = nw.p; c.w_idx_k = wik.p; c.idx_k_norm_w = nk.p; c.rope_freq = dfreq.p;
    c.state_kv = st.skv.p; c.state_score = st.ssc.p; c.comp_kv = st.ckv.p; c.comp_ik = st.cik.p;
    Dev<char> cws(compress_workspace_bytes());
    Dev<int32_t> cpos(std::vector<int32_t>{1, 2, 3, 4});
    bench_line("compress ratio 2 T=1", bench_us([&] { compress(c, x.p, cpos.p, 1, cws.p, nullptr); }, 200), 2.0 * kLatent * D * 2);
    bench_line("compress ratio 2 T=4", bench_us([&] { compress(c, x.p, cpos.p, 4, cws.p, nullptr); }, 200), 2.0 * kLatent * D * 2);
    Dev<float> raw(randn(4 * kLatent, 51, 1.f)), out(4 * kLatent);
    Dev<uint8_t> ring((size_t) kWindowSlots * kWinRowBytes);
    bench_line("kv_window T=4", bench_us([&] { kv_window(raw.p, nw.p, cpos.p, 4, 1e-20f, dfreq.p, out.p, ring.p, CacheFormat::Fp8, nullptr); }, 500), 4.0 * kLatent * 12);
}

}  // namespace

int main(int argc, char** argv) {
    Verdict v;
    for (CacheFormat f : {CacheFormat::Fp8, CacheFormat::Fp16}) {
        g_fmt = f;
        test_kv_window(v);
        test_compress(v);
        test_indexer(v);
    }
    test_frozen(v);
    g_fmt = CacheFormat::Fp8;
    test_top_rows(v);
    bench();
    bench_top_rows();
    (void) argc; (void) argv;
    std::printf(v.failures ? "FAILED (%d)\n" : "PASS\n", v.failures);
    return v.failures ? 1 : 0;
}
