// Parity test of the DeepSeek-V4.1 Engram post-op and the host n-gram hash: random inputs at the real geometry
// against a double-precision host reference, independence from the window width and position, chunked hashing
// against whole-sequence hashing, a microbench, and (with --fixture DIR) real layers against the per-layer
// fixtures, including the hashed table rows read from the pack (--pack PATH).
#include "basalt/artifact/dequant.hpp"
#include "basalt/models/deepseek41/kernels.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cstring>

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

constexpr int D = 5120, HC = 4, KVW = (HC + 1) * D;

void ref_apply(const float* kv, const std::vector<float>& gw, std::vector<double>& x, const EngramArgs& a) {
    const float* value = kv + HC * D;
    for (int c = 0; c < HC; ++c) {
        double xx = 0, kk = 0;
        for (int d = 0; d < D; ++d) { xx += x[c * D + d] * x[c * D + d]; kk += (double) kv[c * D + d] * kv[c * D + d]; }
        const double rx = 1.0 / std::sqrt(xx / D + a.rms_eps), rk = 1.0 / std::sqrt(kk / D + a.rms_eps);
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += x[c * D + d] * rx * (kv[c * D + d] * rk) * gw[c * D + d];
        dot /= std::sqrt((double) D);
        const double g = 1.0 / (1.0 + std::exp(-std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6)), dot)));
        for (int d = 0; d < D; ++d) x[c * D + d] += g * value[d];
    }
}

std::vector<float> run(const Dev<float>& gw, const std::vector<float>& kv, std::vector<float> x, int T) {
    Dev<float> dkv(kv), dx(x);
    engram_apply(dkv.p, gw.p, nullptr, T, EngramArgs{}, dx.p, nullptr);
    return dx.down();
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture, pack;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--fixture") && i + 1 < argc) fixture = argv[++i];
        else if (!std::strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
    }
    Verdict v;
    EngramArgs a;

    // ---- random inputs; keys correlated with the streams to a varied degree so the gate spans both signs
    const int Tmax = 9;
    auto gw_h = randn((size_t) HC * D, 21, 1.f);
    Dev<float> gw(gw_h);
    auto x_h = randn((size_t) Tmax * HC * D, 22, 2.f);
    auto noise = randn((size_t) Tmax * KVW, 23);
    std::vector<float> kv_h(noise);
    for (int t = 0; t < Tmax; ++t)
        for (int c = 0; c < HC; ++c) {
            const float corr = (float) (t % 3 - 1) * 0.15f * (float) (c + 1);
            for (int d = 0; d < D; ++d) kv_h[(size_t) t * KVW + c * D + d] += corr * x_h[((size_t) t * HC + c) * D + d];
        }
    auto got = run(gw, kv_h, x_h, Tmax);
    double max_abs = 0, mx = 0;
    for (int t = 0; t < Tmax; ++t) {
        std::vector<double> x(x_h.begin() + (size_t) t * HC * D, x_h.begin() + (size_t)(t + 1) * HC * D);
        ref_apply(&kv_h[(size_t) t * KVW], gw_h, x, a);
        for (int i = 0; i < HC * D; ++i) {
            max_abs = std::max(max_abs, std::fabs(got[(size_t) t * HC * D + i] - x[i]));
            mx = std::max(mx, std::fabs(x[i]));
        }
    }
    std::printf("random T=%d: max abs %.3g (max |v| %.2f)\n", Tmax, max_abs, mx);
    v.expect(max_abs < 1e-4 * mx, "engram_apply error");

    bool bitwise = true;
    for (int T : {1, 2, 3, 4})
        for (int start : {0, 2, 5}) {
            if (start + T > Tmax) continue;
            std::vector<float> xs(x_h.begin() + (size_t) start * HC * D, x_h.begin() + (size_t)(start + T) * HC * D);
            std::vector<float> ks(kv_h.begin() + (size_t) start * KVW, kv_h.begin() + (size_t)(start + T) * KVW);
            auto o = run(gw, ks, xs, T);
            bitwise = bitwise && !std::memcmp(o.data(), &got[(size_t) start * HC * D], (size_t) T * HC * D * 4);
        }
    std::printf("window/position independence: %s\n", bitwise ? "bitwise identical" : "DIFFERS");
    v.expect(bitwise, "output depends on window width or position");

    // ---- hash: a window-by-window pass equals a whole-sequence pass; dead tokens and the sequence start pad
    EngramHashParams hp;
    hp.token_map.resize(1000);
    for (int i = 0; i < 1000; ++i) hp.token_map[i] = (i % 17 == 5) ? -1 : (i * 7) % 311;
    hp.pad_token_id = 2;
    hp.mult = {76632096046245ull, 4839876093313ull, 35959672319349ull, 73987337458391ull};
    for (int c = 0; c < hp.n_cols(); ++c) { hp.primes.push_back(1000003 + 2 * c); hp.offsets.push_back((uint64_t) c * 1100000); }
    {
        std::vector<int32_t> seq(40);
        for (int i = 0; i < 40; ++i) seq[i] = (i * 37 + 11) % 1000;
        const int nc = hp.n_cols();
        std::vector<int32_t> whole((size_t) 40 * nc), chunk((size_t) 40 * nc);
        engram_hash(hp, 0, seq.data(), 40, 0, nullptr, whole.data());
        for (int s = 0; s < 40; s += 3) {
            const int n = std::min(3, 40 - s);
            int32_t prev[3] = {0, 0, 0};
            for (int j = 0; j < 3; ++j) if (s - 3 + j >= 0) prev[j] = seq[s - 3 + j];
            engram_hash(hp, 0, &seq[s], n, s, s ? prev : nullptr, &chunk[(size_t) s * nc]);
        }
        v.expect(whole == chunk, "chunked hash differs from the whole-sequence hash");
        // token 0 sees only itself: every n-gram column beyond n=1 is built from pad tokens
        std::vector<int32_t> a1(nc), a2(nc);
        int32_t t1[1] = {seq[0]}, t2[1] = {seq[0]};
        engram_hash(hp, 0, t1, 1, 0, nullptr, a1.data());
        engram_hash(hp, 0, t2, 1, 0, nullptr, a2.data());
        v.expect(a1 == a2 && a1 == std::vector<int32_t>(whole.begin(), whole.begin() + nc), "start-of-sequence hash");
        std::printf("hash: chunked == whole sequence: %s\n", whole == chunk ? "yes" : "NO");
    }

    // ---- microbench
    for (int T : {1, 4}) {
        Dev<float> dkv(std::vector<float>(kv_h.begin(), kv_h.begin() + (size_t) T * KVW)),
            dx(std::vector<float>(x_h.begin(), x_h.begin() + (size_t) T * HC * D));
        char name[64];
        std::snprintf(name, sizeof name, "engram_apply T=%d", T);
        bench_line(name, bench_us([&] { engram_apply(dkv.p, gw.p, nullptr, T, a, dx.p, nullptr); }),
                   (double) T * (KVW + 2 * HC * D) * 4 + (double) HC * D * 4);
    }

    // ---- fixtures
    if (!fixture.empty()) {
        std::vector<char> raw;
        EngramHashParams fp;
        uint64_t sec_off = 0;
        std::vector<int32_t> layers;
        bool have_params = read_file(fixture + "/engram_params.bin", raw);
        if (have_params) {
            const char* q = raw.data();
            int32_t hdr[5];
            std::memcpy(hdr, q, 20); q += 20;
            std::memcpy(&sec_off, q, 8); q += 8;
            fp.n_gram = hdr[0]; fp.n_head = hdr[1]; fp.pad_token_id = hdr[2];
            layers.resize(hdr[3]); std::memcpy(layers.data(), q, 4 * layers.size()); q += 4 * layers.size();
            fp.token_map.resize(hdr[4]); std::memcpy(fp.token_map.data(), q, 4 * fp.token_map.size()); q += 4 * fp.token_map.size();
            fp.mult.resize(layers.size() * fp.n_gram); std::memcpy(fp.mult.data(), q, 8 * fp.mult.size()); q += 8 * fp.mult.size();
            fp.primes.resize(layers.size() * fp.n_cols()); std::memcpy(fp.primes.data(), q, 8 * fp.primes.size()); q += 8 * fp.primes.size();
            fp.offsets.resize(fp.primes.size()); std::memcpy(fp.offsets.data(), q, 8 * fp.offsets.size());
        }
        for (int e = 0; e < (int) layers.size(); ++e) {
            const int L = layers[e];
            std::vector<float> egate;
            if (!read_vec(fixture + "/w.blk." + std::to_string(L) + ".engram_gate.weight.bin", egate)) { v.failures++; continue; }
            Dev<float> dg(egate);
            for (const char* p : {"p1", "p2", "p3"})
                for (int step : {0, 1}) {
                    const std::string base = fixture + "/" + p + ".s" + std::to_string(step) + ".";
                    auto fx = [&](const char* name, std::vector<float>& out) {
                        return read_vec(base + name + "." + std::to_string(L) + ".bin", out);
                    };
                    std::vector<float> x_in, kv, ref_out, ref_gate, rows;
                    if (!read_vec(base + "l_out." + std::to_string(L - 1) + ".bin", x_in) || !fx("engram_kv", kv) ||
                        !fx("engram_out", ref_out) || !fx("engram_rows", rows)) {
                        std::printf("fixture %s s%d L%d missing\n", p, step, L);
                        v.failures++;
                        continue;
                    }
                    const int T = (int) (kv.size() / KVW);
                    auto out = run(dg, kv, x_in, T);
                    Err e1 = compare(out, ref_out);
                    std::printf("fixture %s s%d L%-2d T=%2d: engram_out rel rms %.3g max abs %.3g", p, step, L, T, e1.rms_rel, e1.max_abs);
                    v.expect(e1.rms_rel < 1e-4, "fixture engram_out");
                    // the hashed rows against the gathered rows of the fixture
                    if (!pack.empty() && have_params) {
                        std::vector<char> tj;
                        read_file(fixture + "/" + p + ".tokens.json", tj);
                        std::string js(tj.begin(), tj.end());
                        auto grab = [&](const char* key) {
                            std::vector<int32_t> r;
                            size_t b = js.find(std::string("\"") + key + "\"");
                            b = js.find('[', b);
                            const size_t en = js.find(']', b);
                            const char* s = js.c_str() + b + 1;
                            while (s < js.c_str() + en) {
                                char* nx;
                                long val = std::strtol(s, &nx, 10);
                                if (nx == s) { ++s; continue; }
                                r.push_back((int32_t) val);
                                s = nx;
                            }
                            return r;
                        };
                        auto prompt = grab("prompt"), dec = grab("decoded");
                        std::vector<int32_t> all(prompt);
                        all.insert(all.end(), dec.begin(), dec.end());
                        const int n_cols = fp.n_cols();
                        const int pos0 = step == 0 ? 0 : (int) prompt.size() + step - 1;
                        const int n = T;
                        std::vector<int32_t> ids((size_t) n * n_cols);
                        const int32_t* prev = nullptr;
                        if (pos0 > 0) prev = &all[pos0 - (fp.n_gram - 1)];
                        engram_hash(fp, e, &all[pos0], n, pos0, prev, ids.data());
                        std::ifstream f(pack, std::ios::binary);
                        std::vector<float> mine((size_t) n * n_cols * 256);
                        for (size_t r = 0; r < ids.size(); ++r) {
                            uint8_t blob[144];
                            f.seekg((std::streamoff)(sec_off + (uint64_t) ids[r] * 144));
                            f.read((char*) blob, 144);
                            for (int b = 0; b < 8; ++b) basalt::dequantize_iq4_nl(blob + b * 18, &mine[r * 256 + b * 32]);
                        }
                        Err e2 = compare(mine, rows, 1e-2);
                        std::printf("; hashed rows max abs %.3g", e2.max_abs);
                        v.expect(e2.max_abs < 1e-6, "hashed rows differ from the fixture rows");
                    }
                    std::printf("\n");
                }
        }
    }
    std::printf(v.failures ? "engram_parity: FAILED\n" : "engram_parity: ok\n");
    return v.failures ? 1 : 0;
}
