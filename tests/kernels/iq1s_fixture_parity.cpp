// src/kernels/iq1s_fixture_parity.cpp - DeepSeek routed + shared experts on real weights against llama.cpp's tensors.
//
//     iq1s_fixture_parity --gguf <shard-1.gguf> --fixture <dir>/p1 [--layers 0,3,20] [--step 0] [--eps 1e-20] [--limit 10]
//
// Per layer: the MoE input is ffn_norm(hc_ffn_pre) (RMS norm with the layer's weight, computed here in float), the
// router's choices are the fixture's ffn_moe_topk / ffn_moe_weights_scaled, the experts are read from the GGUF.  The
// routed experts run through native_expert_grouped_clamped, the shared expert through gated_mlp_clamped; the weighted
// sum is compared with the fixture's ffn_moe_out, the shared output with ffn_shexp (relative RMS, max abs).
// The fixture is llama.cpp's own CUDA run on the same weights, not ground truth.  Exit 0 when every layer is
// within the q8_1 activation noise (rel RMS < 3e-2).  The tensor-less first shard is skipped (the reader refuses a
// shard whose data section starts past the end of file); the norm epsilon and the limits are flags, defaulting to
// this model's values.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace kg = basalt::kernels;

namespace basalt {
inline double fp16_to_double(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25);
    return (h & 0x8000) ? -v : v;
}
}  // namespace basalt

namespace {

constexpr int H = 5120, FF = 2304, K = 6;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template<typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template<typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d"); }
template<typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

struct FxTensor {
    int step = 0, layer = -1;
    std::string name;
    std::vector<int64_t> shape;
    uint64_t offset = 0, nbytes = 0;
};

std::string field(const std::string& obj, const std::string& key) {
    const size_t k = obj.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    size_t p = obj.find(':', k) + 1;
    while (p < obj.size() && obj[p] == ' ') ++p;
    if (obj[p] == '"') return obj.substr(p + 1, obj.find('"', p + 1) - p - 1);
    if (obj[p] == '[') return obj.substr(p + 1, obj.find(']', p) - p - 1);
    return obj.substr(p, obj.find_first_of(",}", p) - p);
}

std::vector<FxTensor> read_index(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string j = ss.str();
    std::vector<FxTensor> out;
    size_t p = j.find("\"tensors\"");
    p = j.find('[', p);
    while (true) {
        const size_t a = j.find('{', p);
        if (a == std::string::npos) break;
        const size_t b = j.find('}', a);
        const std::string o = j.substr(a, b - a + 1);
        FxTensor t;
        t.step = std::atoi(field(o, "step").c_str());
        t.layer = std::atoi(field(o, "layer").c_str());
        t.name = field(o, "name");
        t.offset = std::strtoull(field(o, "offset").c_str(), nullptr, 10);
        t.nbytes = std::strtoull(field(o, "nbytes").c_str(), nullptr, 10);
        std::stringstream sh(field(o, "shape"));
        std::string tok;
        while (std::getline(sh, tok, ',')) t.shape.push_back(std::atoll(tok.c_str()));
        if (!t.name.empty()) out.push_back(t);
        p = b + 1;
    }
    return out;
}

struct Err {
    double sq = 0, ref_sq = 0, max_abs = 0;
    void add(double got, double ref) {
        sq += (got - ref) * (got - ref);
        ref_sq += ref * ref;
        max_abs = std::max(max_abs, std::fabs(got - ref));
    }
    double rms() const { return std::sqrt(sq / (ref_sq + 1e-30)); }
};

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, fx;
    float eps = 1e-20f, limit = 10.0f;
    bool diag = false;
    std::vector<int> layers{0, 1, 2, 3, 10, 20, 30, 39};
    int step = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gguf") gguf = argv[i + 1];
        else if (a == "--fixture") fx = argv[i + 1];
        else if (a == "--step") step = std::atoi(argv[i + 1]);
        else if (a == "--diag") diag = std::atoi(argv[i + 1]) != 0;
        else if (a == "--eps") eps = (float) std::atof(argv[i + 1]);
        else if (a == "--limit") limit = (float) std::atof(argv[i + 1]);
        else if (a == "--layers") {
            layers.clear();
            std::stringstream ss(argv[i + 1]);
            std::string t;
            while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
        }
    }
    if (gguf.empty() || fx.empty()) {
        std::fprintf(stderr, "usage: iq1s_fixture_parity --gguf <shard-1.gguf> --fixture <dir>/p1 [--layers a,b] [--step n]\n");
        return 2;
    }
    const auto index = read_index(fx + ".json");
    std::ifstream bin(fx + ".bin", std::ios::binary);
    auto fetch = [&](const std::string& name, int layer, std::vector<float>& f, std::vector<int32_t>* iv = nullptr) {
        for (const auto& t : index)
            if (t.step == step && t.layer == layer && t.name == name) {
                bin.clear();
                bin.seekg((std::streamoff) t.offset);
                if (iv) {
                    iv->resize(t.nbytes / 4);
                    bin.read((char*) iv->data(), (std::streamsize) t.nbytes);
                } else {
                    f.resize(t.nbytes / 4);
                    bin.read((char*) f.data(), (std::streamsize) t.nbytes);
                }
                return t.shape;
            }
        std::fprintf(stderr, "fixture: no %s at layer %d step %d\n", name.c_str(), layer, step);
        std::exit(2);
    };

    // every shard that holds tensors: shard N of M is <prefix>-0000N-of-0000M.gguf
    std::vector<std::unique_ptr<basalt::GgufFile>> shards;
    {
        const size_t at = gguf.rfind("-00001-of-");
        if (at == std::string::npos) { std::fprintf(stderr, "--gguf must be the -00001-of-N shard\n"); return 2; }
        const std::string tail = gguf.substr(at + 10);
        const int total = std::atoi(tail.c_str());
        for (int i = 2; i <= total; ++i) {
            char b[16];
            std::snprintf(b, sizeof b, "-%05d-of-", i);
            shards.push_back(std::make_unique<basalt::GgufFile>(gguf.substr(0, at) + b + tail));
        }
    }
    struct Found { const basalt::TensorInfo* t = nullptr; const uint8_t* data = nullptr; };
    auto find = [&](const std::string& name) {
        for (const auto& sh : shards)
            if (const auto* t = sh->find(name)) return Found{t, sh->tensor_data(*t)};
        return Found{};
    };

    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    int bad = 0;
    for (int layer : layers) {
        std::vector<float> pre, w_scaled, moe_out, shexp, dummy;
        std::vector<int32_t> topk;
        const auto sh = fetch("hc_ffn_pre", layer, pre);
        const int T = (int) sh[1];
        fetch("ffn_moe_topk", layer, dummy, &topk);
        fetch("ffn_moe_weights_scaled", layer, w_scaled);
        fetch("ffn_moe_out", layer, moe_out);
        fetch("ffn_shexp", layer, shexp);
        const std::string pfx = "blk." + std::to_string(layer) + ".";
        const auto tn = find(pfx + "ffn_norm.weight");
        if (!tn.t) { std::fprintf(stderr, "layer %d: no ffn_norm\n", layer); return 2; }
        const float* nw = (const float*) tn.data;
        std::vector<float> x((size_t) T * H);
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int i = 0; i < H; ++i) ss += (double) pre[(size_t) t * H + i] * pre[(size_t) t * H + i];
            const float r = 1.0f / std::sqrt((float) (ss / H) + eps);
            for (int i = 0; i < H; ++i) x[(size_t) t * H + i] = pre[(size_t) t * H + i] * r * nw[i];
        }
        auto* dx = dalloc<float>(x.size());
        up(dx, x);
        auto* xq = dalloc<uint8_t>((size_t) T * H / 32 * 36);
        kg::quantize_q8_1_rows(dx, T, H, xq, s);

        // routed experts
        const auto tg = find(pfx + "ffn_gate_exps.weight");
        const auto tu = find(pfx + "ffn_up_exps.weight");
        const auto td = find(pfx + "ffn_down_exps.weight");
        if (!tg.t || !tu.t || !td.t || tg.t->type != tu.t->type) { std::fprintf(stderr, "layer %d: expert tensors\n", layer); return 2; }
        const kg::NativeExpertLayout L = kg::native_expert_layout((int) tg.t->type, (int) td.t->type, H, FF);
        std::vector<int> uniq;
        for (int32_t e : topk)
            if (std::find(uniq.begin(), uniq.end(), e) == uniq.end()) uniq.push_back(e);
        std::sort(uniq.begin(), uniq.end());
        const size_t slot = (L.bytes + 255) & ~(size_t) 255;
        auto* blobs = dalloc<uint8_t>(slot * uniq.size());
        std::vector<uint8_t> hb(L.bytes);
        std::vector<std::vector<uint8_t>> hblobs;
        for (size_t u = 0; u < uniq.size(); ++u) {
            const size_t e = (size_t) uniq[u];
            std::memcpy(hb.data(), tg.data + e * FF * L.gu_row, FF * L.gu_row);
            std::memcpy(hb.data() + L.up_off, tu.data + e * FF * L.gu_row, FF * L.gu_row);
            std::memcpy(hb.data() + L.down_off, td.data + e * H * L.d_row, (size_t) H * L.d_row);
            ck(cudaMemcpy(blobs + slot * u, hb.data(), L.bytes, cudaMemcpyHostToDevice), "blob");
            hblobs.push_back(hb);
        }
        const int n = T * K;
        std::vector<unsigned long long> gptr;
        std::vector<int32_t> start{0}, dst, tok;
        for (size_t u = 0; u < uniq.size(); ++u) {
            for (int i = 0; i < n; ++i)
                if (topk[i] == uniq[u]) { dst.push_back(i); tok.push_back(i / K); }
            gptr.push_back((unsigned long long) (blobs + slot * u));
            start.push_back((int32_t) dst.size());
        }
        const int ng = (int) gptr.size();
        auto* dptr = dalloc<unsigned long long>(ng);
        auto* dstart = dalloc<int32_t>(ng + 1);
        auto* dng = dalloc<int32_t>(1);
        auto* ddst = dalloc<int32_t>(n);
        auto* dtok = dalloc<int32_t>(n);
        up(dptr, gptr); up(dstart, start); up(ddst, dst); up(dtok, tok);
        ck(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice), "ng");
        auto* scr = dalloc<uint8_t>(kg::native_expert_scratch_bytes(n, FF));
        auto* dout = dalloc<float>((size_t) n * H);
        kg::native_expert_grouped_clamped(L, limit, dptr, dstart, dng, ddst, dtok, ng, n, xq, scr, dout, s);
        ck(cudaStreamSynchronize(s), "sync");
        const auto rows = down(dout, (size_t) n * H);
        Err er;
        std::vector<float> sum((size_t) T * H, 0.f);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k)
                for (int i = 0; i < H; ++i) sum[(size_t) t * H + i] += w_scaled[t * K + k] * rows[((size_t) t * K + k) * H + i];
        for (size_t i = 0; i < sum.size(); ++i) er.add(sum[i], moe_out[i]);

        if (diag) {
            // token 0 in double from the dequantized weights: the activation as the kernels see it (q8_1), the SwiGLU
            // output exact (A) or rounded to q8_1 blocks (B); against the kernels' sum and the fixture
            const auto xqh = down(xq, (size_t) T * H / 32 * 36);
            std::vector<double> xh(H);
            for (int b = 0; b < H / 32; ++b) {
                const uint8_t* q = xqh.data() + (size_t) b * 36;
                uint16_t hb;
                std::memcpy(&hb, q, 2);
                const double dd = basalt::fp16_to_double(hb);
                for (int i = 0; i < 32; ++i) xh[b * 32 + i] = dd * (int8_t) q[4 + i];
            }
            std::vector<double> A(H, 0.0), B(H, 0.0), C(H, 0.0), D(H, 0.0);
            auto deq = [&](int type, const uint8_t* w, int64_t nvals) {
                std::vector<float> o((size_t) nvals);
                void* dw = nullptr;
                float* dq = nullptr;
                ck(cudaMalloc(&dw, kg::iq_row_bytes(type, nvals)), "m");
                ck(cudaMalloc(&dq, nvals * 4), "m");
                ck(cudaMemcpy(dw, w, kg::iq_row_bytes(type, nvals), cudaMemcpyHostToDevice), "c");
                kg::iq_dequant_f32(type, dw, nvals, dq, s);
                ck(cudaStreamSynchronize(s), "s");
                ck(cudaMemcpy(o.data(), dq, nvals * 4, cudaMemcpyDeviceToHost), "c");
                cudaFree(dw); cudaFree(dq);
                return o;
            };
            for (int k = 0; k < K; ++k) {
                const size_t e = (size_t) topk[k];
                const auto wg = deq((int) tg.t->type, tg.data + e * FF * L.gu_row, (int64_t) FF * H);
                const auto wu = deq((int) tu.t->type, tu.data + e * FF * L.gu_row, (int64_t) FF * H);
                const auto wd = deq((int) td.t->type, td.data + e * H * L.d_row, (int64_t) H * FF);
                std::vector<double> h(FF);
                for (int r = 0; r < FF; ++r) {
                    double g = 0, u = 0;
                    for (int c = 0; c < H; ++c) { g += (double) wg[(size_t) r * H + c] * xh[c]; u += (double) wu[(size_t) r * H + c] * xh[c]; }
                    g = std::min(g, (double) limit);
                    u = std::max(std::min(u, (double) limit), -(double) limit);
                    h[r] = g / (1.0 + std::exp(-g)) * u;
                }
                std::vector<double> hc(FF), hd(FF);
                for (int r = 0; r < FF; ++r) {
                    double g = 0, u = 0;
                    for (int c = 0; c < H; ++c) { g += (double) wg[(size_t) r * H + c] * xh[c]; u += (double) wu[(size_t) r * H + c] * xh[c]; }
                    hd[r] = g / (1.0 + std::exp(-g)) * u;                                                   // no limit
                    hc[r] = std::min(g / (1.0 + std::exp(-g)), (double) limit) * std::max(std::min(u, (double) limit), -(double) limit);   // limit after silu
                }
                std::vector<double> hqv(h);
                for (int b = 0; b < FF / 32; ++b) {
                    double am = 0;
                    for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(h[b * 32 + i]));
                    const double d = am / 127.0;
                    for (int i = 0; i < 32; ++i) hqv[b * 32 + i] = d > 0 ? std::nearbyint(h[b * 32 + i] / d) * d : 0.0;
                }
                for (int r = 0; r < H; ++r) {
                    double a = 0, bq = 0;
                    for (int c = 0; c < FF; ++c) { a += (double) wd[(size_t) r * FF + c] * h[c]; bq += (double) wd[(size_t) r * FF + c] * hqv[c]; }
                    double cc = 0, dd = 0;
                    for (int c = 0; c < FF; ++c) { cc += (double) wd[(size_t) r * FF + c] * hc[c]; dd += (double) wd[(size_t) r * FF + c] * hd[c]; }
                    C[r] += w_scaled[k] * cc;
                    D[r] += w_scaled[k] * dd;
                    A[r] += w_scaled[k] * a;
                    B[r] += w_scaled[k] * bq;
                }
            }
            Err ga, gb, fa, fb, gf, fc, fd;
            for (int r = 0; r < H; ++r) {
                ga.add(sum[r], A[r]); gb.add(sum[r], B[r]); fa.add(moe_out[r], A[r]); fb.add(moe_out[r], B[r]); fc.add(moe_out[r], C[r]); fd.add(moe_out[r], D[r]); gf.add(sum[r], moe_out[r]);
            }
            {   // the weights that best rebuild the fixture's sum from the kernels' per-expert rows
                double M[K][K + 1] = {};
                for (int i = 0; i < K; ++i) {
                    for (int j = 0; j < K; ++j)
                        for (int r = 0; r < H; ++r) M[i][j] += (double) rows[(size_t) i * H + r] * rows[(size_t) j * H + r];
                    for (int r = 0; r < H; ++r) M[i][K] += (double) rows[(size_t) i * H + r] * moe_out[r];
                }
                for (int c = 0; c < K; ++c) {
                    int piv = c;
                    for (int r = c + 1; r < K; ++r) if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
                    for (int j = 0; j <= K; ++j) std::swap(M[c][j], M[piv][j]);
                    for (int r = 0; r < K; ++r) {
                        if (r == c) continue;
                        const double f = M[r][c] / M[c][c];
                        for (int j = c; j <= K; ++j) M[r][j] -= f * M[c][j];
                    }
                }
                std::printf("  fitted/router weights per expert:");
                Err fit;
                for (int k = 0; k < K; ++k) std::printf(" %.4f/%.4f", M[k][K] / M[k][k], (double) w_scaled[k]);
                for (int r = 0; r < H; ++r) {
                    double v = 0;
                    for (int k = 0; k < K; ++k) v += M[k][K] / M[k][k] * rows[(size_t) k * H + r];
                    fit.add(v, moe_out[r]);
                }
                std::printf("  residual %.2e\n", fit.rms());
            }
            std::printf("  token 0 diag: kernels vs exact-h ref %.2e, vs q8-h ref %.2e | fixture vs exact-h ref %.2e, vs q8-h ref %.2e | kernels vs fixture %.2e | fixture vs limit-after-silu %.2e, vs no limit %.2e\n",
                        ga.rms(), gb.rms(), fa.rms(), fb.rms(), gf.rms(), fc.rms(), fd.rms());
        }

        {   // the CPU tier (ggml's dots on Q8_K activations) on the same entries
            namespace kc = basalt::kernels::cpu;
            kc::NativeFmt f;
            std::string err;
            if (!kc::native_fmt((int) tg.t->type, (int) td.t->type, H, FF, f, err, limit)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
            std::vector<float> csum((size_t) T * H, 0.f), ff(FF), orow(H);
            std::vector<uint8_t> act(kc::kNativeActBytes), hq(kc::kNativeHBytes);
            for (int t = 0; t < T; ++t) {
                kc::native_quant_act(f, &x[(size_t) t * H], act.data());
                for (int k = 0; k < K; ++k) {
                    const size_t u = std::find(uniq.begin(), uniq.end(), topk[t * K + k]) - uniq.begin();
                    const void* ap = act.data();
                    float* fp = ff.data();
                    kc::native_gu_rows(f, hblobs[u].data(), &ap, 1, &fp, 0, FF);
                    kc::native_quant_h(f, ff.data(), hq.data());
                    const void* hp = hq.data();
                    float* op = orow.data();
                    kc::native_down_rows(f, hblobs[u].data(), &hp, 1, &op, 0, H);
                    for (int i = 0; i < H; ++i) csum[(size_t) t * H + i] += w_scaled[t * K + k] * orow[i];
                }
            }
            Err ec, ecg;
            for (size_t i = 0; i < csum.size(); ++i) { ec.add(csum[i], moe_out[i]); ecg.add(csum[i], sum[i]); }
            std::printf("  CPU tier (Q8_K activations) vs ffn_moe_out: rel RMS %.3e (max abs %.2e); vs the GPU kernels %.3e\n", ec.rms(), ec.max_abs, ecg.rms());
        }

        // shared expert
        const auto hg = find(pfx + "ffn_gate_shexp.weight");
        const auto hu = find(pfx + "ffn_up_shexp.weight");
        const auto hd = find(pfx + "ffn_down_shexp.weight");
        if (!hg.t || !hu.t || !hd.t || hg.t->type != hu.t->type) { std::fprintf(stderr, "layer %d: shared tensors\n", layer); return 2; }
        const kg::NativeExpertLayout LS = kg::native_expert_layout((int) hg.t->type, (int) hd.t->type, H, FF);
        if (!kg::native_expert_supported((int) hg.t->type, (int) hd.t->type, H, FF)) {
            std::fprintf(stderr, "layer %d: shared types %u/%u unsupported\n", layer, hg.t->type, hd.t->type);
            return 2;
        }
        std::vector<uint8_t> sb(LS.bytes);
        std::memcpy(sb.data(), hg.data, FF * LS.gu_row);
        std::memcpy(sb.data() + LS.up_off, hu.data, FF * LS.gu_row);
        std::memcpy(sb.data() + LS.down_off, hd.data, (size_t) H * LS.d_row);
        auto* dsb = dalloc<uint8_t>(sb.size());
        up(dsb, sb);
        auto* sscr = dalloc<uint8_t>(kg::gated_mlp_scratch_bytes(T, FF));
        auto* sout = dalloc<float>((size_t) T * H);
        kg::gated_mlp_clamped(LS, limit, dsb, T, T, xq, sscr, sout, s);
        ck(cudaStreamSynchronize(s), "sync");
        const auto sres = down(sout, (size_t) T * H);
        Err es, et;
        for (size_t i = 0; i < sres.size(); ++i) {
            es.add(sres[i], shexp[i]);
            et.add(sum[i] + sres[i], moe_out[i] + shexp[i]);
        }
        const bool ok = er.rms() < 3e-2 && es.rms() < 3e-2;
        std::printf("layer %2d  experts %s/%s  routed vs ffn_moe_out: rel RMS %.3e (max abs %.2e)   shared %s/%s vs ffn_shexp: rel RMS %.3e (max abs %.2e)   sum: %.3e  %s\n",
                    layer, basalt::ggml_type_name(tg.t->type), basalt::ggml_type_name(td.t->type), er.rms(), er.max_abs,
                    basalt::ggml_type_name(hg.t->type), basalt::ggml_type_name(hd.t->type), es.rms(), es.max_abs, et.rms(),
                    ok ? "ok" : "FAIL");
        if (!ok) ++bad;
        cudaFree(dx); cudaFree(xq); cudaFree(blobs); cudaFree(dptr); cudaFree(dstart); cudaFree(dng); cudaFree(ddst);
        cudaFree(dtok); cudaFree(scr); cudaFree(dout); cudaFree(dsb); cudaFree(sscr); cudaFree(sout);
    }
    std::printf("iq1s_fixture_parity: %d bad layers\n", bad);
    return bad ? 1 : 0;
}
