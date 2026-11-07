// tests/prefill/peer_share_parity.cu - a second card's share of a prompt chunk's experts (prefill::PeerShare) against the
// same experts on the main card's MMQ products and a double-precision CPU reference, at the routed-expert geometry of a
// large MoE (4096 x 2048, six experts a token). Half the experts sit in the second card's memory, half in pinned host
// memory. Checks every row, the rows' independence from how the share cuts its launches, and prints the products' rates
// on each card. Exit 77 without two GPUs.
#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/peer_share.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = basalt::prefill::mmq;

constexpr int N = 4096, FF = 2048, K = 6, E = 48, T = 2048;
constexpr ggml_type kGu = GGML_TYPE_Q2_K, kDown = GGML_TYPE_Q3_K;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Geo {
    size_t gu_row = ggml_row_size(kGu, N), d_row = ggml_row_size(kDown, FF);
    size_t down_off = gu_row * 2 * FF, bytes = down_off + d_row * N, stride = 0;
    Geo() {
        const size_t a = ggml_type_size(kGu), b = ggml_type_size(kDown);
        size_t unit = a;
        while (unit % b != 0) unit += a;
        stride = (bytes + unit - 1) / unit * unit;
    }
};

// gate rows, then up rows, then down: N(0, 0.02) weights quantized by ggml's reference quantizers
std::vector<uint8_t> make_blob(const Geo& g, std::mt19937& rng, std::vector<float>& gu, std::vector<float>& dn) {
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> wgu((size_t) 2 * FF * N), wd((size_t) N * FF);
    for (float& v : wgu) v = nd(rng);
    for (float& v : wd) v = nd(rng);
    std::vector<uint8_t> b(g.stride, 0);
    ggml_quantize_chunk(kGu, wgu.data(), b.data(), 0, 2 * FF, N);
    ggml_quantize_chunk(kDown, wd.data(), b.data() + g.down_off, 0, N, FF);
    gu.resize(wgu.size());
    dn.resize(wd.size());
    const auto* tg = ggml_get_type_traits(kGu);
    const auto* td = ggml_get_type_traits(kDown);
    for (int r = 0; r < 2 * FF; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * N, N);
    for (int r = 0; r < N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * FF, FF);
    return b;
}

// silu(gate) * up of split rows
__global__ void k_swiglu(const float* gu, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / FF, k = i - r * FF;
    const float g = gu[r * 2 * FF + k], u = gu[r * 2 * FF + FF + k];
    h[i] = g / (1.f + __expf(-g)) * u;
}
void swiglu(const float* gu, float* h, int64_t rows, void* stream) {
    const int64_t n = rows * FF;
    if (n > 0) k_swiglu<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gu, h, n);
}

struct Routed {
    std::vector<std::vector<int32_t>> pairs;   // [expert] pair = t * K + j
};
Routed route(std::mt19937& rng) {
    Routed r;
    r.pairs.resize(E);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick;
        while ((int) pick.size() < K) {
            // skewed: low ids are hot, so some experts span several launches
            const int e = (rng() % 3 == 0) ? (int) (rng() % 4) : (int) (rng() % E);
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int j = 0; j < K; ++j) r.pairs[(size_t) pick[(size_t) j]].push_back(t * K + j);
    }
    return r;
}

// The main card's rows, the experts in launches of `group` experts; `ms` the products' time.
std::vector<float> main_rows(const Geo& g, const std::vector<std::vector<uint8_t>>& blobs, const Routed& rt, const float* dx,
                             double& ms, int64_t& rows_out, int group) {
    std::vector<int32_t> tok, pair, bounds{0};
    for (int e = 0; e < E; ++e) {
        for (int32_t p : rt.pairs[(size_t) e]) {
            tok.push_back(p / K);
            pair.push_back(p);
        }
        bounds.push_back((int32_t) tok.size());
    }
    const int64_t R = (int64_t) tok.size();
    rows_out = R;
    void *w, *xq, *hq, *dtok, *dpair, *dbnd, *ident;
    float *gu, *h, *rows;
    ck(cudaMalloc(&w, g.stride * E), "w");
    for (int e = 0; e < E; ++e) ck(cudaMemcpy((uint8_t*) w + g.stride * e, blobs[(size_t) e].data(), g.bytes, cudaMemcpyHostToDevice), "w up");
    ck(cudaMalloc(&xq, mmq::q8_bytes(R, N)), "xq");
    ck(cudaMalloc(&hq, mmq::q8_bytes(R, FF)), "hq");
    ck(cudaMalloc(&gu, (size_t) R * 2 * FF * 4), "gu");
    ck(cudaMalloc(&h, (size_t) R * FF * 4), "h");
    ck(cudaMalloc(&rows, (size_t) T * K * N * 4), "rows");
    ck(cudaMalloc(&dtok, (size_t) R * 4), "tok");
    ck(cudaMalloc(&dpair, (size_t) R * 4), "pair");
    ck(cudaMalloc(&dbnd, (size_t) (E + 1) * 4), "bounds");
    ck(cudaMalloc(&ident, (size_t) R * 4), "ident");
    cudaMemcpy(dtok, tok.data(), (size_t) R * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpair, pair.data(), (size_t) R * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbnd, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
    mmq::iota((int32_t*) ident, R, nullptr);
    mmq::Context ctx;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int rep = 0; rep < 3; ++rep) {   // the last one timed
        mmq::quantize(dx, (const int32_t*) dtok, xq, kGu, N, N, R, nullptr);
        cudaEventRecord(a, nullptr);
        mmq::Product p;
        p.type = kGu;
        p.w_rows = 2 * FF;
        p.w_cols = N;
        p.expert_bytes = g.stride;
        p.xq = xq;
        p.ids = (const int32_t*) ident;
        p.total_rows = R;
        p.dst = gu;
        p.ld_dst = 2 * FF;
        auto launches = [&](mmq::Product q) {
            for (int e0 = 0; e0 < E; e0 += group) {
                q.n = std::min(group, E - e0);
                q.bounds = (const int32_t*) dbnd + e0;
                int64_t mr = 0;
                for (int e = e0; e < e0 + q.n; ++e) mr = std::max<int64_t>(mr, (int64_t) rt.pairs[(size_t) e].size());
                q.max_rows = mr;
                mmq::Product r = q;
                r.w = (const uint8_t*) q.w + g.stride * (size_t) e0;
                ctx.run(r, nullptr);
            }
        };
        p.w = w;
        launches(p);
        swiglu(gu, h, R, nullptr);
        mmq::quantize(h, nullptr, hq, kDown, FF, FF, R, nullptr);
        mmq::Product d = p;
        d.w = (uint8_t*) w + g.down_off;
        d.type = kDown;
        d.w_rows = N;
        d.w_cols = FF;
        d.xq = hq;
        d.ids = (const int32_t*) dpair;
        d.dst = rows;
        d.ld_dst = N;
        launches(d);
        cudaEventRecord(b, nullptr);
    }
    ck(cudaDeviceSynchronize(), "main products");
    float t = 0;
    cudaEventElapsedTime(&t, a, b);
    ms = t;
    std::vector<float> out((size_t) T * K * N);
    cudaMemcpy(out.data(), rows, out.size() * 4, cudaMemcpyDeviceToHost);
    for (void* p : {w, xq, hq, (void*) gu, (void*) h, (void*) rows, dtok, dpair, dbnd, ident}) cudaFree(p);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return out;
}

struct Err {
    double max_abs = 0, max_rel = 0;
    int64_t same = 0, n = 0;
};
Err compare(const std::vector<float>& a, const std::vector<float>& b, const Routed& rt) {
    Err r;
    for (int e = 0; e < E; ++e)
        for (int32_t p : rt.pairs[(size_t) e]) {
            const float* x = a.data() + (size_t) p * N;
            const float* y = b.data() + (size_t) p * N;
            double nrm = 0;
            for (int i = 0; i < N; ++i) nrm = std::max(nrm, (double) std::fabs(y[i]));
            for (int i = 0; i < N; ++i) {
                const double d = std::fabs((double) x[i] - (double) y[i]);
                r.max_abs = std::max(r.max_abs, d);
                r.max_rel = std::max(r.max_rel, d / std::max(nrm, 1e-30));
            }
            r.same += std::memcmp(x, y, (size_t) N * 4) == 0;
            ++r.n;
        }
    return r;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 2) {
        std::printf("peer_share_parity: needs two GPUs, skipped\n");
        return 77;
    }
    try {
        std::mt19937 rng(7);
        const Geo g;
        std::vector<std::vector<uint8_t>> blobs;
        std::vector<std::vector<float>> wgu(E), wdn(E);
        for (int e = 0; e < E; ++e) blobs.push_back(make_blob(g, rng, wgu[(size_t) e], wdn[(size_t) e]));
        std::vector<float> x((size_t) T * N);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : x) v = nd(rng);
        const Routed rt = route(rng);

        ck(cudaSetDevice(0), "device 0");
        float* dx = nullptr;
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        {   // gathered rows of a whole quantized layer against quantizing those rows directly: the same bytes
            const int Rg = 1000;
            std::vector<int32_t> ids(Rg);
            for (int i = 0; i < Rg; ++i) ids[(size_t) i] = (int32_t) ((i * 7919) % T);
            int32_t* dids = nullptr;
            void *all = nullptr, *a = nullptr, *b = nullptr;
            ck(cudaMalloc(&dids, Rg * 4), "ids");
            ck(cudaMalloc(&all, mmq::q8_bytes(T, N)), "q8 all");
            ck(cudaMalloc(&a, mmq::q8_bytes(Rg, N)), "q8 a");
            ck(cudaMalloc(&b, mmq::q8_bytes(Rg, N)), "q8 b");
            cudaMemcpy(dids, ids.data(), Rg * 4, cudaMemcpyHostToDevice);
            cudaMemset(a, 0, mmq::q8_bytes(Rg, N));
            cudaMemset(b, 0, mmq::q8_bytes(Rg, N));
            int bad = 0, nt = 0;
            for (ggml_type t : {kGu, kDown, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_0, GGML_TYPE_NVFP4,
                                GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0}) {
                ++nt;
                mmq::quantize(dx, nullptr, all, t, N, N, T, nullptr);
                mmq::gather_q8(all, T, dids, a, Rg, N, nullptr);
                mmq::quantize(dx, dids, b, t, N, N, Rg, nullptr);
                ck(cudaDeviceSynchronize(), "gather");
                std::vector<uint8_t> ha(mmq::q8_bytes(Rg, N)), hb(ha.size());
                cudaMemcpy(ha.data(), a, ha.size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(hb.data(), b, hb.size(), cudaMemcpyDeviceToHost);
                bad += std::memcmp(ha.data(), hb.data(), ha.size()) != 0;
            }
            std::printf("gather_q8 vs quantize with ids, %d formats, %d rows of %d: %s\n", nt, Rg, T, bad ? "DIFFER" : "bitwise identical");
            if (bad) {
                std::printf("peer_share_parity: FAIL\n");
                return 1;
            }
            for (void* p : {(void*) dids, all, a, b}) cudaFree(p);
        }
        double main_ms = 0;
        int64_t R = 0;
        const std::vector<float> ref = main_rows(g, blobs, rt, dx, main_ms, R, E);
        {   // the main card alone, its launches cut another way: how much the cut moves a row there
            double ms2 = 0;
            int64_t r2 = 0;
            const Err c = compare(main_rows(g, blobs, rt, dx, ms2, r2, 3), ref, rt);
            std::printf("main card MMQ, launches of 3 experts vs all %d: %lld/%lld rows bitwise, max abs %.3g, max rel %.3g\n", E,
                        (long long) c.same, (long long) c.n, c.max_abs, c.max_rel);
        }

        // the second card's copy of the first half; the second half in pinned host memory
        std::vector<uint8_t*> peer_blob(E, nullptr);
        {
            const basalt::core::OnDevice od(1);
            for (int e = 0; e < E / 2; ++e) {
                ck(cudaMalloc(&peer_blob[(size_t) e], g.stride), "peer blob");
                cudaMemcpy(peer_blob[(size_t) e], blobs[(size_t) e].data(), g.bytes, cudaMemcpyHostToDevice);
            }
        }
        for (int e = E / 2; e < E; ++e) {
            ck(cudaHostAlloc((void**) &peer_blob[(size_t) e], g.stride, cudaHostAllocPortable), "host blob");
            std::memcpy(peer_blob[(size_t) e], blobs[(size_t) e].data(), g.bytes);
        }
        std::vector<std::vector<int32_t>> tok(E);
        std::vector<basalt::prefill::PeerShare::Expert> ex;
        for (int e = 0; e < E; ++e) {
            for (int32_t p : rt.pairs[(size_t) e]) tok[(size_t) e].push_back(p / K);
            basalt::prefill::PeerShare::Expert q;
            q.blob = peer_blob[(size_t) e];
            q.on_peer = e < E / 2;
            q.tok = tok[(size_t) e].data();
            q.n = (int) tok[(size_t) e].size();
            ex.push_back(q);
        }
        basalt::prefill::PeerShare::Layer L;
        L.gu_type = kGu;
        L.d_type = kDown;
        L.blob_bytes = g.bytes;
        L.stride = g.stride;
        L.gu_stride = g.stride;
        L.down_off = g.down_off;
        L.d_stride = g.stride;
        L.act = [](const float* gu, float* h, const basalt::prefill::PeerShare::Launch& l, void* stream) {
            swiglu(gu, h, l.rows, stream);
        };

        float* ddst = nullptr;
        ck(cudaMalloc(&ddst, (size_t) T * K * N * 4), "dst");
        cudaStream_t st;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        int fail = 0;
        double peer_ms = 0;
        struct Cut { int experts; int64_t rows; };
        std::vector<float> first;
        for (const Cut c : {Cut{8, 2048}, Cut{3, 700}}) {
            basalt::prefill::PeerShare::Dims d;
            d.n_embd = N;
            d.n_ff = FF;
            d.group_bytes = g.stride * (size_t) c.experts;
            d.act_bytes = mmq::q8_bytes(T, N);
            d.max_tokens = T;
            d.max_rows = (int64_t) T * K;
            d.n_expert = E;
            d.launch_experts = c.experts;
            d.launch_rows = c.rows;
            basalt::prefill::PeerShare share;
            std::string err;
            if (!share.open(0, 1, d, err)) throw std::runtime_error(err);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int rep = 0; rep < 3; ++rep) {   // the last one timed: the whole share, transport included
                cudaMemsetAsync(ddst, 0, (size_t) T * K * N * 4, st);
                cudaEventRecord(a, st);
                if (!share.run(L, dx, T, ex, ddst, st, err)) throw std::runtime_error(err);
                if (!share.join(st, err)) throw std::runtime_error(err);
                cudaEventRecord(b, st);
            }
            ck(cudaStreamSynchronize(st), "share");
            float t = 0;
            cudaEventElapsedTime(&t, a, b);
            if (c.experts == 8) peer_ms = t;
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            // the share's rows lie expert by expert; put each at its pair
            std::vector<float> block((size_t) T * K * N), got((size_t) T * K * N);
            cudaMemcpy(block.data(), ddst, block.size() * 4, cudaMemcpyDeviceToHost);
            size_t r = 0;
            for (int e = 0; e < E; ++e)
                for (int32_t p : rt.pairs[(size_t) e]) std::memcpy(got.data() + (size_t) p * N, block.data() + N * r++, (size_t) N * 4);
            const Err m = compare(got, ref, rt);
            std::printf("peer share (%d experts, %lld rows a launch) vs main card MMQ: %lld/%lld rows bitwise, max abs %.3g, max rel %.3g\n",
                        c.experts, (long long) c.rows, (long long) m.same, (long long) m.n, m.max_abs, m.max_rel);
            if (m.max_rel > 1e-2) ++fail;
            if (first.empty()) {
                first = got;
            } else {
                const Err s = compare(got, first, rt);
                std::printf("peer share, launches cut two ways: %lld/%lld rows bitwise, max rel %.3g\n", (long long) s.same,
                            (long long) s.n, s.max_rel);
                if (s.max_rel > 1e-2) ++fail;
            }
        }

        // a double-precision reference on a sample of rows (FP32 activations, ggml's dequantized weights)
        double worst = 0;
        for (int e = 0; e < E; e += 5)
            for (size_t i = 0; i < rt.pairs[(size_t) e].size(); i += 97) {
                const int32_t p = rt.pairs[(size_t) e][i];
                const float* xr = x.data() + (size_t) (p / K) * N;
                std::vector<double> hh(FF);
                for (int k = 0; k < FF; ++k) {
                    double gg = 0, uu = 0;
                    for (int c = 0; c < N; ++c) {
                        gg += (double) wgu[(size_t) e][(size_t) k * N + c] * xr[c];
                        uu += (double) wgu[(size_t) e][(size_t) (FF + k) * N + c] * xr[c];
                    }
                    hh[(size_t) k] = gg / (1.0 + std::exp(-gg)) * uu;
                }
                double nrm = 0, dmax = 0;
                std::vector<double> o(N);
                for (int r = 0; r < N; ++r) {
                    double s = 0;
                    for (int k = 0; k < FF; ++k) s += (double) wdn[(size_t) e][(size_t) r * FF + k] * hh[(size_t) k];
                    o[(size_t) r] = s;
                    nrm = std::max(nrm, std::fabs(s));
                }
                for (int r = 0; r < N; ++r) dmax = std::max(dmax, std::fabs(o[(size_t) r] - (double) first[(size_t) p * N + r]));
                worst = std::max(worst, dmax / nrm);
            }
        std::printf("peer share vs double-precision reference (q8_1 activations against FP32): max rel %.3g\n", worst);
        if (worst > 0.05) ++fail;

        // the products' rates: main card products alone; the second card's whole share (copies in and out included)
        const double macs = (double) R * 3.0 * N * FF;
        const basalt::core::DeviceRates r0 = basalt::core::device_rates(0), r1 = basalt::core::device_rates(1);
        std::printf("main card products: %.0f us for %lld rows, %.1f TFLOP/s, %.0f MACs per SM-cycle\n", main_ms * 1e3,
                    (long long) R, 2 * macs / (main_ms * 1e-3) / 1e12, r0.sm_hz > 0 ? macs / (main_ms * 1e-3) / r0.sm_hz : 0.0);
        std::printf("second card share: %.0f us for %lld rows, %.1f TFLOP/s, %.0f MACs per SM-cycle (transport included)\n",
                    peer_ms * 1e3, (long long) R, 2 * macs / (peer_ms * 1e-3) / 1e12,
                    r1.sm_hz > 0 ? macs / (peer_ms * 1e-3) / r1.sm_hz : 0.0);
        std::printf("links: main %.1f GB/s, second %.1f GB/s (driver facts)\n", r0.link_bytes / 1e9, r1.link_bytes / 1e9);
        if (fail) {
            std::printf("peer_share_parity: FAIL\n");
            return 1;
        }
        std::printf("peer_share_parity: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("peer_share_parity: error: %s\n", e.what());
        return 1;
    }
}
