// tests/kernels/gdn_seg_split_parity.cpp - dec_gdn_gates, then dec_gdn_rec_seg_split (split kernel per segment, then
// the norm), against dec_gdn_rec_seg (a cluster per head) bitwise: h, gates, betas, the raw qkv channels, y, y16 and the q8_1 rows, over
// segment mixes up to the batch window's 16 rows, with and without SegDst, both widths and both output gates, and the
// norm in place (a.o == a.y).  dec_gdn_rec_seg is itself bitwise to dec_gdn_rec per segment (dec_gdn_parity).
// Exit 77 without a GPU.
#include "basalt/kernels/gdn_window.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::kernels;

namespace {

std::mt19937 rng(7);
std::normal_distribution<float> nd(0.f, 1.f);
cudaStream_t S0;
int fails = 0;

template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> randn(size_t n, float sd) {
    std::vector<float> v(n);
    for (auto& x : v) x = sd * nd(rng);
    return v;
}
std::vector<uint16_t> randbf(size_t n, float sd) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) {
        const float f = sd * nd(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = (uint16_t) (u >> 16);
    }
    return v;
}
void* zeros(size_t bytes) {
    void* d = nullptr;
    cudaMalloc(&d, bytes);
    cudaMemset(d, 0xff, bytes);   // unwritten bytes compare too
    return d;
}
bool same(const char* what, const void* a, const void* b, size_t bytes) {
    std::vector<unsigned char> x(bytes), y(bytes);
    cudaStreamSynchronize(S0);
    cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost);
    size_t d = 0;
    for (size_t i = 0; i < bytes; ++i) d += x[i] != y[i];
    if (d) std::printf("  %s: %zu bytes differ\n", what, d);
    return d == 0;
}

struct Geo {
    int n_embd, h_k, h_v;
};

void run(const Geo& g, const std::vector<int>& seg, bool use_dst, GdnGate gate, bool in_place) {
    const int SS = 128, C = 2 * g.h_k * SS + g.h_v * SS, VD = g.h_v * SS, E = g.n_embd;
    int rows = 0, max_n = 0;
    for (int n : seg) { rows += n; max_n = std::max(max_n, n); }
    const size_t sf = (size_t) SS * g.h_v * SS + (size_t) C * 3;
    std::vector<float*> states;
    std::vector<SegRow> first;
    for (size_t k = 0, r0 = 0; k < seg.size(); r0 += seg[k++]) {
        states.push_back(up(randn(sf, 0.05f)));
        first.push_back(SegRow{states.back(), (int32_t) r0, 0, seg[k], seg[k], (int32_t) k, 0});
    }
    SegRow* d_rows = up(first);
    DecRecArgs a;
    a.qkv = up(randn((size_t) rows * C, 1.f));
    a.conv_w = up(randn((size_t) C * 4, 0.5f));
    a.x = up(randn((size_t) rows * E, 1.f));
    const std::vector<uint16_t> wa = randbf((size_t) g.h_v * E, 0.02f), wb = randbf((size_t) g.h_v * E, 0.02f);
    a.w_alpha = up(wa);
    a.w_beta = up(wb);
    a.dt = up(randn(g.h_v, 0.5f));
    a.ssm_a = up(randn(g.h_v, 0.5f));
    a.z = up(randn((size_t) rows * VD, 1.f));
    a.gamma = up(randn(SS, 1.f));
    a.C = C; a.n_embd = E; a.h_k = g.h_k; a.h_v = g.h_v; a.eps = 1e-6f; a.gate_act = gate;
    // one set of outputs per path
    struct Out {
        float *h, *gate, *beta, *y, *o, *qkv;
        uint16_t* y16;
        void* yq;
    } o[2];
    const size_t yq_bytes = (size_t) rows * (VD / 32) * 36;
    for (Out& x : o) {
        x.h = (float*) zeros((size_t) 2 * kDecMaxT * C * 4);
        x.gate = (float*) zeros((size_t) 2 * kDecMaxT * g.h_v * 4);
        x.beta = (float*) zeros((size_t) 2 * kDecMaxT * g.h_v * 4);
        x.qkv = (float*) zeros((size_t) 2 * kDecMaxT * C * 4);
        x.y = (float*) zeros((size_t) rows * VD * 4);
        x.o = in_place ? x.y : (float*) zeros((size_t) rows * VD * 4);
        x.y16 = (uint16_t*) zeros((size_t) rows * VD * 2);
        x.yq = zeros(yq_bytes);
    }
    // with SegDst, segment k's rows go to its own stretch (row 2 k kDecMaxT / seg size .. ) of each buffer
    std::vector<SegDst> dh[2];
    SegDst* dd[2] = {nullptr, nullptr};
    float* gates = (float*) zeros((size_t) 2 * rows * g.h_v * 4);
    for (int p = 0; p < 2; ++p) {
        if (!use_dst) continue;
        for (size_t k = 0; k < seg.size(); ++k) {
            const size_t r = k * (2 * kDecMaxT / seg.size());
            dh[p].push_back(SegDst{o[p].qkv + r * C, o[p].h + r * C, o[p].gate + r * g.h_v, o[p].beta + r * g.h_v});
        }
        dd[p] = up(dh[p]);
    }
    for (int p = 0; p < 2; ++p) {
        DecRecArgs b = a;
        b.h = o[p].h; b.gate = o[p].gate; b.beta = o[p].beta; b.y = o[p].y; b.y16 = o[p].y16; b.yq = o[p].yq;
        if (p == 0) dec_gdn_rec_seg(b, d_rows, 0, (int) seg.size(), max_n, S0, dd[0]);
        else {
            b.o = o[p].o;
            dec_gdn_gates(b, rows, gates, S0);
            dec_gdn_rec_seg_split(b, d_rows, 0, (int) seg.size(), max_n, rows, S0, dd[1], 0, 0, gates);
        }
    }
    bool ok = cudaStreamSynchronize(S0) == cudaSuccess;
    for (int p = 1; p < 2; ++p) {
    ok = same("h", o[0].h, o[p].h, (size_t) 2 * kDecMaxT * C * 4) && ok;
    ok = same("gate", o[0].gate, o[p].gate, (size_t) 2 * kDecMaxT * g.h_v * 4) && ok;
    ok = same("beta", o[0].beta, o[p].beta, (size_t) 2 * kDecMaxT * g.h_v * 4) && ok;
    ok = same("qkv", o[0].qkv, o[p].qkv, (size_t) 2 * kDecMaxT * C * 4) && ok;
    ok = same("y", o[0].y, o[p].y, (size_t) rows * VD * 4) && ok;
    ok = same("y16", o[0].y16, o[p].y16, (size_t) rows * VD * 2) && ok;
    ok = same("yq", o[0].yq, o[p].yq, yq_bytes) && ok;
    }
    std::printf("n_embd %d segments", E);
    for (int n : seg) std::printf(" %d", n);
    std::printf("%s%s%s: %s\n", use_dst ? ", dst" : "", gate == GdnGate::Silu ? ", silu" : "", in_place ? ", in place" : "",
                ok ? "bitwise" : "FAIL");
    if (!ok) ++fails;
    for (float* p : states) cudaFree(p);
    for (const void* p : {(const void*) d_rows, (const void*) a.qkv, (const void*) a.conv_w, (const void*) a.x, (const void*) a.w_alpha,
                          (const void*) a.w_beta, (const void*) a.dt, (const void*) a.ssm_a, (const void*) a.z, (const void*) a.gamma,
                          (const void*) dd[0], (const void*) dd[1], (const void*) gates})
        cudaFree(const_cast<void*>(p));
    for (Out& x : o) {
        for (void* p : {(void*) x.h, (void*) x.gate, (void*) x.beta, (void*) x.qkv, (void*) x.y, (void*) x.y16, x.yq}) cudaFree(p);
        if (!in_place) cudaFree(x.o);
    }
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) return 77;
    cudaStreamCreate(&S0);
    const Geo big{5120, 16, 48}, small{2560, 16, 32};
    const std::vector<std::vector<int>> mixes = {{1, 1}, {5, 3}, {8, 8}, {2, 14}, {16}, {4, 4, 4}, {9, 7}};
    for (const Geo& g : {big, small})
        for (const auto& m : mixes) run(g, m, false, GdnGate::Sigmoid, false);
    for (const auto& m : mixes) run(big, m, true, GdnGate::Silu, true);
    run(small, {6, 3}, true, GdnGate::Sigmoid, true);
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
