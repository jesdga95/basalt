// tests/kernels/window_gemm_two_gpu.cpp - the window gemm on a second card planned as the main card (plan_device): every
// call a decode window of Qwen3.8-27B layers makes (NVFP4 planes: the GDN and attention groups, their AddBf16 outputs, the
// SwiGLU pair with its bf16 rows, the down product), at T = 1..16, gives the main card's bits.  The same calls planned by
// the second card's own SM count are reported (they may round differently).  Exit 77 without two GPUs.
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/kernels/window_gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace wg = basalt::kernels::wgemm;
namespace np = basalt::kernels::nvfp4p;

namespace {

constexpr int R = wg::kRows;

struct Case {
    const char* name;
    int64_t rows[3];
    int n;
    int64_t K;
    wg::Epilogue epi;
};
const Case kCases[] = {
    {"GDN qkv+z 10240,6144 x 5120", {10240, 6144}, 2, 5120, wg::Epilogue::Store},
    {"GDN out 5120 x 6144 +bf16", {5120}, 1, 6144, wg::Epilogue::AddBf16},
    {"attn q+k+v 12288,1024,1024 x 5120", {12288, 1024, 1024}, 3, 5120, wg::Epilogue::Store},
    {"FFN gate/up SwiGLU 17408 x 5120", {17408, 17408}, 2, 5120, wg::Epilogue::SwiGLU},
    {"FFN down 5120 x 17408 +bf16", {5120}, 1, 17408, wg::Epilogue::AddBf16},
};

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// one card's copy of a case's inputs and outputs
struct OnCard {
    int dev = 0;
    std::vector<void*> owned;
    wg::Call c;
    void* act = nullptr;
    void* act_out = nullptr;
    std::vector<float*> y;
    uint16_t* y16 = nullptr;
    wg::Workspace ws;
    template <class T> T* up(const std::vector<T>& h) {
        void* d = nullptr;
        cudaMalloc(&d, h.size() * sizeof(T));
        cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
        owned.push_back(d);
        return (T*) d;
    }
    void* alloc(size_t bytes) {
        void* d = nullptr;
        cudaMalloc(&d, bytes);
        owned.push_back(d);
        return d;
    }
    ~OnCard() {
        cudaSetDevice(dev);
        for (void* p : owned) cudaFree(p);
    }
};

// the case's outputs for rows 0..T-1 hashed, on the current device
uint64_t run(OnCard& oc, const Case& cs, const std::vector<float>& x, const std::vector<uint16_t>& resid, int T) {
    std::vector<float> xt((size_t) R * cs.K, 0.0f);
    std::memcpy(xt.data(), x.data(), (size_t) T * cs.K * sizeof(float));
    float* d_x = oc.up(xt);
    wg::pack(d_x, cs.K, cs.K, T, oc.act, nullptr);
    if (oc.y16) cudaMemcpy(oc.y16, resid.data(), resid.size() * 2, cudaMemcpyHostToDevice);
    for (float* y : oc.y)
        if (y) cudaMemset(y, 0, (size_t) R * 17408 * 4);
    wg::gemm(oc.c, oc.act, T, oc.ws, nullptr);
    cudaDeviceSynchronize();
    uint64_t h = 1469598103934665603ull;
    std::vector<uint8_t> buf;
    for (int i = 0; i < oc.c.n; ++i) {
        if (!oc.y[(size_t) i]) continue;
        buf.resize((size_t) T * cs.rows[i] * 4);
        cudaMemcpy(buf.data(), oc.y[(size_t) i], buf.size(), cudaMemcpyDeviceToHost);
        h = fnv(buf.data(), buf.size(), h);
    }
    if (oc.y16) {
        buf.resize((size_t) T * cs.rows[0] * 2);
        cudaMemcpy(buf.data(), oc.y16, buf.size(), cudaMemcpyDeviceToHost);
        h = fnv(buf.data(), buf.size(), h);
    }
    if (oc.act_out) {
        buf.resize((size_t) T * cs.rows[0] * 2);
        cudaMemcpy(buf.data(), oc.act_out, buf.size(), cudaMemcpyDeviceToHost);
        h = fnv(buf.data(), buf.size(), h);
    }
    return h;
}

void setup(OnCard& oc, int dev, const Case& cs, const std::vector<std::vector<uint8_t>>& mats) {
    oc.dev = dev;
    cudaSetDevice(dev);
    oc.c.fmt = wg::Format::Nvfp4;
    oc.c.K = cs.K;
    oc.c.n = cs.n;
    oc.c.epi = cs.epi;
    const float* sc = oc.up(std::vector<float>{0.03f});
    for (int i = 0; i < cs.n; ++i) {
        float* y = cs.epi == wg::Epilogue::AddBf16 || (cs.epi == wg::Epilogue::SwiGLU && i == 1)
                       ? nullptr
                       : (float*) oc.alloc((size_t) R * 17408 * 4);
        oc.y.push_back(y);
        oc.c.m[i] = {oc.up(mats[(size_t) i]), cs.rows[i], sc, y, cs.rows[i]};
    }
    if (cs.epi == wg::Epilogue::AddBf16) {
        oc.y16 = (uint16_t*) oc.alloc((size_t) R * cs.rows[0] * 2);
        oc.c.m[0].y16 = oc.y16;
    }
    if (cs.epi == wg::Epilogue::SwiGLU) oc.c.act_out = oc.act_out = oc.alloc(wg::act_bytes(cs.rows[0]));
    oc.act = oc.alloc(wg::act_bytes(cs.K));
    size_t p, t;
    wg::workspace_need(oc.c, p, t);
    oc.ws.partial_floats = p;
    oc.ws.tickets_n = t;
    oc.ws.partial = (float*) oc.alloc(std::max<size_t>(p, 1) * 4);
    oc.ws.tickets = (int32_t*) oc.alloc(std::max<size_t>(t, 1) * 4);
    cudaMemset(oc.ws.tickets, 0, std::max<size_t>(t, 1) * 4);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) {
        std::printf("two GPUs needed; skipped\n");
        return 77;
    }
    int sms[2] = {};
    for (int d = 0; d < 2; ++d) cudaDeviceGetAttribute(&sms[d], cudaDevAttrMultiProcessorCount, d);
    std::printf("CUDA0 %d SMs, CUDA1 %d SMs\n", sms[0], sms[1]);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    int fails = 0, own_diff = 0, own_total = 0;
    for (const Case& cs : kCases) {
        std::vector<std::vector<uint8_t>> mats;
        for (int i = 0; i < cs.n; ++i) {
            std::vector<uint8_t> m(np::bytes(cs.rows[i], cs.K));
            const size_t so = np::scale_offset(cs.rows[i], cs.K);
            for (size_t j = 0; j < so; ++j) m[j] = (uint8_t) rng();
            for (size_t j = so; j < m.size(); ++j) m[j] = (uint8_t) (rng() % 0x7f);
            mats.push_back(std::move(m));
        }
        std::vector<float> x((size_t) R * cs.K);
        for (float& v : x) v = nd(rng);
        std::vector<uint16_t> resid((size_t) R * cs.rows[0]);
        for (uint16_t& v : resid) v = (uint16_t) (0x3c00 + rng() % 0x0400);   // bf16 values of about unit size
        int match = 0, own = 0;
        for (int T = 1; T <= R; ++T) {
            wg::plan_device(-1);
            OnCard a;
            setup(a, 0, cs, mats);
            const uint64_t h0 = run(a, cs, x, resid, T);
            wg::plan_device(0);   // the second card planned as the first
            OnCard b;
            setup(b, 1, cs, mats);
            const uint64_t h1 = run(b, cs, x, resid, T);
            wg::plan_device(-1);   // and by its own SM count
            OnCard c;
            setup(c, 1, cs, mats);
            const uint64_t h2 = run(c, cs, x, resid, T);
            match += h0 == h1;
            own += h0 != h2;
            if (h0 != h1) std::printf("  %s T=%d: CUDA0 %016llx, CUDA1 planned as CUDA0 %016llx  FAIL\n", cs.name, T,
                                      (unsigned long long) h0, (unsigned long long) h1);
        }
        std::printf("%-36s planned as CUDA0: %2d of %d windows bitwise; own plan differs in %d\n", cs.name, match, R, own);
        fails += R - match;
        own_diff += own;
        own_total += R;
    }
    std::printf("CUDA1 on its own plan: %d of %d windows differ from CUDA0\n", own_diff, own_total);
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
