// src/models/hy4flash/cpu_experts.cpp - the routed experts of one token on the CPU.
#include "models/hy4flash/cpu_experts.hpp"

#include "basalt/ggml/ggml.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace basalt::models::hy4flash {

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) pool_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    {
        std::lock_guard<std::mutex> g(m_);
        stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : pool_) t.join();
}

void Workers::loop() {
    int seen = 0;
    for (;;) {
        const std::function<void(int)>* job;
        int n;
        {
            std::unique_lock<std::mutex> l(m_);
            wake_.wait(l, [&] { return stop_ || epoch_ != seen; });
            if (stop_) return;
            seen = epoch_;
            job = job_;
            n = n_;
            ++busy_;
        }
        for (int i; (i = next_.fetch_add(1)) < n;) (*job)(i);
        {
            std::lock_guard<std::mutex> g(m_);
            --busy_;
        }
        idle_.notify_all();
    }
}

void Workers::run(int n, const std::function<void(int)>& job) {
    {
        std::lock_guard<std::mutex> g(m_);
        job_ = &job;
        n_ = n;
        next_.store(0);
        ++epoch_;
    }
    wake_.notify_all();
    for (int i; (i = next_.fetch_add(1)) < n;) job(i);
    std::unique_lock<std::mutex> l(m_);
    idle_.wait(l, [&] { return busy_ == 0; });
}

namespace {

constexpr int kRowsPerJob = 32;

struct Dot {
    ggml_type type;
    ggml_vec_dot_t dot;
    ggml_from_float_t quant;
    ggml_type act_type;
    size_t row_bytes;
    size_t act_bytes;
};

Dot dot_for(int type, int n) {
    const ggml_type_traits_cpu* t = ggml_get_type_traits_cpu((ggml_type) type);
    if (!t || !t->vec_dot) throw std::runtime_error("no CPU dot product for ggml type " + std::to_string(type));
    const ggml_type_traits_cpu* a = ggml_get_type_traits_cpu(t->vec_dot_type);
    if (!a || !a->from_float) throw std::runtime_error("no activation quantizer for ggml type " + std::to_string(type));
    return {(ggml_type) type, t->vec_dot, a->from_float, t->vec_dot_type, ggml_row_size((ggml_type) type, n),
            ggml_row_size(t->vec_dot_type, n)};
}

}  // namespace

void cpu_experts(Workers& w, const ExpertBlob* ex, const float* weights, int k, int n, float limit, const float* x,
                 float* out) {
    // every expert of a layer shares its types
    const Dot gu = dot_for(ex[0].gu_type, n), dn = dot_for(ex[0].d_type, n);
    if (gu.act_type != dn.act_type) throw std::runtime_error("the gate/up and down types need different activations");
    thread_local std::vector<uint8_t> xq, hq;
    thread_local std::vector<float> g, u, d;
    xq.resize(gu.act_bytes);
    hq.resize(dn.act_bytes * (size_t) k);
    g.resize((size_t) k * n);
    u.resize((size_t) k * n);
    d.resize((size_t) k * n);
    gu.quant(x, xq.data(), n);
    const size_t mat = (size_t) n * gu.row_bytes;
    const int jobs_per = (n + kRowsPerJob - 1) / kRowsPerJob;
    uint8_t* xqp = xq.data();
    float *gp = g.data(), *up = u.data(), *dp = d.data();
    uint8_t* hqp = hq.data();
    w.run(k * 2 * jobs_per, [&](int j) {
        const int e = j / (2 * jobs_per), m = (j / jobs_per) & 1, r0 = (j % jobs_per) * kRowsPerJob;
        const uint8_t* base = ex[e].bytes + (size_t) m * mat;
        float* dst = (m ? up : gp) + (size_t) e * n;
        for (int r = r0; r < std::min(n, r0 + kRowsPerJob); ++r)
            gu.dot(n, dst + r, 0, base + (size_t) r * gu.row_bytes, 0, xqp, 0, 1);
    });
    for (int e = 0; e < k; ++e) {
        float* h = gp + (size_t) e * n;
        const float* uu = up + (size_t) e * n;
        for (int r = 0; r < n; ++r) {
            const float gv = std::min(h[r], limit), uv = std::min(std::max(uu[r], -limit), limit);
            h[r] = gv / (1.f + std::exp(-gv)) * uv;
        }
        dn.quant(h, hqp + (size_t) e * dn.act_bytes, n);
    }
    w.run(k * jobs_per, [&](int j) {
        const int e = j / jobs_per, r0 = (j % jobs_per) * kRowsPerJob;
        const uint8_t* base = ex[e].bytes + 2 * mat;
        for (int r = r0; r < std::min(n, r0 + kRowsPerJob); ++r)
            dn.dot(n, dp + (size_t) e * n + r, 0, base + (size_t) r * dn.row_bytes, 0, hqp + (size_t) e * dn.act_bytes, 0, 1);
    });
    for (int r = 0; r < n; ++r) {
        float acc = 0.f;
        for (int e = 0; e < k; ++e) acc += weights[e] * dp[(size_t) e * n + r];
        out[r] = acc;
    }
}

}  // namespace basalt::models::hy4flash
