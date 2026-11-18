// src/models/hy4flash/cpu_experts.hpp - the routed experts of one token on the CPU: each expert is three matrices of
// [n][n] (gate, up, down) in ggml K-quants, run with ggml's own row dot products over a Q8_K activation.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::hy4flash {

/// Worker threads that run `n` jobs of a function, the caller joining them.
class Workers {
public:
    explicit Workers(int threads);
    ~Workers();
    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;
    void run(int n, const std::function<void(int)>& job);
    int threads() const { return (int) pool_.size() + 1; }

private:
    void loop();
    std::vector<std::thread> pool_;
    std::mutex m_;
    std::condition_variable wake_, idle_;
    const std::function<void(int)>* job_ = nullptr;
    std::atomic<int> next_{0};
    int n_ = 0, epoch_ = 0, busy_ = 0;
    bool stop_ = false;
};

struct ExpertBlob {
    const uint8_t* bytes = nullptr;   ///< [gate | up | down]
    int gu_type = 0, d_type = 0;      ///< ggml types of gate and up, and of down
};

/// out[r] = sum_k weights[k] * down_k(silu(min(gate_k . x, limit)) * clamp(up_k . x, -limit, limit))[r] for r < n, over
/// k = 1..16 experts, in that order. x: n floats.
void cpu_experts(Workers& w, const ExpertBlob* experts, const float* weights, int k, int n, float limit, const float* x,
                 float* out);

}  // namespace basalt::models::hy4flash
