#pragma once

#include "common.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include <atomic>
#include <vector>

namespace basalt::core {

// The CPU tier's cache warm-up.  After layer l's doorbell the parked pool workers score layer l + 1's router on layer
// l's FFN input (the ring's q8_1 rows; the router in 4-bit with one scale per 32 values, a prediction only: half the
// bytes of an 8-bit copy, so the scoring leaves more of the gap to the read), take every token's top k, rank the
// predicted experts only the CPU holds by their summed weight, and read the likeliest one into their caches.  The next
// batch stops the work; the experts are computed as before.
struct CpuWarm {
    std::vector<uint8_t> rq;  ///< [n_layers][n_expert][n_embd / 2]: per 32 values, value i in the low nibble of byte i
                              ///< and value i + 16 in its high nibble, each the code plus 8
    std::vector<float> rs;    ///< [n_layers][n_expert][n_embd / 32]
    int n_expert = 0, n_embd = 0, k = 0;
    size_t bytes = 0, down_off = 0;   ///< a blob's bytes, where its down rows start
    ExpertSource* src = nullptr;
    // the job in flight
    int layer = 0, n_tok = 0;
    const uint8_t* xq = nullptr;                      ///< the ring's q8_1 rows, tokens back to back
    const int32_t* res0 = nullptr, *res1 = nullptr;   ///< the layer's CUDA0 / peer residency (res1 may be null)
    std::vector<float> logits;                        ///< [kVerifyMaxT][n_expert]
    alignas(64) std::atomic<int> arrive{0};
    alignas(64) std::atomic<int> ready{0};
    const uint8_t* blob = nullptr;                    ///< the expert to read, or null
    std::atomic<uint64_t> sink{0};
};

// the MoE tail (dec_moe.hpp): planned on CUDA0 from device residency tables, the peer card served from mapped rings
// by one persistent kernel per window, the CPU pool by the host loop (only its own entries).

namespace verify_detail {
void warm_job(void* ctx, int w, int W, const std::atomic<bool>& stop);
bool router_codes(const WeightTable& wt, const ModelGeometry& g, std::vector<int8_t>* q8, std::vector<float>* s8,
                  std::vector<uint8_t>* q4, std::vector<float>* s4);
bool warm_router(const std::vector<int8_t>& q, const std::vector<float>& sc, int8_t** rq, float** rs);
}

}  // namespace basalt::core
