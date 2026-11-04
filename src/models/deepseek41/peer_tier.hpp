// src/models/deepseek41/peer_tier.hpp - the routed experts a second card holds.
//
// The card computes its experts with the main card's kernels from the same activation, so an expert gives the same bits
// on either card. The host is the only link between the cards: the activation goes up from pinned memory and the rows
// land in the mapped rows the weighted sum reads, as the CPU's rows do. A prompt reads the card's experts back through
// the host, since its products run on the main card.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class PeerTier {
public:
    /// One layer's share as it is launched: entry i of `blob[0..n)` (device addresses on this card) writes row `row[i]`.
    struct Share {
        const uint8_t* blob[8] = {};
        int32_t row[8] = {};
        int n = 0;
    };

    PeerTier() = default;
    ~PeerTier();
    PeerTier(const PeerTier&) = delete;
    PeerTier& operator=(const PeerTier&) = delete;

    /// Takes the card's buffers and streams. `h_x`: pinned, portable host memory holding the layer's activation;
    /// `h_rows`: pinned, portable, mapped host rows (n_embd floats each) the card writes.
    bool open(int device, int n_layers, int n_expert, int n_embd, int n_ff, const float* h_x, float* h_rows, std::string& err);
    /// The device bytes free for expert slots.
    uint64_t room() const;
    /// Slots of the given sizes; each layer's first slots hold what `admit` gives them.
    bool slots(const std::vector<int64_t>& slot_bytes, std::string& err);
    void admit(int layer, int e, int32_t slot) { res_[(size_t) layer * n_expert_ + e] = slot; }
    /// Copies `bytes` from pinned host memory into a slot and waits (any thread).
    bool fill(int32_t slot, const uint8_t* src, uint64_t bytes);

    int device() const { return device_; }
    double gib() const { return cache_.gib(); }
    /// The device address of (layer, e) on this card, or null.
    const uint8_t* blob(int layer, int e) const {
        const int32_t s = res_[(size_t) layer * n_expert_ + e];
        return s >= 0 ? cache_.device_slot(s) : nullptr;
    }

    /// Starts the layer's share over the activation now in `h_x`.
    bool launch(const kernels::NativeExpertLayout& L, float limit, const Share& share, std::string& err);
    /// Waits for the launched share's rows.
    bool finish(std::string& err);

    /// Queues a copy of a blob of this card into pinned, portable host memory; `read_wait` waits for those queued.
    void read(const uint8_t* blob, uint8_t* dst, uint64_t bytes);
    bool read_wait(std::string& err);

private:
    struct Plan {
        unsigned long long ptr[8];
        int32_t start[16];
        int32_t ng[4];
        int32_t dst[8];
        int32_t tok[8];
        int32_t row[8];
    };
    int device_ = -1, n_embd_ = 0, n_expert_ = 0;
    core::ExpertCache cache_;
    std::vector<int32_t> res_;              ///< [layer * n_expert + e] slot or -1
    cudaStream_t stream_ = nullptr;         ///< the layers' shares
    cudaStream_t read_ = nullptr;           ///< a prompt's reads
    const float* h_x_ = nullptr;
    float* d_rows_ = nullptr;               ///< the mapped rows as this card addresses them
    Plan* h_plan_ = nullptr;
    Plan* d_plan_ = nullptr;
    float *d_x_ = nullptr, *d_out_ = nullptr;
    void *d_xq_ = nullptr, *d_scr_ = nullptr;
    bool launched_ = false;
};

}  // namespace basalt::models::deepseek41
