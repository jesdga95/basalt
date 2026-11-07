// src/models/deepseek41/prompt_host.cpp - see prompt_host.hpp.
#include "models/deepseek41/prompt_host.hpp"

#include "basalt/core/peer_experts.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/models/deepseek41/kernels.hpp"
#include "basalt/models/deepseek41/vision.hpp"
#include "basalt/platform/memory.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/prefill_plan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace basalt::models::deepseek41 {

uint64_t staged_stride(const ExpertLayer& e) {
    uint64_t unit = 1;
    for (int t : {e.gu_type, e.d_type}) {
        const uint64_t bs = ggml_type_size((ggml_type) t);
        unit = unit / std::gcd(unit, bs) * bs;
    }
    return (e.blob_bytes + unit - 1) / unit * unit;
}

PromptHost::~PromptHost() {
    for (void* p : {(void*) ring, (void*) stage[0], (void*) stage[1], (void*) hx, (void*) hrows, (void*) xb})
        if (p) cudaFreeHost(p);
    platform::free_aligned(act);
}

bool PromptHost::open(const DsModel& m, int pieces, int slots, bool vision, std::string& err) {
    const Geometry& g = m.geo();
    const size_t n = (size_t) g.n_embd;
    for (const ExpertLayer& e : m.pack().experts) slot = std::max<size_t>(slot, (size_t) staged_stride(e));
    share_rows = (int) ((prefillp::kAheadPairs + g.n_used - 1) / g.n_used);   // a read past this stages its groups ahead
    auto pin = [&](auto*& p, size_t bytes, unsigned flags) {
        if (bytes == 0) return true;
        void* v = nullptr;
        if (cudaHostAlloc(&v, bytes, flags) != cudaSuccess) return false;
        p = (std::remove_reference_t<decltype(p)>) v;
        pinned_ += bytes;
        return true;
    };
    constexpr unsigned kPortable = cudaHostAllocPortable, kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
    ring_pieces = pieces;
    bool ok = pin(ring, (size_t) pieces * slot, kPortable) &&
              pin(stage[0], (size_t) prefillp::kStageExperts * slot, kPortable) &&
              pin(stage[1], (size_t) prefillp::kStageExperts * slot, kPortable) &&
              pin(hx, (size_t) share_rows * n * 4, kPortable) && pin(hrows, (size_t) prefillp::kShareRows * n * 4, kMapped) &&
              pin(xb, (size_t) core::PeerExperts::kRowBatch * n * 4, kPortable);
    act = ok ? (uint8_t*) platform::alloc_aligned(kernels::cpu::kNativeActBytes * (size_t) share_rows, 64) : nullptr;
    if (!ok || !act) {
        cudaGetLastError();
        err = "deepseek41: out of host memory for the prompt path's buffers";
        return false;
    }
    bytes_ = pinned_ + kernels::cpu::kNativeActBytes * (size_t) share_rows;
    // pages written once now, so the free memory the tier reads already lacks them
    auto take = [&](auto& v, size_t count) {
        v.assign(count, 0);
        bytes_ += count * sizeof(v[0]);
    };
    take(embed, (size_t) prefillp::kEmbedRows * kHcMult * n);
    const EngramInfo& en = m.pack().engram;
    if (!en.layers.empty()) {
        const size_t cols = (size_t) en.hash.n_cols();
        take(engram_rows, (size_t) prefillp::kEngramRows * cols * en.row_values);
        engram_layer = (size_t) prefillp::kMaxChunk * cols * en.row_bytes;
        for (std::vector<uint8_t>& r : engram_raw) take(r, engram_layer * en.layers.size());
    }
    if (vision) {
        images_.resize((size_t) slots);
        for (Image& im : images_) {
            take(im.picture.rgb, (size_t) vision::kMaxImagePixels * 3);
            const size_t merge = (size_t) VisionPlan{}.merge;
            take(im.patches, (size_t) kVisionMaxTokens * merge * merge * kVisionPatchValues);
        }
    }
    return true;
}

bool PromptHost::set_ring(int pieces, std::string& err) {
    if (pieces == ring_pieces) return true;
    if (ring) cudaFreeHost(ring);
    pinned_ -= (uint64_t) ring_pieces * slot;
    bytes_ -= (uint64_t) ring_pieces * slot;
    ring = nullptr;
    ring_pieces = 0;
    if (pieces > 0 && cudaHostAlloc((void**) &ring, (size_t) pieces * slot, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        ring = nullptr;
        err = "deepseek41: out of host memory for the prompt path's ring";
        return false;
    }
    ring_pieces = pieces;
    pinned_ += (uint64_t) pieces * slot;
    bytes_ += (uint64_t) pieces * slot;
    return true;
}

PromptHost::Image* PromptHost::take_image() {
    return images_taken_ < (int) images_.size() ? &images_[(size_t) images_taken_++] : nullptr;
}

}  // namespace basalt::models::deepseek41
