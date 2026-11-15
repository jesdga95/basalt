// include/basalt/core/conversation_stream.hpp - a conversation's state read where it is, piece by piece, without a
// whole copy of it in RAM.
//
// A write to the disk tier reads a ConversationSource: the tokens, pictures and checkpoints by reference, the live
// running state and the K/V as readers of the buffers that hold them, copied out a write buffer at a time.
#pragma once

#include "basalt/core/conversation_cache.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace basalt::core {

/// Bytes read piece by piece: read(offset, dst, n) copies bytes [offset, offset + n) into dst (false when the copy
/// failed); view(offset, n), when set, gives them in place if they are contiguous there (else null).
struct ConversationBytes {
    uint64_t size = 0;
    std::function<bool(uint64_t, uint8_t*, uint64_t)> read;
    std::function<const uint8_t*(uint64_t, uint64_t)> view;
};

/// A K/V layer as a ConversationKv records it, its five buffers (k, v, k scales, v scales, pooled rows) as readers.
struct ConversationKvSource {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<ConversationBytes, 5> parts;
};

/// A conversation to write where it is.  What it refers to and the buffers its readers copy from stay unchanged while
/// it is used.
struct ConversationSource {
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = true;
    const std::vector<int32_t>* ids = nullptr;
    const std::vector<ConversationImageKey>* imgs = nullptr;
    std::vector<const ConversationCheckpoint*> checkpoints;
    std::array<ConversationBytes, 5> live;   ///< the live running state, in the order of a checkpoint's parts
    std::vector<ConversationKvSource> kv;    ///< main layers followed by the draft layer

    /// The state it holds: the live running state, the K/V and the checkpoints' states.
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const auto& p : live) n += p.size;
        for (const auto& k : kv)
            for (const auto& p : k.parts) n += p.size;
        for (const ConversationCheckpoint* c : checkpoints)
            n += c->state_a.size() + c->state_b.size() + c->state_c.size() + c->state_d.size() + c->state_e.size();
        return n;
    }
};

}  // namespace basalt::core
