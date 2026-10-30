// include/basalt/core/conversation_stream.hpp - a conversation's state moved between the session and the disk tier
// piece by piece, without a whole copy of it in RAM.
//
// A write reads a ConversationSource: the tokens, pictures and checkpoints by reference and the live running state and
// K/V as readers of the session's own buffers, copied out a write buffer at a time.  A read leaves the K/V in its files
// (a ConversationKvRead) until the restore copies it into the session's buffers through a ConversationKvSink.
#pragma once

#include "basalt/core/conversation_cache.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
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

/// A conversation to write where it is.  The referenced vectors and the buffers the readers copy from stay unchanged
/// until the write returns.
struct ConversationSource {
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = true;
    const std::vector<int32_t>* ids = nullptr;
    const std::vector<ConversationImageKey>* imgs = nullptr;
    const std::vector<ConversationCheckpoint>* checkpoints = nullptr;
    std::array<ConversationBytes, 5> live;   ///< the live running state: gdn, ple, tails, dead, block_pos
    std::vector<ConversationKvSource> kv;    ///< main layers followed by the draft layer

    uint64_t bytes() const {
        uint64_t n = 0;
        for (const auto& p : live) n += p.size;
        for (const auto& k : kv)
            for (const auto& p : k.parts) n += p.size;
        if (checkpoints)
            for (const auto& c : *checkpoints)
                n += c.gdn.size() + c.ple.size() + c.tails.size() + c.dead.size() + c.block_pos.size();
        return n;
    }
};

/// Where a read puts K/V bytes: sink(layer, part, offset, src, n) copies n bytes from src to bytes [offset, offset + n)
/// of buffer `part` (as in ConversationKvSource) of K/V layer `layer`; false when the copy failed.  It may be called
/// from several threads at once, each piece once.
using ConversationKvSink = std::function<bool(size_t, int, uint64_t, const uint8_t*, uint64_t)>;

/// The K/V of a conversation read from the disk tier, still in its files: the image it belongs to carries the K/V
/// layers' shapes with empty buffers, and bytes() the sizes the buffers would have.
class ConversationKvRead {
public:
    virtual ~ConversationKvRead() = default;
    virtual size_t layers() const = 0;
    virtual uint64_t bytes(size_t layer, int part) const = 0;
    /// Reads every byte once into `sink`; false at the first failed read, check or sink call (the file is then
    /// dropped, as a failed whole read drops it).
    virtual bool read(const ConversationKvSink& sink, std::string& error) = 0;
};

}  // namespace basalt::core
