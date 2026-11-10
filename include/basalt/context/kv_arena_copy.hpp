// DeviceCopy over the engine's elastic KV pools (basalt::core::KvArena).
//
// The arena holds one session's pool arrays; the first `cells` cells of every array (whole pages, page-major) are that
// session's KV. A context's flat byte range is those prefixes back to back: array 0's first n pages, then array 1's,
// and so on (n = cells / page_cells). Ranges beyond the pages (the indexer's spare tail rows) are not part of it.
//
// One context is resident at a time, because the arena is one session's memory:
//   adopt(): the arena's current backing becomes the context (gpu -> host: read it out);
//   allocate(): backs the pages for `bytes` (host -> gpu: restore it); fails when another context is resident.
//   free(): gives the backing back (unmap to 0) and forgets the context.
// Linked by engine programs only (needs basalt::core::KvArena and the CUDA runtime).
#pragma once

#include "basalt/context/device_tier_mover.hpp"
#include "basalt/core/kv_arena.hpp"

#include <mutex>
#include <vector>

namespace basalt::context {

class KvArenaCopy : public DeviceCopy {
public:
    /// `page_bytes` must be the vector the arena was opened with (bytes of one page of each array).
    KvArenaCopy(core::KvArena& arena, std::vector<uint64_t> page_bytes);

    uint64_t bytes_per_page() const { return per_page_; }
    /// The arena's current backing as `context`; returns its size in bytes, 0 with `error` when nothing is mapped or
    /// another context is resident.
    uint64_t adopt(const ContextId& context, std::string& error);

    bool allocate(const ContextId& context, uint64_t bytes, std::string& error) override;
    bool download(const ContextId& context, uint64_t offset, void* dst, size_t n, std::string& error) override;
    bool upload(const ContextId& context, uint64_t offset, const void* src, size_t n, std::string& error) override;
    void free(const ContextId& context) override;

private:
    struct Span { size_t array; uint64_t from, n; };
    bool spans(uint64_t offset, size_t n, std::vector<Span>& out, std::string& error) const;  ///< needs mu_
    core::KvArena& arena_;
    std::vector<uint64_t> page_bytes_;
    uint64_t per_page_ = 0;
    mutable std::mutex mu_;
    ContextId resident_;
    uint64_t bytes_ = 0;
};

}  // namespace basalt::context
