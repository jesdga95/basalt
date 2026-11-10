#include "basalt/context/kv_arena_copy.hpp"

#include <cuda_runtime.h>

namespace basalt::context {

KvArenaCopy::KvArenaCopy(core::KvArena& arena, std::vector<uint64_t> page_bytes)
    : arena_(arena), page_bytes_(std::move(page_bytes)) {
    for (uint64_t b : page_bytes_) per_page_ += b;
}

uint64_t KvArenaCopy::adopt(const ContextId& context, std::string& error) {
    if (!valid_context_id(context)) { error = "invalid context id"; return 0; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!resident_.empty() && resident_ != context) { error = resident_ + " is resident in the arena"; return 0; }
    if (arena_.cells() <= 0 || arena_.page_cells() <= 0 || per_page_ == 0) { error = "nothing is mapped"; return 0; }
    resident_ = context;
    bytes_ = (uint64_t) (arena_.cells() / arena_.page_cells()) * per_page_;
    return bytes_;
}

bool KvArenaCopy::allocate(const ContextId& context, uint64_t bytes, std::string& error) {
    if (!valid_context_id(context) || bytes == 0 || per_page_ == 0) { error = "invalid context or size"; return false; }
    if (bytes % per_page_ != 0) { error = "state is not a whole number of pages"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!resident_.empty()) { error = resident_ + " is resident in the arena"; return false; }
    const uint64_t pages = bytes / per_page_;
    if (pages > (uint64_t) arena_.max_cells() / (uint64_t) arena_.page_cells()) { error = "state exceeds the arena"; return false; }
    if (!arena_.map((int64_t) (pages * (uint64_t) arena_.page_cells()), error)) return false;
    // map() backs whole granules, so the arena may now hold more pages than the context; the context is its first pages
    resident_ = context;
    bytes_ = bytes;
    return true;
}

bool KvArenaCopy::spans(uint64_t offset, size_t n, std::vector<Span>& out, std::string& error) const {
    if (offset > bytes_ || n > bytes_ - offset) { error = "range past the context"; return false; }
    const uint64_t pages = bytes_ / per_page_;
    uint64_t at = 0;
    for (size_t i = 0; i < page_bytes_.size() && n; ++i) {
        const uint64_t len = pages * page_bytes_[i];
        if (offset < at + len) {
            const uint64_t from = offset - at, take = std::min<uint64_t>(len - from, n);
            out.push_back({i, from, take});
            offset += take;
            n -= (size_t) take;
        }
        at += len;
    }
    return true;
}

bool KvArenaCopy::download(const ContextId& context, uint64_t offset, void* dst, size_t n, std::string& error) {
    std::vector<Span> sp;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (resident_ != context) { error = context + " is not resident"; return false; }
        if (!spans(offset, n, sp, error)) return false;
    }
    uint8_t* out = (uint8_t*) dst;
    for (const Span& s : sp) {
        if (cudaError_t e = cudaMemcpy(out, arena_.array(s.array) + s.from, (size_t) s.n, cudaMemcpyDeviceToHost); e != cudaSuccess) {
            error = std::string("cudaMemcpy D2H: ") + cudaGetErrorString(e);
            return false;
        }
        out += s.n;
    }
    return true;
}

bool KvArenaCopy::upload(const ContextId& context, uint64_t offset, const void* src, size_t n, std::string& error) {
    std::vector<Span> sp;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (resident_ != context) { error = context + " is not resident"; return false; }
        if (!spans(offset, n, sp, error)) return false;
    }
    const uint8_t* in = (const uint8_t*) src;
    for (const Span& s : sp) {
        if (cudaError_t e = cudaMemcpy(arena_.array(s.array) + s.from, in, (size_t) s.n, cudaMemcpyHostToDevice); e != cudaSuccess) {
            error = std::string("cudaMemcpy H2D: ") + cudaGetErrorString(e);
            return false;
        }
        in += s.n;
    }
    return true;
}

void KvArenaCopy::free(const ContextId& context) {
    std::lock_guard<std::mutex> lk(mu_);
    if (resident_ != context) return;
    std::string err;
    arena_.unmap(0, err);
    resident_.clear();
    bytes_ = 0;
}

}  // namespace basalt::context
