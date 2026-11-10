// GPU <-> HOST data path for TierManager over an abstract DeviceCopy, plus the router that sends each move to the
// right mover. Host-only: the CUDA implementation of DeviceCopy (cudaMemcpyAsync against the engine's KV/state
// arrays) belongs to the engine, which owns which arena region a context occupies.
#pragma once

#include "basalt/context/file_tier_mover.hpp"

namespace basalt::context {

/// The device side of one context's state, as a flat byte range of `bytes`. Calls for one context never overlap.
class DeviceCopy {
public:
    virtual ~DeviceCopy() = default;
    /// Reserves the context's device range; false with `error` when the device has no room.
    virtual bool allocate(const ContextId& context, uint64_t bytes, std::string& error) = 0;
    virtual bool download(const ContextId& context, uint64_t offset, void* dst, size_t n, std::string& error) = 0;
    virtual bool upload(const ContextId& context, uint64_t offset, const void* src, size_t n, std::string& error) = 0;
    virtual void free(const ContextId& context) = 0;
};

class DeviceTierMover : public TierMover {
public:
    DeviceTierMover(PageStore& store, DeviceCopy& device) : store_(store), device_(device) {}
    Outcome copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest, std::string& error) override;
    void release(const ContextId& context, Tier tier) override;

private:
    PageStore& store_;
    DeviceCopy& device_;
};

/// gpu<->host to `device`, host<->disk to `file`; anything else (gpu<->disk goes through host) fails.
class CompositeTierMover : public TierMover {
public:
    CompositeTierMover(TierMover& device, TierMover& file) : device_(device), file_(file) {}
    Outcome copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest, std::string& error) override;
    void release(const ContextId& context, Tier tier) override;

private:
    TierMover& device_;
    TierMover& file_;
};

}  // namespace basalt::context
