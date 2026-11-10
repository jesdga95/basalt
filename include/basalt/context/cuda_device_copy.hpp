// DeviceCopy over the CUDA runtime. Two ways to give a context a device range:
//   allocate(): this object cudaMallocs it and cudaFrees it on free();
//   bind():     the engine hands over a range it already owns (a KV/state region); free() only forgets it.
// Copies are synchronous cudaMemcpy on the legacy stream, so a finished call means the bytes are in place.
// Linked only by programs that already link CUDA (basalt_context_cuda); basalt_context itself stays host-only.
#pragma once

#include "basalt/context/device_tier_mover.hpp"

#include <map>
#include <mutex>

namespace basalt::context {

class CudaDeviceCopy : public DeviceCopy {
public:
    /// `max_owned_bytes`: ceiling for what allocate() may hold at once (0: no ceiling). bind() ranges do not count.
    explicit CudaDeviceCopy(uint64_t max_owned_bytes = 0) : max_owned_(max_owned_bytes) {}
    ~CudaDeviceCopy() override;
    CudaDeviceCopy(const CudaDeviceCopy&) = delete;
    CudaDeviceCopy& operator=(const CudaDeviceCopy&) = delete;

    bool allocate(const ContextId& context, uint64_t bytes, std::string& error) override;
    bool download(const ContextId& context, uint64_t offset, void* dst, size_t n, std::string& error) override;
    bool upload(const ContextId& context, uint64_t offset, const void* src, size_t n, std::string& error) override;
    void free(const ContextId& context) override;

    /// Non-owning: `device` must stay valid until free(context). False when the context already has a range.
    bool bind(const ContextId& context, void* device, uint64_t bytes, std::string& error);
    uint64_t owned_bytes() const;

private:
    struct Range { uint8_t* ptr = nullptr; uint64_t bytes = 0; bool owned = false; };
    mutable std::mutex mu_;
    std::map<ContextId, Range> ranges_;
    uint64_t owned_ = 0, max_owned_ = 0;
};

}  // namespace basalt::context
