#include "basalt/context/cuda_device_copy.hpp"

#include <cuda_runtime.h>

namespace basalt::context {
namespace {
std::string cuda_error(const char* what, cudaError_t e) { return std::string(what) + ": " + cudaGetErrorString(e); }
}  // namespace

CudaDeviceCopy::~CudaDeviceCopy() {
    for (auto& [id, r] : ranges_) if (r.owned) cudaFree(r.ptr);
}

uint64_t CudaDeviceCopy::owned_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return owned_;
}

bool CudaDeviceCopy::allocate(const ContextId& context, uint64_t bytes, std::string& error) {
    if (!valid_context_id(context) || bytes == 0) { error = "invalid context or size"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (ranges_.count(context)) { error = context + " already has a device range"; return false; }
    if (max_owned_ && (bytes > max_owned_ || owned_ > max_owned_ - bytes)) {
        error = "device ceiling reached: " + std::to_string(owned_) + " + " + std::to_string(bytes) + " > " + std::to_string(max_owned_);
        return false;
    }
    void* p = nullptr;
    if (cudaError_t e = cudaMalloc(&p, (size_t) bytes); e != cudaSuccess) {
        cudaGetLastError();   // clear the sticky error: a full device is an answer, not a fault
        error = cuda_error("cudaMalloc", e);
        return false;
    }
    ranges_.emplace(context, Range{(uint8_t*) p, bytes, true});
    owned_ += bytes;
    return true;
}

bool CudaDeviceCopy::bind(const ContextId& context, void* device, uint64_t bytes, std::string& error) {
    if (!valid_context_id(context) || !device || bytes == 0) { error = "invalid context, pointer or size"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!ranges_.emplace(context, Range{(uint8_t*) device, bytes, false}).second) {
        error = context + " already has a device range";
        return false;
    }
    return true;
}

bool CudaDeviceCopy::download(const ContextId& context, uint64_t offset, void* dst, size_t n, std::string& error) {
    uint8_t* src = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = ranges_.find(context);
        if (it == ranges_.end()) { error = "no device range for " + context; return false; }
        if (offset > it->second.bytes || n > it->second.bytes - offset) { error = "read past the device range"; return false; }
        src = it->second.ptr + offset;
    }
    if (cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost); e != cudaSuccess) { error = cuda_error("cudaMemcpy D2H", e); return false; }
    return true;
}

bool CudaDeviceCopy::upload(const ContextId& context, uint64_t offset, const void* src, size_t n, std::string& error) {
    uint8_t* dst = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = ranges_.find(context);
        if (it == ranges_.end()) { error = "no device range for " + context; return false; }
        if (offset > it->second.bytes || n > it->second.bytes - offset) { error = "write past the device range"; return false; }
        dst = it->second.ptr + offset;
    }
    if (cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice); e != cudaSuccess) { error = cuda_error("cudaMemcpy H2D", e); return false; }
    return true;
}

void CudaDeviceCopy::free(const ContextId& context) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = ranges_.find(context);
    if (it == ranges_.end()) return;
    if (it->second.owned) { cudaFree(it->second.ptr); owned_ -= it->second.bytes; }
    ranges_.erase(it);
}

}  // namespace basalt::context
