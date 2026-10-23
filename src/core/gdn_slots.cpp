#include "basalt/core/gdn_slots.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace basalt::core {
namespace {
bool fail(std::string& error, const char* what) {
    error = what;
    return false;
}
}  // namespace

struct GdnSlots::Impl {
    std::vector<void*> buffers;
};

GdnSlots::~GdnSlots() {
    if (impl_ != nullptr) {
        for (void* p : impl_->buffers) cudaFree(p);
        delete impl_;
    }
}

bool GdnSlots::reserve(int n, size_t bytes, std::string& error) {
    if (impl_ != nullptr) return true;   // reserved once
    if (n <= 0 || bytes == 0) return true;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        return fail(error, "cannot read the current device for the GDN slots");
    }
    auto* impl = new Impl;
    for (int i = 0; i < n; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            cudaGetLastError();
            for (void* q : impl->buffers) cudaFree(q);
            delete impl;
            return fail(error, "the GDN slots do not fit in device memory");
        }
        if (cudaMemset(p, 0, bytes) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(p);
            for (void* q : impl->buffers) cudaFree(q);
            delete impl;
            return fail(error, "cannot zero a GDN slot");
        }
        impl->buffers.push_back(p);
    }
    impl_ = impl;
    device_ = dev;
    bytes_ = bytes;
    book_.reset(n);
    return true;
}

void* GdnSlots::ptr(int slot) const {
    if (impl_ == nullptr || slot < 0 || slot >= (int) impl_->buffers.size()) return nullptr;
    return impl_->buffers[(size_t) slot];
}

}  // namespace basalt::core
