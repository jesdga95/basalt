// The host side of the MMQ kernels: error reporting, the properties of each device and the cache of scratch buffers
// that the stream-k fixup of a launch takes from.
#include "mmq_common.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include "basalt/log.hpp"

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    basalt::log::error("basalt", "mmq: %s: %s\n  in %s at %s:%d", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        for (int id = 0; id < n; ++id) {
            // cudaDeviceGetAttribute is ABI-stable across runtime versions, a cudaDeviceProp is not.
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            auto & d = in.devices[id];
            d.smpbo = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
            d.nsm = attr(cudaDevAttrMultiProcessorCount);
            d.warp_size = attr(cudaDevAttrWarpSize);
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, size));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) cudaFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
