// src/core/region_copy.cu - see include/basalt/core/region_copy.hpp.
#include "basalt/core/region_copy.hpp"

#include <algorithm>

namespace basalt::core {
namespace {

constexpr int kLaunchRegions = 256;   // regions one launch takes (its arguments by value)
constexpr int kThreads = 256;
constexpr int kBlocksPerRegion = 32;

struct Batch {
    int n = 0;
    uint8_t* dev[kLaunchRegions] = {};
    uint64_t at[kLaunchRegions] = {};      // each region's offset in the packed buffer
    uint64_t bytes[kLaunchRegions] = {};
};

// Block (x, region): the region's bytes in 16-byte words when the region, its packed place and its size allow, else
// bytes; blocks x stride over the region.
template<bool GATHER>
__global__ void __launch_bounds__(kThreads) region_copy_kernel(Batch b, uint8_t* packed) {
    const int r = blockIdx.y;
    uint8_t* dev = b.dev[r];
    uint8_t* pk = packed + b.at[r];
    const uint64_t n = b.bytes[r];
    uint8_t* to = GATHER ? pk : dev;
    const uint8_t* from = GATHER ? dev : pk;
    const uint64_t stride = (uint64_t) gridDim.x * kThreads;
    const uint64_t first = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
    if ((((uintptr_t) to | (uintptr_t) from | n) & 15) == 0) {
        const uint64_t words = n >> 4;
        for (uint64_t i = first; i < words; i += stride)
            reinterpret_cast<uint4*>(to)[i] = reinterpret_cast<const uint4*>(from)[i];
    } else {
        for (uint64_t i = first; i < n; i += stride) to[i] = from[i];
    }
}

template<bool GATHER>
cudaError_t region_copy(const std::vector<DeviceRegion>& regions, uint8_t* packed, cudaStream_t stream) {
    uint64_t at = 0;
    size_t i = 0;
    while (i < regions.size()) {
        Batch b;
        uint64_t most = 0;
        for (; i < regions.size() && b.n < kLaunchRegions; ++i) {
            const DeviceRegion& r = regions[i];
            if (r.bytes == 0) continue;
            b.dev[b.n] = static_cast<uint8_t*>(r.dev);
            b.at[b.n] = at;
            b.bytes[b.n] = r.bytes;
            most = std::max<uint64_t>(most, r.bytes);
            at += r.bytes;
            ++b.n;
        }
        if (b.n == 0) break;
        const uint64_t want = (most / 16 + kThreads - 1) / kThreads;
        const unsigned gx = (unsigned) std::max<uint64_t>(1, std::min<uint64_t>(want, kBlocksPerRegion));
        region_copy_kernel<GATHER><<<dim3(gx, (unsigned) b.n), kThreads, 0, stream>>>(b, packed);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

}  // namespace

cudaError_t regions_gather(const std::vector<DeviceRegion>& regions, uint8_t* dst, cudaStream_t stream) {
    return region_copy<true>(regions, dst, stream);
}

cudaError_t regions_scatter(const std::vector<DeviceRegion>& regions, const uint8_t* src, cudaStream_t stream) {
    return region_copy<false>(regions, const_cast<uint8_t*>(src), stream);
}

}  // namespace basalt::core
