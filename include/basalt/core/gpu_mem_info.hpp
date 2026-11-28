// include/basalt/core/gpu_mem_info.hpp - the current device's free and total memory, as a plan may spend it.
//
// Header-only, like device_copy.hpp; the host's available memory comes from `platform`.
#pragma once

#include "basalt/core/device_copy.hpp"
#include "basalt/platform/memory.hpp"
#ifdef _WIN32
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/system.hpp"
#endif

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#ifdef _WIN32
#include <array>
#include <cstdint>
#include <mutex>
#endif

namespace basalt::core {

/// Whether the current device is an integrated GPU (cudaDevAttrIntegrated); false when the runtime cannot say.
inline bool current_device_integrated() {
    if (!any_integrated_device()) return false;
    int d = 0, integrated = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, d) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return integrated != 0;
}

#ifdef _WIN32
namespace detail {

/// The free memory of one card as the management library reports it, for the Windows driver model that the CUDA
/// runtime cannot answer for. The library is opened once per process and never closed (DynLib); a card's handle is
/// looked up once by its PCI bus id. No NVML, no answer: the caller keeps the runtime's number.
class NvmlFreeMemory {
public:
    NvmlFreeMemory() {
        lib_ = basalt::platform::DynLib::open_nvml();
        if (!lib_) return;
        init_ = lib_.sym<int (*)()>("nvmlInit_v2");
        by_bus_ = lib_.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
        get_mem_ = lib_.sym<int (*)(void*, NvmlMemory*)>("nvmlDeviceGetMemoryInfo");
        ready_ = init_ != nullptr && by_bus_ != nullptr && get_mem_ != nullptr && init_() == 0;
    }

    /// The free memory of CUDA device `dev` in bytes; false when NVML cannot report it, and the caller falls back
    /// to the runtime's number.
    bool query(int dev, uint64_t& free_bytes) {
        if (!ready_ || dev < 0 || dev >= (int) handles_.size()) return false;
        void* handle = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!tried_[(size_t) dev]) {
                char bus[64] = {};
                if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) == cudaSuccess) {
                    void* h = nullptr;
                    if (by_bus_(bus, &h) == 0) handles_[(size_t) dev] = h;
                } else {
                    (void) cudaGetLastError();
                }
                tried_[(size_t) dev] = true;
            }
            handle = handles_[(size_t) dev];
        }
        if (handle == nullptr) return false;
        NvmlMemory m{};
        if (get_mem_(handle, &m) != 0) return false;
        free_bytes = (uint64_t) m.free;
        return true;
    }

private:
    // nvmlMemory_t, as in gpu_select.cpp and bench_main.cpp.
    struct NvmlMemory {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };

    basalt::platform::DynLib lib_;
    bool ready_ = false;
    int (*init_)() = nullptr;
    int (*by_bus_)(const char*, void**) = nullptr;
    int (*get_mem_)(void*, NvmlMemory*) = nullptr;
    std::mutex mu_;
    std::array<void*, 16> handles_{};
    std::array<bool, 16> tried_{};
};

}  // namespace detail
#endif

/// cudaMemGetInfo, with two corrections for the drivers that misreport free memory.
///
/// An integrated GPU's memory is the machine's. Its driver (a GB10) reports MemFree as free, so the page cache counts as
/// used: right after a large download or file read it reported 8 GiB free of 122 GiB while 120 GiB were available, and
/// every plan refused. Its allocations take page cache back on demand (40 GiB allocated and written with 8 GiB "free"
/// succeeded, the cache shrinking under it), so MemAvailable is what a plan can spend.
///
/// A discrete card under Windows WDDM reports its memory budget shrunk by the shared-memory fallback the driver
/// reserves, so cudaMemGetInfo's free is well under what the card can actually hold; the management library reports
/// the card's own memory, and the larger of the two is what a plan can spend. Everywhere else the card is reported
/// as the runtime reports it.
inline cudaError_t gpu_mem_info(size_t* free_bytes, size_t* total_bytes) {
    const cudaError_t e = cudaMemGetInfo(free_bytes, total_bytes);
    if (e != cudaSuccess) return e;
    if (current_device_integrated()) {
        *free_bytes = std::min(*total_bytes, std::max<size_t>(*free_bytes, (size_t) basalt::platform::available_memory()));
        return e;
    }
#ifdef _WIN32
    if (basalt::platform::gpu_memory_pages()) {
        static detail::NvmlFreeMemory nvml;
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            uint64_t nvml_free = 0;
            if (nvml.query(dev, nvml_free) && nvml_free > (uint64_t) *free_bytes) {
                *free_bytes = (size_t) std::min<uint64_t>(nvml_free, (uint64_t) *total_bytes);
            }
        }
    }
#endif
    return e;
}

}  // namespace basalt::core
