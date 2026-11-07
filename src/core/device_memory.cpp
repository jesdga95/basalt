// src/core/device_memory.cpp - see include/basalt/core/device_memory.hpp.
#include "basalt/core/device_memory.hpp"

#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <cstdint>

namespace basalt::core {

namespace {

// The management library's report of a device's dedicated memory, opened once for the process.
class DedicatedMemory {
public:
    DedicatedMemory() {
        lib_ = basalt::platform::DynLib::open_nvml();
        if (!lib_) return;
        const auto init = lib_.sym<int (*)()>("nvmlInit_v2");
        by_bus_ = lib_.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
        mem_ = lib_.sym<int (*)(void*, MemInfo*)>("nvmlDeviceGetMemoryInfo");
        ok_ = init != nullptr && by_bus_ != nullptr && mem_ != nullptr && init() == 0;
    }
    DedicatedMemory(const DedicatedMemory&) = delete;
    DedicatedMemory& operator=(const DedicatedMemory&) = delete;

    // CUDA device `dev`'s dedicated memory; false when the library reports none
    bool query(int dev, uint64_t& free_b, uint64_t& total_b) const {
        if (!ok_) return false;
        char bus[64] = {};
        if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        void* h = nullptr;
        MemInfo m{};
        if (by_bus_(bus, &h) != 0 || mem_(h, &m) != 0 || m.total == 0) return false;
        free_b = m.free;
        total_b = m.total;
        return true;
    }

private:
    struct MemInfo {
        unsigned long long total, free, used;
    };
    basalt::platform::DynLib lib_;
    int (*by_bus_)(const char*, void**) = nullptr;
    int (*mem_)(void*, MemInfo*) = nullptr;
    bool ok_ = false;
};

const DedicatedMemory& dedicated_memory() {
    static const DedicatedMemory d;
    return d;
}

bool current_device(int& dev) {
    if (cudaGetDevice(&dev) == cudaSuccess) return true;
    (void) cudaGetLastError();
    return false;
}

bool integrated(int dev) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, dev) == cudaSuccess) return v != 0;
    (void) cudaGetLastError();
    return false;
}

}  // namespace

bool device_integrated() {
    int dev = 0;
    return current_device(dev) && integrated(dev);
}

bool device_memory_is_host_ram() {
    int dev = 0;
    uint64_t f = 0, t = 0;
    return current_device(dev) && integrated(dev) && !dedicated_memory().query(dev, f, t);
}

cudaError_t device_mem_info(size_t* free_b, size_t* total_b) {
    const cudaError_t e = cudaMemGetInfo(free_b, total_b);
    int dev = 0;
    if (e != cudaSuccess || !current_device(dev) || !integrated(dev)) return e;
    uint64_t ded_free = 0, ded_total = 0;
    if (dedicated_memory().query(dev, ded_free, ded_total)) {
        *total_b = (size_t) std::min<uint64_t>(*total_b, ded_total);
        *free_b = (size_t) std::min<uint64_t>(*free_b, ded_free);
    } else {
        const uint64_t ram = basalt::platform::total_physical_memory();
        const uint64_t avail = basalt::platform::available_memory();
        const uint64_t reserve = basalt::platform::ram_reserve();
        if (ram > 0) *total_b = (size_t) std::min<uint64_t>(*total_b, ram);
        if (avail > 0) *free_b = (size_t) std::min<uint64_t>(*free_b, avail > reserve ? avail - reserve : 0);
    }
    *free_b = std::min(*free_b, *total_b);
    return e;
}

}  // namespace basalt::core
