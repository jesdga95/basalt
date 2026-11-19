// src/core/device_memory.cpp - see the header.
#include "basalt/core/device_memory.hpp"

#include "basalt/core/ram_load.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace basalt::core {
namespace {

// The management library, loaded once and kept for the process: the readings are taken on serving paths.
struct Nvml {
    struct MemInfo { unsigned long long total, free, used; };
    platform::DynLib lib;
    int (*by_bus)(const char*, void**) = nullptr;
    int (*by_uuid)(const char*, void**) = nullptr;
    int (*mem)(void*, MemInfo*) = nullptr;
    bool ok = false;

    Nvml() {
        lib = platform::DynLib::open_nvml();
        if (!lib) return;
        const auto init = lib.sym<int (*)()>("nvmlInit_v2");
        by_bus = lib.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
        by_uuid = lib.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByUUID");
        mem = lib.sym<int (*)(void*, MemInfo*)>("nvmlDeviceGetMemoryInfo");
        ok = init != nullptr && mem != nullptr && (by_bus != nullptr || by_uuid != nullptr) && init() == 0;
    }
};

Nvml& nvml() {
    static Nvml n;
    return n;
}

// The management library's handle of CUDA device `dev`: by its PCI address, else by its UUID.
void* nvml_handle(int dev) {
    Nvml& n = nvml();
    if (!n.ok) return nullptr;
    void* h = nullptr;
    char bus[32] = {0};
    if (n.by_bus && cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) == cudaSuccess && n.by_bus(bus, &h) == 0) return h;
    (void) cudaGetLastError();
    cudaDeviceProp p{};
    if (n.by_uuid && cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
        const unsigned char* u = (const unsigned char*) p.uuid.bytes;
        char id[48];
        std::snprintf(id, sizeof id, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
                      u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        if (n.by_uuid(id, &h) == 0) return h;
    }
    (void) cudaGetLastError();
    return nullptr;
}

struct Device {
    bool probed = false;
    DeviceMemoryModel model;
    void* handle = nullptr;            // the management library's
    char luid[8] = {0};                // the OS adapter (Windows)
    unsigned node_mask = 0;
    bool based = false;                // the two figures below are taken
    uint64_t others0 = 0;              // what every process had written at this process's first reading
    uint64_t own0 = 0;                 // what this process held then
    bool commit_told = false;          // the log has said once that the commit limit binds
};

constexpr int kMaxDevices = 64;
std::mutex g_mu;
Device g_dev[kMaxDevices];

// Fills `d` for CUDA device `dev` (g_mu held).
void probe(int dev, Device& d) {
    d.probed = true;
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    d.model.integrated = integrated != 0;
    if (!d.model.integrated) return;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
        std::memcpy(d.luid, p.luid, sizeof d.luid);
        d.node_mask = p.luidDeviceNodeMask;
    } else {
        (void) cudaGetLastError();
    }
    d.handle = nvml_handle(dev);
    Nvml::MemInfo m{};
    if (d.handle != nullptr && nvml().mem(d.handle, &m) == 0) d.model.dedicated = m.total;
    platform::AdapterMemory a;
    if (platform::adapter_memory(d.luid, d.node_mask, a)) {
        d.model.os_usage = true;
        if (d.model.dedicated == 0) d.model.dedicated = a.dedicated;
    }
    // where the OS keeps a commit limit (Windows), an integrated GPU's allocations count against it (measured on the
    // N1X: cudaMalloc and cuMemCreate each raise the commit charge by their full size when made)
    uint64_t avail = 0, limit = 0;
    d.model.commit_bound = platform::commit_headroom(avail, limit);
}

}  // namespace

cudaError_t device_mem_reading(DeviceMemoryReading& r) {
    r = DeviceMemoryReading{};
    size_t cf = 0, ct = 0;
    const cudaError_t e = cudaMemGetInfo(&cf, &ct);
    if (e != cudaSuccess) return e;
    r.device_free = cf;
    r.total = ct;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) {
        (void) cudaGetLastError();
        return e;
    }
    const std::lock_guard<std::mutex> lock(g_mu);
    Device& d = g_dev[dev];
    if (!d.probed) probe(dev, d);
    if (!d.model.integrated) return e;   // a discrete card: the runtime's figures stand

    const uint64_t D = d.model.dedicated;
    if (D != 0 && D < ct) {   // a carve-out below the runtime's total: the carve-out is the device's memory
        uint64_t touched = 0;   // written device pages, every process
        Nvml::MemInfo m{};
        if (d.handle != nullptr && nvml().mem(d.handle, &m) == 0) touched = m.used;
        uint64_t own = ct - cf;   // what this process holds
        platform::AdapterMemory a;
        if (d.model.os_usage && platform::adapter_memory(d.luid, d.node_mask, a)) own = a.local_usage;
        if (!d.based) {
            d.based = true;
            d.others0 = touched;
            d.own0 = own;
        }
        const uint64_t held = d.others0 + (own > d.own0 ? own - d.own0 : 0);
        const uint64_t used = std::min<uint64_t>(D, std::max<uint64_t>(touched, held));
        r.device_free = D - used;
        r.total = D;
    }
    // The OS charges device memory to the commit limit: past it nothing more is allocated, carve-out or not.  What the
    // RAM tiers leave of the RAM, the device leaves of the commit: the reserve the RAM guard holds, plus what the engine
    // takes after its plan (plan_reserve), so the settled headroom is above the guard and not on it.
    uint64_t avail = 0, limit = 0;
    if (d.model.commit_bound && platform::commit_headroom(avail, limit)) {
        const uint64_t reserve = plan_reserve(platform::ram_reserve());
        r.commit_room = avail > reserve ? avail - reserve : 0;
        if (r.commit_room < r.device_free && !d.commit_told) {
            d.commit_told = true;
            constexpr double GiB = 1073741824.0;
            basalt::log::info("basalt",
                              "device memory: the commit limit binds before the GPU's own memory: %.2f GiB free of "
                              "%.2f GiB of commit (RAM plus page file), %.2f GiB of it kept (--ram-reserve-mib plus "
                              "what loads later), against %.2f GiB of the GPU's memory free. A larger page file lets "
                              "the engine use more of it",
                              (double) avail / GiB, (double) limit / GiB, (double) reserve / GiB,
                              (double) r.device_free / GiB);
        }
    }
    return e;
}

cudaError_t device_mem_info(size_t* free_b, size_t* total_b) {
    DeviceMemoryReading r;
    const cudaError_t e = device_mem_reading(r);
    if (e != cudaSuccess) return e;
    if (free_b) *free_b = (size_t) std::min(r.device_free, r.commit_room);
    if (total_b) *total_b = (size_t) r.total;
    return e;
}

DeviceMemoryModel device_memory_model() {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) {
        (void) cudaGetLastError();
        return {};
    }
    const std::lock_guard<std::mutex> lock(g_mu);
    Device& d = g_dev[dev];
    if (!d.probed) probe(dev, d);
    return d.model;
}

}  // namespace basalt::core
