// Isolate transfers and device access over adjacent, single and re-registered host ranges.
#include "basalt/platform/memory.hpp"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* phase) {
    std::printf("%s: %s (%d)\n", phase, cudaGetErrorName(e), (int) e);
    std::fflush(stdout);
    if (e != cudaSuccess) std::exit(1);
}
__device__ void copy_stage(const unsigned char* src, unsigned char* dst, size_t n, size_t i) {
    if (i < n) dst[i] = src[i];
}
__global__ void copy_bytes(const unsigned char* src, unsigned char* dst, size_t n) {
    copy_stage(src, dst, n, (size_t) blockIdx.x * blockDim.x + threadIdx.x);
}
void read(const std::string& op, const unsigned char* host, unsigned char* device, size_t n, cudaStream_t stream) {
    if (op == "kernel-read" || op == "graph-kernel") {
        copy_bytes<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(host, device, n);
        ck(cudaGetLastError(), "kernel read launch");
    } else {
        ck(cudaMemcpyAsync(device, host, n, cudaMemcpyDefault, stream), "async H2D enqueue");
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::puts("usage: kv_host_registration_test adjacent|single|regrow OP"); return 1; }
    const std::string layout = argv[1], op = argv[2];
    if (layout != "adjacent" && layout != "single" && layout != "regrow") return 1;
    const bool graph_op = op == "graph-copy" || op == "graph-kernel";
    if (!graph_op && op != "copy-h2d" && op != "copy-d2h" && op != "async-h2d" && op != "async-d2h" &&
        op != "kernel-read" && op != "kernel-write") return 1;
    int count = 0, device_id = 0, identity = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    ck(cudaGetDevice(&device_id), "device");
    ck(cudaDeviceGetAttribute(&identity, cudaDevAttrCanUseHostPointerForRegisteredMem, device_id), "identity capability");
    if (!identity) return 77;
    namespace p = basalt::platform;
    const size_t page = (size_t) p::vm_page_size();
    if (page == 0) return 1;
    const size_t half = ((4ull << 20) + page - 1) / page * page, n = half * 2;
    auto* host = (unsigned char*) p::vm_reserve(n);
    const size_t first = layout == "single" ? n : half;
    if (host == nullptr || !p::vm_commit(host, first)) return 1;
    const unsigned flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
    ck(cudaHostRegister(host, first, flags), "register initial range");
    void* alias = nullptr;
    ck(cudaHostGetDevicePointer(&alias, host, 0), "first alias");
    if (alias != host) { ck(cudaHostUnregister(host), "unregister alias"); p::vm_release(host, n); return 77; }
    std::vector<unsigned char> expected(n), got(n);
    for (size_t i = 0; i < n; ++i) expected[i] = (unsigned char) ((i * 37 + i / 7) % 251);
    std::memcpy(host, expected.data(), half);
    unsigned char* device = nullptr;
    ck(cudaMalloc(&device, n), "device allocation");
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream), "stream");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    if (graph_op) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin old capture");
        read(op, host, device, half, stream);
        ck(cudaStreamEndCapture(stream, &graph), "end old capture");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate old graph");
    }
    if (layout != "single" && !p::vm_commit(host + half, half)) return 1;
    if (layout == "adjacent") {
        ck(cudaHostRegister(host + half, half, flags), "register adjacent prefix");
        ck(cudaHostGetDevicePointer(&alias, host + half, 0), "adjacent alias");
        if (alias != host + half) return 77;
    } else if (layout == "regrow") {
        ck(cudaHostUnregister(host), "unregister first prefix");
        ck(cudaHostRegister(host, n, flags), "register whole prefix");
        ck(cudaHostGetDevicePointer(&alias, host, 0), "whole alias");
        if (alias != host) return 77;
    }
    // Existing bytes must survive the registration change before any device operation can rewrite them.
    if (std::memcmp(host, expected.data(), half) != 0) return 1;
    std::memcpy(host + half, expected.data() + half, half);
    ck(cudaDeviceSynchronize(), "clean device before operation");
    ck(cudaGetLastError(), "clean runtime before operation");
    std::printf("layout=%s operation=%s bytes=%zu boundary=%zu\n", layout.c_str(), op.c_str(), n, half);
    if (graph_op && exec != nullptr) {
        ck(cudaGraphLaunch(exec, stream), "old graph launch");
        ck(cudaStreamSynchronize(stream), "old graph execution");
        ck(cudaMemcpy(got.data(), device, half, cudaMemcpyDeviceToHost), "old graph readback");
        if (std::memcmp(got.data(), expected.data(), half) != 0) return 1;
        ck(cudaGraphExecDestroy(exec), "destroy old executable");
        ck(cudaGraphDestroy(graph), "destroy old graph");
    }
    if (graph_op) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin full capture");
        read(op, host, device, n, stream);
        ck(cudaStreamEndCapture(stream, &graph), "end full capture");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate full graph");
        ck(cudaGraphLaunch(exec, stream), "full graph launch");
    } else if (op == "copy-h2d") {
        ck(cudaMemcpy(device, host, n, cudaMemcpyDefault), "sync H2D call");
    } else if (op == "async-h2d" || op == "kernel-read") {
        read(op, host, device, n, stream);
    } else {
        ck(cudaMemcpy(device, expected.data(), n, cudaMemcpyHostToDevice), "prepare device source");
        std::memset(host, 0xff, n);
        if (op == "copy-d2h") ck(cudaMemcpy(host, device, n, cudaMemcpyDefault), "sync D2H call");
        else if (op == "async-d2h") ck(cudaMemcpyAsync(host, device, n, cudaMemcpyDefault, stream), "async D2H enqueue");
        else {
            copy_bytes<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(device, host, n);
            ck(cudaGetLastError(), "kernel write launch");
        }
    }
    ck(cudaStreamSynchronize(stream), "operation execution");
    const bool write = op == "copy-d2h" || op == "async-d2h" || op == "kernel-write";
    if (write) std::memcpy(got.data(), host, n);
    else ck(cudaMemcpy(got.data(), device, n, cudaMemcpyDeviceToHost), "operation readback");
    if (got != expected) { std::puts("byte mismatch"); return 1; }
    if (graph_op) {
        ck(cudaGraphExecDestroy(exec), "destroy executable");
        ck(cudaGraphDestroy(graph), "destroy graph");
    }
    ck(cudaStreamDestroy(stream), "destroy stream");
    ck(cudaFree(device), "free device");
    if (layout == "adjacent") ck(cudaHostUnregister(host + half), "unregister adjacent");
    ck(cudaHostUnregister(host), "unregister host");
    p::vm_release(host, n);
    std::puts("kv_host_registration_test: PASS");
    return 0;
}
