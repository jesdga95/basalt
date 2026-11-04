// Re-registered growth versus a copied allocation, across every array and layer, with transfers and a live graph.
#include "basalt/core/kv_host_pool.hpp"
#include "basalt/core/session.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace c = basalt::core;
namespace k = basalt::kernels;
namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::printf("%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "zero");
    return p;
}
struct Span { uint8_t* p; size_t n; };
std::vector<Span> spans(const k::KvHostPools& h, const c::ModelGeometry& g, int64_t cells) {
    const size_t rows = (size_t) ((cells + 3) / 4 * 4 * g.n_head_kv), codes = rows * g.head_dim;
    if (h.k_q != nullptr)
        return {{(uint8_t*) h.k_q, codes}, {(uint8_t*) h.v_q, codes},
                {(uint8_t*) h.k_scale, codes / 32}, {(uint8_t*) h.v_scale, codes / 32}};
    if (h.k_q4 != nullptr) {
        const size_t bytes = rows * k::kv_q4_bytes_per_head((int) g.head_dim);
        return {{h.k_q4, bytes}, {h.v_q4, bytes}};
    }
    return {{(uint8_t*) h.k_pool, codes * 2}, {(uint8_t*) h.v_pool, codes * 2}};
}

int run(int fmt, bool ahead) {
    const c::ModelGeometry g{};
    c::QsaState layers[12];
    for (auto& st : layers) { st.kv_int8 = fmt == k::kKvInt8; st.kv_q4 = fmt == k::kKvQ4; }
    c::SessionState ss, legacy;
    ss.qsa_states = legacy.qsa_states = layers;
    ss.qsa_alloc = legacy.qsa_alloc = 12;
    ss.max_cells = 33797;
    std::string err;
    c::KvHostPool pool;
    // a prefaulted pool is registered in place by the reserve
    if (ahead && !pool.prefault(ss, g, 8192, err)) {
        std::puts("registered identity mappings unavailable: skipped");
        return 77;
    }
    if (!pool.reserve(ss, g, 8192, err)) { std::printf("reserve: %s\n", err.c_str()); return 1; }
    if (!pool.grows_in_place()) { std::puts("registered identity mappings unavailable: skipped"); return 77; }
    std::mt19937 rng(86);
    for (int64_t j = 0; j < pool.layers(); ++j)
        for (const auto& a : spans(pool.layer(j), g, pool.cells()))
            for (size_t i = 0; i < a.n; ++i) a.p[i] = (uint8_t) rng();
    const auto first = spans(pool.layer(0), g, pool.cells())[0];
    const std::vector<uint8_t> captured_reference(first.p, first.p + first.n);
    auto* device = dalloc<uint8_t>(spans(pool.layer(0), g, 33800)[0].n);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    ck(cudaStreamCreate(&stream), "stream");
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    ck(cudaMemcpyAsync(device, first.p, first.n, cudaMemcpyDefault, stream), "captured read");
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
    int bad = 0;
    for (int64_t cells : {16384, 32768, 33001, 33797}) {
        c::KvHostPool copied;
        const int64_t old_cells = pool.cells();
        if (!copied.reserve(legacy, g, cells, err) || !copied.copy_from(pool, old_cells, err)) {
            std::printf("copied pool: %s\n", err.c_str()); return 1;
        }
        std::vector<std::vector<Span>> before;
        for (int64_t j = 0; j < pool.layers(); ++j) before.push_back(spans(pool.layer(j), g, old_cells));
        const uint64_t bytes = pool.bytes(), delta = pool.growth_bytes(cells);
        if (!pool.grow(cells, err)) { std::printf("grow: %s\n", err.c_str()); return 1; }
        bad += pool.bytes() != bytes + delta;
        // Replay the same executable after re-registration; its source address must remain usable.
        ck(cudaGraphLaunch(executable, stream), "replay old graph");
        std::vector<uint8_t> captured(first.n);
        ck(cudaMemcpyAsync(captured.data(), device, first.n, cudaMemcpyDefault, stream), "graph output");
        ck(cudaStreamSynchronize(stream), "graph sync");
        bad += captured != captured_reference;
        for (int64_t j = 0; j < pool.layers(); ++j) {
            const auto arrays = spans(pool.layer(j), g, pool.cells()), ref = spans(copied.layer(j), g, pool.cells());
            for (size_t a = 0; a < arrays.size(); ++a) {
                const auto old = before[(size_t) j][a];
                bad += arrays[a].p != old.p;
                std::vector<uint8_t> reference(arrays[a].n), readback(arrays[a].n);
                ck(cudaMemcpy(reference.data(), ref[a].p, old.n, cudaMemcpyDefault), "copied prefix");
                bad += std::memcmp(arrays[a].p, reference.data(), old.n) != 0;
                for (size_t i = old.n; i < reference.size(); ++i) reference[i] = (uint8_t) rng();
                // DMA both ways over the whole re-registered prefix, then compare every byte with the copied pool.
                ck(cudaMemcpyAsync(device, reference.data(), reference.size(), cudaMemcpyDefault, stream), "upload");
                ck(cudaMemcpyAsync(arrays[a].p, device, reference.size(), cudaMemcpyDefault, stream), "write span");
                ck(cudaMemcpyAsync(ref[a].p, device, reference.size(), cudaMemcpyDefault, stream), "write copied");
                ck(cudaMemcpyAsync(device, arrays[a].p, reference.size(), cudaMemcpyDefault, stream), "read span");
                ck(cudaMemcpyAsync(readback.data(), device, reference.size(), cudaMemcpyDefault, stream), "download");
                ck(cudaStreamSynchronize(stream), "span sync");
                bad += readback != reference;
                bad += std::memcmp(arrays[a].p, reference.data(), reference.size()) != 0;
                ck(cudaMemcpy(readback.data(), ref[a].p, reference.size(), cudaMemcpyDefault), "read copied");
                bad += readback != reference;
            }
        }
        std::printf("format %d%s: %lld -> %lld cells, %llu extra bytes, %d failures\n", fmt,
                    ahead ? " (prefaulted)" : "", (long long) old_cells, (long long) pool.cells(),
                    (unsigned long long) delta, bad);
    }
    ck(cudaGraphExecDestroy(executable), "destroy executable");
    ck(cudaGraphDestroy(graph), "destroy graph");
    ck(cudaStreamDestroy(stream), "destroy stream");
    ck(cudaFree(device), "free");
    return bad ? 1 : 0;
}
}  // namespace

#include "kv_host_read_bench.hpp"

// The first pin of a 131072-cell INT8 pool over 12 layers: fresh pages, and pages prefaulted before the pin.
int pin_bench(double power_w) {
    if (power_w <= 0) { std::puts("--pin-bench requires the measured GPU power limit in watts"); return 1; }
    const c::ModelGeometry g{};
    c::QsaState layers[12];
    for (auto& st : layers) st.kv_int8 = true;
    c::SessionState ss;
    ss.qsa_states = layers;
    ss.qsa_alloc = 12;
    ss.max_cells = 262144;
    const int64_t cells = 131072;
    std::string err;
    using Ms = std::chrono::duration<double, std::milli>;
    for (int arm : {0, 1, 1, 0, 0, 1}) {
        c::KvHostPool pool;
        double ahead_ms = 0;
        if (arm == 1) {
            if (!pool.prefault(ss, g, cells, err)) { std::printf("prefault: %s\n", err.c_str()); return 77; }
            ahead_ms = pool.prefault_wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!pool.reserve(ss, g, cells, err) || !pool.grows_in_place()) {
            std::printf("reserve: %s\n", err.c_str());
            return 77;
        }
        const double pin_ms = Ms(std::chrono::steady_clock::now() - t0).count();
        std::printf("pin bench: %s, %.2f GiB, pin %.1f ms, prefault %.1f ms, power limit %.0f W\n",
                    arm ? "prefaulted" : "fresh", (double) pool.bytes() / 1073741824.0, pin_ms, ahead_ms, power_w);
    }
    return 0;
}

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    if (argc == 3 && std::string(argv[1]) == "--bench") return host_read_bench(std::atof(argv[2]));
    if (argc == 3 && std::string(argv[1]) == "--pin-bench") return pin_bench(std::atof(argv[2]));
    if (argc != 1) { std::puts("usage: kv_host_grow_test [--bench POWER_W | --pin-bench POWER_W]"); return 1; }
    for (bool ahead : {false, true})
        for (int fmt : {k::kKvInt8, k::kKvQ4, k::kKvF16}) {
            const int result = run(fmt, ahead);
            if (result) return result;
        }
    std::puts("kv_host_grow_test: PASS");
    return 0;
}
