// tests/model/batch_graph_memory_test.cu - a composition's graph that finds no device memory (GPU): its window already
// ran direct, the batch goes on without a graph for that composition, and a composition captured once the card has room
// again gets its graph.
#include "basalt/model/batch_driver.hpp"

#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

constexpr int kLaunches = 4000;   // nodes: enough that instantiating the graph needs device memory of its own

__global__ void touch(int* p, int i) { p[i % 64] += 1; }

class Fake final : public basalt::model::BatchSession {
public:
    Fake(basalt::model::BatchShared* sh, cudaStream_t st, int* buf) : sh_(sh), st_(st), buf_(buf) {}
    basalt::model::Result decode_window(const basalt::model::DecodeRequest&) override { return {}; }
    basalt::model::Result prefill_chunk(const basalt::model::PrefillRequest&) override { return {}; }
    int forwards = 0;

protected:
    const char* batch_name() const override { return "fake"; }
    int64_t batch_context() const override { return 4096; }
    basalt::model::BatchShared& batch_shared() override { return *sh_; }
    cudaStream_t batch_stream() const override { return st_; }
    int batch_rows() const override { return 16; }
    bool batch_reserve(const basalt::model::DecodeRequest&, std::string&) override { return true; }
    bool batch_stage(const basalt::model::BatchView&, basalt::model::GraphLru<basalt::core::CapturedGraph>::Key&, std::string&) override { return true; }
    bool batch_forward(const basalt::model::BatchView&, std::string&) override {
        ++forwards;
        for (int i = 0; i < kLaunches; ++i) touch<<<1, 1, 0, st_>>>(buf_, i);
        return cudaGetLastError() == cudaSuccess;
    }
    void batch_take(const basalt::model::BatchView&, int) override {}
    const int32_t* batch_draw(const basalt::model::BatchView&, int, std::string&) override { return pick_; }

private:
    basalt::model::BatchShared* sh_;
    cudaStream_t st_;
    int* buf_;
    int32_t pick_[4] = {7, 7, 7, 7};
};

}  // namespace

int main() {
    using namespace basalt::model;
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("skip: no CUDA device\n");
        return 77;
    }
    cudaStream_t st;
    cudaStreamCreate(&st);
    int* buf = nullptr;
    cudaMalloc(&buf, 64 * sizeof(int));
    cudaMemset(buf, 0, 64 * sizeof(int));
    BatchShared sh;
    Fake a(&sh, st, buf), b(&sh, st, buf);
    ModelSession* peers[2] = {&a, &b};
    const int32_t t0[1] = {1}, t1[1] = {2};
    int32_t out0 = 0, out1 = 0;
    const auto window = [&](int64_t pos) {
        DecodeRequest r[2];
        r[0].tokens = t0, r[0].n_tokens = 1, r[0].pos = pos, r[0].next_token = &out0;
        r[1].tokens = t1, r[1].n_tokens = 1, r[1].pos = pos, r[1].next_token = &out1;
        return a.decode_batch(peers, r, 2);
    };

    // the card's free memory taken: a graph of this many launches cannot be instantiated
    std::vector<void*> hog;
    for (size_t chunk = size_t(1) << 30; chunk >= (size_t(1) << 16); chunk >>= 1) {
        void* p = nullptr;
        while (cudaMalloc(&p, chunk) == cudaSuccess) hog.push_back(p);
        (void) cudaGetLastError();
    }
    bool ok = true;
    for (int i = 0; i < 4; ++i) ok = window(10 + i).ok() && ok;
    check(ok, "windows of a composition whose graph finds no memory all complete");
    check(sh.graphs.size() == 0, "no graph is kept for it");
    check(a.forwards == 1 + 2 + 1 + 1, "its windows ran direct (the second also built the failed capture)");
    check(out0 == 7 && out1 == 7, "the picks came back");

    for (void* p : hog) cudaFree(p);
    (void) cudaGetLastError();
    // the same composition stays graphless; another one, captured with room, gets its graph
    ok = true;
    int before = a.forwards;
    for (int i = 0; i < 2; ++i) ok = window(2000 + i).ok() && ok;
    check(ok && a.forwards == before + 2 && sh.graphs.size() == 0, "the graphless composition keeps running direct, one forward a window");
    const int32_t w0[2] = {1, 1}, w1[2] = {2, 2};
    ok = true;
    before = a.forwards;
    for (int i = 0; i < 3; ++i) {
        DecodeRequest r[2];
        r[0].tokens = w0, r[0].n_tokens = 2, r[0].pos = 100 + i, r[0].next_token = &out0;
        r[1].tokens = w1, r[1].n_tokens = 2, r[1].pos = 100 + i, r[1].next_token = &out1;
        ok = a.decode_batch(peers, r, 2).ok() && ok;
    }
    check(ok && sh.graphs.size() == 1, "a composition captured with room again keeps its graph");
    check(a.forwards == before + 1 + 2, "its windows: direct, direct plus capture, then the graph");
    cudaStreamDestroy(st);
    std::printf(failures == 0 ? "all ok\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
