// tests/model/gpu_check_test.cpp - the GPU check's trace file and its comparison (no GPU): equal runs are bitwise, one flipped
// bit is not, and a different argmax shows in the picks, the top-1 and the worst step KLD.
#include "basalt/model/gpu_check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace basalt::model;

static int g_fail = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++g_fail;                                                             \
        }                                                                         \
    } while (0)

static GpuTrace make(int steps, int64_t V) {
    GpuTrace t;
    t.n_vocab = V;
    t.logits.resize((size_t) steps * (size_t) V);
    for (int i = 0; i < steps; ++i) {
        for (int64_t v = 0; v < V; ++v) t.logits[(size_t) i * (size_t) V + (size_t) v] = (float) ((v * 7 + i * 3) % 11) * 0.37f;
        t.tokens.push_back((int32_t) ((i * 3 + 10) % V));   // the argmax of the row built above
    }
    return t;
}

int main() {
    const GpuTrace a = make(5, 64);
    GpuTrace b = a;
    GpuCompare c = compare_gpu_traces(a, b);
    CHECK(c.same() && c.steps == 5 && c.bitwise == 5 && c.kld == 0 && c.top1 == 1.0);
    // one low bit of one logit: not bitwise, picks and top-1 unchanged
    b.logits[3 * 64 + 5] = std::nextafterf(b.logits[3 * 64 + 5], 1e9f);
    c = compare_gpu_traces(a, b);
    CHECK(!c.same() && c.bitwise == 4 && c.first_diff == 3 && c.first_token_diff < 0 && c.top1 == 1.0 && c.max_abs > 0);
    // another pick
    b = a;
    b.tokens[2] = 1;
    c = compare_gpu_traces(a, b);
    CHECK(!c.same() && c.first_token_diff == 2 && c.bitwise == 5);
    // a flipped argmax moves the KLD and the top-1
    b = a;
    b.logits[1 * 64 + 0] += 50.f;
    c = compare_gpu_traces(a, b);
    CHECK(c.kld > 1.0 && c.top1 < 1.0 && c.max_kld > c.kld);   // one step of five holds all of it
    // the file round-trips
    const std::string path = (std::filesystem::temp_directory_path() / "gpu_check_test.trace").string();
    std::string err;
    CHECK(write_gpu_trace(path, a, err));
    GpuTrace r;
    CHECK(read_gpu_trace(path, r, err) && r.tokens == a.tokens && r.logits == a.logits && r.n_vocab == 64);
    std::remove(path.c_str());
    CHECK(!read_gpu_trace(path, r, err));
    CHECK(gpu_compare_line(compare_gpu_traces(a, a)).find("BITWISE") != std::string::npos);
    std::printf(g_fail ? "gpu_check_test: FAILED\n" : "gpu_check_test: ok\n");
    return g_fail ? 1 : 0;
}
