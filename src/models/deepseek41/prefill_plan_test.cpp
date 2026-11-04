// src/models/deepseek41/prefill_plan_test.cpp - the prompt chunk choice and the cache give-back rule (host only).
#include "models/deepseek41/prefill_plan.hpp"

#include <cstdio>

using namespace basalt::models::deepseek41::prefillp;

static int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

int main() {
    constexpr uint64_t GiB = 1ull << 30;
    Dims d;
    d.n_embd = 5120; d.n_used = 6; d.n_expert = 256; d.n_ff_exp = 2304; d.engram_cols = 8 * 256;
    d.expert_blob_bytes = 6ull << 20;

    CHECK(per_token_bytes(d) > 0 && fixed_bytes(d) > 0);
    CHECK(workspace_bytes(d, 2048) - workspace_bytes(d, 1024) == per_token_bytes(d) * 1024);

    // plenty of memory: the full chunk
    Fit f = plan_chunk(d, 24 * GiB, kMargin);
    CHECK(f.chunk == kMaxChunk && f.error.empty());
    // less memory: a smaller power-of-two fraction whose workspaces and margin fit
    const uint64_t w4096 = workspace_bytes(d, 4096) + kMargin;
    f = plan_chunk(d, w4096, kMargin);
    CHECK(f.chunk == 4096);
    f = plan_chunk(d, w4096 - 1, kMargin);
    CHECK(f.chunk == 2048);
    // never below the least chunk, with a message naming the memory
    f = plan_chunk(d, 1 * GiB, kMargin);
    CHECK(f.chunk == 0 && f.error.find("not enough free device memory") != std::string::npos);
    f = plan_chunk(d, 0, 0);
    CHECK(f.chunk == 0 && !f.error.empty());
    // an explicit range
    f = plan_chunk(d, 24 * GiB, kMargin, 1024, 512);
    CHECK(f.chunk == 1024);

    // give-back: a short context keeps what it has; a long one drops to the kept positions
    const std::vector<int> ratios = {2, 2, 2, 1};
    CHECK(keep_positions(1000, ratios) == 1000);
    CHECK(keep_positions(kKeepPositions, ratios) == kKeepPositions);
    CHECK(keep_positions(kKeepPositions + 4096, ratios) == kKeepPositions + 4096);   // too little to free
    CHECK(keep_positions(700000, ratios) == kKeepPositions);
    CHECK(keep_positions(1048576, ratios) == kKeepPositions);

    if (g_fail) return 1;
    std::printf("prefill_plan_test: ok\n");
    return 0;
}
