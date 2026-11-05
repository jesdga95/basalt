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
    // a read's rows left short: nothing from one window before its stop on reads one, at the last layer either
    {
        constexpr int kLayers = 40, kSkip = 20, kWin = 128;
        constexpr int64_t kStop = 100000;
        const int64_t cut = far_below(kStop, kLayers, kSkip, kWin);
        const int64_t whole = whole_from(kStop, kWin);
        CHECK(cut == kStop - 2668);
        // the last layer's input is wrong below this; a token at `whole` reads its window from whole - (kWin - 1)
        CHECK(whole - (kWin - 1) >= cut + (kLayers - 1 - kSkip) * (kWin - 1));
        CHECK(whole_from(50, kWin) == 0);
        // saves and resumes: at the stop and past it, not inside the part
        CHECK(may_save(kStop, whole) && may_save(kStop + 7, whole) && may_save(whole, whole));
        CHECK(!may_save(whole - 1, whole) && !may_save(cut, whole) && !may_save(cut - 1, whole));
        CHECK(may_resume(kStop, kStop, whole) && may_resume(whole, kStop, whole));
        CHECK(!may_resume(whole - 1, kStop, whole) && !may_resume(cut, kStop, whole) && !may_resume(1, kStop, whole));
        // forward through the part, and a new sequence, need nothing behind them
        CHECK(may_resume(cut, cut, whole) && may_resume(0, kStop, whole));
    }
    Dims d;
    d.n_embd = 5120; d.n_used = 6; d.n_expert = 256; d.n_ff_exp = 2304; d.engram_cols = 8 * 256;
    d.expert_blob_bytes = 6ull << 20;

    CHECK(per_token_bytes(d) > 0 && fixed_bytes(d, 1024) > 0);
    CHECK(workspace_bytes(d, 8192) - workspace_bytes(d, 4096) == per_token_bytes(d) * 4096);
    // a short chunk's launches hold its rows only; the long ones are capped
    CHECK(stage_rows(6, 256) == kShortChunk * 6 && stage_rows(6, 1024) == kShortChunk * 6);
    CHECK(stage_rows(6, 2048) == 2048 * 6 && stage_rows(6, 16384) == kStageRows);
    CHECK(fixed_bytes(d, 1024) < fixed_bytes(d, 4096));

    constexpr int kLong = 1 << 20;   // tokens of a prompt longer than any chunk
    // plenty of memory: the full chunk
    Fit f = plan_chunk(d, 24 * GiB, kMargin, kLong);
    CHECK(f.chunk == kMaxChunk && f.error.empty());
    // less memory: a smaller power-of-two fraction whose workspaces, staged groups and margin fit
    const uint64_t w4096 = chunk_bytes(d, 4096, kLong) + kMargin;
    f = plan_chunk(d, w4096, kMargin, kLong);
    CHECK(f.chunk == 4096);
    f = plan_chunk(d, w4096 - 1, kMargin, kLong);
    CHECK(f.chunk == 2048);
    // never below the least chunk, with a message naming the memory
    f = plan_chunk(d, 1 * GiB, kMargin, kLong);
    CHECK(f.chunk == 0 && f.error.find("not enough free device memory") != std::string::npos);
    f = plan_chunk(d, 0, 0, kLong);
    CHECK(f.chunk == 0 && !f.error.empty());
    // an explicit range
    f = plan_chunk(d, 24 * GiB, kMargin, kLong, 1024, 512);
    CHECK(f.chunk == 1024);
    // a prompt a smaller chunk holds takes that one
    CHECK(plan_chunk(d, 24 * GiB, kMargin, 33).chunk == kMinChunk);
    CHECK(plan_chunk(d, 24 * GiB, kMargin, 554).chunk == 1024);
    CHECK(plan_chunk(d, 24 * GiB, kMargin, kMaxChunk / 2).chunk == kMaxChunk / 2);
    CHECK(plan_chunk(d, 24 * GiB, kMargin, kMaxChunk / 2 + 1).chunk == kMaxChunk);
    // the staged groups count from the pairs a chunk of this prompt holds
    CHECK(chunk_bytes(d, 1024, 554) == workspace_bytes(d, 1024));
    CHECK(chunk_bytes(d, 1024, 5000) == workspace_bytes(d, 1024));
    CHECK(chunk_bytes(d, 16384, 20000) == workspace_bytes(d, 16384) + ahead_bytes(d));
    // a chunk that fits only without its staged groups does not fit
    const int64_t pairs_chunk = (kAheadPairs + d.n_used - 1) / d.n_used;
    CHECK(chunk_bytes(d, (int) pairs_chunk, kLong) == workspace_bytes(d, (int) pairs_chunk) + ahead_bytes(d));
    CHECK(plan_chunk(d, workspace_bytes(d, 16384) + kMargin, kMargin, kLong).chunk < 16384);

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
