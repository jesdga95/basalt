// basalt/program/serve/loans.cpp - see loans.hpp.
#include "loans.hpp"

#include <algorithm>
#include <limits>

#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/program/multi_gpu_plan.hpp"   // prompt_one_card_chunks, prompt_chunk_list

#include "../engine_local.hpp"   // biggest_chunk

namespace basalt::program::serve {

int64_t cache_slots_for(const basalt::core::ExpertCache& xc, uint64_t need) {
    if (xc.slot_offsets() != nullptr) {   // sized slots: from the end until they hold `need`
        int64_t k = 0;
        while (k < xc.slots() &&
               (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[xc.slots() - k]) < need) ++k;
        return k;
    }
    const int64_t blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
    return (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
}

int64_t lend_slots(const basalt::core::ExpertCache& xc, const basalt::core::ModelGeometry& g,
                   const basalt::core::SessionState& ss, int64_t cells) {
    return cache_slots_for(xc, basalt::prefill::Prefill::bytes_needed(g, ss, cells));
}

int64_t request_chunk(int64_t tokens, int64_t max_chunk) {
    if (tokens <= 0 || max_chunk <= 0) return 0;
    const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                ? tokens
                                : ((tokens + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}

std::pair<int64_t, int64_t> one_card_chunks(int64_t tokens, int64_t cap, bool head_ok, bool auto_fit) {
    constexpr int64_t kHeadChunk = 16384;   // the first chunk of a prompt above the chunk cap
    return basalt::program::prompt_one_card_chunks(tokens, cap, head_ok && auto_fit, kHeadChunk);
}

std::vector<int64_t> chunk_list(int64_t tokens, int64_t chunk, int64_t head) {
    return basalt::program::prompt_chunk_list(tokens, chunk, head);
}

int64_t plan_lend(int64_t& chunk, const basalt::core::ExpertCache& xc, const basalt::core::ModelGeometry& g,
                  const basalt::core::SessionState& ss, bool prefill_auto, int64_t auto_ceiling, int auto_lend_pct) {
    const auto slots_for = [&](int64_t cells) { return lend_slots(xc, g, ss, cells); };
    if (prefill_auto) {
        // The scan bisects on the 256-token grid `request_chunk` rounds to: `bytes_needed` is a sum of (T x positive
        // constant) terms plus a max of such sums, so it rises monotonically with T and so does every test below.

        const int64_t c = basalt::program::engine_local::biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
            const int64_t k = slots_for(t);
            return k + 128 <= xc.slots() && k * 100 <= auto_lend_pct * xc.slots();
        });
        if (c > 0) { chunk = c; return slots_for(c); }
        return 0;
    }
    for (int64_t c = chunk; c >= 256; c /= 2) {
        const int64_t k = slots_for(c);
        if (k + 128 <= xc.slots()) { chunk = c; return k; }
    }
    return 0;
}

}  // namespace basalt::program::serve
