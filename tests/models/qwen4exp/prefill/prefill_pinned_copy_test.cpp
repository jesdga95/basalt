#include "models/qwen4exp/prefill/stream_copy.hpp"

#include <cstdio>
#include <string>
#include <string_view>

using basalt::prefill::detail::PinnedCopyDecision;
using basalt::prefill::detail::StreamEntry;

namespace {

struct Routing {
    Routing(int64_t recorded_layer, int32_t ready_layer, bool completes_on_wait = false)
        : recorded(recorded_layer), ready(ready_layer), ready_after_wait(completes_on_wait) {}

    int64_t recorded;
    int32_t ready;
    bool ready_after_wait;
    const uint32_t counts[2] = {0, 7};
    std::string calls;

    bool skip(PinnedCopyDecision& decision, const StreamEntry& entry, bool snapshot, bool slot_live) {
        calls.clear();
        return decision.skip(entry, snapshot, slot_live,
            [&] {
                calls += 'W';
                if (ready_after_wait) ready = entry.l;
            },
            [&] {
                calls += 'R';
                return recorded;
            },
            [&](int32_t layer) -> const uint32_t* {
                calls += 'Q';
                return ready == layer ? counts : nullptr;
            });
    }
};

}  // namespace

int main() {
    constexpr int32_t layer = 4;
    const uint8_t blob = 0;
    const StreamEntry cold{layer, 0, &blob, -1, nullptr};
    const StreamEntry hot{layer, 1, &blob, -1, nullptr};
    const StreamEntry next_layer{layer + 1, 0, &blob, -1, nullptr};
    const StreamEntry staged{layer, 0, &blob, 0, nullptr};
    const StreamEntry peer{layer, 0, nullptr, -1, &blob};
    int failures = 0;
    auto check = [&](const char* name, PinnedCopyDecision& decision, Routing& routing,
                     const StreamEntry& entry, bool snapshot, bool slot_live,
                     bool expected_skip, std::string_view expected_calls) {
        const bool skip = routing.skip(decision, entry, snapshot, slot_live);
        if (skip == expected_skip && routing.calls == expected_calls) {
            std::printf("PASS %s\n", name);
        } else {
            std::printf("FAIL %s: skip=%d, calls='%s'; expected skip=%d, calls='%.*s'\n",
                        name, skip, routing.calls.c_str(), expected_skip,
                        (int) expected_calls.size(), expected_calls.data());
            ++failures;
        }
    };

    PinnedCopyDecision reused;
    Routing finishes_during_wait{layer, -1, true};
    check("reused slot, cold route completes during wait", reused, finishes_during_wait, cold, true, true, true, "WRQ");

    PinnedCopyDecision hot_decision;
    Routing complete{layer, layer};
    check("reused slot, hot route", hot_decision, complete, hot, true, true, false, "WRQ");

    PinnedCopyDecision pending_decision;
    Routing pending{layer, -1};
    check("recorded route still pending after wait", pending_decision, pending, cold, true, true, false, "WRQ");

    PinnedCopyDecision fresh;
    check("fresh slot, ready cold route", fresh, complete, cold, true, false, true, "RQ");
    check("reuse same-layer counts without another wait or query", fresh, complete, cold, true, true, true, "");
    check("discard previous layer's counts on slot reuse", fresh, complete, next_layer, true, true, false, "WR");

    PinnedCopyDecision bypass;
    check("missing route snapshot", bypass, complete, cold, false, true, false, "");
    check("staged copy bypasses route probing", bypass, complete, staged, true, true, false, "");
    check("peer copy bypasses route probing", bypass, complete, peer, true, true, false, "");
    return failures != 0;
}
