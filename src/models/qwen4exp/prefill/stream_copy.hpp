#pragma once

#include <cstdint>

namespace basalt::prefill::detail {

struct StreamEntry {
    int32_t l, e;
    const uint8_t* blob;
    int job;
    const uint8_t* pdev;
};

class PinnedCopyDecision {
public:
    template<class WaitForSlot, class RecordedLayer, class ReadyCounts>
    bool skip(const StreamEntry& entry, bool has_snapshot, bool slot_live,
              WaitForSlot wait_for_slot, RecordedLayer recorded_layer, ReadyCounts ready_counts) {
        if (!has_snapshot || entry.job >= 0 || entry.pdev != nullptr) return false;
        if (layer_ != entry.l) {
            layer_ = entry.l;
            counts_ = nullptr;
        }
        // A reused slot already depends on its previous consumer. Wait there before probing routing.
        if (!counts_ && slot_live) wait_for_slot();
        if (!counts_ && layer_ <= recorded_layer()) counts_ = ready_counts(layer_);
        return counts_ && counts_[entry.e] == 0;
    }

private:
    int32_t layer_ = -1;
    const uint32_t* counts_ = nullptr;
};

}  // namespace basalt::prefill::detail
