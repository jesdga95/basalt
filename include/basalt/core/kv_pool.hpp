// include/basalt/core/kv_pool.hpp - the device bytes every session of a model backs its per-position KV cells from on one
// card. A session reserves the address range of a whole context per cache (ElasticArray) and maps it as its conversation
// grows; the pool counts what all of them have mapped, so one session can hold most of it while others run short ones.
#pragma once

#include <cstdint>

namespace basalt::core {

class KvPool {
public:
    void set_budget(uint64_t bytes) { budget_ = bytes; }
    uint64_t budget() const { return budget_; }
    uint64_t taken() const { return taken_; }
    uint64_t free() const { return taken_ < budget_ ? budget_ - taken_ : 0; }
    /// Counts `bytes` as mapped; false, with nothing counted, when fewer than that are free.
    bool take(uint64_t bytes) {
        if (bytes > free()) return false;
        taken_ += bytes;
        return true;
    }
    /// Counts `bytes` as given back (never more than are taken).
    void give(uint64_t bytes) { taken_ -= bytes < taken_ ? bytes : taken_; }

private:
    uint64_t budget_ = 0, taken_ = 0;
};

}  // namespace basalt::core
