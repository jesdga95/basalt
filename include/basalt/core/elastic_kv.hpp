// include/basalt/core/elastic_kv.hpp - the elastic KV's shared figures: the cells a session backs at start (more come out of
// the expert cache when a request needs them and go back when it does not), and the start line that states the ceiling.
#pragma once

#include "basalt/log.hpp"

#include <cstdint>

namespace basalt::core {

constexpr int64_t kElasticKvBase = 32768;       ///< cells a session backs at start, with a context above it
constexpr int64_t kElasticKvBaseBatch = 4096;   ///< the same for each session of --batch's slots

/// The start line: `cells` fit in `sessions` sessions (`full` at full context) before the expert cache reaches the
/// `floor_slots` slots a prompt chunk needs.
inline void log_elastic_kv_ceiling(int64_t cells, int sessions, int64_t full, int64_t floor_slots) {
    basalt::log::info("basalt serve", "elastic KV: about %lld cells fit in the %d session%s (%lld at full context) before the expert cache "
                      "reaches the %lld slots a prompt chunk needs",
                      (long long) cells, sessions, sessions == 1 ? "" : "s", (long long) full, (long long) floor_slots);
}

}  // namespace basalt::core
