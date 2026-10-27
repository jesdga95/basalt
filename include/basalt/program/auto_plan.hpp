// include/basalt/program/auto_plan.hpp - the start-up picks for the settings the user left out. One pure function from
// plain values; the engine reads the machine, fills AutoInputs, and applies the AutoPlan.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::program {

enum class AutoAffinity { All, Auto, PCores };

/// Which settings the user set (config file or command line), and the values they gave.
struct AutoGiven {
    bool peer_exclusive = false;
    bool resident = false;
    bool pool_affinity = false;
    bool pool_workers = false;
    bool peer_exclusive_value = false;
    double resident_gb = 0;               ///< GiB
    AutoAffinity pool_affinity_value = AutoAffinity::All;
    int pool_workers_value = 0;
};

struct AutoInputs {
    uint64_t ram_total = 0;               ///< bytes
    uint64_t ram_available = 0;           ///< bytes
    std::vector<uint64_t> vram_free;      ///< bytes per card, card 0 first
    bool peer = false;                    ///< a second card serves as the peer expert tier
    bool serve = false;                   ///< peer-exclusive needs the serving mode
    // CPU topology: worker_cores = cores left for pool workers once the host thread has its own.
    int worker_cores = 1;
    int p_cores = 0;
    int e_cores = 0;
    bool hybrid = false;
    double cpu_quota = 0;                 ///< cgroup CPU quota in CPUs, 0 = none
    // Pack sizes in bytes.
    uint64_t complement_full = 0;         ///< experts no GPU tier holds, with the peer's also in RAM
    uint64_t complement_peer_excl = 0;    ///< the same with the peer's experts held only on the peer
    uint64_t extra_full = 0;              ///< optional copies beside complement_full (lent slots, adapt spares): kept as far as RAM allows
    uint64_t extra_peer_excl = 0;         ///< the same beside complement_peer_excl
    uint64_t floor_bytes = 0;             ///< RAM that must stay free beside the copy when every complement expert is in it
    // The floor when some experts are read from the file instead (platform::ram_floor, plus late_bytes).
    uint64_t margin_bytes = 0;
    uint64_t late_bytes = 0;              ///< RAM the engine takes after the copy
    uint64_t layer_bytes = 0;             ///< every expert of the largest layer
    uint64_t prompt_file_bytes = 0;       ///< what a prompt chunk reads from the file for lent slots the copy leaves out
    AutoGiven given;
};

enum class AutoSource { Auto, Given };

struct AutoPlan {
    bool peer_exclusive = false;
    double resident_gb = 0;               ///< GiB of experts held in RAM
    uint64_t resident_bytes = 0;          ///< the same in bytes, exact: what the RAM copy is sized by
    AutoAffinity pool_affinity = AutoAffinity::All;
    int pool_workers = 1;
    AutoSource peer_exclusive_src = AutoSource::Auto;
    AutoSource resident_src = AutoSource::Auto;
    AutoSource pool_affinity_src = AutoSource::Auto;
    AutoSource pool_workers_src = AutoSource::Auto;
};

/// The pool size used when no worker count is given.
int auto_pool_workers(const AutoInputs& in, AutoAffinity affinity);

AutoPlan choose_auto(const AutoInputs& in);

const char* auto_affinity_name(AutoAffinity a);

/// "auto flags: peer-exclusive off, resident 51.2 GB, pool-affinity all, pool-workers 15", only the auto-picked keys;
/// empty when none was.
std::string auto_flags_line(const AutoPlan& plan);

}  // namespace basalt::program
