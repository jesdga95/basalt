// include/basalt/program/auto_plan.hpp - the start-up picks for the settings the user left out. One pure function from
// plain values; the engine reads the machine, fills AutoInputs, and applies the AutoPlan.
#pragma once

#include <cstdint>
#include <functional>
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

/// Where one request's K/V lives: in pinned host RAM (streamed, the VRAM tail stays expert slots) or in the elastic
/// VRAM tail.  Decided per request, before its prompt is read.
struct KvHostInputs {
    int64_t cells = 0;                    ///< the cells the request can hold: prompt + new tokens + slack
    int64_t crossover_cells = 0;          ///< below this many cells the VRAM tail is the faster place
    uint64_t bytes_per_cell = 0;          ///< host K/V bytes per cell over every QSA layer
    uint64_t ram_available = 0;           ///< bytes free now
    uint64_t ram_kept = 0;                ///< bytes that stay free beside the pin (kv_host_keep)
    bool ram_low = false;                 ///< the RAM guard is below its floor: a pool held is given back, not kept
    int64_t read_cells = 0;               ///< the prompt cells the request reads (the rest it reuses)
    int64_t fresh_cells = 0;              ///< a session not streamed yet takes a pool only for a request that reads at
                                          ///< most this many (0: any): a long streamed read costs more than it gains
};

/// The RAM a pin leaves free beside it: the guard's floor and what serving can still take, i.e. its budget less what
/// the session already holds of it (its checkpoints, the parked conversations).
uint64_t kv_host_keep(uint64_t guard_floor, uint64_t serve_budget, uint64_t serve_held);

struct KvHostChoice {
    bool host = false;
    uint64_t pin_bytes = 0;               ///< what the host K/V takes when `host`
    const char* why = "";                 ///< "shallow", "fresh", "no RAM" or "host"
};

KvHostChoice choose_kv_host(const KvHostInputs& in);

/// The cells a request's host K/V is pinned for: the longer of its prompt and the snapshot it resumes (`need`), its
/// new tokens and `slack`, rounded up to `grid` cells so a conversation that grows a little keeps its pool, at most
/// `max_cells`.
int64_t kv_host_cells(int64_t need, int64_t max_new, int64_t slack, int64_t grid, int64_t max_cells);

/// What a request's admission does with the session's host K/V.
enum class KvHostAct {
    none,   ///< stays in the VRAM tail, no host K/V held
    take,   ///< pins a pool and moves the session's cells into it
    keep,   ///< the pool held covers the request: streamed as it is
    grow,   ///< a larger pool; the cells move over, the old one is freed
    drop,   ///< back to the VRAM tail; the cells move back, the pool is freed
};

struct KvHostPlan {
    bool host = false;                    ///< the request runs streamed
    KvHostAct act = KvHostAct::none;
    int64_t cells = 0;                    ///< the pool's cells after the step (0: none)
    uint64_t pin_bytes = 0;               ///< RAM the step pins on top of what the session holds
    const char* why = "";
};

/// `pool_cells`: the cells of the pool the session holds now (0: none, the session is in the VRAM tail).  A pool that
/// covers a deep request is kept (it is pinned already) unless `ram_low`; a larger one must fit beside it.
KvHostPlan plan_kv_host(const KvHostInputs& in, int64_t pool_cells);
/// Whether a pool of `pool_cells` can still serve a request that starts decoding at `at` cells: its prompt and the
/// next decode window (`window` cells); the output past it grows the pool between windows or stops the request.
bool kv_host_pool_serves(int64_t pool_cells, int64_t at, int64_t window);
const char* kv_host_act_name(KvHostAct a);

/// Whether cutting the expert cache to its first `cut` slots gives up an expert RAM holds no copy of: `res[i]` is the
/// slot of expert i = layer * n_expert + expert (negative: none), `has_copy(layer, expert)` whether RAM holds it.
bool cut_gives_up_bare(const std::vector<int32_t>& res, int64_t cut, int64_t n_expert,
                       const std::function<bool(int64_t, int64_t)>& has_copy);

/// "auto flags: peer-exclusive off, resident 51.2 GB, pool-affinity all, pool-workers 15", only the auto-picked keys;
/// empty when none was.
std::string auto_flags_line(const AutoPlan& plan);

}  // namespace basalt::program
