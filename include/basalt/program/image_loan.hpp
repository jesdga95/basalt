// include/basalt/program/image_loan.hpp - the expert-cache room an image encode may borrow, the order the elastic KV
// gives its spare cells back in when that room is short, and the elastic KV's ceiling.  Host only: no GPU runtime,
// no model.
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace basalt::program {

/// The first cache slot an encode may borrow from: every slot from it up to `live` is empty or holds an expert its
/// refill can read back from RAM.  `slot_of[i]`: the slot expert i sits in (negative: none).  `in_ram[i]`: expert i
/// has a RAM copy; empty means every expert has one.
int64_t image_lend_floor(const std::vector<int32_t>& slot_of, const std::vector<uint8_t>& in_ram, int64_t live);

/// One session's elastic KV: the cells it backs now, the cells it must keep (what it holds, and a decoding slot's
/// next window), and whether a request is decoding in it.
struct KvHold {
    int64_t backed = 0;
    int64_t keep = 0;
    bool active = false;
};

/// The sessions that give spare cells back for an encode, in order: idle ones first, then decoding ones, the largest
/// spare first within each.  Each entry is (session index, cells it keeps), never below its `keep` nor `floor`; a
/// session with nothing past that is left out.
std::vector<std::pair<int, int64_t>> kv_giveback_order(const std::vector<KvHold>& holds, int64_t floor);

/// The most granules the elastic KV can take off the expert cache's tail while the cache keeps `floor` slots, as the
/// KV's own fit tests it: `mapped` is the bytes the cache backs now (whole granules of `gran` bytes), and
/// `slots_after(keep)` the slots a shrink to `keep` bytes leaves.
int64_t kv_cache_spare_granules(int64_t mapped, int64_t gran, int64_t floor,
                                const std::function<int64_t(int64_t)>& slots_after);

/// The elastic KV's ceiling: the cells the sessions can hold together when each in turn backs what it can of `spare`
/// bytes, on the 256-cell grid and up to its context.  `backed[i]`: the cells session i backs now (negative: it is not
/// elastic and holds its whole context); `max_cells[i]`: its context; `grow(i, cells)`: the bytes session i would
/// add to back `cells`.
int64_t kv_ceiling_cells(const std::vector<int64_t>& backed, const std::vector<int64_t>& max_cells, int64_t spare,
                         const std::function<int64_t(size_t, int64_t)>& grow);

}  // namespace basalt::program
