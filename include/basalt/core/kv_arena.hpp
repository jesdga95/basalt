// include/basalt/core/kv_arena.hpp - the elastic KV pools of one QSA layer.
//
// Each pool array (K and V codes, their scales) gets an address range sized for the whole context, but only the
// pages in use are backed by VRAM.  The addresses never move, so the captured graphs and every kernel argument that
// points into a pool stay valid; mapping more pages or giving the tail back changes only the backing.  The pools'
// layout is page-major ([page][kv_head][page_size][...]), so the first `n` pages of every array are a prefix of it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

class KvArena {
public:
    KvArena() = default;
    ~KvArena();
    KvArena(const KvArena&) = delete;
    KvArena& operator=(const KvArena&) = delete;

    /// Reserves one range per array, `page_bytes[i] * pages + tail_bytes[i]` bytes each (rounded up to the
    /// granularity).  The tail is a few rows past the last page that an array keeps once it has any page (the
    /// indexer's spare rows).  Nothing is backed yet.  False with `err` when the driver offers no virtual memory
    /// management.
    bool open(const std::vector<uint64_t>& page_bytes, int64_t pages, int64_t page_cells, std::string& err,
              const std::vector<uint64_t>& tail_bytes = {});
    /// Array `i` is a table, not a cache: every byte `map` backs is filled from `bytes` (from the same offset)
    /// instead of zeroed, as far as `bytes` reaches.
    void set_contents(size_t i, std::vector<uint8_t> bytes);
    /// `map` splits each array's new backing at the grid lines of `cells` cells (0: one mapping per call), so a mapping
    /// starts on a grid line or where an earlier `map` stopped; `unmap` gives back every mapping that starts at or above
    /// its cut, however few calls backed it.
    void set_extent_cells(int64_t cells) { extent_cells_ = std::max<int64_t>(cells, 0); }
    void close();
    bool valid() const { return !arrays_.empty(); }

    uint8_t* array(size_t i) const { return i < arrays_.size() ? arrays_[i].base : nullptr; }
    int64_t page_cells() const { return page_cells_; }
    int64_t max_cells() const { return pages_ * page_cells_; }
    /// The cells every array has backed now (a whole number of pages).
    int64_t cells() const { return cells_; }
    /// VRAM backing the arrays now, and what backing the first `cells` cells takes.
    uint64_t mapped_bytes() const;
    uint64_t bytes_for(int64_t cells) const;
    /// The VRAM `map(cells)` would add (0 when `cells` is covered already).
    uint64_t grow_bytes(int64_t cells) const;

    /// Backs at least the first `cells` cells (whole pages, at most the context).  The new bytes are zeroed on the
    /// legacy stream before this returns, so a fresh page reads as a page a fresh session zeroed.  On failure what was
    /// mapped stays and `err` says why.
    bool map(int64_t cells, std::string& err);
    /// Gives back the backing past the first `cells` cells, as far as whole mappings allow (a mapping that also holds
    /// cells below `cells` stays).  Waits for the device first.
    bool unmap(int64_t cells, std::string& err);

private:
    struct Extent { uint64_t off, bytes; unsigned long long handle; };
    struct Array {
        uint8_t* base = nullptr;
        uint64_t page_bytes = 0;
        uint64_t tail_bytes = 0;
        uint64_t reserved = 0;
        std::vector<uint8_t> contents;
        std::vector<Extent> extents;   // back to back from offset 0
        uint64_t mapped() const { return extents.empty() ? 0 : extents.back().off + extents.back().bytes; }
    };
    uint64_t need(const Array& a, int64_t cells) const;
    void recount();
    std::vector<Array> arrays_;
    int64_t pages_ = 0;
    int64_t page_cells_ = 0;
    int64_t cells_ = 0;
    int64_t extent_cells_ = 0;
    uint64_t gran_ = 0;
};

}  // namespace basalt::core
