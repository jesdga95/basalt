// include/basalt/core/gdn_slot_book.hpp - which device slot holds which checkpoint's recurrence, and who holds it.
//
// A slot is a device mirror of a checkpoint's GDN state: restoring through one copies device to device instead of
// over the host.  The pool is fixed-size, so an index means something only while exactly one checkpoint holds it.
// These two classes are that rule; the buffers behind it are `GdnSlots` (gdn_slots.hpp).
#pragma once

#include <cstdint>
#include <vector>

namespace basalt::core {

/// Which slot holds which seqlen, and which are free.  Pure bookkeeping: the caller owns the buffers, so the
/// allocation policy can be walked in a test without a device.
class GdnSlotBook {
public:
    /// `slots` free slots, none holding a state.
    void reset(int slots) {
        seqlen_.assign((size_t) (slots > 0 ? slots : 0), -1);
        free_.clear();
        for (int i = (int) seqlen_.size(); i-- > 0;) free_.push_back(i);
    }
    int slots() const { return (int) seqlen_.size(); }
    /// A free slot for a checkpoint of `seqlen` tokens, or -1 when none is free.
    int take(int64_t seqlen) {
        if (free_.empty()) return -1;
        const int slot = free_.back();
        free_.pop_back();
        seqlen_[(size_t) slot] = seqlen;
        return slot;
    }
    void release(int slot) {
        if (slot < 0 || slot >= slots() || seqlen_[(size_t) slot] < 0) return;
        seqlen_[(size_t) slot] = -1;
        free_.push_back(slot);
    }
    /// The seqlen the slot holds, or -1 for a free or invalid slot.
    int64_t seqlen(int slot) const {
        return slot >= 0 && slot < slots() ? seqlen_[(size_t) slot] : -1;
    }
    int used() const {
        int n = 0;
        for (int64_t s : seqlen_) n += s >= 0 ? 1 : 0;
        return n;
    }
    /// Slots a checkpoint may still take.  `slots() - used()` is the same number; this names it for the tests that
    /// check a dropped checkpoint gives its slot back.
    int free() const { return (int) free_.size(); }

private:
    std::vector<int64_t> seqlen_;   // -1: free
    std::vector<int> free_;
};

/// One slot index, held by exactly one checkpoint.
///
/// A copy holds none: the source keeps its slot, so a checkpoint copied out of a chain cannot name a buffer the
/// pool will lend again.  A move transfers the index, so replacing, pruning or dropping a chain gives its slots
/// back without the caller remembering to.  The moves are `noexcept` because they must be the ones a growing
/// vector takes: copying instead would leave the surviving element naming no slot, and a checkpoint with a slot
/// keeps no host bytes, so it would stop being restorable.
class GdnSlotRef {
public:
    /// The held index, or -1.
    int index() const { return index_; }
    /// Takes a free slot for a state of `seqlen` tokens, giving back any slot this already holds.  False when the
    /// pool has none free: the checkpoint then keeps host bytes.
    bool acquire(GdnSlotBook& book, int64_t seqlen) {
        const int slot = book.take(seqlen);
        if (slot < 0) return false;
        reset();
        book_ = &book;
        index_ = slot;
        return true;
    }
    /// Gives the slot back now rather than at destruction: a checkpoint that publishes host bytes returns its slot
    /// before the chain that owned it is done with it, so the next save can take it.
    void release() {
        if (book_ != nullptr) book_->release(index_);
        reset();
    }

    GdnSlotRef() = default;
    GdnSlotRef(const GdnSlotRef&) {}                        // a copy owns nothing
    GdnSlotRef& operator=(const GdnSlotRef&) { release(); return *this; }
    GdnSlotRef(GdnSlotRef&& o) noexcept : book_(o.book_), index_(o.index_) { o.reset(); }
    GdnSlotRef& operator=(GdnSlotRef&& o) noexcept {
        if (this != &o) { release(); book_ = o.book_; index_ = o.index_; o.reset(); }
        return *this;
    }
    ~GdnSlotRef() { release(); }

private:
    void reset() { book_ = nullptr; index_ = -1; }

    GdnSlotBook* book_ = nullptr;
    int index_ = -1;
};

}  // namespace basalt::core
