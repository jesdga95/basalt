// include/basalt/core/gdn_slots.hpp - the device buffers behind a checkpoint's GDN slots.
//
// The recurrence cannot be recomputed from K/V, so a checkpoint without it is unusable, and it is the largest part of
// one.  A slot keeps a checkpoint's state device-resident, so restoring it copies device to device instead
// of over the host.  The session keeps its own buffer: the captured commit graph binds that buffer's address,
// so a slot is a mirror of the state, not a stand-in for it.  Which checkpoint holds which index - and what
// happens to the index when the checkpoint is copied, moved or dropped - is GdnSlotRef (gdn_slot_book.hpp).
#pragma once

#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/gdn_slot_book.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::core {

/// The device buffers behind the book, on one device.
class GdnSlots {
public:
    GdnSlots() = default;
    ~GdnSlots();
    GdnSlots(const GdnSlots&) = delete;
    GdnSlots& operator=(const GdnSlots&) = delete;

    /// Allocates `n` buffers of `bytes` each on the current device, and a book that knows them.  False (and none
    /// allocated) when they do not fit: a checkpoint then keeps host bytes.
    bool reserve(int n, size_t bytes, std::string& error);
    bool empty() const { return book_.slots() == 0; }
    int slots() const { return book_.slots(); }
    uint64_t bytes() const { return (uint64_t) book_.slots() * bytes_; }
    /// The slot's device buffer (null when the slot is invalid or the pool is empty).
    void* ptr(int slot) const;
    GdnSlotBook& book() { return book_; }
    const GdnSlotBook& book() const { return book_; }
    int device() const { return device_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    GdnSlotBook book_;
    size_t bytes_ = 0;
    int device_ = -1;
};

}  // namespace basalt::core
