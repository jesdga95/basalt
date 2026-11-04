// src/core/ram_load.cpp - the shared RAM expert loader: unbuffered reads on a thread pool, then the pin.
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

namespace basalt::core {

void add_ram_blob(std::vector<RamRun>& runs, uint8_t* dst, uint64_t file_pos, uint64_t blob_bytes, int64_t expert,
                  uint64_t progress) {
    if (!runs.empty()) {
        RamRun& r = runs.back();
        if (r.dst != nullptr && r.file_pos + r.bytes == file_pos && r.dst + r.bytes == dst && r.blob_bytes == blob_bytes) {
            r.bytes += blob_bytes;
            ++r.count;
            r.progress += progress;
            return;
        }
    }
    RamRun r;
    r.dst = dst;
    r.file_pos = file_pos;
    r.bytes = blob_bytes;
    r.blob_bytes = blob_bytes;
    r.first = expert;
    r.count = 1;
    r.progress = progress;
    runs.push_back(r);
}

std::vector<std::vector<RamRun>> ram_arena_runs(const RamGrid& grid, uint8_t* base, const std::vector<uint64_t>& offsets,
                                                uint64_t no_place, const std::vector<uint8_t>& no_bar) {
    std::vector<std::vector<RamRun>> groups((size_t) grid.n_layers);
    for (int64_t l = 0; l < grid.n_layers; ++l) {
        const uint64_t blob = grid.blob_bytes[(size_t) l];
        for (int64_t e = 0; e < grid.n_expert; ++e) {
            const size_t i = (size_t) l * (size_t) grid.n_expert + (size_t) e;
            if (offsets[i] == no_place) continue;
            add_ram_blob(groups[(size_t) l], base + (size_t) offsets[i], grid.file_pos(l, e), blob, e, no_bar[i] ? 0 : blob);
        }
    }
    return groups;
}

std::vector<std::vector<RamRun>> ram_layer_runs(const RamGrid& grid, const std::vector<uint8_t*>& spans,
                                                const std::vector<int32_t>& slots, const std::vector<int32_t>& sunk) {
    std::vector<std::vector<RamRun>> groups((size_t) grid.n_layers);
    for (int64_t l = 0; l < grid.n_layers; ++l) {
        const uint64_t blob = grid.blob_bytes[(size_t) l];
        for (int64_t e = 0; e < grid.n_expert; ++e) {
            const size_t i = (size_t) l * (size_t) grid.n_expert + (size_t) e;
            if (slots[i] >= 0)
                add_ram_blob(groups[(size_t) l], spans[(size_t) l] + (size_t) slots[i] * blob, grid.file_pos(l, e), blob, e, blob);
            if (!sunk.empty() && sunk[i] >= 0) {
                RamRun r;
                r.file_pos = grid.file_pos(l, e);
                r.bytes = blob;
                r.blob_bytes = blob;
                r.first = e;
                r.count = 1;
                groups[(size_t) l].push_back(r);
            }
        }
    }
    return groups;
}

bool read_ram_runs(const RamLoad& L, RamLoadStats& stats, std::string& err) {
    constexpr uint64_t kAlign = 4096;
    const auto t0 = std::chrono::steady_clock::now();
    const int direct = L.unbuffered ? platform::open_unbuffered(L.fd) : -1;
    // what a sink's run needs at once, whatever the chunk size
    uint64_t stage_bytes = L.chunk;
    for (const auto& g : L.groups)
        for (const RamRun& r : g)
            if (r.dst == nullptr) stage_bytes = std::max(stage_bytes, r.bytes);
    stage_bytes += 2 * kAlign;

    std::atomic<int64_t> next{0}, groups_done{0};
    std::atomic<uint64_t> copied{0}, unbuffered{0};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    std::string fail_msg;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = m;
        failed.store(true);
    };
    auto worker = [&]() {
        uint8_t* stage = nullptr;   // for ranges that are not block aligned
        auto take_stage = [&]() {
            if (stage != nullptr) return;
            if (!L.stage_pinned) stage = (uint8_t*) platform::alloc_aligned((size_t) stage_bytes, kAlign);
            else if (cudaHostAlloc((void**) &stage, (size_t) stage_bytes, cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                stage = nullptr;
            }
        };
        struct StageFree {
            uint8_t*& p;
            bool pinned;
            ~StageFree() {
                if (p == nullptr) return;
                if (pinned) (void) cudaFreeHost(p);
                else platform::free_aligned(p);
            }
        } stage_free{stage, L.stage_pinned};
        const int64_t n_groups = (int64_t) L.groups.size();
        for (;;) {
            const int64_t g = next.fetch_add(1);
            if (g >= n_groups || failed.load()) return;
            const std::vector<RamRun>& runs = L.groups[(size_t) g];
            if (L.read_ahead && (direct < 0 || !platform::read_ahead_enabled())) L.read_ahead(g, runs);
            bool fell_back = false;
            std::string e;
            for (const RamRun& r : runs) {
                if (r.dst == nullptr) {   // one blob for a sink, through the staging
                    take_stage();
                    if (stage == nullptr) {
                        fail("no memory for the expert staging");
                        return;
                    }
                    const uint64_t a0 = r.file_pos / kAlign * kAlign;
                    const uint64_t a1 = (r.file_pos + r.bytes + kAlign - 1) / kAlign * kAlign;
                    const uint8_t* got = nullptr;
                    if (direct >= 0 && platform::pread_all(direct, stage, a1 - a0, a0)) got = stage + (r.file_pos - a0);
                    else if (platform::pread_all(L.fd, stage, r.bytes, r.file_pos)) got = stage;
                    if (got == nullptr) {
                        fail("short read at file offset " + std::to_string(r.file_pos));
                        return;
                    }
                    if (!L.sink(g, r, got, e)) {
                        fail(e);
                        return;
                    }
                    copied.fetch_add(r.bytes);
                    continue;
                }
                uint64_t done = 0;
                if (direct >= 0) {
                    const bool aligned = r.file_pos % kAlign == 0 && ((uintptr_t) r.dst) % kAlign == 0 && r.bytes % kAlign == 0;
                    if (aligned) {   // straight into RAM
                        while (done < r.bytes) {
                            const uint64_t n = std::min<uint64_t>(L.chunk, r.bytes - done);
                            const int64_t got = platform::pread(direct, r.dst + done, (size_t) n, r.file_pos + done);
                            if (got <= 0 || (uint64_t) got % kAlign != 0) break;   // the rest takes the fallback
                            done += (uint64_t) got;
                        }
                    } else {   // block-aligned reads into the staging, then the wanted bytes are copied out
                        take_stage();
                        while (stage != nullptr && done < r.bytes) {
                            const uint64_t pos = r.file_pos + done;
                            const uint64_t n = std::min<uint64_t>(L.chunk, r.bytes - done);
                            const uint64_t a0 = pos - pos % kAlign;
                            const uint64_t a1 = (pos + n + kAlign - 1) / kAlign * kAlign;
                            const int64_t got = platform::pread(direct, stage, (size_t) (a1 - a0), a0);
                            if (got <= 0 || (uint64_t) got < pos + n - a0) break;
                            std::memcpy(r.dst + done, stage + (pos - a0), (size_t) n);
                            done += n;
                        }
                    }
                    unbuffered.fetch_add(done);
                }
                if (done < r.bytes) {
                    fell_back = true;
                    if (L.fallback) {
                        if (!L.fallback(g, r, done, e)) {
                            fail(e);
                            return;
                        }
                    } else if (!platform::pread_all(L.fd, r.dst + done, r.bytes - done, r.file_pos + done)) {
                        fail("short read at file offset " + std::to_string(r.file_pos + done));
                        return;
                    }
                }
                copied.fetch_add(r.bytes);
                if (L.progress != nullptr && r.progress > 0) L.progress->add(r.progress);
            }
            const int64_t done_groups = groups_done.fetch_add(1) + 1;
            if (L.after_group && !L.after_group(g, done_groups, copied.load(), fell_back, e)) {
                fail(e);
                return;
            }
        }
    };
    {
        std::vector<std::thread> pool;
        for (int i = 1; i < L.threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    if (direct >= 0) platform::close_fd(direct);
    stats.copied = copied.load();
    stats.unbuffered = unbuffered.load();
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (failed.load()) {
        err = fail_msg;
        return false;
    }
    return true;
}

RamPin pin_ram(void* base, uint64_t register_bytes, uint64_t lock_bytes, const std::vector<uint64_t>& bounds, bool mapped,
               const uint8_t** device) {
    RamPin p;
    const unsigned flags = cudaHostRegisterPortable | (mapped ? cudaHostRegisterMapped : 0u);
    const auto lock_rest = [&](uint64_t from) {
        const platform::LockResult lr = platform::lock_resident((uint8_t*) base + from, lock_bytes - from);
        p.locked_bytes = lr.locked_bytes;
        p.lock_off = from;
        p.lock_note = lr.note;
    };
    if (register_bytes == 0) {
        p.refused = "over the pin budget";
        lock_rest(0);
        return p;
    }
    const cudaError_t e = cudaHostRegister(base, (size_t) register_bytes, flags);
    if (e == cudaSuccess) {
        if (device == nullptr) {
            p.registered_bytes = register_bytes;
            return p;
        }
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, base, 0) == cudaSuccess && alias != nullptr) {
            *device = (const uint8_t*) alias;
            p.registered_bytes = register_bytes;
            return p;
        }
        (void) cudaHostUnregister(base);
    }
    p.refused = cudaGetErrorString(e);
    // read here: an error left set would be reported by the next unrelated call that checks
    (void) cudaGetLastError();
    if (bounds.size() < 2) {
        lock_rest(0);
        return p;
    }
    // Pin slice by slice from the start, stopping below the shared-memory budget rather than at the driver's refusal.
    p.sliced = true;
    std::string why;
    uint64_t seg = 0;
    if (platform::pin_budget(seg, why) && seg < register_bytes) {
        p.capped = true;
        p.cap = seg;
        p.cap_why = std::move(why);
    }
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
        if (p.capped && (off > p.cap || n > p.cap - off)) break;
        if (cudaHostRegister((uint8_t*) base + off, (size_t) n, flags) != cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        p.slice_starts.push_back(off);
        p.registered_bytes = off + n;
        ++p.slices;
    }
    if (p.registered_bytes < lock_bytes) lock_rest(p.registered_bytes);
    return p;
}

void unpin_ram(void* base, const RamPin& p) {
    if (p.locked_bytes > 0) platform::unlock_resident((uint8_t*) base + p.lock_off, p.locked_bytes);
    if (p.sliced) {
        for (uint64_t off : p.slice_starts) (void) cudaHostUnregister((uint8_t*) base + off);
    } else if (p.registered_bytes > 0) {
        (void) cudaHostUnregister(base);
    }
}

uint64_t ram_padded(uint64_t bytes) {
    constexpr uint64_t kHuge = 2ull << 20;
    return (bytes + kHuge - 1) / kHuge * kHuge;
}

uint8_t* alloc_ram(uint64_t bytes) {
    return (uint8_t*) platform::alloc_aligned((size_t) ram_padded(bytes), 2ull << 20);
}

std::vector<RamPin> pin_ram_spans(const std::vector<uint8_t*>& spans, const std::vector<uint64_t>& bytes, bool mapped) {
    uint64_t budget = 0, total = 0;
    std::string why;
    for (uint64_t b : bytes) total += ram_padded(b);
    const bool limited = platform::pin_budget(budget, why) && budget < total;
    std::vector<RamPin> pins(spans.size());
    uint64_t pinned = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i] == nullptr) continue;
        const uint64_t padded = ram_padded(bytes[i]);
        const bool room = !limited || (pinned <= budget && padded <= budget - pinned);
        pins[i] = pin_ram(spans[i], room ? padded : 0, bytes[i], {}, mapped);
        pinned += pins[i].registered_bytes;
    }
    return pins;
}

}  // namespace basalt::core
