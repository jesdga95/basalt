// src/core/loader.cpp - see loader.hpp.
#include "basalt/core/loader.hpp"

#include "basalt/core/load_progress.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace basalt::core {

namespace {

constexpr uint64_t kAlign = 4096;

uint64_t round_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

/// `out` bytes of destination for the file bytes [0, x) of an item of `bytes` whose bar count is `count`; exact at both ends.
uint64_t scaled(uint64_t count, uint64_t x, uint64_t bytes) {
    if (x >= bytes) return count;
    return (uint64_t) ((long double) count * (long double) x / (long double) bytes);
}

void convert(Conv conv, const uint8_t* src, uint64_t n, uint8_t* out) {
    switch (conv) {
        case Conv::Copy: std::memcpy(out, src, (size_t) n); break;
        case Conv::Bf16High:   // the high half of the f32 image is the bf16 encoding
            for (uint64_t i = 0; i < n / 4; ++i) std::memcpy(out + i * 2, src + i * 4 + 2, 2);
            break;
        case Conv::ToF16:
            for (uint64_t i = 0; i < n / 4; ++i) {
                float v;
                std::memcpy(&v, src + i * 4, 4);
                const uint16_t h = basalt::kernels::f16_from_f32(v);
                std::memcpy(out + i * 2, &h, 2);
            }
            break;
        case Conv::WidenF16:
            for (uint64_t i = 0; i < n / 2; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                const float v = basalt::kernels::f32_from_f16(h);
                std::memcpy(out + i * 4, &v, 4);
            }
            break;
        case Conv::Bf16ToF32:
            for (uint64_t i = 0; i < n / 2; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                const uint32_t v = (uint32_t) h << 16;
                std::memcpy(out + i * 4, &v, 4);
            }
            break;
        case Conv::Bf16ToQ8_0: {
            uint16_t h[32];
            for (uint64_t b = 0; b < n / 64; ++b) {
                std::memcpy(h, src + b * 64, 64);
                quantize_bf16_q8_0(h, 32, out + b * 34);
            }
            break;
        }
    }
}

uint64_t conv_unit(Conv c) {   // the bytes of one source element (of a block, for a block format)
    switch (c) {
        case Conv::Bf16High:
        case Conv::ToF16: return 4;
        case Conv::WidenF16:
        case Conv::Bf16ToF32: return 2;
        case Conv::Bf16ToQ8_0: return 64;
        default: return 1;
    }
}

}  // namespace

uint64_t conv_out_bytes(Conv conv, uint64_t in) {
    switch (conv) {
        case Conv::Bf16High:
        case Conv::ToF16: return in / 2;
        case Conv::WidenF16:
        case Conv::Bf16ToF32: return in * 2;
        case Conv::Bf16ToQ8_0: return in / 64 * 34;
        default: return in;
    }
}

void quantize_bf16_q8_0(const uint16_t* bf16, uint64_t n, uint8_t* blocks) {
    for (uint64_t b = 0; b < n / 32; ++b) {
        float x[32], amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const uint32_t u = (uint32_t) bf16[b * 32 + (uint64_t) i] << 16;
            std::memcpy(&x[i], &u, 4);
            amax = std::max(amax, std::fabs(x[i]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = blocks + b * 34;
        const uint16_t dh = basalt::kernels::f16_from_f32(d);
        std::memcpy(o, &dh, 2);
        for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::round(x[i] * id);
    }
}

int LoadPlan::file(const std::string& path) {
    for (size_t i = 0; i < files_.size(); ++i)
        if (files_[i].fd < 0 && files_[i].path == path) return (int) i;
    files_.push_back(File{path, -1});
    return (int) files_.size() - 1;
}

int LoadPlan::file_fd(int fd) {
    for (size_t i = 0; i < files_.size(); ++i)
        if (files_[i].fd == fd) return (int) i;
    files_.push_back(File{std::string(), fd});
    return (int) files_.size() - 1;
}

void LoadPlan::add(LoadItem it) {
    if (it.bytes == 0) return;
    if (it.count == LoadItem::kOut) it.count = conv_out_bytes(it.conv, it.bytes);
    if (!items_.empty()) {
        LoadItem& p = items_.back();
        const uint64_t out = conv_out_bytes(p.conv, p.bytes);
        // only host items join: two device allocations that happen to touch are still two (one copy cannot span them)
        const bool same_dst = p.host != nullptr && it.host != nullptr && p.host + out == it.host;
        const bool same_src = p.from == nullptr && it.from == nullptr && p.file == it.file && p.off + p.bytes == it.off;
        if (same_src && same_dst && p.conv == it.conv) {
            p.bytes += it.bytes;
            p.count += it.count;
            return;
        }
    }
    items_.push_back(it);
}

void LoadPlan::after_reads(std::function<bool(std::string&)> step) { steps_.push_back(std::move(step)); }

void LoadPlan::append(const LoadPlan& other) {
    steps_.insert(steps_.end(), other.steps_.begin(), other.steps_.end());
    for (LoadItem it : other.items_) {
        if (it.from == nullptr) {
            const File& f = other.files_[(size_t) it.file];
            it.file = f.fd >= 0 ? file_fd(f.fd) : file(f.path);
        }
        add(it);
    }
}

void LoadPlan::append_reusing(const LoadPlan& other) {
    // this plan's device reads by file and range: "path" or "fd:<n>", offset, bytes, conversion
    const auto key = [](const File& f, const LoadItem& it) {
        return (f.fd >= 0 ? "fd:" + std::to_string(f.fd) : f.path) + "@" + std::to_string(it.off) + "+" + std::to_string(it.bytes) +
               "/" + std::to_string((int) it.conv);
    };
    std::map<std::string, std::pair<void*, int>> on_device;   // its destination and device
    steps_.insert(steps_.end(), other.steps_.begin(), other.steps_.end());
    for (const LoadItem& it : items_)
        if (it.from == nullptr && it.dev != nullptr) on_device.emplace(key(files_[(size_t) it.file], it), std::make_pair(it.dev, it.device));
    for (LoadItem it : other.items_) {
        const auto found = it.from == nullptr ? on_device.find(key(other.files_[(size_t) it.file], it)) : on_device.end();
        if (found == on_device.end()) {
            if (it.from == nullptr) {
                const File& f = other.files_[(size_t) it.file];
                it.file = f.fd >= 0 ? file_fd(f.fd) : file(f.path);
            }
            add(it);
            continue;
        }
        it.from = (const uint8_t*) found->second.first;
        it.from_device = found->second.second;
        it.bytes = conv_out_bytes(it.conv, it.bytes);
        it.conv = Conv::Copy;
        it.count = 0;
        add(it);
    }
}

void LoadPlan::to_host(const void* dev_base, uint64_t bytes, uint8_t* host) {
    const uint8_t* lo = (const uint8_t*) dev_base;
    for (LoadItem& it : items_) {
        const uint8_t* d = (const uint8_t*) it.dev;
        if (d == nullptr || d < lo || d >= lo + bytes) continue;
        it.host = host + (d - lo);
        it.dev = nullptr;
    }
}

uint64_t LoadPlan::counted() const {
    uint64_t n = 0;
    for (const LoadItem& it : items_) n += it.count;
    return n;
}

namespace {

struct FileHandle {
    int direct = -1;     // reads around the file cache
    int buffered = -1;   // ordinary reads: the fallback
    bool owns_buffered = true;
};

// A staging buffer: read target and, for a conversion, its output; guarded by the copy that last read from it.
struct Slot {
    uint8_t* raw_alloc = nullptr;
    uint8_t* raw = nullptr;
    uint8_t* out_alloc = nullptr;
    uint8_t* out = nullptr;
};

struct Worker {
    Slot slot[2];
};

struct Task {
    uint32_t item;
    uint64_t off, len;   // the piece of the item's file bytes
};

// A worker's copy stream on one device, and the events that guard its two staging slots.
struct DevState {
    cudaStream_t stream = nullptr;
    cudaEvent_t ev[2] = {nullptr, nullptr};
};

}  // namespace

struct Loader::Impl {
    int threads;
    std::map<std::string, FileHandle> files;   // by path, or "fd:<n>"
    std::vector<std::unique_ptr<Worker>> workers;
    bool pinned = false;
    uint64_t cap = 0, out_cap = 0;   // the largest piece a slot reads, and what a conversion makes of one
    bool all_direct = true;                  // every file opened unbuffered
    std::atomic<uint64_t> buffered_bytes{0};   // bytes that were read the ordinary way after all
    std::vector<std::map<int, DevState>> streams;   // per worker, its copy stream on each device it wrote to

    explicit Impl(int t) : threads(std::max(1, t)), streams((size_t) std::max(1, t)) {}

    // worker `wi`'s stream on device `d`, made on first use
    DevState* stream(int wi, int d) {
        auto& m = streams[(size_t) wi];
        if (auto it = m.find(d); it != m.end()) return &it->second;
        DevState s;
        if (cudaSetDevice(d) != cudaSuccess || cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.ev[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.ev[1], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        return &m.emplace(d, s).first->second;
    }

    ~Impl() {
        int prev = 0;
        const bool had = cudaGetDevice(&prev) == cudaSuccess;
        for (auto& m : streams)
            for (auto& [d, s] : m) {
                (void) cudaSetDevice(d);
                (void) cudaEventDestroy(s.ev[0]);
                (void) cudaEventDestroy(s.ev[1]);
                (void) cudaStreamDestroy(s.stream);
            }
        if (had) (void) cudaSetDevice(prev);   // the caller's device is as it was
        free_staging();
        for (auto& [path, f] : files) {
            if (f.direct >= 0) platform::close_fd(f.direct);
            if (f.buffered >= 0 && f.owns_buffered) platform::close_fd(f.buffered);
        }
    }

    void free_staging() {
        for (auto& w : workers)
            for (Slot& s : w->slot) {
                release(s.raw_alloc);
                release(s.out_alloc);
                s = Slot{};
            }
        workers.clear();
    }

    void release(uint8_t* p) {
        if (p == nullptr) return;
        if (pinned) (void) cudaFreeHost(p);
        else platform::free_aligned(p);
    }

    bool alloc(uint8_t*& alloc, uint8_t*& aligned, uint64_t bytes) {
        if (pinned) {
            if (cudaHostAlloc((void**) &alloc, (size_t) (bytes + kAlign), cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                alloc = nullptr;
                return false;
            }
            aligned = alloc + (round_up((uintptr_t) alloc) - (uintptr_t) alloc);
        } else {
            alloc = (uint8_t*) platform::alloc_aligned((size_t) bytes, kAlign);
            if (alloc == nullptr) return false;
            aligned = alloc;
        }
        return true;
    }

    // `n` workers' slots for pieces of up to `piece` bytes: sized to the plan, so a small phase takes small buffers
    bool staging(int n, uint64_t piece, uint64_t out_piece, bool want_pinned, std::string& err) {
        want_pinned = want_pinned || pinned;   // pinned memory serves host reads too
        if ((int) workers.size() >= n && pinned == want_pinned && piece <= cap && out_piece <= out_cap) return true;
        if (pinned != want_pinned || piece > cap || out_piece > out_cap) free_staging();
        pinned = want_pinned;
        cap = std::max(cap, piece);
        out_cap = std::max(out_cap, out_piece);
        while ((int) workers.size() < n) workers.push_back(std::make_unique<Worker>());
        for (auto& w : workers)
            for (Slot& s : w->slot) {
                if (s.raw == nullptr && !alloc(s.raw_alloc, s.raw, cap + 2 * kAlign)) {
                    err = std::string("no ") + (pinned ? "pinned " : "") + "memory for the file read staging (" +
                          std::to_string(workers.size() * 2 * (cap + 2 * kAlign) >> 20) + " MiB): free memory or load again";
                    return false;
                }
                if (out_cap > 0 && s.out == nullptr && !alloc(s.out_alloc, s.out, out_cap)) {
                    err = "no memory for the file read staging";
                    return false;
                }
            }
        return true;
    }

    bool open(const LoadPlan::File& file, FileHandle*& out, std::string& err) {
        const std::string key = file.fd >= 0 ? "fd:" + std::to_string(file.fd) : file.path;
        auto it = files.find(key);
        if (it == files.end()) {
            FileHandle f;
            if (file.fd >= 0) {
                f.direct = platform::open_unbuffered(file.fd);
                f.buffered = file.fd;
                f.owns_buffered = false;
            } else {
                f.direct = platform::open_unbuffered(file.path);
                f.buffered = platform::open_fd_utf8(file.path);
            }
            if (f.direct < 0 && f.buffered < 0) {
                err = "cannot open " + (file.fd >= 0 ? key : file.path);
                return false;
            }
            it = files.emplace(key, f).first;
        }
        if (it->second.direct < 0) all_direct = false;
        out = &it->second;
        return true;
    }
};

Loader::Loader(int threads) : impl_(new Impl(threads)) {}

bool Loader::prepare(const std::vector<int>& devices, std::string& err) {
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) prev = 0;
    for (int wi = 0; wi < impl_->threads; ++wi)
        for (int d : devices)
            if (impl_->stream(wi, d < 0 ? prev : d) == nullptr) {
                err = "cannot create the copy stream of device " + std::to_string(d);
                (void) cudaSetDevice(prev);
                return false;
            }
    (void) cudaSetDevice(prev);
    return true;
}
Loader::~Loader() = default;
void Loader::release_staging() { impl_->free_staging(); }
bool Loader::unbuffered() const { return impl_->all_direct && impl_->buffered_bytes.load() == 0; }

namespace {

// `n` bytes of the file at `pos` in `buf` (aligned), the unbuffered way when the file allows it; where they start in `buf`
const uint8_t* read_staged(const FileHandle& f, uint8_t* buf, uint64_t pos, uint64_t n, std::atomic<uint64_t>& buffered) {
    if (f.direct >= 0) {
        const uint64_t a0 = pos - pos % kAlign, a1 = round_up(pos + n), need = pos + n - a0;
        uint64_t got = 0;
        while (got < need) {
            const int64_t r = platform::pread(f.direct, buf + got, (size_t) (a1 - a0 - got), a0 + got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;
            got += (uint64_t) r;
        }
        if (got >= need) return buf + (pos - a0);
    }
    if (f.buffered >= 0 && platform::pread_all(f.buffered, buf, n, pos)) {
        buffered.fetch_add(n);
        return buf;
    }
    return nullptr;
}

// `n` bytes at `pos` straight into `dst`: all three aligned
bool read_direct(const FileHandle& f, uint8_t* dst, uint64_t pos, uint64_t n, std::atomic<uint64_t>& buffered) {
    uint64_t done = 0;
    if (f.direct >= 0) {
        while (done < n) {
            const int64_t r = platform::pread(f.direct, dst + done, (size_t) (n - done), pos + done);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0 || ((uint64_t) r % kAlign != 0 && done + (uint64_t) r < n)) break;
            done += (uint64_t) r;
        }
    }
    if (done >= n) return true;
    if (f.buffered >= 0 && platform::pread_all(f.buffered, dst + done, n - done, pos + done)) {
        buffered.fetch_add(n - done);
        return true;
    }
    return false;
}

}  // namespace

bool Loader::run(const LoadPlan& plan, std::string& err) {
    Impl& m = *impl_;
    if (plan.empty()) return true;
    const auto& items = plan.items();

    // the plan's shape: what it needs of the staging, which devices it writes
    bool any_dev = false;
    std::vector<int> devices;
    std::vector<FileHandle*> handles(plan.files().size(), nullptr);
    for (const LoadItem& it : items) {
        const bool file_src = it.from == nullptr;
        if ((it.host != nullptr) == (it.dev != nullptr)) { err = "load plan: an item needs one destination"; return false; }
        if (it.from_device >= 0 && it.conv != Conv::Copy) { err = "load plan: a device source is copied as is"; return false; }
        if (file_src && (it.file < 0 || it.file >= (int) handles.size())) { err = "load plan: an item names no file"; return false; }
        if (it.bytes % conv_unit(it.conv) != 0) { err = "load plan: a converted range is not whole elements"; return false; }
        for (const int d : {it.dev != nullptr ? it.device : -1, it.from_device})
            if (d >= 0) {
                any_dev = true;
                if (std::find(devices.begin(), devices.end(), d) == devices.end()) devices.push_back(d);
            }
    }
    for (size_t i = 0; i < handles.size(); ++i)
        if (!m.open(plan.files()[i], handles[i], err)) return false;

    // two rounds: the file reads, then the copies from memory, whose sources those reads fill.  The file reads that land
    // on a card come first, each sweep in plan order: the cards get their share before RAM fills, whatever order the
    // model queued its items in, and a RAM tier's pages are taken only after the cards' bytes are read.
    std::vector<Task> rounds[2], host_reads;
    uint64_t piece = 0, out_piece = 0, device_bytes = 0;
    size_t last_card_item = 0;   // the plan's last card read: RAM reads queued before it may run beside the card reads
    for (size_t i = 0; i < items.size(); ++i)
        if (items[i].from == nullptr && items[i].dev != nullptr) last_card_item = i;
    size_t early_ram = 0;        // the tasks of those RAM reads, the first of host_reads
    for (size_t i = 0; i < items.size(); ++i) {
        // a converted range is read in smaller pieces: its staging holds the raw bytes and what they become
        const uint64_t step = items[i].conv == Conv::Copy ? kChunk : kChunk / 4;
        const bool from_file = items[i].from == nullptr;
        if (from_file && items[i].dev != nullptr) device_bytes += items[i].bytes;
        std::vector<Task>& tasks = !from_file ? rounds[1] : items[i].dev != nullptr ? rounds[0] : host_reads;
        for (uint64_t off = 0; off < items[i].bytes; off += step) {
            tasks.push_back(Task{(uint32_t) i, off, std::min<uint64_t>(step, items[i].bytes - off)});
            if (&tasks == &host_reads && i < last_card_item) early_ram = host_reads.size();
            piece = std::max(piece, tasks.back().len);
            if (items[i].conv != Conv::Copy) out_piece = std::max(out_piece, conv_out_bytes(items[i].conv, tasks.back().len));
        }
    }
    // one card after the other, in the engine's card order (the main card, the current device, then the others by
    // ordinal), so each card's share is complete before the next begins: the drive is slower than any card's link
    int main_card = 0;
    if (cudaGetDevice(&main_card) != cudaSuccess) main_card = 0;
    const auto card_rank = [&](int d) { return d == main_card ? -1 : d; };
    std::stable_sort(rounds[0].begin(), rounds[0].end(), [&](const Task& a, const Task& b) {
        return card_rank(items[a.item].device) < card_rank(items[b.item].device);
    });
    const size_t device_tasks = rounds[0].size();
    rounds[0].insert(rounds[0].end(), host_reads.begin(), host_reads.end());
    const int n_workers = (int) std::min<size_t>((size_t) m.threads, std::max(rounds[0].size(), rounds[1].size()));
    if (!m.staging(n_workers, round_up(piece), out_piece > 0 ? round_up(out_piece) + kAlign : 0, any_dev, err)) return false;

    int prev_device = 0;
    if (any_dev) {
        if (cudaGetDevice(&prev_device) != cudaSuccess) prev_device = 0;
        // fills made on the legacy stream (zeroed arenas) are ordered before the copies below
        for (int d : devices)
            if (cudaSetDevice(d) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                err = "cannot select device " + std::to_string(d) + " for the load";
                (void) cudaSetDevice(prev_device);
                return false;
            }
    }

    // the last device read queued: the bar reads RAM, and the log says when the cards had their share
    const auto run_t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> device_left{device_tasks};
    auto device_done = [&]() {
        if (device_tasks == rounds[0].size()) return;   // no RAM reads follow
        LoadProgress::into("ram");
        basalt::log::info("basalt loader", "the cards' share is read (%.2f GiB in %.1f s); RAM next",
                          (double) device_bytes / 1073741824.0,
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - run_t0).count());
    };
    // the bar's label in the device sweep: the first card in that order whose share is not yet placed
    std::mutex card_mu;
    std::vector<std::pair<int, size_t>> card_left;   // (card, its tasks not yet placed), in the sweep's order
    for (size_t ti = 0; ti < device_tasks; ++ti) {
        const int d = items[rounds[0][ti].item].device;
        if (card_left.empty() || card_left.back().first != d) card_left.emplace_back(d, 0);
        ++card_left.back().second;
    }
    auto card_label = [&]() {   // under card_mu
        for (const auto& [d, left] : card_left)
            if (left > 0) return LoadProgress::into(LoadProgress::cuda(d));
    };
    auto card_placed = [&](int d) {
        std::lock_guard<std::mutex> lk(card_mu);
        for (auto& [c, left] : card_left)
            if (c == d && --left == 0) card_label();
    };
    {
        std::vector<std::string> stages;
        for (const auto& [d, left] : card_left) stages.push_back(LoadProgress::cuda(d));
        if (device_tasks < rounds[0].size()) stages.push_back("ram");
        LoadProgress::stages(stages);
    }
    if (device_tasks > 0) {
        std::lock_guard<std::mutex> lk(card_mu);
        card_label();
    }
    if (device_tasks == 0 && !rounds[0].empty()) LoadProgress::into("ram");
    // Round 0's workers: most read the cards' share first, in the sweep's order, and RAM once it is all taken.  When
    // the plan queued RAM reads among its card reads (a tier whose experts go to RAM and to a card layer by layer), a few
    // workers (kRamFirst, of a pool at least twice that) read those from the start: RAM reads land in fresh pages and
    // run below the drive's speed on their own, so beside the card reads the drive stays busy and the pass takes no
    // longer than the plan's order would, while the cards' share still completes sooner than in that order.  RAM reads
    // queued after every card read (a RAM copy planned after the cards) wait for the cards.
    constexpr int kRamFirst = 3;
    const size_t early_end = device_tasks + early_ram;
    std::atomic<size_t> ram_next{device_tasks};
    auto take_ram = [&](bool early_only) -> size_t {   // the next RAM task, or none (rounds[0].size())
        size_t i = ram_next.load();
        for (;;) {
            if (i >= rounds[0].size() || (early_only && i >= early_end)) return rounds[0].size();
            if (ram_next.compare_exchange_weak(i, i + 1)) return i;
        }
    };
    auto take = [&](int wi, const std::vector<Task>& tasks, std::atomic<size_t>& next) -> size_t {
        if (&tasks != &rounds[0] || device_tasks == 0) return next.fetch_add(1);
        if (n_workers >= 2 * kRamFirst && wi >= n_workers - kRamFirst && next.load() < device_tasks)
            if (const size_t i = take_ram(true); i < tasks.size()) return i;
        if (const size_t i = next.fetch_add(1); i < device_tasks) return i;
        return take_ram(false);
    };
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    std::string fail_msg;
    auto fail = [&](const std::string& what) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = what;
        failed.store(true);
    };

    auto work = [&](int wi, const std::vector<Task>& tasks, std::atomic<size_t>& next) {
        Worker& w = *m.workers[(size_t) wi];
        std::map<int, DevState*> devs;   // the devices this round's copies of this worker went to
        bool pending[2] = {false, false};
        int pending_dev[2] = {0, 0};
        int turn = 0;

        auto dev_state = [&](int d) -> DevState* {
            if (auto it = devs.find(d); it != devs.end()) return it->second;
            DevState* s = m.stream(wi, d);
            if (s == nullptr) {
                fail("cannot create the copy stream of device " + std::to_string(d));
                return nullptr;
            }
            return devs.emplace(d, s).first->second;
        };
        // the slot to use next, free of the copy that last read from it
        auto take_slot = [&]() -> int {
            const int s = turn;
            turn ^= 1;
            if (pending[s]) {
                (void) cudaEventSynchronize(devs[pending_dev[s]]->ev[s]);
                pending[s] = false;
            }
            return s;
        };

        for (;;) {
            const size_t ti = take(wi, tasks, next);
            if (ti >= tasks.size() || failed.load()) break;
            const Task& t = tasks[ti];
            const LoadItem& it = items[t.item];
            const uint64_t out_off = conv_out_bytes(it.conv, t.off);
            const uint64_t out_len = conv_out_bytes(it.conv, t.len);
            const uint64_t pos = it.off + t.off;

            if (it.from_device >= 0) {   // device memory to host memory or to a device, on the source's stream
                DevState* ds = dev_state(it.from_device);
                if (ds == nullptr) break;
                const uint8_t* src = it.from + t.off;
                cudaError_t e = cudaSetDevice(it.from_device);
                if (e == cudaSuccess)
                    e = it.host != nullptr ? cudaMemcpyAsync(it.host + t.off, src, (size_t) t.len, cudaMemcpyDeviceToHost, ds->stream)
                                           : cudaMemcpyPeerAsync((uint8_t*) it.dev + t.off, it.device, src, it.from_device,
                                                                 (size_t) t.len, ds->stream);
                if (e != cudaSuccess) { fail(std::string("copy from device ") + std::to_string(it.from_device) + ": " + cudaGetErrorString(e)); break; }
            } else if (it.host != nullptr && it.from != nullptr) {   // host memory to host memory
                convert(it.conv, it.from + t.off, t.len, it.host + out_off);
            } else if (it.host != nullptr) {
                uint8_t* dst = it.host + out_off;
                const FileHandle& f = *handles[(size_t) it.file];
                if (it.conv == Conv::Copy && pos % kAlign == 0 && (uintptr_t) dst % kAlign == 0 && t.len % kAlign == 0) {
                    if (!read_direct(f, dst, pos, t.len, m.buffered_bytes)) { fail("short read at file offset " + std::to_string(pos)); break; }
                } else {
                    const int s = take_slot();
                    const uint8_t* src = read_staged(f, w.slot[s].raw, pos, t.len, m.buffered_bytes);
                    if (src == nullptr) { fail("short read at file offset " + std::to_string(pos)); break; }
                    convert(it.conv, src, t.len, dst);
                }
            } else {
                DevState* ds = dev_state(it.device);
                if (ds == nullptr) break;
                const int s = take_slot();
                const uint8_t* src = it.from != nullptr ? it.from + t.off : nullptr;
                if (src == nullptr) {
                    src = read_staged(*handles[(size_t) it.file], w.slot[s].raw, pos, t.len, m.buffered_bytes);
                    if (src == nullptr) { fail("short read at file offset " + std::to_string(pos)); break; }
                }
                if (it.conv != Conv::Copy) {
                    convert(it.conv, src, t.len, w.slot[s].out);
                    src = w.slot[s].out;
                }
                const cudaError_t e = cudaSetDevice(it.device) != cudaSuccess
                                          ? cudaErrorInvalidDevice
                                          : cudaMemcpyAsync((uint8_t*) it.dev + out_off, src, (size_t) out_len,
                                                            cudaMemcpyHostToDevice, ds->stream);
                if (e != cudaSuccess) { fail(std::string("copy to device ") + std::to_string(it.device) + ": " + cudaGetErrorString(e)); break; }
                (void) cudaEventRecord(ds->ev[s], ds->stream);
                pending[s] = true;
                pending_dev[s] = it.device;
            }
            LoadProgress::placed(scaled(it.count, t.off + t.len, it.bytes) - scaled(it.count, t.off, it.bytes));
            if (&tasks == &rounds[0] && ti < device_tasks) {
                card_placed(it.device);
                if (device_left.fetch_sub(1) == 1) device_done();
            }
        }

        // every copy this worker issued is complete before the plan is
        for (auto& [d, s] : devs) {
            (void) cudaSetDevice(d);
            const cudaError_t e = cudaStreamSynchronize(s->stream);
            if (e != cudaSuccess) fail(std::string("copy to device ") + std::to_string(d) + ": " + cudaGetErrorString(e));
        }
    };

    for (const std::vector<Task>& tasks : rounds) {   // each round's copies are complete before the next round reads them
        if (&tasks == &rounds[1] && !failed.load())   // the reads have landed: the plan's steps, before the copies
            for (const auto& step : plan.steps()) {
                std::string e;
                if (!step(e)) { fail(e); break; }
            }
        if (tasks.empty() || failed.load()) continue;
        std::atomic<size_t> next{0};
        const int n = (int) std::min<size_t>((size_t) n_workers, tasks.size());
        std::vector<std::thread> pool;
        for (int i = 1; i < n; ++i) pool.emplace_back(work, i, std::cref(tasks), std::ref(next));
        work(0, tasks, next);
        for (auto& t : pool) t.join();
    }
    if (any_dev) (void) cudaSetDevice(prev_device);
    if (failed.load()) {
        err = fail_msg;
        return false;
    }
    return true;
}

bool place_start(Loader& loader, const LoadPlan& plan, std::string& err) {
    LoadProgress::announce_total(plan.counted());
    const bool ok = loader.run(plan, err);
    loader.release_staging();
    return ok;
}

}  // namespace basalt::core
