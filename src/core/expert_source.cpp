// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "basalt/core/expert_source.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include "basalt/core/pinned.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/direct_file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/quantize_act.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::core {

namespace detail {

bool parse_cgroup_memory_stat(const std::string& text, uint64_t current, CgroupMemoryStat& stat) {
    stat = CgroupMemoryStat{};
    bool inactive_file = false, file_dirty = false, file_writeback = false, file = false;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        auto take = [&](bool& seen, uint64_t& out) {
            if (seen) return false;
            seen = true;
            out = value;
            return true;
        };
        if (key == "file") { if (!take(file, stat.file)) return false; }
        else if (key == "inactive_file") { if (!take(inactive_file, stat.inactive_file)) return false; }
        else if (key == "file_dirty") { if (!take(file_dirty, stat.file_dirty)) return false; }
        else if (key == "file_writeback") { if (!take(file_writeback, stat.file_writeback)) return false; }
    }
    if (!inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // Clean file pages are reclaimable.  The page cache is the active and inactive lists together (`file`; the
    // inactive list alone when the file total is not reported), because the container's own reads move pages to
    // the active list.  The counters can race memory.current, so bound them to charged usage first.
    uint64_t reclaimable = std::min(std::max(stat.file, stat.inactive_file), stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;
    // Keep a margin: part of the cache is mapped or locked and does not come back on demand.
    reclaimable -= std::min(reclaimable / 2, limit / 64);

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from) {
    if (base_bytes > budget) return -1;
    const int64_t slots = (int64_t) slot_bytes.size();
    if (lend_from < 0 || lend_from > slots) lend_from = slots;   // no lend region: only the experts no slot holds
    int64_t keep = slots;
    uint64_t bytes = base_bytes;
    while (keep > lend_from) {
        const uint64_t b = slot_bytes[(size_t) keep - 1];
        if (b > budget - bytes) break;
        bytes += b;
        --keep;
    }
    return keep;
}

int64_t choose_resident_room(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                             int64_t lend_from, uint64_t spare_bytes, bool& spares_fit) {
    spares_fit = spare_bytes > 0 && base_bytes <= budget && spare_bytes <= budget - base_bytes;
    return choose_resident_keep_from(slot_bytes, base_bytes, spares_fit ? budget - spare_bytes : budget, lend_from);
}

bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out) {
    if (in == out || in >= offsets.size() || out >= offsets.size() || offsets[in] == kNoCacheComplement ||
        offsets[out] != kNoCacheComplement) return false;
    offsets[out] = offsets[in];
    offsets[in] = kNoCacheComplement;
    return true;
}

int home_region_find(const std::vector<HomeRegion>& regions, uint64_t off, uint64_t bytes) {
    for (size_t i = 0; i < regions.size(); ++i) {
        const HomeRegion& r = regions[i];
        if (off >= r.base && off - r.base <= r.bytes && bytes <= r.bytes - (off - r.base)) return (int) i;
    }
    return -1;
}

bool home_region_used(const HomeRegion& r, const std::vector<uint64_t>& offsets) {
    for (const uint64_t o : offsets)
        if (o != kNoCacheComplement && o >= r.base && o - r.base < r.bytes) return true;
    return false;
}

}  // namespace detail

namespace {

bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;
    std::ostringstream text;
    text << input.rdbuf();
    return text.good() && detail::parse_cgroup_memory_stat(text.str(), current, stat);
}

// cgroup v1's memory.stat: the group's hierarchical counters (total_*), the same allowance as v2's.  False when the
// file or total_inactive_file is missing (the usage is then taken as it is).
bool read_cgroup_v1_memory_stat(const std::filesystem::path& path, uint64_t usage, detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;
    bool inactive_file = false;
    std::string key;
    uint64_t value = 0, active_file = 0;
    while (input >> key >> value) {
        if (key == "total_inactive_file") { stat.inactive_file = value; inactive_file = true; }
        else if (key == "total_active_file") active_file = value;
        else if (key == "total_dirty") stat.file_dirty = value;
        else if (key == "total_writeback") stat.file_writeback = value;
    }
    if (!inactive_file) return false;
    stat.file = stat.inactive_file + active_file;
    stat.current = usage;
    stat.valid = true;
    return true;
}

}  // namespace

namespace detail {

bool host_available_memory(HostMemory& m, const std::string& meminfo_arg, const std::string& self_cgroup_arg,
                           const std::string& cgroup_root_arg) {
    m = HostMemory{};
    const std::string meminfo = meminfo_arg.empty() ? basalt::platform::meminfo_path() : meminfo_arg;
    const std::string self_cgroup = self_cgroup_arg.empty() ? basalt::platform::self_cgroup_path() : self_cgroup_arg;
    const std::string cgroup_root = cgroup_root_arg.empty() ? basalt::platform::cgroup_root() : cgroup_root_arg;
    if (meminfo.empty()) {
        // No memory accounting files and no cgroup hierarchy here: the OS reading is the whole answer, and there is
        // no container limit tighter than it.
        const uint64_t avail = basalt::platform::available_memory();
        if (avail == 0) return false;
        m.available = avail;
        m.cgroup_limit = ~0ull;
        return true;
    }
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info(meminfo);
    std::string line;
    uint64_t bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // The tightest cgroup limit of this process's group and its ancestors.  A group whose path is not visible here
    // (a container's own cgroup namespace shows its group as the mount's root) is read at that root; a group without
    // a readable limit has none of its own.  No cgroup file or line: MemAvailable alone.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v2_path, v1_path;   // the unified group; the v1 memory controller's group
    if (std::ifstream groups(self_cgroup); groups) {
        while (std::getline(groups, line)) {
            if (line.rfind("0::/", 0) == 0) {
                v2 = true;
                v2_path = line.substr(4);
                continue;
            }
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "memory") v1_path = line.substr(c2 + 1);
        }
    }
    // the group's directory under `top`, or `top` itself when the group is not visible there
    auto visible_group = [](const std::filesystem::path& top, const std::string& rel) {
        const std::string sub = rel.substr(rel.rfind('/', 0) == 0 ? 1 : 0);
        if (sub.empty()) return top;
        auto path = (top / sub).lexically_normal();
        if (path.string().rfind(top.string(), 0) != 0 || !std::filesystem::is_directory(path)) return top;
        return path;
    };
    if (!v1_path.empty()) {
        // cgroup v1 (also the hybrid layout, where the unified line carries no memory controller): the memory
        // controller's memory.limit_in_bytes - memory.usage_in_bytes, with the clean inactive file cache counted as
        // free as for v2.  An unlimited group says a number near 2^63 (rounded to its page size).
        const std::filesystem::path mroot = root / "memory";
        for (auto path = visible_group(mroot, v1_path); path.string().rfind(mroot.string(), 0) == 0;
             path = path.parent_path()) {
            std::ifstream limit_file(path / "memory.limit_in_bytes"), usage_file(path / "memory.usage_in_bytes");
            uint64_t cap = 0, usage = 0;
            if (limit_file >> cap && usage_file >> usage && cap < (1ull << 62)) {
                uint64_t room = usage < cap ? cap - usage : 0;
                CgroupMemoryStat stat;
                if (read_cgroup_v1_memory_stat(path, usage, stat) && !cgroup_available_bytes(cap, stat, room)) return false;
                bytes = std::min(bytes, room);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
        }
    } else if (v2) {
        for (auto path = visible_group(root, v2_path); path.string().rfind(root.string(), 0) == 0;
             path = path.parent_path()) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            // the host's root group, and a group whose parent does not enable the memory controller, have no limit
            if (limit_file >> limit && current_file >> current && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                } catch (...) { return false; }
            }
            if (path == root) break;
        }
    }
    m.available = bytes;
    return true;
}

}  // namespace detail

namespace {

bool available_memory_bytes(uint64_t& bytes) {
    detail::HostMemory m;
    if (!detail::host_available_memory(m)) return false;
    bytes = m.available;
    return bytes > 0;
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::FileExpertSource() = default;
FileExpertSource::~FileExpertSource() { advise_worker_.reset(); close(); }

bool FileExpertSource::open(const FileRange& experts, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = basalt::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
        layout.fmt.size() != (size_t) n_layers) {
        err = "FileExpertSource: the native expert layout is incomplete";
        return false;
    }
    uint64_t at = 0;
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const size_t i = (size_t) layer;
        const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
        if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
            bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
            err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
            return false;
        }
        const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
        if (at > want || layer_bytes > want - at) {
            err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                  std::to_string(layer);
            return false;
        }
        layer_offsets[i] = layout.offset[i];
        layer_blob_bytes[i] = bytes;
        at += layer_bytes;
    }
    if (at != want) {
        err = "FileExpertSource: the native expert layout has an inconsistent size";
        return false;
    }
    const std::string& path = experts.path;
    uint64_t fsize = 0;
    if (!platform::file_size(path, fsize)) { err = "FileExpertSource: cannot open " + path; return false; }
    // a whole file is exactly the experts; a section must be, and lie inside the file
    const uint64_t have = experts.bytes != 0 ? experts.bytes : fsize;
    if (have != want || experts.offset > fsize || have > fsize - experts.offset) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) have,
                      (unsigned long long) want);
        err = buf;
        return false;
    }
    platform::MapOptions mo;
    mo.offset = experts.offset;
    mo.bytes = want;
    mo.shared = true;
    std::string map_err;
    auto view_map = platform::MappedFile::open_ro(path, &map_err, mo);
    if (!view_map) { err = "FileExpertSource: " + map_err; return false; }
    // A plain descriptor for the prompt path's positional reads: the view stays read-only for the compute.  The
    // bulk fills open the same path unbuffered themselves (read_blobs).  A failure here is not fatal - the caller
    // falls back to copying from the view.
    fd_ = platform::open_fd_utf8(path);
    path_ = path;
    view_ = std::move(*view_map);
    file_off_ = experts.offset;
    base_ = view_.data();
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    return true;
}

void FileExpertSource::close() {
    repin_join();
    if (complement_arena_ != nullptr) {
        if (complement_pinned_ && !complement_partial_) (void) cudaFreeHost(complement_arena_);
        else {
            if (complement_partial_) unregister_complement();
            if (complement_locked_ > 0)
                basalt::platform::unlock_resident((uint8_t*) complement_arena_ + complement_lock_off_, complement_locked_);
            basalt::platform::free_aligned(complement_arena_);
        }
    }
    if (top_arena_ != nullptr) {
        if (top_pinned_) (void) cudaFreeHost(top_arena_);
        else {
            if (top_locked_ > 0) basalt::platform::unlock_resident(top_arena_, top_locked_);
            std::free(top_arena_);
        }
    }
    for (const detail::HomeRegion& r : homes_) {
        if (r.pinned) (void) cudaFreeHost(r.arena);
        else {
            if (r.locked > 0) basalt::platform::unlock_resident(r.arena, r.locked);
            std::free(r.arena);
        }
    }
    homes_.clear();
    homes_pending_.clear();
    homes_next_ = 0;
    top_arena_ = nullptr;
    top_device_ = nullptr;
    top_bytes_ = top_locked_ = 0;
    top_pinned_ = false;
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
    }
    xstage_ = nullptr;
    xstage_pinned_ = false;
    xstage_cap_ = 0;
    xstage_blob_ = 0;
    override_.clear();
    staged_.clear();
    exchanges_ = 0;
    spare_free_.clear();
    file_reads_.store(0);
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    ++places_gen_;
    complement_pinned_ = false;
    unpinned_ = false;
    pin_units_.clear();
    pin_state_.reset();
    complement_partial_ = false;
    complement_pin_limit_ = 0;
    complement_lock_off_ = 0;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
    ram_reads_.store(0);
    file_read_bytes_.store(0);
    file_us_.store(0);
    basalt::platform::ram_guard_reclaim_view(nullptr, 0);   // before the view is unmapped
    view_.reset();
    platform::close_fd(fd_);
    fd_ = -1;
    path_.clear();
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    file_off_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    reads_ = 0;
}


bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

bool FileExpertSource::copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const {
    if (dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return false;
    std::memcpy(dst, b, (size_t) layer_blob_bytes_[(size_t) layer]);
    return true;
}

bool ExpertSource::read_blobs(const std::pair<int32_t, int32_t>* pairs, int64_t n, const BlobSink& sink,
                              std::string& err) {
    constexpr int64_t kAhead = 256;
    const bool advise = advise_pairs(pairs, std::min<int64_t>(kAhead, n));
    for (int64_t i = 0; i < n; ++i) {
        if (advise && i + kAhead < n) (void) advise_pairs(pairs + i + kAhead, 1);
        const uint8_t* b = blob(pairs[i].first, pairs[i].second);
        if (b == nullptr) {
            err = "expert blob " + std::to_string(pairs[i].first) + "/" + std::to_string(pairs[i].second) +
                  " is not readable";
            return false;
        }
        if (!sink(i, 0, b, err)) return false;
    }
    return true;
}

bool FileExpertSource::read_blobs(const std::pair<int32_t, int32_t>* pairs, int64_t n,
                                 const BlobSink& sink, std::string& err) {
    // A blob copied out of the mapping leaves its pages in this process, so each one goes back to the OS as soon as
    // the sink has it: a fill of every expert must not leave the whole pack resident.
    const BlobSink drop_after = [&](int64_t i, int w, const uint8_t* blob, std::string& e) {
        if (!sink(i, w, blob, e)) return false;
        if (blob >= base_ && blob < base_ + mapped_bytes_)
            (void) basalt::platform::drop_mapped_view(base_, mapped_bytes_, (uint64_t) (blob - base_),
                                                      layer_blob_bytes_[(size_t) pairs[i].first]);
        return true;
    };
    // With the compact RAM copy ready the experts are not read from the file, so there is nothing to give back.
    const bool give_back = n > 0 && !complement_ready_ && base_ != nullptr;
    const BlobSink mapped_sink = give_back ? drop_after : sink;
    auto mapped_read = [&] { return ExpertSource::read_blobs(pairs, n, mapped_sink, err); };
    if (complement_ready_ || !base_ || path_.empty() || n <= 0) return mapped_read();

    uint64_t max_blob = 0;
    for (int64_t i = 0; i < n; ++i) {
        const auto [layer, expert] = pairs[i];
        if (!mapped_blob(layer, expert)) {
            err = "read_blobs: invalid expert";
            return false;
        }
        max_blob = std::max(max_blob, layer_blob_bytes_[(size_t) layer]);
    }
    constexpr uint64_t alignment = platform::DirectFile::alignment();
    if (max_blob > UINT32_MAX - 2 * alignment) return mapped_read();
    const uint64_t stage_bytes = (max_blob + 2 * (alignment - 1)) & ~(alignment - 1);
    const int readers = (int) std::min<int64_t>(kBlobReaders, n);
    struct Stage {
        uint8_t* alloc = nullptr;   // what cudaHostAlloc returned
        uint8_t* data = nullptr;    // `alloc` rounded up to the read alignment
        int64_t index = -1;
        uint32_t prefix = 0, required = 0;
        ~Stage() { if (alloc) cudaFreeHost(alloc); }
    } stages[kBlobReaders];
    // The file drains outstanding writes before the staging buffers are released.  A pack on a file system without
    // unbuffered reads, or one that cannot be reopened, is read through the mapping.
    platform::DirectFile file;
    std::string direct_err;
    if (!file.open(path_, direct_err, readers)) return mapped_read();
    for (int i = 0; i < readers; ++i) {
        // Portable: the fill writes the peer card's slots straight from these buffers.  A small pinned allocation
        // can be carved from a shared block at less than the read alignment, so each buffer has one alignment of
        // slack and starts at the next aligned address inside it.
        if (cudaHostAlloc((void**) &stages[i].alloc, (size_t) (stage_bytes + alignment), cudaHostAllocPortable) !=
            cudaSuccess) {
            stages[i].alloc = nullptr;
            (void) cudaGetLastError();
            file.close();
            return mapped_read();
        }
        const uintptr_t at = (uintptr_t) stages[i].alloc;
        stages[i].data = stages[i].alloc + (((at + alignment - 1) & ~(uintptr_t) (alignment - 1)) - at);
    }

    auto submit = [&](int slot, int64_t index) {
        const auto [layer, expert] = pairs[index];
        const uint8_t* mapped = mapped_blob(layer, expert);
        const uint64_t pos = file_off_ + (uint64_t) (mapped - base_);
        Stage& stage = stages[slot];
        stage.index = index;
        stage.prefix = (uint32_t) (pos & (alignment - 1));
        stage.required = stage.prefix + (uint32_t) layer_blob_bytes_[(size_t) layer];
        const uint32_t length = (uint32_t) (((uint64_t) stage.required + alignment - 1) & ~(alignment - 1));
        return file.submit(pos - stage.prefix, stage.data, length, (uint64_t) slot, err);
    };
    int64_t next = 0;
    for (int slot = 0; slot < readers; ++slot)
        if (!submit(slot, next++)) return false;
    int64_t finished = 0;
    platform::Completion completions[kBlobReaders];
    while (finished < n) {
        const int count = file.wait(completions, readers, -1);
        if (count <= 0) {
            err = "read_blobs: direct read completion failed";
            return false;
        }
        for (int j = 0; j < count; ++j) {
            const auto& done = completions[j];
            if (done.tag == platform::DirectFile::WAKE_TAG) continue;
            if (done.tag >= (uint64_t) readers || stages[done.tag].index < 0) {
                err = "read_blobs: invalid direct read completion";
                return false;
            }
            Stage& stage = stages[done.tag];
            // A read that came back short or failed (a truncated pack, a drive that refused it) hands over this
            // blob's mapped copy, and the fill carries on with the rest.
            const uint8_t* data = done.ok && done.bytes >= stage.required
                                      ? stage.data + stage.prefix
                                      : mapped_blob(pairs[stage.index].first, pairs[stage.index].second);
            if (!drop_after(stage.index, (int) done.tag, data, err)) return false;
            stage.index = -1;
            ++finished;
            if (next < n && !submit((int) done.tag, next++)) return false;
        }
    }
    return true;
}

bool fill_cache_slots(ExpertSource& src, ExpertCache& cache, int device, const std::pair<int32_t, int32_t>* pairs,
                      const int32_t* slots, int64_t n, const std::function<void(uint64_t)>& on_bytes, std::string& err) {
    std::vector<cudaStream_t> streams((size_t) ExpertSource::kBlobReaders, nullptr);
    struct Destroy {
        std::vector<cudaStream_t>& v;
        ~Destroy() { for (cudaStream_t s : v) if (s) cudaStreamDestroy(s); }
    } destroy{streams};
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(device) != cudaSuccess) { err = "cannot select the device for the cache fill"; return false; }
    // the cache's arena was zeroed on the legacy stream, which the non-blocking streams below are not ordered against
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    for (cudaStream_t& s : streams)
        ok = ok && cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) {
        err = "cannot create the streams for the cache fill";
        cudaSetDevice(prev);
        return false;
    }
    const auto& lay = basalt::kernels::cpu::expert_layout();
    const bool done = src.read_blobs(pairs, n, [&](int64_t i, int w, const uint8_t* blob, std::string& e) {
        if (cudaSetDevice(device) != cudaSuccess) { e = "cannot select the device for the cache fill"; return false; }
        const int64_t bytes = (int64_t) lay.blob_bytes(pairs[i].first);
        if (!cache.fill_slot(slots[i], blob, streams[(size_t) w], e, bytes)) return false;
        const cudaError_t sync = cudaStreamSynchronize(streams[(size_t) w]);   // the staging buffer is reused next
        if (sync != cudaSuccess) { e = std::string("cache fill: ") + cudaGetErrorString(sync); return false; }
        if (on_bytes) on_bytes((uint64_t) bytes);
        return true;
    }, err);
    cudaSetDevice(prev);
    return done;
}

bool FileExpertSource::advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
    if (base_ == nullptr || !basalt::platform::read_ahead_enabled()) return false;
    for (int64_t i = 0; pairs != nullptr && i < n; ++i) {
        const int64_t l = pairs[i].first, e = pairs[i].second;
        if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) continue;
        // an expert held in RAM is not read from the file
        if (complement_ready_ && complement_offsets_[(size_t) l * (size_t) n_expert_ + (size_t) e] != kNoComplement)
            continue;
        if (const uint8_t* b = mapped_blob(l, e)) basalt::platform::advise_willneed(b, layer_blob_bytes_[(size_t) l]);
    }
    return true;
}

uint64_t FileExpertSource::expert_bytes() const {
    uint64_t total = 0;
    for (uint64_t b : layer_blob_bytes_) total += b * (uint64_t) n_expert_;
    return total;
}

bool FileExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (base_ == nullptr || dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_)
        return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    if (complement_ready_) {
        const uint8_t* held = complement_offsets_[index] != kNoComplement
                                  ? host_at(complement_offsets_[index], bytes) : nullptr;
        if (held == nullptr && !override_.empty()) held = override_[index];
        if (held != nullptr) {
            std::memcpy(dst, held, (size_t) bytes);
            return true;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, expert, dst)) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

bool FileExpertSource::file_extent(int64_t layer, int64_t expert, int& fd, uint64_t& offset) const {
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr || fd_ < 0) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    // a blob held in RAM (the complement, or a staged exchange) is not read from the file
    if (complement_ready_ && (complement_offsets_[index] != kNoComplement || (!override_.empty() && override_[index] != nullptr)))
        return false;
    fd = fd_;
    offset = file_off_ + (uint64_t) (b - base_);
    return true;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, bool pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs, int64_t lend_from_slot,
    uint64_t headroom_bytes, uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>* rank) {
    err.clear();
    headroom_bytes = std::max(headroom_bytes, platform::ram_reserve());   // never lock into the reserve
    const auto pin_t0 = std::chrono::steady_clock::now();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + cudaGetErrorString(sync);
        (void) cudaGetLastError();
        return false;
    }

    // The GPU cache's experts, and the bytes each slot's expert takes here (for the lend region below).
    const int64_t n_slots = cache.slots();
    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    std::vector<int32_t> pair_slot;
    std::vector<uint64_t> slot_bytes((size_t) std::max<int64_t>(n_slots, 0), 0);
    std::vector<uint8_t> on_gpu((size_t) n_layers_ * (size_t) n_expert_, 0);   // held by the cache (counted when filled)
    uint64_t gpu_bytes = 0;
    primary_gpu_pairs.reserve((size_t) cache.resident());
    pair_slot.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const int32_t slot = cache.slot_of(layer, expert);
            if (slot == kNotResident) continue;
            primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
            on_gpu[(size_t) layer * (size_t) n_expert_ + (size_t) expert] = 1;
            gpu_bytes += layer_blob_bytes_[(size_t) layer];
            pair_slot.push_back(slot);
            if (slot >= 0 && slot < n_slots) slot_bytes[(size_t) slot] = layer_blob_bytes_[(size_t) layer];
        }
    }
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;
    const bool what_fits = budget_bytes == kResidentWhatFits;   // the soft mode's second try
    uint64_t budget_physical = 0;   // the RAM reading a budget was sized from (0: no budget)
    if (budget_bytes > 0) {
        // CS-T: a RAM budget.  The complement's experts in `rank` order (the expert profile, hottest first) while
        // they fit, the rest left on the mapped files; clamped to what the RAM has room for.
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for --resident-budget-gib";
            return false;
        }
        budget_physical = physical;
        const uint64_t room = ram_tier_room(physical, headroom_bytes, pin);
        if (budget_bytes > room) {
            if (what_fits)
                basalt::log::info("basalt", "FileExpertSource: RAM room for the complement: %.2f GiB (%.2f GiB available "
                                     "minus %.0f GiB reserve and the page tracking)",
                             (double) room / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0);
            else
                basalt::log::warn("basalt", "FileExpertSource: --resident-budget-gib %.2f is more than the RAM has room for "
                                     "(%.2f GiB available minus %.0f GiB reserve and the page tracking): %.2f GiB",
                             (double) budget_bytes / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0, (double) room / 1073741824.0);
            budget_bytes = room;
        }
        std::vector<uint64_t> ranked(offsets.size(), kNoComplement);
        uint64_t at = 0;
        int64_t held = 0;
        if (rank != nullptr)
            for (const auto& pr : *rank) {
                if (pr.first < 0 || pr.second < 0 || pr.first >= n_layers_ || pr.second >= n_expert_) continue;
                const size_t i = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                if (offsets[i] == kNoComplement || ranked[i] != kNoComplement) continue;   // on a GPU, or twice
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) continue;
                ranked[i] = at;
                at += b;
                ++held;
            }
        if (what_fits && held == 0) {   // nothing to keep - the caller's plain mmap fallback, not an empty copy
            err = "FileExpertSource: the RAM has no room for any expert of the complement";
            return false;
        }
        basalt::log::info("basalt", "FileExpertSource: RAM budget %.2f GiB: %lld of the %.2f GiB of experts the GPU cache does "
                             "not hold, by profile rank; the rest are read from the files",
                     (double) budget_bytes / 1073741824.0, (long long) held, (double) bytes / 1073741824.0);
        offsets.swap(ranked);
        bytes = at;
        lend_from_slot = -1;
    }

    const bool lend = lend_from_slot >= 0 && lend_from_slot < n_slots;
    uint64_t budget = std::numeric_limits<uint64_t>::max();
    if (bytes > 0 || lend) {
        // with a budget, the reading it was sized from - a second reading a few MB lower (the engine's own allocations,
        // the file cache) could fail a budget the first one had clamped.  (A budget turns `lend` off.)
        uint64_t physical = budget_physical;
        if (physical == 0 && !available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for the resident-memory safety check";
            return false;
        }
        budget = ram_tier_room(physical, headroom_bytes, pin);
        if (bytes > budget) {
            char message[320];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available RAM (%.2f GiB) minus the "
                          "%.0f GiB safety headroom",
                          (double) bytes / 1073741824.0, (double) physical / 1073741824.0,
                          (double) headroom_bytes / 1073741824.0);
            err = message;
            return false;
        }
    }
    // The free places after the experts (see set_spares).  A swap takes a place of its layer's blob size and gives one
    // of the same size back, so each size keeps its own share of the n places: its layers' share, at least 8.
    std::vector<std::pair<uint64_t, int64_t>> spare_sizes;   // (blob bytes, places)
    uint64_t spare_bytes = 0;
    if (spares_want_ > 0 && n_layers_ > 0) {
        for (const uint64_t b : layer_blob_bytes_) {
            if (b == 0) continue;
            auto it = std::find_if(spare_sizes.begin(), spare_sizes.end(), [&](const auto& x) { return x.first == b; });
            if (it == spare_sizes.end()) spare_sizes.emplace_back(b, 1);
            else ++it->second;
        }
        for (auto& [b, n] : spare_sizes) {
            n = std::max<int64_t>(8, (spares_want_ * n + n_layers_ - 1) / n_layers_);
            spare_bytes += (uint64_t) n * b;
        }
    }

    // The prompt path's lend region: its slots' experts are streamed from here during a prompt and copied back into
    // their slots after it, so the ones that fit next to the free places are kept here too (from the last slot down:
    // a short prompt lends only the last few).  The rest keep the mapped-file fallback.
    int64_t keep_from = n_slots;
    if (lend) {
        bool spares_fit = false;
        keep_from = detail::choose_resident_room(slot_bytes, bytes, budget, lend_from_slot, spare_bytes, spares_fit);
        if (keep_from < 0) keep_from = n_slots;
        if (keep_from < n_slots) {
            std::vector<std::pair<int32_t, int32_t>> core;
            core.reserve(primary_gpu_pairs.size());
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] < keep_from) core.push_back(primary_gpu_pairs[i]);
            if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, core,
                                                    additional_gpu_pairs, offsets, bytes, err)) return false;
        }
    }

    const uint64_t spare_at = bytes;
    if (bytes == 0 || bytes + spare_bytes > budget) spare_sizes.clear();   // no room: swaps use exchange buffers
    else bytes += spare_bytes;

    // The experts the GPU cache was filled from sit in the file cache and the mapping; give them back first, so the
    // arena below does not have to reclaim them while the drive is being read.
    if (bytes > 0) {
        (void) basalt::platform::drop_mapped_view(base_, mapped_bytes_, 0, mapped_bytes_);
        basalt::platform::advise_dontneed(fd_, file_off_, mapped_bytes_);
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    bool pinned_ok = false;
    uint64_t locked = 0;
    uint64_t partial_pin = 0;   ///< a registered prefix of the arena (all of it once registered after the read)
    bool register_after = false;   ///< the arena is registered with the driver after it is filled
    uint64_t lock_off = 0;      ///< where the working-set lock starts (after the registered prefix)
    std::string note;
    // One alignment for both arena paths, so one call releases it whichever way it was chosen.
    constexpr size_t kArenaAlign = 2ull << 20;
    auto release = [&]() {
        if (arena == nullptr) return;
        if (pinned_ok) (void) cudaFreeHost(arena);
        else {
            if (partial_pin > 0) (void) cudaHostUnregister(arena);
            if (locked > 0) basalt::platform::unlock_resident((uint8_t*) arena + lock_off, locked);
            basalt::platform::free_aligned(arena);
        }
        arena = nullptr;
    };
    if (bytes > 0) {
        basalt::log::info("basalt", "FileExpertSource: allocating %.2f GiB %s cache complement",
                     (double) bytes / 1073741824.0, pin ? "page-locked" : "pageable resident");
        if (pin) {
            // Ordinary memory, aligned and padded to 2 MiB (whole large pages, so the registered range ends on a page
            // boundary), registered with the driver once it is filled: the reads fault the pages in while the drive
            // works, instead of the driver zeroing and locking all of it before the first read.
            arena = alloc_ram(bytes);
            if (arena != nullptr) {
                register_after = true;
            } else {
                note = "no aligned memory for registration";
            }
        }
        if (arena == nullptr) {
            arena = basalt::platform::alloc_aligned((size_t) bytes, kArenaAlign);
            if (arena == nullptr) {
                err = "FileExpertSource: pageable resident complement allocation failed";
                return false;
            }
            if (pin) {
                // CS-T, a RAM budget: its bytes are in profile order, hottest first, so the driver is asked to
                // register the largest prefix it takes (from the cap down in 2 GiB steps).  Those experts can be
                // read by the GPU over PCIe (the PCIe share) and copied by DMA; only the rest is locked in the working
                // set (the registered prefix is page-locked by the driver already - locking it twice made the next
                // device allocation fail).
                lock_off = partial_pin;
                const basalt::platform::LockResult lr =
                    basalt::platform::lock_resident((uint8_t*) arena + lock_off, bytes - lock_off);
                locked = lr.locked_bytes;
                note += (note.empty() ? "" : "; ") + lr.note;
            }
        }
        host = (const uint8_t*) arena;
    }

    const double alloc_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - pin_t0).count();
    // Progress counts the experts the GPU cache does not hold (those it holds, kept here for the prompt path's
    // loan, are already counted by its fill); experts left on the file are not placed at all.
    uint64_t progress_total = 0;
    for (int64_t layer = 0; layer < n_layers_; ++layer)
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
            if (offsets[index] == kNoComplement) continue;
            const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
            if (offsets[index] > bytes || blob_bytes > bytes - offsets[index] || mapped_blob(layer, expert) == nullptr) {
                err = "FileExpertSource: invalid blob bounds while building the cache complement";
                release();
                return false;
            }
            if (!on_gpu[index]) progress_total += blob_bytes;
        }
    // experts another GPU holds alone are placed by that tier's own phase
    uint64_t extra_gpu_bytes = 0;
    for (const auto& pr : additional_gpu_pairs) extra_gpu_bytes += layer_blob_bytes_[(size_t) pr.first];
    const uint64_t all_experts = expert_bytes();
    if (all_experts > gpu_bytes + extra_gpu_bytes + progress_total)
        LoadProgress::skipped(all_experts - gpu_bytes - extra_gpu_bytes - progress_total);
    LoadProgress::plan("ram", progress_total);
    LoadProgress progress("ram", progress_total);
    // Experts are read straight into the arena with unbuffered reads, so the file cache neither copies them nor
    // evicts anything; ranges that are not block-aligned, or a file system without unbuffered reads, take the mapped copy.
    RamGrid grid;
    grid.n_layers = n_layers_;
    grid.n_expert = n_expert_;
    grid.file_base = file_off_;
    grid.layer_offset.assign(layer_offsets_.begin(), layer_offsets_.end());
    grid.blob_bytes.assign(layer_blob_bytes_.begin(), layer_blob_bytes_.end());
    const auto copy_t0 = std::chrono::steady_clock::now();
    RamLoad load;
    load.fd = fd_;
    load.unbuffered = bytes > 0;
    load.threads = (int) std::max<int64_t>(1, std::min<int64_t>(6, n_layers_));
    load.progress = &progress;
    load.groups = ram_arena_runs(grid, (uint8_t*) host, offsets, kNoComplement, on_gpu);
    load.read_ahead = [&](int64_t layer, const std::vector<RamRun>& runs) {
        std::vector<std::pair<int32_t, int32_t>> ahead;
        for (const RamRun& r : runs)
            for (int64_t k = 0; k < r.count; ++k) ahead.emplace_back((int32_t) layer, (int32_t) (r.first + k));
        (void) advise_pairs(ahead.data(), (int64_t) ahead.size());
    };
    load.fallback = [&](int64_t layer, const RamRun& r, uint64_t done, std::string& e) {
        for (int64_t k = (int64_t) (done / r.blob_bytes); k < r.count; ++k)
            if (!copy_from_files(layer, r.first + k, r.dst + (size_t) k * (size_t) r.blob_bytes)) {
                e = "FileExpertSource: invalid blob bounds while building the cache complement";
                return false;
            }
        return true;
    };
    load.after_group = [&](int64_t layer, int64_t done, uint64_t copied, bool mapped_used, std::string& e) {
        if (mapped_used) {   // the layer's pages are not kept in the file cache once copied
            const uint64_t layer_offset = layer_offsets_[(size_t) layer];
            const uint64_t layer_bytes = layer_blob_bytes_[(size_t) layer] * (uint64_t) n_expert_;
            if (!basalt::platform::drop_mapped_view(base_, mapped_bytes_, layer_offset, layer_bytes)) {
                e = "FileExpertSource: could not release mapped expert layer " + std::to_string(layer);
                return false;
            }
            basalt::platform::advise_dontneed(fd_, file_off_ + layer_offset, layer_bytes);
        }
        if (done % 8 == 0 || done == n_layers_)
            basalt::log::info("basalt", "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB, %.0f s)",
                         (long long) done, (long long) n_layers_, (double) copied / 1073741824.0,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count());
        return true;
    };
    RamLoadStats read;
    if (!read_ram_runs(load, read, err)) {
        release();
        return false;
    }
    const double read_done_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count();
    if (register_after) {
        const RamPin rp = pin_ram(arena, ram_padded(bytes), bytes, {}, /*mapped=*/true, &device);
        if (device != nullptr) {
            partial_pin = bytes;   // released through the registered-prefix path
            lock_off = bytes;
            note = "page-locked and mapped";
        } else {
            note = "page-locking refused (" + rp.refused + ")";
            locked = rp.locked_bytes;
            note += "; " + rp.lock_note;
        }
    }
    progress.finish();
    {
        const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count();
        basalt::log::info("basalt", "FileExpertSource: cache complement: allocation %.1f s, read %.1f s (%.2f GB/s, %.1f%% unbuffered), registration %.1f s",
                     alloc_s, read_done_s, read_done_s > 0 ? (double) read.copied / 1e9 / read_done_s : 0.0,
                     read.copied > 0 ? 100.0 * (double) read.unbuffered / (double) read.copied : 0.0,
                     read_s - read_done_s);
    }

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    ++places_gen_;
    complement_pinned_ = (pinned_ok || partial_pin > 0) && bytes > 0;
    complement_pin_limit_ = pinned_ok ? bytes : partial_pin;
    complement_partial_ = !pinned_ok && partial_pin > 0;
    complement_locked_ = locked;
    complement_lock_off_ = lock_off;
    complement_lent_slots_ = lend ? n_slots - keep_from : 0;
    spare_free_.clear();
    {
        uint64_t at = spare_at;
        for (const auto& [b, n] : spare_sizes)
            for (int64_t i = 0; i < n; ++i, at += b) spare_free_.emplace_back(at, b);
        if (!spare_sizes.empty())
            basalt::log::info("basalt", "FileExpertSource: %lld free places for the adaptive swaps (%zu blob sizes, %.0f MiB)",
                         (long long) spare_free_.size(), spare_sizes.size(), (double) (at - spare_at) / 1048576.0);
    }
    complement_ready_ = true;
    // The GPU cache's experts are not in the complement: refilling a lent slot reads them through the view again.
    basalt::platform::ram_guard_reclaim_view(base_, mapped_bytes_);
    basalt::log::info("basalt", "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB in %.0f s%s%s",
                 complement_pinned_ ? "mapped pinned" : pin ? "locked resident" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - pin_t0).count(),
                 note.empty() ? "" : "; ", note.c_str());
    if (lend)
        basalt::log::info("basalt", "FileExpertSource: %lld of the prompt path's %lld lendable slots keep their experts in RAM "
                             "too%s", (long long) complement_lent_slots_, (long long) (n_slots - lend_from_slot),
                     complement_lent_slots_ < n_slots - lend_from_slot
                         ? " (the others are read from the file when lent: not enough RAM for them)" : "");
    if (!additional_gpu_pairs.empty()) {
        basalt::log::info("basalt", "FileExpertSource: %zu experts held only by another GPU are not kept here (read from the "
                             "file when needed)",
                     additional_gpu_pairs.size());
    }
    return true;
}

const uint8_t* FileExpertSource::host_at(uint64_t off, uint64_t bytes) const {
    if (off < complement_bytes_)
        return complement_host_ != nullptr && bytes <= complement_bytes_ - off ? complement_host_ + (size_t) off : nullptr;
    const uint64_t o2 = off - complement_bytes_;
    if (off >= homes_start() && !homes_.empty()) {
        const int r = detail::home_region_find(homes_, off, bytes);
        return r < 0 ? nullptr : (const uint8_t*) homes_[(size_t) r].arena + (size_t) (off - homes_[(size_t) r].base);
    }
    return top_arena_ != nullptr && o2 <= top_bytes_ && bytes <= top_bytes_ - o2 ? (const uint8_t*) top_arena_ + (size_t) o2
                                                                                 : nullptr;
}

// Main thread only, between requests, after wait_commit / apply_pending / xq_drain: no pool, keep-alive or read-ahead
// thread reads blob() or the offsets then.  Regions are appended in the offset space (homes_next_ only grows).
bool FileExpertSource::homes_take(const std::vector<std::pair<int32_t, int32_t>>& experts, std::string& err,
                                  std::vector<uint8_t*>& dst) {
    err.clear();
    dst.clear();
    if (!complement_ready_ || !homes_pending_.empty()) { err = "no resident copy to give homes to"; return false; }
    if (!staged_.empty()) { err = "an exchange is in flight"; return false; }
    uint64_t total = 0;
    for (const auto& [l, e] : experts) {
        if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) { err = "an expert outside the model"; return false; }
        total += layer_blob_bytes_[(size_t) l];
    }
    if (total == 0) return true;
    // the RAM left over keeps the homes reserve free above the reserve (what the guard stops a request at)
    const uint64_t budget = basalt::platform::ram_budget();
    if (total > budget || homes_reserve_ > budget - total) {
        const uint64_t room = budget > homes_reserve_ ? budget - homes_reserve_ : 0;
        err = "the experts the cache gives back need " + std::to_string((total + (1u << 20) - 1) >> 20) +
              " MiB of RAM, and " + std::to_string(room >> 20) + " MiB are free above the RAM floor";
        return false;
    }
    // regions of a bounded size: one that no expert uses any more can then be returned on its own
    constexpr uint64_t kRegionBytes = 96ull << 20;
    if (homes_next_ < homes_start()) homes_next_ = homes_start();
    size_t k = 0;
    while (k < experts.size()) {
        uint64_t bytes = 0;
        size_t n = 0;
        while (k + n < experts.size() && (n == 0 || bytes < kRegionBytes))
            bytes += layer_blob_bytes_[(size_t) experts[k + n++].first];
        detail::HomeRegion r;
        r.bytes = bytes;
        if (complement_pinned_ || top_pinned_) {
            void* arena = nullptr;
            if (cudaHostAlloc(&arena, (size_t) bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
                void* alias = nullptr;
                if (cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
                    r.arena = arena;
                    r.device = (const uint8_t*) alias;
                    r.pinned = true;
                } else {
                    (void) cudaFreeHost(arena);
                }
            }
            (void) cudaGetLastError();
        }
        if (r.arena == nullptr) {
            r.arena = std::malloc((size_t) bytes);
            if (r.arena == nullptr) {
                err = "the RAM for the experts the cache gives back could not be allocated";
                dst.clear();
                homes_abort();
                return false;
            }
            if (complement_locked_ > 0 || top_locked_ > 0)
                r.locked = basalt::platform::lock_resident(r.arena, bytes).locked_bytes;
        }
        r.base = homes_next_;
        homes_next_ += bytes;
        homes_.push_back(r);
        uint64_t at = 0;
        for (size_t m = 0; m < n; ++m) {
            const auto& [l, e] = experts[k + m];
            homes_pending_.emplace_back((size_t) l * (size_t) n_expert_ + (size_t) e, r.base + at);
            dst.push_back((uint8_t*) r.arena + (size_t) at);
            at += layer_blob_bytes_[(size_t) l];
        }
        k += n;
    }
    return true;
}

void FileExpertSource::homes_commit() {
    for (const auto& [i, off] : homes_pending_)
        if (i < complement_offsets_.size()) complement_offsets_[i] = off;
    homes_pending_.clear();
    ++places_gen_;
}

void FileExpertSource::homes_abort() {
    homes_pending_.clear();
    (void) homes_reclaim();
}

bool FileExpertSource::home_release(int64_t layer, int64_t expert) {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (i >= complement_offsets_.size() || complement_offsets_[i] == kNoComplement || complement_offsets_[i] < homes_start())
        return false;
    spare_free_.emplace_back(complement_offsets_[i], layer_blob_bytes_[(size_t) layer]);
    complement_offsets_[i] = kNoComplement;
    ++places_gen_;
    return true;
}

uint64_t FileExpertSource::homes_reclaim() {
    uint64_t freed = 0;
    for (size_t k = homes_.size(); k-- > 0;) {
        const detail::HomeRegion r = homes_[k];
        if (detail::home_region_used(r, complement_offsets_)) continue;
        bool pending = false;
        for (const auto& p : homes_pending_) pending = pending || (p.second >= r.base && p.second - r.base < r.bytes);
        if (pending) continue;
        spare_free_.erase(std::remove_if(spare_free_.begin(), spare_free_.end(),
                                         [&](const std::pair<uint64_t, uint64_t>& f) {
                                             return f.first >= r.base && f.first - r.base < r.bytes;
                                         }),
                          spare_free_.end());
        if (r.pinned) (void) cudaFreeHost(r.arena);
        else {
            if (r.locked > 0) basalt::platform::unlock_resident(r.arena, r.locked);
            std::free(r.arena);
        }
        freed += r.bytes;
        homes_.erase(homes_.begin() + (ptrdiff_t) k);
    }
    return freed;
}

uint64_t FileExpertSource::homes_compact() {
    if (!complement_ready_ || homes_.empty() || !homes_pending_.empty() || !staged_.empty() || exchange_live()) return 0;
    std::vector<std::pair<int32_t, int32_t>> live;
    uint64_t used = 0;
    for (size_t i = 0; i < complement_offsets_.size(); ++i) {
        if (complement_offsets_[i] == kNoComplement || complement_offsets_[i] < homes_start()) continue;
        const int32_t l = (int32_t) (i / (size_t) n_expert_);
        live.emplace_back(l, (int32_t) (i % (size_t) n_expert_));
        used += layer_blob_bytes_[(size_t) l];
    }
    constexpr uint64_t kSlackBytes = 96ull << 20;   // one region of slack is not worth a copy
    const uint64_t held = homes_bytes();
    if (held < used + kSlackBytes) return 0;
    if (live.empty()) return homes_reclaim();
    std::string err;
    std::vector<uint8_t*> dst;
    if (!homes_take(live, err, dst)) return 0;
    // the copies are made from the old places; the offsets move only at the commit
    for (size_t k = 0; k < live.size(); ++k) {
        const size_t i = (size_t) live[k].first * (size_t) n_expert_ + (size_t) live[k].second;
        const uint64_t bytes = layer_blob_bytes_[(size_t) live[k].first];
        const uint8_t* from = host_at(complement_offsets_[i], bytes);
        if (from == nullptr) { homes_abort(); return 0; }
        std::memcpy(dst[k], from, (size_t) bytes);
    }
    homes_commit();
    (void) homes_reclaim();
    return held - homes_bytes();
}

bool FileExpertSource::exchange_live() const {
    return std::any_of(override_.begin(), override_.end(), [](const uint8_t* p) { return p != nullptr; });
}

uint64_t FileExpertSource::homes_bytes() const {
    uint64_t n = 0;
    for (const auto& r : homes_) n += r.bytes;
    return n;
}

uint64_t FileExpertSource::top_up(const std::vector<std::pair<int32_t, int32_t>>& rank,
                                  const std::function<bool(int32_t, int32_t)>& skip, bool pin, int64_t& count) {
    count = 0;
    if (!complement_ready_ || top_arena_ != nullptr || !override_.empty() || !staged_.empty()) return 0;
    const uint64_t avail = basalt::platform::available_memory();
    if (avail == 0) return 0;
    const uint64_t room = ram_tier_room(avail, basalt::platform::ram_reserve(), pin);
    // the experts no tier holds, hottest first, as many as fit
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<uint64_t> at;
    std::vector<uint8_t> seen(complement_offsets_.size(), 0);
    uint64_t total = 0;
    auto consider = [&](int32_t l, int32_t e) {
        if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) return true;
        const size_t i = (size_t) l * (size_t) n_expert_ + (size_t) e;
        if (seen[i]) return true;
        seen[i] = 1;
        if (complement_offsets_[i] != kNoComplement || skip(l, e)) return true;
        const uint64_t b = layer_blob_bytes_[(size_t) l];
        if (b > room - total) return false;
        pick.emplace_back(l, e);
        at.push_back(total);
        total += b;
        return true;
    };
    bool more = true;
    for (const auto& pr : rank) if (!(more = consider(pr.first, pr.second))) break;
    for (int32_t l = 0; more && l < (int32_t) n_layers_; ++l)
        for (int32_t e = 0; more && e < (int32_t) n_expert_; ++e) more = consider(l, e);
    if (total == 0) return 0;
    void* arena = nullptr;
    const uint8_t* device = nullptr;
    bool pinned_ok = false;
    uint64_t locked = 0;
    if (pin && cudaHostAlloc(&arena, (size_t) total, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
            device = (const uint8_t*) alias;
            pinned_ok = true;
        } else {
            (void) cudaFreeHost(arena);
            arena = nullptr;
        }
    }
    (void) cudaGetLastError();
    if (arena == nullptr) {
        arena = std::malloc((size_t) total);
        if (arena == nullptr) return 0;
        if (pin) locked = basalt::platform::lock_resident(arena, total).locked_bytes;
    }
    RamLoad load;
    load.fd = fd_;
    load.unbuffered = false;
    load.threads = (int) std::min<size_t>(6, pick.size());
    load.groups.resize(pick.size());
    for (size_t k = 0; k < pick.size(); ++k) {
        const uint64_t b = layer_blob_bytes_[(size_t) pick[k].first];
        add_ram_blob(load.groups[k], (uint8_t*) arena + (size_t) at[k], 0, b, pick[k].second, 0);
    }
    load.fallback = [&](int64_t k, const RamRun& r, uint64_t, std::string&) {
        return copy_from_files(pick[(size_t) k].first, r.first, r.dst);
    };
    RamLoadStats read;
    std::string read_err;
    if (!read_ram_runs(load, read, read_err)) {
        if (pinned_ok) (void) cudaFreeHost(arena);
        else {
            if (locked > 0) basalt::platform::unlock_resident(arena, locked);
            std::free(arena);
        }
        return 0;
    }
    top_arena_ = arena;
    top_device_ = device;
    top_bytes_ = total;
    top_locked_ = locked;
    top_pinned_ = pinned_ok;
    for (size_t k = 0; k < pick.size(); ++k)
        complement_offsets_[(size_t) pick[k].first * (size_t) n_expert_ + (size_t) pick[k].second] = complement_bytes_ + at[k];
    ++places_gen_;
    count = (int64_t) pick.size();
    return total;
}

bool FileExpertSource::has_resident(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement;
}

bool FileExpertSource::resident_in_top_up(int64_t layer, int64_t expert) const {
    if (!has_resident(layer, expert)) return false;
    const uint64_t off = complement_offsets_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
    return off >= complement_bytes_ && off < homes_start();
}

bool FileExpertSource::reserve_exchanges(int64_t n, std::string& err) {
    err.clear();
    if (n <= xstage_cap_) return true;
    if (!staged_.empty()) { err = "FileExpertSource: exchange buffers are in use"; return false; }
    uint64_t blob = 0;
    for (const uint64_t b : layer_blob_bytes_) blob = std::max(blob, b);
    if (blob == 0 || n <= 0) { err = "FileExpertSource: no expert geometry for the exchange buffers"; return false; }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
        xstage_ = nullptr;
        xstage_cap_ = 0;
    }
    const size_t total = (size_t) n * (size_t) blob;
    if (total > platform::ram_budget()) {
        err = "the exchange buffers need " + std::to_string(total >> 20) + " MiB of RAM, which is not available";
        return false;
    }
    void* p = nullptr;
    if (cudaHostAlloc(&p, total, cudaHostAllocDefault) == cudaSuccess && p != nullptr) {
        xstage_pinned_ = true;
    } else {
        (void) cudaGetLastError();
        p = std::malloc(total);
        xstage_pinned_ = false;
        if (p == nullptr) { err = "FileExpertSource: cannot allocate the exchange buffers"; return false; }
    }
    xstage_ = (uint8_t*) p;
    xstage_cap_ = n;
    xstage_blob_ = blob;
    return true;
}

uint8_t* FileExpertSource::spare_take(int64_t layer, uint64_t& off) {
    if (!complement_ready_ || layer < 0 || layer >= n_layers_) return nullptr;
    const uint64_t need = layer_blob_bytes_[(size_t) layer];
    size_t pick = spare_free_.size();
    for (size_t i = spare_free_.size(); i-- > 0;)   // a place outside the homes first: those regions can then be freed
        if (spare_free_[i].second == need && (pick == spare_free_.size() || spare_free_[i].first < homes_start())) {
            pick = i;
            if (spare_free_[i].first < homes_start()) break;
        }
    if (pick == spare_free_.size()) return nullptr;
    off = spare_free_[pick].first;
    spare_free_[pick] = spare_free_.back();
    spare_free_.pop_back();
    return (uint8_t*) host_at(off, need);
}

bool FileExpertSource::spare_land(int64_t layer, int64_t out, uint64_t off) {
    if (!complement_ready_ || layer < 0 || out < 0 || layer >= n_layers_ || out >= n_expert_) return false;
    const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (i >= complement_offsets_.size() || complement_offsets_[i] != kNoComplement ||
        host_at(off, layer_blob_bytes_[(size_t) layer]) == nullptr) return false;
    complement_offsets_[i] = off;
    ++exchanges_;
    ++places_gen_;
    return true;
}

bool FileExpertSource::spare_release(int64_t layer, int64_t in) {
    if (!complement_ready_ || layer < 0 || in < 0 || layer >= n_layers_ || in >= n_expert_) return false;
    const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    if (i >= complement_offsets_.size() || complement_offsets_[i] == kNoComplement) return false;
    spare_free_.emplace_back(complement_offsets_[i], layer_blob_bytes_[(size_t) layer]);
    complement_offsets_[i] = kNoComplement;
    ++places_gen_;
    return true;
}

bool FileExpertSource::equals_file(int64_t layer, int64_t expert, const uint8_t* bytes) const {
    if (bytes == nullptr || layer < 0 || layer >= n_layers_) return false;
    std::vector<uint8_t> f((size_t) layer_blob_bytes_[(size_t) layer]);
    return copy_from_files(layer, expert, f.data()) && std::memcmp(f.data(), bytes, f.size()) == 0;
}

void FileExpertSource::spare_return(int64_t layer, uint64_t off) {
    if (layer >= 0 && layer < n_layers_) spare_free_.emplace_back(off, layer_blob_bytes_[(size_t) layer]);
}

uint8_t* FileExpertSource::exchange_buffer(int64_t q) const {
    if (xstage_ == nullptr || q < 0 || q >= xstage_cap_) return nullptr;
    return xstage_ + (size_t) q * (size_t) xstage_blob_;
}

bool FileExpertSource::stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q) {
    if (!has_resident(layer, in) || has_resident(layer, out) || exchange_buffer(q) == nullptr) return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (override_.empty()) override_.assign((size_t) blobs_, nullptr);
    if (override_[i_out] != nullptr) return false;
    for (const Exchange& x : staged_)
        if (x.in == i_in || x.q == q) return false;
    override_[i_out] = exchange_buffer(q);
    staged_.push_back({i_in, i_out, q, layer_blob_bytes_[(size_t) layer]});
    return true;
}

bool FileExpertSource::copy_exchanges(uint64_t max_bytes) {
    uint64_t base = 0;
    const uint64_t end = xcopy_at_ + max_bytes;
    for (const Exchange& x : staged_) {
        const uint64_t lo = std::max(base, xcopy_at_), hi = std::min(base + x.bytes, end);
        const uint8_t* src = override_[x.out];
        const uint64_t at = complement_offsets_[x.in];
        const uint8_t* dst = at != kNoComplement ? host_at(at, x.bytes) : nullptr;
        if (lo < hi && src != nullptr && dst != nullptr)
            std::memcpy((uint8_t*) dst + (size_t) (lo - base), src + (lo - base), (size_t) (hi - lo));
        base += x.bytes;
    }
    xcopy_at_ = std::min(end, base);
    xcopied_ = xcopy_at_ >= base;
    return xcopied_;
}

int64_t FileExpertSource::commit_exchanges() {
    int64_t n = 0;
    const bool copied = xcopied_;
    xcopied_ = false;
    if (!copied && xcopy_at_ > 0) copy_exchanges(~0ull >> 1);   // the rest of a partial copy (the same bytes)
    const bool done = xcopied_ || copied;
    xcopied_ = false;
    xcopy_at_ = 0;
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        const uint64_t at = complement_offsets_[x.in];
        const uint8_t* dst = at != kNoComplement ? host_at(at, x.bytes) : nullptr;
        if (src != nullptr && dst != nullptr) {
            if (!done) std::memcpy((uint8_t*) dst, src, (size_t) x.bytes);
            if (detail::exchange_cache_complement(complement_offsets_, x.in, x.out)) ++n;
        }
        override_[x.out] = nullptr;
    }
    if (!staged_.empty()) ++places_gen_;
    staged_.clear();
    exchanges_ += n;
    return n;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* result = nullptr;
    bool from_files = true;
    if (complement_ready_) {
        result = complement_offsets_[index] != kNoComplement
                     ? host_at(complement_offsets_[index], layer_blob_bytes_[(size_t) layer]) : nullptr;
        if (result != nullptr) {
            ram_reads_.fetch_add(1, std::memory_order_relaxed);
            from_files = false;
        } else if (!override_.empty() && override_[index] != nullptr) {
            result = override_[index];
            from_files = false;
        }
    }
    if (from_files) {
        result = mapped_blob(layer, expert);
        if (result != nullptr)
            file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        if (complement_ready_ && result != nullptr) file_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    if (result != nullptr) ++reads_;
    return result;
}

const uint8_t* FileExpertSource::resident_blob(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_offsets_[index] != kNoComplement)
        return host_at(complement_offsets_[index], layer_blob_bytes_[(size_t) layer]);
    return !override_.empty() ? override_[index] : nullptr;
}

// A helper thread issues the read-ahead so the caller never waits on the kernel.  The queue is bounded; a request
// that does not fit is dropped (the pages are then read on first touch, as without the advice).
struct FileExpertSource::AdviseWorker {
    explicit AdviseWorker(const FileExpertSource* s) : src(s), th([this] { run(); }) {}
    ~AdviseWorker() {
        { std::lock_guard<std::mutex> g(mu); stop = true; }
        cv.notify_one();
        th.join();
    }
    void run() {
        std::vector<std::pair<int32_t, int32_t>> batch;
        for (;;) {
            {
                std::unique_lock<std::mutex> g(mu);
                cv.wait(g, [this] { return stop || !q.empty(); });
                if (q.empty()) return;
                batch.swap(q);
            }
            (void) src->advise_pairs(batch.data(), (int64_t) batch.size());
            batch.clear();
        }
    }
    static constexpr size_t kMaxQueued = 512;
    const FileExpertSource* src;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::pair<int32_t, int32_t>> q;
    bool stop = false;
    std::thread th;
};

void FileExpertSource::advise_cpu(int64_t layer, const int32_t* ids, int64_t n) const {
    if (base_ == nullptr || ids == nullptr || n <= 0 || !basalt::platform::read_ahead_enabled()) return;
    std::call_once(advise_once_, [this] { advise_worker_.reset(new AdviseWorker(this)); });
    AdviseWorker& w = *advise_worker_;
    {
        std::lock_guard<std::mutex> g(w.mu);
        if (w.q.size() + (size_t) n > AdviseWorker::kMaxQueued) return;
        for (int64_t i = 0; i < n; ++i) w.q.emplace_back((int32_t) layer, ids[i]);
    }
    w.cv.notify_one();
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (index >= complement_offsets_.size() || complement_offsets_[index] == kNoComplement) return false;
    if (complement_offsets_[index] >= homes_start()) {
        const int r = detail::home_region_find(homes_, complement_offsets_[index], layer_blob_bytes_[(size_t) layer]);
        return r >= 0 && homes_[(size_t) r].pinned;
    }
    if (complement_offsets_[index] >= complement_bytes_) return top_pinned_;
    if (!complement_pinned_ || complement_host_ == nullptr) return false;
    // a partial pin (CS-T): only the registered prefix
    return !complement_partial_ ||
           complement_offsets_[index] + layer_blob_bytes_[(size_t) layer] <= complement_pin_limit_;
}

void FileExpertSource::repin_join() {
    for (auto& t : pin_threads_) t.join();
    pin_threads_.clear();
}

// Each range registered by itself: the whole arena (as loaded), or the ranges of an earlier WAKE.
void FileExpertSource::unregister_complement() {
    if (unpinned_ || complement_arena_ == nullptr) return;
    if (pin_units_.size() >= 2 && pin_state_) {
        for (size_t i = 0; i + 1 < pin_units_.size(); ++i)
            if (pin_state_[i].load() == 1) (void) cudaHostUnregister((uint8_t*) complement_arena_ + pin_units_[i]);
    } else {
        (void) cudaHostUnregister(complement_arena_);
    }
    (void) cudaGetLastError();
}

bool FileExpertSource::unpin_complement(std::string& err) {
    if (unpinned_) return true;
    if (!complement_ready_ || complement_arena_ == nullptr || !complement_partial_ ||
        complement_pin_limit_ != complement_bytes_ || complement_device_ == nullptr ||
        complement_device_ != complement_host_) {
        err = "the RAM copy is not one registered range read at its host addresses";
        return false;
    }
    repin_join();
    unregister_complement();
    unpinned_ = true;
    pin_units_.clear();   // laid out again at the next WAKE (adaptive swaps may have moved blobs since)
    pin_state_.reset();
    return true;
}

void FileExpertSource::repin_start(int threads) {
    if (!unpinned_ || !pin_threads_.empty()) return;
    if (pin_units_.empty()) {
        // the ranges end where a blob starts on a page boundary, about 1 GiB apart (a blob never spans two ranges,
        // so every copy from one stays a copy from registered memory); the last one ends where the load registered
        constexpr uint64_t kPage = 4096, kUnit = 1ull << 30, kHuge = 2ull << 20;
        const uint64_t end = (complement_bytes_ + kHuge - 1) / kHuge * kHuge;
        std::vector<uint64_t> starts;
        for (uint64_t off : complement_offsets_)
            if (off != kNoComplement && off < complement_bytes_ && off % kPage == 0) starts.push_back(off);
        std::sort(starts.begin(), starts.end());
        pin_units_.push_back(0);
        for (uint64_t s : starts)
            if (s >= pin_units_.back() + kUnit) pin_units_.push_back(s);
        pin_units_.push_back(end);
    }
    const size_t n = pin_units_.size() - 1;
    pin_state_.reset(new std::atomic<int>[n]);
    for (size_t i = 0; i < n; ++i) pin_state_[i].store(0);
    pin_next_.store(0);
    for (int t = 0; t < std::max(1, threads); ++t)
        pin_threads_.emplace_back([this, n] {
            for (;;) {
                const size_t i = pin_next_.fetch_add(1);
                if (i >= n) return;
                const cudaError_t e = cudaHostRegister((uint8_t*) complement_arena_ + pin_units_[i],
                                                       (size_t) (pin_units_[i + 1] - pin_units_[i]),
                                                       cudaHostRegisterPortable | cudaHostRegisterMapped);
                if (e != cudaSuccess) (void) cudaGetLastError();
                pin_state_[i].store(e == cudaSuccess ? 1 : 2, std::memory_order_release);
            }
        });
}

bool FileExpertSource::repin_wait(const void* p, uint64_t n) const {
    if (!unpinned_ || !pin_state_ || pin_units_.size() < 2) return true;
    const uint8_t* base = (const uint8_t*) complement_arena_;
    const uint8_t* q = (const uint8_t*) p;
    if (q < base || q >= base + pin_units_.back()) return true;   // not in the copy (a RAM home)
    const uint64_t lo = (uint64_t) (q - base), hi = std::min<uint64_t>(lo + n, pin_units_.back());
    size_t i = (size_t) (std::upper_bound(pin_units_.begin(), pin_units_.end(), lo) - pin_units_.begin()) - 1;
    for (; i + 1 < pin_units_.size() && pin_units_[i] < hi; ++i) {
        int s;
        while ((s = pin_state_[i].load(std::memory_order_acquire)) == 0) std::this_thread::yield();
        if (s != 1) return false;
    }
    return true;
}

bool FileExpertSource::repin_finish(std::string& err) {
    if (!unpinned_) return true;
    repin_join();
    if (!pin_state_) { err = "the RAM copy was not registered again"; return false; }
    for (size_t i = 0; i + 1 < pin_units_.size(); ++i)
        if (pin_state_[i].load() != 1) {
            err = "the driver refused to register the RAM copy again";
            return false;
        }
    unpinned_ = false;
    return true;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert)) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t off = complement_offsets_[index];
    if (off < complement_bytes_ && complement_device_ == nullptr) return nullptr;
    if (off >= homes_start()) {
        const int r = detail::home_region_find(homes_, off, layer_blob_bytes_[(size_t) layer]);
        return r >= 0 ? homes_[(size_t) r].device + (size_t) (off - homes_[(size_t) r].base) : nullptr;
    }
    return off >= complement_bytes_ ? top_device_ + (size_t) (off - complement_bytes_) : complement_device_ + (size_t) off;
}

bool FileExpertSource::pcie_layer(int64_t layer) const {
    if (complement_ready_ && complement_pinned_ && complement_device_ != nullptr)
        return layer >= 0 && layer < n_layers_;
    return device_alias(layer, 0) != nullptr;
}

namespace {

void hash_u64(uint64_t& h, uint64_t v) {
    h = fnv1a64((const uint8_t*) &v, sizeof v, h);
}

void hash_text(uint64_t& h, const std::string& s) {
    h = fnv1a64((const uint8_t*) s.data(), (uint64_t) s.size(), h);
}

// The fingerprint of the model's experts: the index and the layout in full, the experts sampled.
bool shared_arena_range_hash(const ExpertFiles& files, const basalt::kernels::cpu::ExpertLayout& lay, uint64_t& out,
                             std::string& err) {
    uint64_t h = 1469598103934665603ull;
    hash_text(h, "basalt-shared-expert-arena-pack-v1");
    hash_u64(h, (uint64_t) lay.n_layers);
    hash_u64(h, (uint64_t) lay.n_expert);
    hash_u64(h, lay.total);
    hash_u64(h, lay.max_blob);
    hash_u64(h, lay.native ? 1 : 0);
    const std::pair<const char*, const FileRange*> small[] = {{"index", &files.index}, {"layout", &files.layout}};
    for (const auto& [name, r] : small) {
        hash_text(h, name);
        std::vector<uint8_t> raw;
        std::string e;
        if (r->empty() || !read_range(*r, raw, e)) { hash_u64(h, 0); continue; }
        hash_u64(h, 1);
        h = fnv1a64(raw.data(), (uint64_t) raw.size(), h);
    }
    const uint64_t bytes = files.experts.size();
    hash_text(h, "experts");
    hash_u64(h, bytes);
    constexpr uint64_t sample = 64u << 10;
    const uint64_t starts[3] = {0, bytes / 2, bytes > sample ? bytes - sample : 0};
    std::vector<uint8_t> buf((size_t) std::min<uint64_t>(sample, bytes));
    for (uint64_t off : starts) {
        if (buf.empty()) break;
        const uint64_t at = std::min<uint64_t>(off, bytes - (uint64_t) buf.size());
        if (!read_range_at(files.experts, at, buf.data(), buf.size(), err)) {
            err = "ArenaExpertSource: cannot sample the experts: " + err;
            return false;
        }
        hash_u64(h, at);
        h = fnv1a64(buf.data(), (uint64_t) buf.size(), h);
    }
    out = h == 0 ? 1 : h;
    return true;
}

}  // namespace

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const ExpertFiles& files_in, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err,
                             const std::string& shared_arena_file) {
    close();
    const std::string& path = files_in.experts.path;
    // the layout (the per-layer blobs) was loaded by the driver
    const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong file should name the two numbers rather than spend a
    // minute of loading first.
    {
        if (!files_in.experts.exists()) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = files_in.experts.size();
        if (got != want && got != want + (uint64_t) blob) {   // + one blob: a file padded for whole-slot copies
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the model this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // THE RAM BEFORE THE ALLOCATION.  The arena is an anonymous mapping that succeeds whatever the host has;
    // its pages are committed as the load writes them, so a container whose memory limit is below the arena would be
    // killed by the OOM killer part-way through the load, without a message.  A hard cgroup limit below it is certain
    // to end that way: refused, with the numbers.  Less RAM available than the arena (another program, an engine
    // still exiting) is refused as well, with the numbers: the arena is locked resident, so the room is needed up front.
    ram_warning_.clear();
    // (a shared arena, --shared-expert-arena, may already be in RAM for another engine: not checked)
    if (detail::HostMemory hm; shared_arena_file.empty() && detail::host_available_memory(hm)) {
        const uint64_t need = want + (uint64_t) blob;
        const double gib = 1073741824.0;
        char buf[512];
        if (hm.cgroup_limit < need) {
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: the expert arena needs %.2f GiB of RAM but this process's memory limit "
                          "(cgroup memory.max / memory.limit_in_bytes) is %.2f GiB: it would be killed while loading. "
                          "Raise the container's limit, or run with less RAM: --mmap-experts with "
                          "--resident-budget-gib N keeps only the hottest experts in RAM",
                          (double) need / gib, (double) hm.cgroup_limit / gib);
            err = buf;
            return false;
        }
        // the arena is locked resident: it must leave the reserve free, or the machine swaps
        bool known = false;
        const uint64_t budget = platform::ram_budget(&known);
        if ((known && need > budget) || hm.available < need) {
            std::snprintf(buf, sizeof buf,
                          "the experts need %.1f GiB of RAM, %.1f GiB is available; use --resident-experts or "
                          "--mmap-experts", (double) need / gib, (double) hm.available / gib);
            err = buf;
            return false;
        }
    }

    uint64_t arena_hash = 0;
    if (!shared_arena_file.empty() && !shared_arena_range_hash(files_in, lay, arena_hash, err)) return false;

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds, shared_arena_file,
                                     arena_hash);
    if (!a->valid()) {
        const std::string why = a->note;
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" +
              std::to_string(want + (uint64_t) blob) + " B)" +
              (why.empty() ? std::string{} : ": " + why);
        return false;
    }
    // read through the file cache, which a warm restart serves from
    const int fd = basalt::platform::open_fd_utf8(path);
    if (fd < 0) {
        delete a;
        err = "ArenaExpertSource: cannot open " + path;
        return false;
    }
    constexpr uint64_t kChunk = 8ull << 20;
    LoadProgress progress("ram", want);
    RamLoad load;
    load.fd = fd;
    load.unbuffered = false;
    load.threads = std::max(threads, 1);
    load.chunk = kChunk;
    load.progress = &progress;
    load.groups.resize((size_t) n_layers);
    for (size_t l = 0; l < (size_t) n_layers; ++l)
        for (uint64_t pos = 0; pos < lbytes[l]; pos += kChunk) {
            RamRun r;
            r.dst = a->data() + (size_t) (loff[l] + pos);
            r.file_pos = files_in.experts.offset + loff[l] + pos;
            r.bytes = std::min(kChunk, lbytes[l] - pos);
            r.progress = r.bytes;
            load.groups[l].push_back(r);
        }
    RamLoadStats st;
    std::string read_err;
    const bool read_ok = read_ram_runs(load, st, read_err);
    basalt::platform::close_fd(fd);
    basalt::log::info("basalt generate", "expert arena read through the file cache (buffered)");
    if (!read_ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + read_err;
        return false;
    }
    progress.finish();
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.seconds > 0 ? (double) st.copied / 1073741824.0 / st.seconds : 0.0;
    return true;
}

void ArenaExpertSource::close() {
    map_.reset();
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

void ArenaExpertSource::prefetch(int64_t layer, int64_t expert) {
    if (!map_ || layer < 0 || expert < 0 || expert >= n_expert_) return;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    map_.prefetch(lay.blob_offset(layer, expert), lay.blob_bytes(layer));
}

// Only the pages wholly inside the blob: a page shared with a neighbour the CPU may still read stays (see
// MappedFile::release).
uint64_t ArenaExpertSource::release(int64_t layer, int64_t expert) {
    if (!map_ || layer < 0 || expert < 0 || expert >= n_expert_) return 0;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    return map_.release(lay.blob_offset(layer, expert), lay.blob_bytes(layer));
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::resident_blob(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_ || layer * n_expert_ + expert >= blobs_)
        return nullptr;
    return base_ + basalt::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return base_ + basalt::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

}  // namespace basalt::core
