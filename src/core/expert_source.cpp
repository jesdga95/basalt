// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "basalt/core/expert_source.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/remote_experts.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include "basalt/core/pinned.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"

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
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) basalt::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) basalt::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) basalt::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
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
    // A plain descriptor for the prompt path's positional reads: the view stays read-only for the compute, while the
    // staging buffers are filled with reads that do not fault mapping pages in. A failure here is not fatal - the
    // caller falls back to copying from the view.
    fd_ = platform::open_fd_utf8(path);
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
    if (complement_arena_ != nullptr) {
        if (complement_pinned_ && !complement_partial_) (void) cudaFreeHost(complement_arena_);
        else {
            if (complement_partial_) (void) cudaHostUnregister(complement_arena_);
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
    complement_pinned_ = false;
    complement_partial_ = false;
    complement_pin_limit_ = 0;
    complement_lock_off_ = 0;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
    ram_reads_.store(0);
    file_read_bytes_.store(0);
    file_us_.store(0);
    view_.reset();
    platform::close_fd(fd_);
    fd_ = -1;
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

bool FileExpertSource::read_blobs(const std::pair<int32_t, int32_t>* pairs, int64_t n, const BlobSink& sink,
                                  std::string& err) {
    constexpr uint64_t kAlign = 4096;
    int dfd = -1;
    if (n > 0 && !complement_ready_ && base_ != nullptr && fd_ >= 0) dfd = basalt::platform::open_unbuffered(fd_);
    uint64_t max_blob = 0;
    for (int64_t i = 0; dfd >= 0 && i < n; ++i) {
        const int64_t l = pairs[i].first;
        if (l < 0 || l >= n_layers_) { basalt::platform::close_fd(dfd); dfd = -1; break; }
        max_blob = std::max(max_blob, layer_blob_bytes_[(size_t) l]);
    }
    // one staging buffer per reader: the blob's block-aligned span, pinned so the copy to a GPU runs at link speed
    const int readers = (int) std::max<int64_t>(1, std::min<int64_t>(kBlobReaders, n));
    const size_t stage_bytes = (size_t) ((max_blob + 2 * kAlign + kAlign - 1) / kAlign * kAlign);
    std::vector<uint8_t*> stage((size_t) readers, nullptr);
    struct Free {
        std::vector<uint8_t*>& v;
        int& fd;
        ~Free() {
            for (uint8_t* p : v) if (p) cudaFreeHost(p);
            if (fd >= 0) basalt::platform::close_fd(fd);
        }
    } cleanup{stage, dfd};
    for (int w = 0; dfd >= 0 && w < readers; ++w) {
        void* m = nullptr;
        if (cudaHostAlloc(&m, stage_bytes, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            basalt::platform::close_fd(dfd);
            dfd = -1;
            break;
        }
        stage[(size_t) w] = (uint8_t*) m;
    }
    if (dfd < 0) {
        // Without unbuffered reads the blobs come through the mapping, whose pages stay in this process after the copy.
        // Each is given back once the sink has it, so a fill of every expert does not leave the whole file resident.
        const bool give_back = n > 0 && !complement_ready_ && base_ != nullptr;
        if (!give_back) return ExpertSource::read_blobs(pairs, n, sink, err);
        const BlobSink and_drop = [&](int64_t i, int w, const uint8_t* blob, std::string& e) {
            if (!sink(i, w, blob, e)) return false;
            if (blob >= base_ && blob < base_ + mapped_bytes_)
                (void) basalt::platform::drop_mapped_view(base_, mapped_bytes_, (uint64_t) (blob - base_),
                                                          layer_blob_bytes_[(size_t) pairs[i].first]);
            return true;
        };
        return ExpertSource::read_blobs(pairs, n, and_drop, err);
    }

    std::atomic<int64_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    std::string fail_msg;
    auto worker = [&](int w) {
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= n || failed.load()) return;
            const int64_t l = pairs[i].first, e = pairs[i].second;
            const uint8_t* mapped = mapped_blob(l, e);
            std::string msg;
            if (mapped == nullptr) {
                msg = "expert blob " + std::to_string(l) + "/" + std::to_string(e) + " is out of range";
            } else {
                const uint64_t bytes = layer_blob_bytes_[(size_t) l];
                const uint64_t pos = file_off_ + (uint64_t) (mapped - base_);
                const uint64_t a0 = pos - pos % kAlign;
                const uint64_t a1 = (pos + bytes + kAlign - 1) / kAlign * kAlign;
                const int64_t got = basalt::platform::pread(dfd, stage[(size_t) w], (size_t) (a1 - a0), a0);
                const uint8_t* data = got > 0 && (uint64_t) got >= pos + bytes - a0 ? stage[(size_t) w] + (pos - a0)
                                                                                    : mapped;   // short read: mapped copy
                if (sink(i, w, data, msg)) continue;
            }
            std::lock_guard<std::mutex> lock(fail_mu);
            if (fail_msg.empty()) fail_msg = msg;
            failed.store(true);
            return;
        }
    };
    std::vector<std::thread> pool;
    for (int w = 1; w < readers; ++w) pool.emplace_back(worker, w);
    worker(0);
    for (auto& t : pool) t.join();
    if (failed.load()) {
        err = fail_msg;
        return false;
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
    headroom_bytes = std::max(headroom_bytes, platform::ram_margin());   // never lock into the machine's last RAM
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
        const uint64_t room = detail::resident_room(physical, headroom_bytes);
        if (budget_bytes > room) {
            if (what_fits)
                std::fprintf(stderr, "FileExpertSource: RAM room for the complement: %.2f GiB (%.2f GiB available "
                                     "minus %.0f GiB headroom and the page overhead)\n",
                             (double) room / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0);
            else
                std::fprintf(stderr, "FileExpertSource: --resident-budget-gib %.2f is more than the RAM has room for "
                                     "(%.2f GiB available minus %.0f GiB headroom and the page overhead): %.2f GiB\n",
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
        std::fprintf(stderr, "FileExpertSource: RAM budget %.2f GiB: %lld of the %.2f GiB of experts the GPU cache does "
                             "not hold, by profile rank; the rest are read from the files\n",
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
        budget = detail::resident_room(physical, headroom_bytes);
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
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, pin ? "page-locked" : "pageable resident");
        std::fflush(stderr);
        if (pin) {
            // Ordinary memory, aligned and padded to 2 MiB (whole large pages, so the registered range ends on a page
            // boundary), registered with the driver once it is filled: the reads fault the pages in while the drive
            // works, instead of the driver zeroing and locking all of it before the first read.
            const size_t padded = ((size_t) bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
            arena = basalt::platform::alloc_aligned(padded, kArenaAlign);
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
    // Experts are read straight into the arena with unbuffered reads, so the file cache neither copies them nor
    // evicts anything; ranges that are not block-aligned, or a file system without unbuffered reads, take the mapped copy.
    constexpr uint64_t kDirectAlign = 4096, kDirectChunk = 16ull << 20;
    // A descriptor whose reads bypass the OS cache, or -1 where the platform offers none: the copy then comes from
    // the mapping for every range, which is what the file's pages were read through before.
    int direct_fd = -1;
    if (bytes > 0) direct_fd = basalt::platform::open_unbuffered(fd_);
    // Copied layer by layer on a few threads; the layer's contiguous experts are read as one range.
    std::atomic<int64_t> next_layer{0}, layers_done{0};
    std::atomic<uint64_t> copied{0};
    std::atomic<uint64_t> direct_bytes{0};
    // Progress counts the experts the GPU cache does not hold (those it holds, kept here for the prompt path's
    // loan, are already counted by its fill); experts left on the file are not placed at all.
    uint64_t progress_total = 0;
    for (int64_t layer = 0; layer < n_layers_; ++layer)
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
            if (offsets[index] != kNoComplement && !on_gpu[index]) progress_total += layer_blob_bytes_[(size_t) layer];
        }
    // experts another GPU holds alone are placed by that tier's own phase
    uint64_t extra_gpu_bytes = 0;
    for (const auto& pr : additional_gpu_pairs) extra_gpu_bytes += layer_blob_bytes_[(size_t) pr.first];
    const uint64_t all_experts = expert_bytes();
    if (all_experts > gpu_bytes + extra_gpu_bytes + progress_total)
        LoadProgress::skipped(all_experts - gpu_bytes - extra_gpu_bytes - progress_total);
    LoadProgress::plan("ram", progress_total);
    LoadProgress progress("ram", progress_total);
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    const auto copy_t0 = std::chrono::steady_clock::now();
    std::string fail_msg;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = m;
        failed.store(true);
    };
    auto worker = [&]() {
        uint8_t* stage = nullptr;   // staging for ranges that are not block-aligned
        struct StageFree { uint8_t*& p; ~StageFree() { basalt::platform::free_aligned(p); } } stage_free{stage};
        for (;;) {
            const int64_t layer = next_layer.fetch_add(1);
            if (layer >= n_layers_ || failed.load()) return;
            const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
            bool mapped_used = false;
            // one run: consecutive experts adjacent in both the file and the arena
            struct Run { uint64_t file_pos, dst_off, len; int64_t first, count; };
            std::vector<Run> runs;
            for (int64_t expert = 0; expert < n_expert_; ++expert) {
                const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
                const uint64_t offset = offsets[index];
                if (offset == kNoComplement) continue;
                if (offset > bytes || blob_bytes > bytes - offset || mapped_blob(layer, expert) == nullptr) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                const uint64_t fpos = file_off_ + layer_offsets_[(size_t) layer] + (uint64_t) expert * blob_bytes;
                if (!runs.empty() && runs.back().file_pos + runs.back().len == fpos &&
                    runs.back().dst_off + runs.back().len == offset) {
                    runs.back().len += blob_bytes;
                    ++runs.back().count;
                } else {
                    runs.push_back(Run{fpos, offset, blob_bytes, expert, 1});
                }
            }
            if (direct_fd < 0 || !basalt::platform::read_ahead_enabled()) {
                // mapped copy: ask for the layer's blobs before copying them
                std::vector<std::pair<int32_t, int32_t>> ahead;
                for (const Run& r : runs)
                    for (int64_t k = 0; k < r.count; ++k) ahead.emplace_back((int32_t) layer, (int32_t) (r.first + k));
                (void) advise_pairs(ahead.data(), (int64_t) ahead.size());
            }
            for (const Run& r : runs) {
                uint8_t* dst = (uint8_t*) host + (size_t) r.dst_off;
                uint64_t done_bytes = 0;
                if (direct_fd >= 0) {
                    const bool aligned = r.file_pos % kDirectAlign == 0 && ((uintptr_t) dst) % kDirectAlign == 0 &&
                                         r.len % kDirectAlign == 0;
                    if (aligned) {   // straight into the arena
                        while (done_bytes < r.len) {
                            const uint64_t n = std::min<uint64_t>(kDirectChunk, r.len - done_bytes);
                            const int64_t got = basalt::platform::pread(direct_fd, dst + done_bytes, (size_t) n,
                                                                        r.file_pos + done_bytes);
                            if (got <= 0 || (uint64_t) got % kDirectAlign != 0) break;   // the rest takes the mapped copy
                            done_bytes += (uint64_t) got;
                        }
                    } else {   // block-aligned reads into a staging buffer, then the wanted bytes are copied out
                        if (stage == nullptr)
                            stage = (uint8_t*) basalt::platform::alloc_aligned(kDirectChunk + 2 * kDirectAlign,
                                                                              kDirectAlign);
                        while (stage != nullptr && done_bytes < r.len) {
                            const uint64_t pos = r.file_pos + done_bytes;
                            const uint64_t n = std::min<uint64_t>(kDirectChunk, r.len - done_bytes);
                            const uint64_t a0 = pos - pos % kDirectAlign;
                            const uint64_t a1 = (pos + n + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
                            const int64_t got = basalt::platform::pread(direct_fd, stage, (size_t) (a1 - a0), a0);
                            if (got <= 0 || (uint64_t) got < pos + n - a0) break;
                            std::memcpy(dst + done_bytes, stage + (pos - a0), (size_t) n);
                            done_bytes += n;
                        }
                    }
                    direct_bytes.fetch_add(done_bytes);
                }
                // whatever the unbuffered reads did not place, per expert from the mapped file
                for (int64_t k = done_bytes / blob_bytes; k < r.count; ++k) {
                    if (!copy_from_files(layer, r.first + k, dst + (size_t) k * (size_t) blob_bytes)) {
                        fail("FileExpertSource: invalid blob bounds while building the cache complement");
                        return;
                    }
                    mapped_used = true;
                }
                for (int64_t k = 0; k < r.count; ++k) {
                    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) (r.first + k);
                    copied.fetch_add(blob_bytes);
                    if (!on_gpu[index]) progress.add(blob_bytes);
                }
            }
            if (mapped_used) {   // the layer's pages are not kept in the file cache once copied
                const uint64_t layer_offset = layer_offsets_[(size_t) layer];
                const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
                if (!basalt::platform::drop_mapped_view(base_, mapped_bytes_, layer_offset, layer_bytes)) {
                    fail("FileExpertSource: could not release mapped expert layer " + std::to_string(layer));
                    return;
                }
                basalt::platform::advise_dontneed(fd_, file_off_ + layer_offset, layer_bytes);
            }
            const int64_t done = layers_done.fetch_add(1) + 1;
            if (done % 8 == 0 || done == n_layers_)
                std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB, %.0f s)\n",
                             (long long) done, (long long) n_layers_, (double) copied.load() / 1073741824.0,
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count());
        }
    };
    {
        const int threads = (int) std::max<int64_t>(1, std::min<int64_t>(6, n_layers_));
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    std::fflush(stderr);
    if (direct_fd >= 0) basalt::platform::close_fd(direct_fd);
    const double read_done_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count();
    if (register_after && !failed.load()) {
        constexpr size_t kHuge = 2ull << 20;
        const size_t padded = ((size_t) bytes + kHuge - 1) / kHuge * kHuge;
        const cudaError_t reg = cudaHostRegister(arena, padded, cudaHostRegisterMapped | cudaHostRegisterPortable);
        void* alias = nullptr;
        if (reg == cudaSuccess && cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
            device = (const uint8_t*) alias;
            partial_pin = bytes;   // released through the registered-prefix path
            lock_off = bytes;
            note = "page-locked and mapped";
        } else {
            if (reg == cudaSuccess) (void) cudaHostUnregister(arena);
            note = std::string("page-locking refused (") + cudaGetErrorString(reg) + ")";
            (void) cudaGetLastError();
            const basalt::platform::LockResult lr = basalt::platform::lock_resident(arena, bytes);
            locked = lr.locked_bytes;
            note += "; " + lr.note;
        }
    }
    if (failed.load()) {
        err = fail_msg;
        release();
        return false;
    }
    progress.finish();
    {
        const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count();
        std::fprintf(stderr, "FileExpertSource: cache complement: allocation %.1f s, read %.1f s (%.2f GB/s, %.1f%% unbuffered), registration %.1f s\n",
                     alloc_s, read_done_s, read_done_s > 0 ? (double) copied.load() / 1e9 / read_done_s : 0.0,
                     copied.load() > 0 ? 100.0 * (double) direct_bytes.load() / (double) copied.load() : 0.0,
                     read_s - read_done_s);
    }

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
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
            std::fprintf(stderr, "FileExpertSource: %lld free places for the adaptive swaps (%zu blob sizes, %.0f MiB)\n",
                         (long long) spare_free_.size(), spare_sizes.size(), (double) (at - spare_at) / 1048576.0);
    }
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB in %.0f s%s%s\n",
                 complement_pinned_ ? "mapped pinned" : pin ? "locked resident" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - pin_t0).count(),
                 note.empty() ? "" : "; ", note.c_str());
    if (lend)
        std::fprintf(stderr, "FileExpertSource: %lld of the prompt path's %lld lendable slots keep their experts in RAM "
                             "too%s\n", (long long) complement_lent_slots_, (long long) (n_slots - lend_from_slot),
                     complement_lent_slots_ < n_slots - lend_from_slot
                         ? " (the others are read from the file when lent: not enough RAM for them)" : "");
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu experts held only by another GPU are not kept here (read from the "
                             "file when needed)\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

const uint8_t* FileExpertSource::host_at(uint64_t off, uint64_t bytes) const {
    if (off < complement_bytes_)
        return complement_host_ != nullptr && bytes <= complement_bytes_ - off ? complement_host_ + (size_t) off : nullptr;
    const uint64_t o2 = off - complement_bytes_;
    return top_arena_ != nullptr && o2 <= top_bytes_ && bytes <= top_bytes_ - o2 ? (const uint8_t*) top_arena_ + (size_t) o2
                                                                                 : nullptr;
}

uint64_t FileExpertSource::top_up(const std::vector<std::pair<int32_t, int32_t>>& rank,
                                  const std::function<bool(int32_t, int32_t)>& skip, bool pin, int64_t& count,
                                  uint64_t later) {
    count = 0;
    if (!complement_ready_ || top_arena_ != nullptr || !override_.empty() || !staged_.empty()) return 0;
    bool known = false;
    const uint64_t budget = basalt::platform::ram_budget(&known);
    if (!known) return 0;
    const uint64_t guard = (256ull << 20) + later;   // the engine's own allocations after this reading
    const uint64_t room = budget > guard ? budget - guard : 0;
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
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    auto worker = [&]() {
        for (size_t k; (k = next.fetch_add(1)) < pick.size() && !failed.load();)
            if (!copy_from_files(pick[k].first, pick[k].second, (uint8_t*) arena + (size_t) at[k])) failed.store(true);
    };
    {
        const size_t threads = std::min<size_t>(6, pick.size());
        std::vector<std::thread> pool;
        for (size_t i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    if (failed.load()) {
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
    return complement_offsets_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] >= complement_bytes_;
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
    for (size_t i = spare_free_.size(); i-- > 0;)
        if (spare_free_[i].second == need) { pick = i; break; }
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
    return true;
}

bool FileExpertSource::spare_release(int64_t layer, int64_t in) {
    if (!complement_ready_ || layer < 0 || in < 0 || layer >= n_layers_ || in >= n_expert_) return false;
    const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    if (i >= complement_offsets_.size() || complement_offsets_[i] == kNoComplement) return false;
    spare_free_.emplace_back(complement_offsets_[i], layer_blob_bytes_[(size_t) layer]);
    complement_offsets_[i] = kNoComplement;
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
    if (complement_offsets_[index] >= complement_bytes_) return top_pinned_;
    if (!complement_pinned_ || complement_host_ == nullptr) return false;
    // a partial pin (CS-T): only the registered prefix
    return !complement_partial_ ||
           complement_offsets_[index] + layer_blob_bytes_[(size_t) layer] <= complement_pin_limit_;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert)) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t off = complement_offsets_[index];
    if (off < complement_bytes_ && complement_device_ == nullptr) return nullptr;
    return off >= complement_bytes_ ? top_device_ + (size_t) (off - complement_bytes_) : complement_device_ + (size_t) off;
}

bool FileExpertSource::pcie_layer(int64_t layer) const {
    if (complement_ready_ && complement_pinned_ && complement_device_ != nullptr)
        return layer >= 0 && layer < n_layers_;
    return device_alias(layer, 0) != nullptr;
}

// ================================ THE ADAPTER ================================

// Read-ahead for what the CPU tier may read from the file: every routed id that no GPU holds.
static void advise_routed(ExpertDispatch& d, const int32_t* ids, int64_t n) {
    int32_t want[128];
    int64_t m = 0;
    for (int64_t i = 0; i < n && m < 128; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= d.n_expert) continue;
        if (d.host_res != nullptr && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) continue;
        if (d.peer != nullptr && d.peer->has(d.layers, e)) continue;
        bool seen = false;
        for (int64_t j = 0; j < m && !seen; ++j) seen = want[j] == e;
        if (!seen) want[m++] = e;
    }
    if (m > 0) d.src->advise_cpu(d.layers, want, m);
}

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace basalt::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    advise_routed(d, ids, k);
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  Deciding here instead would make `Launch` compute the PREVIOUS
    // layer's experts into this layer's rows.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // rows of every expert across all threads (bitwise the same as `run`).
    d.pool->run_split(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`)
// are fixed arrays of this many entries: MAXT tokens of the model's 10 routed experts must fit, and a larger k is
// refused at run time rather than written past them.
constexpr int64_t kMaxWindowEntries = 128;
static_assert(basalt::kernels::cpu::MAXT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace basalt::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (k < 1 || n_tok * k > kMaxWindowEntries) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    const auto c0 = std::chrono::steady_clock::now();
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- the GPU's share, decided and published FIRST so the GPU starts while the CPU works.  Distinct experts in
    // routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    if (n <= kMaxWindowEntries) advise_routed(d, ids, n);
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe
    // the device planned the layer; only its CPU entries are the pool's
    const bool ext = d.plan != nullptr && d.plan->ext_kind != nullptr && n <= kMaxWindowEntries;
    if (ext) {
        for (int64_t i = 0; i < n; ++i) kind[i] = d.plan->ext_kind[i];
    } else if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0 &&
                    !(d.peer != nullptr && d.peer->has(d.layers, e))) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->pcie_layer(d.layers);
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else if (d.peer != nullptr && d.peer->has(d.layers, e)) {
                    kd = 2;                        // multi-GPU: the second GPU computes it
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->pinned(d.layers, e) ? d.src->blob(d.layers, e) : nullptr;
                        if (src != nullptr) {
                            kd = 1;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: the arena alias, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = (unsigned long long) d.src->device_alias(d.layers, ids[i0]);
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (P.publish) P.publish(P.ctx);
        if (P.staged) P.staged(P.ctx);
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0
                    : (e >= 0 && e < d.n_expert && d.peer != nullptr && d.peer->has(d.layers, e)) ? 2 : -1;
        }
    }
    if (d.peer != nullptr && !ext) {         // multi-GPU: start the second GPU's share before the CPU's own work
        std::string perr;
        if (!d.peer->launch(d.layers, x_f, ids, n_tok, k, kind, perr, out)) {
            std::fprintf(stderr, "basalt: %s (layer %lld)\n", perr.c_str(), (long long) d.layers);
            d.failed = true;
            d.fail = "the peer GPU's experts could not be launched";
            d.fail_layer = d.layers;
            return;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    // the GPU's copy_rows_from_mapped zeroes the rows it computed itself
    const bool gpu_zeroes_hits = (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap);
    bool any_cpu = false;
    for (int64_t i = 0; i < n; ++i)
        if (kind[i] < 0) { any_cpu = true; break; }
    const auto c1 = std::chrono::steady_clock::now();
    if (any_cpu) {
        // a token whose experts all run on a GPU (CUDA0, the peer or a helper) needs no CPU activation
        for (int64_t t = 0; t < n_tok; ++t) {
            if (std::all_of(kind + t * k, kind + (t + 1) * k, [](int32_t v) { return v >= 0; })) continue;
            if (native && basalt::kernels::cpu::q2_native_kernels(lay.fmt[(size_t) d.layers].gu_type))   // a native Q2_0 pack: the Q2_0 kernels' activations
                act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
            else if (native)
                native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
            else
                act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
        }
    }
    const auto c2 = std::chrono::steady_clock::now();
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote/peer result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                else ++d.offload_entries;   // PCIe or another GPU
                if (kind[i] == 2 && d.peer != nullptr) ++d.peer_entries;
                // multi-GPU: a direct peer launch is writing this row right now - zeroing it would race it;
                // dec_batch: copy_rows_from_mapped_kernel already zeroes kind 0/1 rows on the GPU;
                // ext: the GPU reads only the CPU rows
                if (!(ext || (kind[i] == 2 && d.peer != nullptr && d.peer->launched_direct()) ||
                      (gpu_zeroes_hits && (kind[i] == 0 || kind[i] == 1))))
                    std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    if (njobs > 0) {
        if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
        else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    if (d.peer != nullptr && !ext) {         // multi-GPU: the second GPU's rows, into the same mapped rows
        std::string perr;
        if (!d.peer->finish(out, perr)) {
            std::fprintf(stderr, "basalt: %s (layer %lld)\n", perr.c_str(), (long long) d.layers);
            d.failed = true;
            d.fail = "the peer GPU's experts failed";
            d.fail_layer = d.layers;
            return;
        }
    }
    const auto c4 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, which differs by up to 2^-11 relative per chunk.  `quantize_q8_0_scaled` adopts
        // the CPU's rule and scale, and the kernel takes the fp32 array.  Falling back to fp16 `d` would
        // reintroduce the divergence, so the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales";
            return;
        }
        basalt::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, basalt::kernels::cpu::H,
                                              cs);
        basalt::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) {
            cudaEventRecord((cudaEvent_t) d.hit_done, cs);
            // ONE driver entry here: if the work was waiting for the host to enter the driver, this is what lets it
            // start while the pool runs.
            (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        }
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    basalt::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA ================================

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
                             std::string& err, uint64_t max_pinned_bytes,
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
        // the arena is locked resident: it must leave the margin free, or the machine swaps
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
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds, max_pinned_bytes,
                                     shared_arena_file, arena_hash);
    if (!a->valid()) {
        const std::string why = a->note;
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" +
              std::to_string(want + (uint64_t) blob) + " B)" +
              (why.empty() ? std::string{} : ": " + why);
        return false;
    }
    // unbuffered when the drive is read anyway and the file cache could not keep the experts for the next start
    // either; otherwise the buffered readers, which a warm restart serves from the cache
    std::vector<std::string> files{path};
    std::string why;
    const bool unbuffered = experts_unbuffered(files, want + (uint64_t) blob, why);
    const int readers = unbuffered ? std::max(threads, 16) : threads;   // 16 keep a PCIe 5 drive's queue full
    LoadStats st;
    if (unbuffered) st = load_experts_direct(path, a->data(), loff, lbytes, readers, /*chunk=*/8u << 20,
                                             files_in.experts.offset);
    if (!unbuffered || (!st.ok && st.error.empty()))   // unaligned ranges: the buffered reader
        st = load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20,
                                 files_in.experts.offset);
    std::fprintf(stderr, "basalt generate: expert arena read %s (%s)\n", unbuffered ? "unbuffered" : "through the file cache",
                 why.c_str());
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
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
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
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
