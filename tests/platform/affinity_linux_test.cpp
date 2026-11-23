#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "basalt/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "affinity_linux.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

namespace cpu = basalt::kernels::cpu;

class DynamicCpuSet {
public:
    explicit DynamicCpuSet(size_t count)
        : count_(count), bytes_(CPU_ALLOC_SIZE(count)), set_(CPU_ALLOC(count)) {
        if (set_) CPU_ZERO_S(bytes_, set_);
    }

    ~DynamicCpuSet() {
        if (set_) CPU_FREE(set_);
    }

    DynamicCpuSet(const DynamicCpuSet&) = delete;
    DynamicCpuSet& operator=(const DynamicCpuSet&) = delete;

    cpu_set_t* get() { return set_; }
    const cpu_set_t* get() const { return set_; }
    size_t bytes() const { return bytes_; }
    size_t count() const { return count_; }
    explicit operator bool() const { return set_ != nullptr; }

private:
    size_t count_;
    size_t bytes_;
    cpu_set_t* set_;
};

static size_t initial_cpu_count() {
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    return (std::max)(size_t(128), configured > 0 ? (size_t) configured : size_t(0));
}

// Query with CPU_ALLOC rather than cpu_set_t: this must observe CPUs beyond CPU_SETSIZE too.
static bool get_affinity(std::vector<int>& ids) {
    size_t count = initial_cpu_count();
    for (int attempt = 0; attempt < 8; ++attempt) {
        DynamicCpuSet set(count);
        if (!set) return false;
        const int error = pthread_getaffinity_np(pthread_self(), set.bytes(), set.get());
        if (error == 0) {
            ids.clear();
            for (size_t i = 0; i < set.count(); ++i)
                if (CPU_ISSET_S(i, set.bytes(), set.get())) ids.push_back((int) i);
            return !ids.empty();
        }
        if (error != EINVAL || count > (size_t(1) << 20)) {
            std::fprintf(stderr, "pthread_getaffinity_np failed: %s\n", std::strerror(error));
            return false;
        }
        count *= 2;
    }
    std::fprintf(stderr, "pthread_getaffinity_np still needs a larger CPU set\n");
    return false;
}

static bool set_affinity(const std::vector<int>& ids) {
    if (ids.empty()) return false;
    const int highest = *std::max_element(ids.begin(), ids.end());
    if (highest < 0) return false;
    const size_t count = (std::max)(initial_cpu_count(), (size_t) highest + 1);
    DynamicCpuSet set(count);
    if (!set) return false;
    for (int id : ids) {
        if (id < 0 || (size_t) id >= count) return false;
        CPU_SET_S((size_t) id, set.bytes(), set.get());
    }
    const int error = pthread_setaffinity_np(pthread_self(), set.bytes(), set.get());
    if (error != 0) {
        std::fprintf(stderr, "pthread_setaffinity_np failed: %s\n", std::strerror(error));
        return false;
    }
    return true;
}

class OriginalAffinity {
public:
    bool capture() { return get_affinity(ids_); }
    const std::vector<int>& ids() const { return ids_; }
    bool restore() { return ids_.empty() || set_affinity(ids_); }
    ~OriginalAffinity() { if (!ids_.empty()) set_affinity(ids_); }

private:
    std::vector<int> ids_;
};

static bool expect_affinity(const std::vector<int>& expected, const char* label) {
    std::vector<int> actual;
    if (!get_affinity(actual)) {
        std::fprintf(stderr, "affinity test: %s: cannot query current affinity\n", label);
        return false;
    }
    if (actual != expected) {
        std::fprintf(stderr, "affinity test: %s: expected %zu CPUs, observed %zu\n",
                     label, expected.size(), actual.size());
        return false;
    }
    return true;
}

static bool topology_respects_current_mask() {
    std::vector<int> allowed;
    if (!get_affinity(allowed)) return false;
    const auto topology = cpu::detect_cpu_topology(false);
    if (topology.worker_cores.empty()) return false;
    for (int core : topology.worker_cores) {
        if (!std::binary_search(allowed.begin(), allowed.end(), core)) {
            std::fprintf(stderr, "affinity test: topology selected disallowed CPU %d\n", core);
            return false;
        }
    }
    return true;
}

// A CPU list the kernel publishes ("0-15", "0-3,8-11", "5"): the processors it names that this run may run on.  Empty
// when the file is missing, unreadable or names none of them.
static std::vector<int> listed_cpus(const char* path, const std::vector<int>& allowed) {
    std::vector<int> got;
    std::FILE* f = std::fopen(path, "r");
    if (!f) return got;
    char buf[1024];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = 0;
    char* s = buf;
    while (*s) {
        char* end = nullptr;
        const long lo = std::strtol(s, &end, 10);
        if (end == s) break;
        long hi = lo;
        s = end;
        if (*s == '-') {
            hi = std::strtol(s + 1, &end, 10);
            if (end == s + 1) break;
            s = end;
        }
        for (int id : allowed)
            if (id >= lo && id <= hi && std::find(got.begin(), got.end(), id) == got.end()) got.push_back(id);
        if (*s == ',') ++s; else break;
    }
    return got;
}

// How many PHYSICAL cores such a list covers, counted by (physical_package_id, core_id): a second reading of sysfs, not
// the topology's own sibling bookkeeping.  A processor whose core ids cannot be read counts as a core of its own.
static size_t listed_cores(const std::vector<int>& cpus) {
    auto read = [](const std::string& path) -> long {
        long v = -1;
        if (std::FILE* f = std::fopen(path.c_str(), "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    std::vector<std::pair<long, long>> seen;
    for (int id : cpus) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(id) + "/topology/";
        long pkg = read(base + "physical_package_id"), core = read(base + "core_id");
        if (pkg < 0 || core < 0) pkg = core = -(long) id - 1;   // negative ids are never real ones, and unique per CPU
        const std::pair<long, long> key{pkg, core};
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) seen.push_back(key);
    }
    return seen.size();
}

// The fast-class count against the OS's own numbers.  Where the kernel states its classes outright - the two hybrid
// PMUs' CPU lists, each naming a processor this run may run on - those lists are what the topology is held to, and the
// capacities are not consulted: a kernel that publishes `cpu_capacity` at all can report the SAME number for every
// processor, which separates nothing, and reading that uniformity as one class is the mistake that left a hybrid
// processor with one class.  With no lists the capacities are the only statement this kernel makes: one capacity per
// PHYSICAL core (a sibling shares its core's), the fast class being the cores above the midpoint between the extremes.
// Reading sysfs here rather than trusting the topology is the point either way - a second reading, not the same
// bookkeeping twice.  A run masked to one class - `taskset` over one list's processors - names no processor of the other
// class it may run on, and so keeps one class; that is checked too, by masking to the performance CPUs and back.
static bool topology_core_classes() {
    auto read = [](const std::string& path) -> long {
        long v = -1;
        if (std::FILE* f = std::fopen(path.c_str(), "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    std::vector<int> allowed;
    if (!get_affinity(allowed)) return false;

    const std::vector<int> core_list = listed_cpus("/sys/devices/cpu_core/cpus", allowed);
    const std::vector<int> atom_list = listed_cpus("/sys/devices/cpu_atom/cpus", allowed);
    if (!core_list.empty() && !atom_list.empty()) {
        const size_t list_p = listed_cores(core_list), list_e = listed_cores(atom_list);
        const auto topology = cpu::detect_cpu_topology(false, cpu::PoolAffinity::Auto);
        if (!topology.is_hybrid || (size_t) topology.p_cores != list_p ||
            (size_t) topology.e_cores + (size_t) topology.p_cores != list_p + list_e) {
            std::fprintf(stderr,
                         "affinity test: the kernel's hybrid CPU lists name %zu performance and %zu efficient physical "
                         "cores this run may use, and the topology says %d fast / %d slow%s\n",
                         list_p, list_e, topology.p_cores, topology.e_cores,
                         topology.is_hybrid ? "" : " (and no core classes at all)");
            return false;
        }
        std::printf("core classes: %d fast / %d slow, matching the kernel's hybrid CPU lists (%zu / %zu processors)\n",
                    topology.p_cores, topology.e_cores, core_list.size(), atom_list.size());

        // The same run with its mask narrowed to the performance list's processors - what `taskset` on the P-core CPUs
        // does - may run on no efficient processor at all, so it is one class of cores: the fast count is that class's
        // core count, there are no slow cores, and neither the default list nor a `p-cores` list reaches into the
        // efficient CPUs it cannot run on.  The mask is restored before returning either way.
        const auto named_p = [&](int id) {
            return std::find(core_list.begin(), core_list.end(), id) != core_list.end();
        };
        if (!set_affinity(core_list) || !expect_affinity(core_list, "establish performance-only mask")) return false;
        const auto one_class = cpu::detect_cpu_topology(false, cpu::PoolAffinity::Auto);
        const auto p_only = cpu::detect_cpu_topology(true, cpu::PoolAffinity::PCores);
        const bool sized = !one_class.is_hybrid && one_class.e_cores == 0 &&
                           (size_t) one_class.p_cores == list_p && (size_t) one_class.p_threads == core_list.size() &&
                           one_class.worker_cores.size() == list_p &&
                           p_only.worker_cores.size() + 1 == list_p && named_p(p_only.host_core);
        bool clean = true;
        const auto workers_named = [&](const cpu::CpuTopology& t) {
            for (int id : t.worker_cores) {
                if (!named_p(id)) {
                    std::fprintf(stderr, "affinity test: a performance-only mask gave worker CPU %d, which the kernel's "
                                         "performance-core list does not name\n", id);
                    return false;
                }
            }
            return true;
        };
        clean = workers_named(one_class) && workers_named(p_only);
        const bool restored = set_affinity(allowed) && expect_affinity(allowed, "restore the full mask");
        if (!sized || !clean) {
            std::fprintf(stderr,
                         "affinity test: masked to the %zu performance-core processors (%zu physical cores), the topology "
                         "says %d fast / %d slow with %zu worker(s), and `p-cores` with %zu\n",
                         core_list.size(), list_p, one_class.p_cores, one_class.e_cores, one_class.worker_cores.size(),
                         p_only.worker_cores.size());
            return false;
        }
        std::printf("core classes under a performance-only mask: one class of %d cores, %zu worker(s); `p-cores` sizes %zu\n",
                    one_class.p_cores, one_class.worker_cores.size(), p_only.worker_cores.size());
        return restored;
    }

    std::vector<std::pair<long, long>> seen;
    std::vector<long> caps;
    for (int id : allowed) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(id);
        const long capacity = read(base + "/cpu_capacity");
        if (capacity <= 0) continue;   // this CPU publishes no capacity: nothing to compare against
        const long pkg = read(base + "/topology/physical_package_id");
        const long core = read(base + "/topology/core_id");
        if (pkg >= 0 && core >= 0) {
            const std::pair<long, long> key{pkg, core};
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
            seen.push_back(key);
        }
        caps.push_back(capacity);
    }
    if (caps.size() < 2) {
        std::puts("SKIP core classes: fewer than two cores publish a capacity");
        return true;
    }
    const long lo = *std::min_element(caps.begin(), caps.end());
    const long hi = *std::max_element(caps.begin(), caps.end());
    const long mid = (lo + hi) / 2;
    const size_t fast = (size_t) std::count_if(caps.begin(), caps.end(), [mid](long c) { return c > mid; });
    const auto topology = cpu::detect_cpu_topology(false, cpu::PoolAffinity::Auto);
    if (!topology.is_hybrid) {
        std::printf("core classes: one class (%zu cores, capacity %ld..%ld)\n", caps.size(), lo, hi);
        return true;
    }
    if ((size_t) topology.p_cores != fast || (size_t) topology.e_cores + (size_t) topology.p_cores != caps.size()) {
        std::fprintf(stderr,
                     "affinity test: %zu cores above the capacity midpoint %ld (%ld..%ld) but the topology counted %d "
                     "fast / %d slow of %zu\n",
                     fast, mid, lo, hi, topology.p_cores, topology.e_cores, caps.size());
        return false;
    }
    std::printf("core classes: %d fast / %d slow, matching the capacities (%ld..%ld, midpoint %ld)\n", topology.p_cores,
                topology.e_cores, lo, hi, mid);
    return true;
}

static bool topology_under_masks(const std::vector<int>& allowed, bool require_high) {
    if (!topology_respects_current_mask()) return false;
    if (allowed.size() >= 2) {
        const std::vector<int> sparse{allowed.front(), allowed.back()};
        if (!set_affinity(sparse) || !expect_affinity(sparse, "establish sparse mask") ||
            !topology_respects_current_mask()) return false;
        if (!set_affinity(allowed) || !expect_affinity(allowed, "restore full mask")) return false;
    } else {
        std::puts("SKIP sparse mask: fewer than two allowed CPUs");
    }
    const auto it = std::find_if(allowed.begin(), allowed.end(), [](int id) { return id >= 64; });
    if (it == allowed.end()) {
        if (require_high) {
            std::fprintf(stderr, "affinity test: --require-high-cpu requested, but no allowed CPU ID is >= 64\n");
            return false;
        }
        std::puts("SKIP high-CPU mask: no allowed CPU ID is >= 64 (use --require-high-cpu to require it)");
        return true;
    }
    const std::vector<int> high_only{*it};
    if (!set_affinity(high_only) || !expect_affinity(high_only, "establish high-only mask") ||
        !topology_respects_current_mask()) return false;
    return set_affinity(allowed) && expect_affinity(allowed, "restore full mask after high-only");
}

int main(int argc, char** argv) {
    bool require_high = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-high-cpu") == 0) require_high = true;
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    OriginalAffinity original;
    if (!original.capture()) {
        std::fprintf(stderr, "affinity test: cannot capture original thread affinity\n");
        return 1;
    }
    const auto allowed = original.ids();

    // The pool takes its size from the caller as given (the sizing rules live in program/auto_plan).
    {
        cpu::ExpertPool three(3, cpu::PoolAffinity::All);
        cpu::ExpertPool none(0, cpu::PoolAffinity::Auto);
        if (three.workers() != 3 || none.workers() != 1) {
            std::fprintf(stderr, "affinity test: pool sizes %d / %d, expected 3 / 1\n", three.workers(), none.workers());
            return 1;
        }
    }

    const bool ok = topology_under_masks(allowed, require_high) && topology_core_classes();
    const bool restored = original.restore() && expect_affinity(allowed, "final original affinity cleanup");
    if (!ok || !restored) return 1;

    std::printf("affinity test: topology under full, sparse and high-CPU masks, and cleanup checks passed (%zu allowed CPUs)\n",
                allowed.size());
    return 0;
}

