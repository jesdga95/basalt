// src/program/auto_plan.cpp - see auto_plan.hpp.
#include "basalt/program/auto_plan.hpp"

#include <algorithm>
#include <cstdio>

#include "basalt/platform/memory.hpp"

namespace basalt::program {

namespace {
constexpr double kGiB = 1073741824.0;

bool fits(uint64_t complement, const AutoInputs& in) {
    return in.ram_available >= in.floor_bytes && complement <= in.ram_available - in.floor_bytes;
}

// The RAM that stays free beside a copy of up to `complement` bytes: the headroom when all of it fits, else the
// file cache's floor, which depends on how much is left to the file, which depends on the floor.  The iteration
// starts from everything on the file and only goes down, so the floor it ends on never undercounts.
uint64_t floor_for(uint64_t complement, const AutoInputs& in) {
    if (fits(complement, in)) return in.floor_bytes;
    uint64_t file = complement, floor = in.floor_bytes;
    for (int i = 0; i < 16; ++i) {
        floor = (std::max)(in.floor_bytes, basalt::platform::ram_floor(in.margin_bytes, file, complement, in.layer_bytes,
                                                                       in.prompt_file_bytes) + in.late_bytes);
        file = complement - (std::min)(complement, in.ram_available > floor ? in.ram_available - floor : 0);
    }
    return floor;
}
}  // namespace

const char* auto_affinity_name(AutoAffinity a) {
    return a == AutoAffinity::All ? "all" : a == AutoAffinity::Auto ? "auto" : "p-cores";
}

int auto_pool_workers(const AutoInputs& in, AutoAffinity affinity) {
    int n = in.worker_cores;
    if (in.hybrid && affinity != AutoAffinity::All) {
        // The P-cores but the host's. The auto affinity adds half the E-cores: they run the expert kernels slower,
        // but a layer's CPU share is split into many small tasks, so each one still shortens it.
        n = (std::max)(1, in.p_cores - 1);
        if (affinity == AutoAffinity::Auto) n += in.e_cores / 2;
    }
    // A CPU quota throttles the whole group once spent: the workers and the host thread together must fit it.
    if (in.cpu_quota > 0) {
        const int fit = (std::max)(1, (int) in.cpu_quota - 1);
        if (fit < n) n = fit;
    }
    return (std::max)(1, n);
}

AutoPlan choose_auto(const AutoInputs& in) {
    const AutoGiven& g = in.given;
    AutoPlan p;

    // Peer-exclusive only when the full complement does not fit and holding the peer's experts only on the peer
    // leaves nothing to read from the SSD.
    if (g.peer_exclusive) {
        p.peer_exclusive = g.peer_exclusive_value;
        p.peer_exclusive_src = AutoSource::Given;
    } else {
        p.peer_exclusive = in.peer && in.serve && !fits(in.complement_full, in) && fits(in.complement_peer_excl, in);
    }

    // Resident: what fits, capped at the complement and its optional copies.
    if (g.resident) {
        p.resident_gb = g.resident_gb;
        p.resident_bytes = (uint64_t) (g.resident_gb * kGiB);
        p.resident_src = AutoSource::Given;
    } else {
        const uint64_t complement = p.peer_exclusive ? in.complement_peer_excl : in.complement_full;
        const uint64_t extra = p.peer_exclusive ? in.extra_peer_excl : in.extra_full;
        const uint64_t floor = floor_for(complement, in);
        const uint64_t room = in.ram_available > floor ? in.ram_available - floor : 0;
        p.resident_bytes = (std::min)(room, complement + extra);
        p.resident_gb = (double) p.resident_bytes / kGiB;
    }

    if (g.pool_affinity) {
        p.pool_affinity = g.pool_affinity_value;
        p.pool_affinity_src = AutoSource::Given;
    } else {
        p.pool_affinity = AutoAffinity::Auto;
    }

    if (g.pool_workers && g.pool_workers_value > 0) {
        p.pool_workers = g.pool_workers_value;
        p.pool_workers_src = AutoSource::Given;
    } else {
        p.pool_workers = auto_pool_workers(in, p.pool_affinity);
    }
    return p;
}

std::string auto_flags_line(const AutoPlan& p) {
    std::string out;
    auto add = [&](const std::string& s) { out += out.empty() ? "auto flags: " : ", "; out += s; };
    char buf[64];
    if (p.peer_exclusive_src == AutoSource::Auto) add(std::string("peer-exclusive ") + (p.peer_exclusive ? "on" : "off"));
    if (p.resident_src == AutoSource::Auto) {
        std::snprintf(buf, sizeof buf, "resident %.1f GB", p.resident_gb);
        add(buf);
    }
    if (p.pool_affinity_src == AutoSource::Auto) add(std::string("pool-affinity ") + auto_affinity_name(p.pool_affinity));
    if (p.pool_workers_src == AutoSource::Auto) {
        std::snprintf(buf, sizeof buf, "pool-workers %d", p.pool_workers);
        add(buf);
    }
    return out;
}

}  // namespace basalt::program
