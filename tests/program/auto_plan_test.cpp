// tests/program/auto_plan_test.cpp - the start-up picks: RAM sizes, CPU shapes, peer or not, given or left out.
#include "basalt/program/auto_plan.hpp"

#include <cstdio>
#include <string>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)
#define CHECK_STR(got, want)                                                                         \
    do {                                                                                             \
        const std::string g_ = (got);                                                                \
        if (g_ != (want)) { std::printf("FAIL %s:%d: got \"%s\" want \"%s\"\n", __FILE__, __LINE__, g_.c_str(), want); ++g_fail; } \
    } while (0)

static constexpr uint64_t G = 1ull << 30;

// A complement of 60 GiB, 40 GiB with the peer's experts held only on the peer, and a 6 GiB floor.
static AutoInputs base(uint64_t ram_gib, bool peer) {
    AutoInputs in;
    in.ram_total = ram_gib * G;
    in.ram_available = ram_gib * G - 4 * G;
    in.vram_free = peer ? std::vector<uint64_t>{30 * G, 14 * G} : std::vector<uint64_t>{30 * G};
    in.peer = peer;
    in.serve = true;
    in.worker_cores = 15;
    in.complement_full = 60 * G;
    in.complement_peer_excl = 40 * G;
    in.floor_bytes = 6 * G;
    return in;
}

int main() {
    // Empty config: all four keys are auto, per RAM size.
    {
        AutoPlan p = choose_auto(base(16, true));   // 12 avail - 6 floor = 6 GiB
        CHECK(!p.peer_exclusive);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive off, resident 6.0 GB, pool-affinity auto, pool-workers 15");
        p = choose_auto(base(32, true));            // 28 - 6 = 22
        CHECK(!p.peer_exclusive);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive off, resident 22.0 GB, pool-affinity auto, pool-workers 15");
        p = choose_auto(base(64, true));            // 60 - 6 = 54: below the full 60, above the exclusive 40
        CHECK(p.peer_exclusive);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive on, resident 40.0 GB, pool-affinity auto, pool-workers 15");
        p = choose_auto(base(128, true));           // everything fits
        CHECK(!p.peer_exclusive);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive off, resident 60.0 GB, pool-affinity auto, pool-workers 15");
    }
    // No peer card, no serving mode, or exclusive that still leaves reads: stays off.
    {
        AutoPlan p = choose_auto(base(64, false));
        CHECK(!p.peer_exclusive);
        CHECK(p.resident_gb == 54.0);
        AutoInputs in = base(64, true);
        in.serve = false;
        CHECK(!choose_auto(in).peer_exclusive);
        in = base(64, true);
        in.complement_peer_excl = 55 * G;
        CHECK(!choose_auto(in).peer_exclusive);
    }
    // CPU shapes.
    {
        AutoInputs in = base(128, false);
        in.hybrid = true; in.p_cores = 8; in.e_cores = 16; in.worker_cores = 23;
        CHECK(choose_auto(in).pool_workers == 15);   // auto affinity: P-cores - 1, plus half the E-cores
        in.given.pool_affinity = true; in.given.pool_affinity_value = AutoAffinity::All;
        CHECK(choose_auto(in).pool_workers == 23);
        in.given.pool_affinity_value = AutoAffinity::PCores;
        CHECK(choose_auto(in).pool_workers == 7);
        in = base(128, false);
        in.p_cores = 8; in.e_cores = 0; in.worker_cores = 7;
        CHECK(choose_auto(in).pool_workers == 7);   // plain CPU: every worker core
        in.cpu_quota = 4.5;
        CHECK(choose_auto(in).pool_workers == 3);   // quota fit
        in.cpu_quota = 1.0;
        CHECK(choose_auto(in).pool_workers == 1);
        in.cpu_quota = 16.0;
        CHECK(choose_auto(in).pool_workers == 7);   // a quota above the cores does not bind
        in = base(128, false);
        in.hybrid = true; in.p_cores = 8; in.e_cores = 16; in.worker_cores = 23;
        in.cpu_quota = 5.0;
        CHECK(choose_auto(in).pool_workers == 4);   // hybrid: the quota fit caps the P-core count
        in.given.pool_affinity = true; in.given.pool_affinity_value = AutoAffinity::All;
        CHECK(choose_auto(in).pool_workers == 4);   // and the all-cores count
        CHECK(auto_pool_workers(in, AutoAffinity::PCores) == 4);
        in.cpu_quota = 0;
        CHECK(auto_pool_workers(in, AutoAffinity::All) == 23 && auto_pool_workers(in, AutoAffinity::Auto) == 15);
        CHECK(auto_pool_workers(in, AutoAffinity::PCores) == 7);
        in.p_cores = 6; in.e_cores = 4; in.worker_cores = 9;
        CHECK(auto_pool_workers(in, AutoAffinity::Auto) == 7);   // 5 P-cores and 2 of the 4 E-cores
    }
    // Each setting given: it is kept, marked Given, and left out of the line.
    {
        AutoInputs in = base(64, true);
        in.given.peer_exclusive = true; in.given.peer_exclusive_value = false;
        AutoPlan p = choose_auto(in);
        CHECK(!p.peer_exclusive && p.peer_exclusive_src == AutoSource::Given);
        CHECK(p.resident_gb == 54.0);
        CHECK_STR(auto_flags_line(p), "auto flags: resident 54.0 GB, pool-affinity auto, pool-workers 15");

        in = base(64, true);
        in.given.resident = true; in.given.resident_gb = 12.5;
        p = choose_auto(in);
        CHECK(p.resident_gb == 12.5 && p.resident_src == AutoSource::Given && p.peer_exclusive);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive on, pool-affinity auto, pool-workers 15");

        in = base(64, true);
        in.given.pool_affinity = true; in.given.pool_affinity_value = AutoAffinity::PCores;
        p = choose_auto(in);
        CHECK(p.pool_affinity == AutoAffinity::PCores && p.pool_affinity_src == AutoSource::Given);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive on, resident 40.0 GB, pool-workers 15");

        in = base(64, true);
        in.given.pool_workers = true; in.given.pool_workers_value = 8;
        p = choose_auto(in);
        CHECK(p.pool_workers == 8 && p.pool_workers_src == AutoSource::Given);
        CHECK_STR(auto_flags_line(p), "auto flags: peer-exclusive on, resident 40.0 GB, pool-affinity auto");

        // Everything given: no line.
        in = base(64, true);
        in.given = {true, true, true, true, true, 20.0, AutoAffinity::All, 12};
        p = choose_auto(in);
        CHECK_STR(auto_flags_line(p), "");
        CHECK(p.peer_exclusive && p.resident_gb == 20.0 && p.pool_affinity == AutoAffinity::All && p.pool_workers == 12);
    }
    // Floor above the available RAM: nothing resident, no underflow.
    {
        AutoInputs in = base(16, false);
        in.floor_bytes = 100 * G;
        CHECK(choose_auto(in).resident_gb == 0.0);
    }
    if (g_fail) { std::printf("auto_plan_test: %d failures\n", g_fail); return 1; }
    std::printf("auto_plan_test: ok\n");
    return 0;
}
