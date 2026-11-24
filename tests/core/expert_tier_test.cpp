// tests/core/expert_tier_test.cpp - the shared expert tier on a synthetic pack: its per-layer LRU slots follow a routing
// stream with an admission threshold exactly as a host model of the rule does, and every resident slot holds the pack's bytes.
#include "basalt/core/expert_tier.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace basalt;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                         \
        }                                                                     \
    } while (0)

namespace {

constexpr int kLayers = 3, kExperts = 48, kSlots = 8, kDim = 256, kQ4K = 12;

struct Lcg {
    uint64_t s;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t) (s >> 33);
    }
};

// a skewed draw: the low ids are hot
int draw(Lcg& r) {
    const uint32_t a = r.next() % kExperts, b = r.next() % kExperts;
    return (int) std::min(a, b);
}

// the rule on the host: per layer an LRU of kSlots experts seeded 0..kSlots-1 (0 the oldest), a miss counted per expert,
// the k-th use admitting it in place of the least recently used, the counts halving at every 128-token boundary
struct Model {
    int k;
    uint64_t token = 0, tick = 0;
    std::vector<std::vector<std::pair<int, uint64_t>>> res;   // [layer] (expert, stamp)
    std::vector<uint16_t> uses;
    int64_t hits = 0, misses = 0;
    explicit Model(int k_) : k(k_), res(kLayers), uses((size_t) kLayers * kExperts, 0) {
        for (auto& l : res)
            for (int e = 0; e < kSlots; ++e) l.push_back({e, (uint64_t) e});
        tick = (uint64_t) kLayers * kSlots;   // the seeding's own ticks: only their order matters
    }
    void begin() {
        if (token / 128 != (token + 1) / 128)
            for (uint16_t& u : uses) u >>= 1;
        ++token;
    }
    void look(int layer, int e) {
        const uint64_t now = ++tick;
        for (auto& [ex, st] : res[(size_t) layer])
            if (ex == e) {
                st = now;
                ++hits;
                return;
            }
        ++misses;
        uint16_t& u = uses[(size_t) layer * kExperts + e];
        if (++u < (uint16_t) k) return;
        u = 0;
        auto victim = res[(size_t) layer].begin();
        for (auto it = res[(size_t) layer].begin(); it != res[(size_t) layer].end(); ++it)
            if (it->second < victim->second) victim = it;
        *victim = {e, now};
    }
};

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("expert_tier_test: no GPU, skipped\n");
        return 77;
    }
    const kernels::NativeExpertLayout L = kernels::native_expert_layout(kQ4K, kQ4K, kDim, kDim);
    const uint64_t blob = L.bytes;
    const std::string path = std::string(BASALT_TEST_TMPDIR) + "/expert_tier_test.bin";
    std::vector<uint8_t> data((size_t) kLayers * kExperts * blob);
    {
        Lcg r{7};
        for (uint8_t& b : data) b = (uint8_t) r.next();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) data.data(), (std::streamsize) data.size());
    }
    const int fd = platform::open_fd_utf8(path);
    CHECK(fd >= 0);
    if (fd < 0) return 1;

    for (int k : {1, 2, 3}) {
        core::ExpertTierSpec spec;
        spec.area = "basalt test";
        spec.name = "test";
        spec.n_layers = kLayers;
        spec.n_expert = kExperts;
        spec.top_k = 1;
        spec.n_embd = kDim;
        spec.n_ff = kDim;
        for (int l = 0; l < kLayers; ++l) spec.layers.push_back({kQ4K, kQ4K, (uint64_t) l * kExperts * blob, blob, 0.f});
        spec.path = path;
        spec.fd = fd;
        spec.range = core::FileRange(path, 0, data.size());
        spec.diag_tier_ram = kExperts;   // every expert in RAM: the slots are copies the stream moves
        spec.diag_tier_vram = kSlots;
        spec.admit_uses = k;
        core::ExpertTier tier(spec);
        float* h_rows = nullptr;
        CHECK(cudaHostAlloc((void**) &h_rows, 128 * kDim * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
        core::LoadPlan plan;
        std::string err;
        core::ExpertTier::Reserve reserve;
        bool ok = tier.open(reserve, h_rows, plan, err);
        if (ok) {
            core::Loader loader;
            ok = loader.run(plan, err) && tier.attach(err);
        }
        if (!ok) std::fprintf(stderr, "k=%d: %s\n", k, err.c_str());
        CHECK(ok);
        if (!ok) return 1;
        CHECK(tier.slots() == (int64_t) kLayers * kSlots);

        // single-expert lookups: the tier's hits equal the rule's, token by token
        Model model(k);
        Lcg r{1234};
        const int kTokens = 3000;
        for (int t = 0; t < kTokens; ++t) {
            tier.begin_token(1);
            model.begin();
            for (int l = 0; l < kLayers; ++l) {
                const int32_t id = draw(r);
                core::ExpertTier::Lookup lk;
                CHECK(tier.lookup(l, &id, 1, lk, err));
                tier.promote(l, lk);
                tier.pump(-1, core::ExpertTier::kAllPieces);
                model.look(l, id);
            }
            if (tier.hits() != model.hits) {
                std::fprintf(stderr, "k=%d: token %d: tier %lld hits, rule %lld\n", k, t, (long long) tier.hits(), (long long) model.hits);
                CHECK(false);
                break;
            }
        }
        tier.settle();
        CHECK(tier.lookups() == (int64_t) kTokens * kLayers);
        std::printf("expert_tier_test: k=%d hit rate %.3f (%lld of %d)\n", k, (double) tier.hits() / (double) tier.lookups(),
                    (long long) tier.hits(), kTokens * kLayers);

        // every expert the tier says a slot holds is the pack's bytes
        int checked = 0;
        std::vector<uint8_t> back(blob);
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e) {
                const core::ExpertTier::Source src = tier.locate(l, e);
                if (!src.dev) continue;
                CHECK(cudaMemcpy(back.data(), src.dev, blob, cudaMemcpyDeviceToHost) == cudaSuccess);
                CHECK(std::memcmp(back.data(), data.data() + ((size_t) l * kExperts + (size_t) e) * blob, blob) == 0);
                ++checked;
            }
        CHECK(checked == kLayers * kSlots);

        // windows of several experts at once keep the slots full and the bytes right
        for (int t = 0; t < 500; ++t) {
            tier.begin_token(4);
            for (int l = 0; l < kLayers; ++l) {
                int32_t ids[12];
                int n = 0;
                while (n < 12) {
                    const int e = draw(r);
                    bool dup = false;
                    for (int j = 0; j < n; ++j) dup = dup || ids[j] == e;
                    if (!dup) ids[n++] = e;
                }
                core::ExpertTier::Lookup lk;
                CHECK(tier.lookup(l, ids, 12, lk, err));
                tier.fetch(l, lk);
                tier.await_fetch(nullptr);
                tier.promote(l, lk);
                tier.mark_gap(nullptr);
                tier.pump(l, 3);
            }
        }
        tier.settle();
        checked = 0;
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e) {
                const core::ExpertTier::Source src = tier.locate(l, e);
                if (!src.dev) continue;
                CHECK(cudaMemcpy(back.data(), src.dev, blob, cudaMemcpyDeviceToHost) == cudaSuccess);
                CHECK(std::memcmp(back.data(), data.data() + ((size_t) l * kExperts + (size_t) e) * blob, blob) == 0);
                ++checked;
            }
        CHECK(checked == kLayers * kSlots);
        cudaFreeHost(h_rows);
    }
    // a bounded lend (ExpertTierSpec::lend_segment) with copies weighted by layer: half the arena goes, every layer keeps a
    // prefix of its slots in proportion to its share and its most recently used experts, every slot it says it holds has
    // the pack's bytes, and reclaim maps the slots again
    {
        core::ExpertTierSpec spec;
        spec.area = "basalt test";
        spec.name = "test";
        spec.n_layers = kLayers;
        spec.n_expert = kExperts;
        spec.top_k = 1;
        spec.n_embd = kDim;
        spec.n_ff = kDim;
        for (int l = 0; l < kLayers; ++l) spec.layers.push_back({kQ4K, kQ4K, (uint64_t) l * kExperts * blob, blob, 0.f});
        spec.path = path;
        spec.fd = fd;
        spec.range = core::FileRange(path, 0, data.size());
        spec.diag_tier_ram = kExperts;
        spec.diag_tier_vram = 24;
        spec.admit_uses = 1;
        spec.layer_weights = {1.5f, 1.f, 0.5f};
        spec.lend_segment = 2ull << 20;
        core::ExpertTier tier(spec);
        float* h_rows = nullptr;
        CHECK(cudaHostAlloc((void**) &h_rows, 128 * kDim * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
        core::LoadPlan plan;
        std::string err;
        core::ExpertTier::Reserve reserve;
        bool ok = tier.open(reserve, h_rows, plan, err);
        if (ok) {
            core::Loader loader;
            ok = loader.run(plan, err) && tier.attach(err);
        }
        if (!ok) std::fprintf(stderr, "bounded lend: %s\n", err.c_str());
        CHECK(ok);
        CHECK(tier.partial_lend());
        auto held = [&](int l) {
            int n = 0;
            for (int e = 0; e < kExperts; ++e) n += tier.locate(l, e).dev != nullptr;
            return n;
        };
        auto bytes_right = [&] {
            std::vector<uint8_t> back(blob);
            bool good = true;
            for (int l = 0; l < kLayers; ++l)
                for (int e = 0; e < kExperts; ++e) {
                    const core::ExpertTier::Source src = tier.locate(l, e);
                    if (!src.dev) continue;
                    good = good && cudaMemcpy(back.data(), src.dev, blob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                           std::memcmp(back.data(), data.data() + ((size_t) l * kExperts + (size_t) e) * blob, blob) == 0;
                }
            return good;
        };
        Lcg r{4321};
        int32_t last[kLayers][4] = {};
        auto run = [&](int tokens) {
            for (int t = 0; t < tokens; ++t) {
                tier.begin_token(1);
                for (int l = 0; l < kLayers; ++l) {
                    const int32_t id = (int32_t) (r.next() % kExperts);   // uniform: the slots churn
                    last[l][t % 4] = id;
                    core::ExpertTier::Lookup lk;
                    CHECK(tier.lookup(l, &id, 1, lk, err));
                    tier.promote(l, lk);
                    tier.pump(-1, core::ExpertTier::kAllPieces);
                }
            }
            tier.settle();
        };
        if (ok) {
            run(400);
            const int64_t full = tier.slots();
            int before[kLayers];
            for (int l = 0; l < kLayers; ++l) before[l] = held(l);
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t arena = (uint64_t) full * blob;
            CHECK(tier.lend((uint64_t) free_b - tier.free_floor() + arena / 2));
            const int64_t kept = tier.slots();
            std::printf("expert_tier_test: bounded lend keeps %lld of %lld slots (by layer %d/%d, %d/%d, %d/%d)\n", (long long) kept,
                        (long long) full, held(0), before[0], held(1), before[1], held(2), before[2]);
            CHECK(kept < full && kept > 0);
            for (int l = 0; l < kLayers; ++l) {
                CHECK(held(l) > 0 && held(l) < before[l]);
                // a layer keeps about its share of what stays (the arena's tail holds every layer's last slots)
                const double share = (double) held(l) / before[l], all = (double) kept / full;
                CHECK(share > all - 0.2 && share < all + 0.2);
                // its last expert stays (the most recently used)
                CHECK(tier.locate(l, last[l][3]).dev != nullptr);
            }
            CHECK(bytes_right());
            run(200);   // the kept slots keep working
            CHECK(bytes_right());
            CHECK(tier.slots() == kept);
            // A retry can borrow another segment while preserving the newest experts in the smaller arena.
            CHECK(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess);
            CHECK(tier.lend((uint64_t) free_b - tier.free_floor() + spec.lend_segment));
            CHECK(tier.slots() > 0 && tier.slots() < kept);
            for (int l = 0; l < kLayers; ++l) CHECK(tier.locate(l, last[l][3]).dev != nullptr);
            CHECK(bytes_right());
            CHECK(tier.reclaim(err));
            CHECK(tier.slots() == full);
            run(400);
            CHECK(bytes_right());
            for (int l = 0; l < kLayers; ++l) CHECK(held(l) == before[l]);
            // a lend that the arena cannot cover gives nothing up
            cudaMemGetInfo(&free_b, &total_b);
            CHECK(!tier.lend((uint64_t) free_b - tier.free_floor() + 4 * arena));
            CHECK(tier.slots() == full);
            // a part lent, then all of it (a prompt that needs more): reclaim opens the whole arena, which lends in parts again
            cudaMemGetInfo(&free_b, &total_b);
            CHECK(tier.lend((uint64_t) free_b - tier.free_floor() + arena / 2));
            tier.lend_all();
            CHECK(tier.slots() == 0 && !tier.partial_lend());
            CHECK(tier.reclaim(err));
            CHECK(tier.slots() == full && tier.partial_lend());
            run(400);
            CHECK(bytes_right());
            cudaMemGetInfo(&free_b, &total_b);
            CHECK(tier.lend((uint64_t) free_b - tier.free_floor() + arena / 2));
            CHECK(tier.slots() > 0 && tier.slots() < full);
            CHECK(bytes_right());
            // a reclaim that restores: the slots come back holding the experts the lend took off, with the pack's bytes
            CHECK(tier.reclaim(err, /*restore=*/true));
            CHECK(tier.slots() == full);
            for (int l = 0; l < kLayers; ++l) CHECK(held(l) == before[l]);
            CHECK(bytes_right());
            // and after lend_all (a sleep, a picture's encoder): every expert the arena held is back in it
            run(100);
            std::vector<uint8_t> had((size_t) kLayers * kExperts);
            for (int l = 0; l < kLayers; ++l)
                for (int e = 0; e < kExperts; ++e) had[(size_t) l * kExperts + e] = tier.locate(l, e).dev != nullptr;
            tier.lend_all();
            CHECK(tier.slots() == 0);
            CHECK(tier.reclaim(err, /*restore=*/true));
            CHECK(tier.slots() == full);
            bool same = true;
            for (int l = 0; l < kLayers; ++l)
                for (int e = 0; e < kExperts; ++e) same = same && had[(size_t) l * kExperts + e] == (tier.locate(l, e).dev != nullptr);
            CHECK(same);
            CHECK(bytes_right());
            run(200);   // and the restored tier keeps working
            CHECK(bytes_right());
        }
        cudaFreeHost(h_rows);
    }
    // a second card holding homes of its own: a lookup of one of them is a second card hit, and its slots hold the pack's bytes
    if (n_dev >= 2) {
        core::ExpertTierSpec spec;
        spec.area = "basalt test";
        spec.name = "test";
        spec.n_layers = kLayers;
        spec.n_expert = kExperts;
        spec.top_k = 6;
        spec.n_embd = kDim;
        spec.n_ff = kDim;
        for (int l = 0; l < kLayers; ++l) spec.layers.push_back({kQ4K, kQ4K, (uint64_t) l * kExperts * blob, blob, 0.f});
        spec.path = path;
        spec.fd = fd;
        spec.range = core::FileRange(path, 0, data.size());
        spec.diag_tier_ram = kExperts;
        spec.diag_tier_vram = kSlots;
        spec.admit_uses = 2;
        spec.share_stride = blob;
        spec.share_max_tokens = 64;
        spec.gpus.mode = basalt::program::GpuMode::Peer;   // GPU 0 main, GPU 1 its peer
        spec.gpus.cards.resize(2);
        spec.gpus.cards[1].device = 1;
        // the card is left room for only some of the homes
        size_t free1 = 0, total1 = 0;
        cudaSetDevice(1);
        cudaMemGetInfo(&free1, &total1);
        cudaSetDevice(0);
        const uint64_t keep = (uint64_t) 20 << 20;   // what the card leaves for homes above its own buffers: a few of the 48
        spec.peer_reserve_mib = (int) ((free1 - keep) >> 20);
        core::ExpertTier tier(spec);
        float* h_rows = nullptr;
        CHECK(cudaHostAlloc((void**) &h_rows, 128 * kDim * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
        core::LoadPlan plan;
        std::string err;
        core::ExpertTier::Reserve reserve;
        bool ok = tier.open(reserve, h_rows, plan, err);
        if (ok) {
            core::Loader loader;
            ok = loader.run(plan, err) && tier.attach(err);
        }
        if (!ok) std::fprintf(stderr, "two cards: %s\n", err.c_str());
        CHECK(ok);
        if (ok) {
            Lcg r{99};
            for (int t = 0; t < 400; ++t) {
                tier.begin_token(1);
                for (int l = 0; l < kLayers; ++l) {
                    int32_t ids[6];
                    int n = 0;
                    while (n < 6) {
                        const int e = (int) (r.next() % kExperts);
                        bool dup = false;
                        for (int j = 0; j < n; ++j) dup = dup || ids[j] == e;
                        if (!dup) ids[n++] = e;
                    }
                    core::ExpertTier::Lookup lk;
                    CHECK(tier.lookup(l, ids, 6, lk, err));
                    tier.fetch(l, lk);
                    tier.await_fetch(nullptr);
                    tier.promote(l, lk);
                    tier.mark_gap(nullptr);
                    tier.pump(l, 3);
                }
            }
            tier.settle();
            CHECK(tier.offloaded() > 0);
            int peer_seen = 0;
            std::vector<uint8_t> back(blob);
            for (int l = 0; l < kLayers; ++l)
                for (int e = 0; e < kExperts; ++e) {
                    const core::ExpertTier::Source src = tier.locate(l, e);
                    const uint8_t* where = src.dev ? src.dev : src.peer;
                    if (!where) continue;
                    CHECK(cudaMemcpy(back.data(), where, blob, cudaMemcpyDefault) == cudaSuccess);
                    CHECK(std::memcmp(back.data(), data.data() + ((size_t) l * kExperts + (size_t) e) * blob, blob) == 0);
                    peer_seen += src.peer != nullptr;
                }
            CHECK(peer_seen > 0);
            std::printf("expert_tier_test: two cards, %lld second-card or bus-fetched lookups, %d experts held by the second card\n",
                        (long long) tier.offloaded(), peer_seen);
        }
        cudaFreeHost(h_rows);
    }
    platform::close_fd(fd);
    std::printf("expert_tier_test: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
