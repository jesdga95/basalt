// src/core/expert_tier.cpp - the routed-expert tiers.
#include "basalt/log.hpp"
#include "basalt/core/expert_tier.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/exact_slots.hpp"

#include "basalt/core/load_progress.hpp"
#include "basalt/core/route_report.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/vmm.hpp"
#include <optional>

#include "basalt/core/expert_source.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace basalt::core {
namespace {

constexpr size_t kMaxPending = 128;              // copies claimed a window
constexpr size_t kEvents = 2 * kMaxPending;      // a window's and the last one's copies
}  // namespace

ExpertTier::ExpertTier(ExpertTierSpec spec) : s_(std::move(spec)) {}

ExpertTier::~ExpertTier() {
    std::string read_err;
    await_reads(read_err);   // the reads write into the RAM tier freed below
    report("final");
    if (copy_stream_) issue(-1, kAllPieces);
    if (!peer_moves_.empty()) {   // every card: an idle card's sync is cheap, a skipped one leaves a copy in flight
        std::string e;
        for (const auto& p : peers_) p->read_wait(e);
    }
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (out_stream_) cudaStreamSynchronize(out_stream_);
    if (fetch_stream_) cudaStreamSynchronize(fetch_stream_);
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
    for (cudaEvent_t e : out_events_) cudaEventDestroy(e);
    if (fetch_ev_) cudaEventDestroy(fetch_ev_);
    if (gap_ev_) cudaEventDestroy(gap_ev_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (out_stream_) cudaStreamDestroy(out_stream_);
    if (fetch_stream_) cudaStreamDestroy(fetch_stream_);
    for (size_t l = 0; l < ram_.size(); ++l) {
        if (ram_[l] == nullptr) continue;
        if (l < ram_pins_.size()) core::unpin_ram(ram_[l], ram_pins_[l]);
    }
    ram_allocations_.clear();
    platform::free_aligned(act_);
    if (held_ev_) {
        cudaEventSynchronize(held_ev_);
        cudaEventDestroy(held_ev_);
    }
    cudaFree(held_dev_);
    cudaFreeHost(held_pin_);
    cudaFree(route_counts_);
    cudaFree(zero_row_);
}

bool ExpertTier::open(const Reserve& reserve, float* h_rows, core::LoadPlan& reads, std::string& err) {
    n_layers_ = s_.n_layers;
    n_expert_ = s_.n_expert;
    uint64_t one_each = 0, one_each_ram = 0;   // bytes of one slot in every layer, of one RAM place in every layer
    for (const ExpertTierSpec::Layer& e : s_.layers) {
        one_each += (e.blob_bytes + 255) / 256 * 256;
        one_each_ram += e.blob_bytes;
    }
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = s_.name + ": cannot query the free device memory: " + why;
        return false;
    }
    one_each_ = one_each;
    auto slots_in = [&](uint64_t room, uint64_t left_free) {
        return (int) std::min<uint64_t>(room > left_free ? (room - left_free) / one_each : 0, (uint64_t) n_expert_);
    };
    if (!load_rank(err)) return false;
    const uint64_t asked_free = (uint64_t) s_.vram_reserve_mib << 20;
    int copies = slots_in(free_b, reserve.copies + asked_free);
    int main_homes = slots_in(free_b, reserve.homes + asked_free);
    start_floor_ = free_floor_ = asked_free;
    if (s_.expert_cache > 0) {
        cap_per_layer_ = s_.expert_cache / n_layers_;
        exact_per_layer_ = std::min<int64_t>(cap_per_layer_, n_expert_);   // honoured or refused (plan_ext), never trimmed
    }
    if (s_.diag_tier_vram >= 0) cap_per_layer_ = cap_per_layer_ >= 0 ? std::min<int64_t>(cap_per_layer_, s_.diag_tier_vram) : s_.diag_tier_vram;
    if (cap_per_layer_ >= 0) {
        copies = (int) std::min<int64_t>(copies, cap_per_layer_);
        main_homes = (int) std::min<int64_t>(main_homes, cap_per_layer_);
    }

    // The second cards: each owns a block of layers and holds that block's second-card homes, so every layer has one.
    // They open here, before the prompt buffers and the RAM reading below, so the RAM tier sees their host buffers.
    int peer_slots = 0;
    layer_card_.assign((size_t) n_layers_, 0);
    if (!s_.peer_devices.empty()) {
        const std::vector<int>& devs = s_.peer_devices;
        std::string pe;
        bool ok = true;
        for (size_t c = 0; c < devs.size() && ok; ++c) {
            auto p = std::make_unique<core::PeerExperts>();
            ok = p->open_plain(devs[c], n_layers_, n_expert_, s_.n_embd, s_.n_ff, h_rows, pe);
            if (ok) {
                peers_.push_back(std::move(p));
                ok = open_share(c, pe);
            }
        }
        if (ok) {
            // every card is open with its share's buffers, so its room is what its slots will get: the blocks are sized
            // to the rooms (bigger late layers go to more cards), then each block's prompt-share prices are set
            std::vector<uint64_t> rooms, layer_bytes;   // one slot of each layer, as one_each counts them
            for (const auto& p : peers_) rooms.push_back(p->room());
            for (int l = 0; l < n_layers_; ++l) layer_bytes.push_back((s_.layers[(size_t) l].blob_bytes + 255) / 256 * 256);
            const uint64_t reserve_peer = (uint64_t) s_.peer_reserve_mib << 20;
            blocks_ = core::policy::split_by_bytes(layer_bytes, rooms, reserve_peer, n_expert_, peer_slots);
            for (size_t c = 0; c < blocks_.lo.size(); ++c)
                for (int l = blocks_.lo[c]; l < blocks_.hi[c]; ++l) layer_card_[(size_t) l] = (int) c;
            fill_share_rates();
        } else {
            shares_.clear();
            basalt::log::warn(s_.area.c_str(), "second card not used: %s", pe.c_str());
            peers_.clear();
            layer_card_.assign((size_t) n_layers_, 0);
        }
    }

    // The prompt and image buffers first, so the free memory the tier reads already lacks them. The ring of file reads
    // comes only when experts may stay in the file.
    platform::HostMemory hm;
    auto available = [&] { return platform::host_available_memory(hm) ? hm.available : 0; };
    const uint64_t ram_reserve = platform::ram_reserve();
    uint64_t tier_reserve = ram_reserve;   // and the file cache's floor once experts stay in the file
    // the image encoder's cuBLAS handle takes host memory when created, so it is taken before the reading
    if (s_.host.open && !s_.host.open(err)) return false;
    host_bytes_ = s_.host.bytes ? s_.host.bytes() : 0;
    uint64_t avail = available();
    // places per layer above the reserve, less the spare each layer keeps, in `bytes` of RAM
    auto places_in = [&](uint64_t bytes) {
        const uint64_t places = core::ram_tier_room(bytes, tier_reserve, true) / one_each_ram;
        return places > 0 ? places - 1 : 0;
    };
    const bool ram_given = s_.resident_budget > 0 || s_.diag_tier_ram >= 0;
    auto ram_in = [&] {
        int slots = (int) std::min<uint64_t>(places_in(avail), (uint64_t) n_expert_);
        if (s_.resident_budget > 0)
            slots = std::min(slots, (int) std::min<uint64_t>(s_.resident_budget / one_each_ram, (uint64_t) n_expert_));
        if (s_.diag_tier_ram >= 0) slots = std::min(s_.diag_tier_ram, n_expert_);
        return slots;
    };
    int ram_slots = ram_in();
    // what the machine can give the tier whatever else runs: its RAM (or the container's) less the engine's own buffers.
    // The home-versus-copies choice reads this alone, so the cache state at start never moves it.
    const uint64_t total = std::min(platform::total_physical_memory(), hm.cgroup_limit);
    const int ram_cap = ram_given ? -1 : (int) std::min<uint64_t>(places_in(total > host_bytes_ ? total - host_bytes_ : 0), 1u << 20);
    homes_ = core::policy::plan_homes(n_expert_, main_homes, peer_slots, ram_slots, ram_cap);
    if (homes_.file > 0) {   // experts stay in the file: the ring of its reads takes RAM before the tier is sized
        if (s_.host.set_ring && !s_.host.set_ring(s_.ring_pieces, err)) return false;
        avail = available();
        // and they read through the OS file cache, which keeps its floor beside the reserve (the ring holds a prompt's
        // reads, so only the layers count)
        core::FileCache cache;
        for (const auto& e : s_.layers) cache.layer_bytes = std::max<uint64_t>(cache.layer_bytes, (uint64_t) n_expert_ * e.blob_bytes);
        if (!ram_given)
            tier_reserve = ram_reserve + core::file_cache_keep(avail, ram_reserve, true,
                                                               (uint64_t) (homes_.ram + homes_.file + 1) * one_each_ram, cache);
        ram_slots = ram_in();
        homes_ = core::policy::plan_homes(n_expert_, main_homes, peer_slots, ram_slots, ram_cap);
    }
    const uint64_t ram_floor =
        tier_reserve + (homes_.ram > 0 ? core::pin_tracking_bytes((uint64_t) (homes_.ram + 1) * one_each_ram) : 0);
    if (homes_.peer == 0) {
        shares_.clear();
        peers_.clear();
    }

    // the base arena (homes, or copies a long prompt borrows) leaves its reserve free; plan_ext() sizes what fills it
    // beside a short prompt
    reserve_ = reserve.beside_homes;
    modules_ = reserve.modules;
    per_layer_ = homes_.main > 0 ? homes_.main : copies;
    if (homes_.main == 0 && per_layer_ < s_.top_k) {
        err = s_.name + ": not enough device memory for the expert tier";
        return false;
    }
    if (!open_base(per_layer_, err)) return false;
    res_.assign((size_t) n_layers_ * n_expert_, -1);
    stamp_.assign((size_t) n_layers_ * n_expert_, 0);
    owner_.assign((size_t) base_slots(), -1);
    busy_.assign((size_t) base_slots(), 0);
    int lo_prio = 0, hi_prio = 0;
    cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio);
    if (cudaEventCreateWithFlags(&fetch_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&gap_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&out_stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithPriority(&fetch_stream_, cudaStreamNonBlocking, hi_prio) != cudaSuccess) {
        err = s_.name + ": cannot create the promotion stream";
        return false;
    }
    claim_cap_ = (int) kMaxPending;
    const size_t n_events = std::max(kEvents, (size_t) 2 * (size_t) std::max(s_.claims_per_row, 0) * kMaxWindow);
    for (size_t i = 0; i < n_events; ++i) {
        cudaEvent_t e, o;
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = s_.name + ": cannot create a promotion event";
            return false;
        }
        events_.push_back(e);
        if (cudaEventCreateWithFlags(&o, cudaEventDisableTiming) != cudaSuccess) {
            err = s_.name + ": cannot create a promotion event";
            return false;
        }
        out_events_.push_back(o);
        free_events_.push_back((int) i);
    }

    // the CPU side: the file view, the per-layer formats, the pool
    if (homes_.file > 0) {
        std::optional<platform::MappedFile> mf = platform::MappedFile::open_ro(s_.path, &err);
        if (!mf) return false;
        file_ = std::move(*mf);
    }
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertTierSpec::Layer& el = s_.layers[(size_t) l];
        kernels::cpu::NativeFmt f;
        if (!kernels::cpu::native_fmt(el.gu_type, el.d_type, s_.n_embd, s_.n_ff, f, err, s_.layers[(size_t) l].clamp))
            return false;
        fmt_.push_back(f);
    }
    act_ = (uint8_t*) platform::alloc_aligned(kernels::cpu::kNativeActBytes * kMaxWindow, 64);
    const kernels::cpu::PoolAffinity affinity = s_.pool_affinity_given ? s_.pool_affinity : kernels::cpu::PoolAffinity::All;
    int workers = s_.pool_workers;
    if (workers <= 0) {
        // The start-up pick the pool's own rule makes: a hybrid CPU runs on its P-cores and half the E-cores, and the
        // cgroup quota caps the workers and the host thread together.  It holds whatever the quant and whatever the peer
        // cards, so no model decides it.
        const kernels::cpu::CpuTopology topo = kernels::cpu::detect_cpu_topology(true);
        const kernels::cpu::PoolAffinity auto_affinity = !s_.pool_affinity_given || affinity == kernels::cpu::PoolAffinity::Auto
                                                             ? kernels::cpu::PoolAffinity::Auto
                                                         : affinity == kernels::cpu::PoolAffinity::PCores
                                                             ? kernels::cpu::PoolAffinity::PCores
                                                             : kernels::cpu::PoolAffinity::All;
        workers = kernels::cpu::auto_pool_workers((int) topo.worker_cores.size(), topo.p_cores, topo.e_cores, topo.is_hybrid,
                                                  kernels::cpu::cgroup_cpu_quota(), auto_affinity);
    }
    pool_ = std::make_unique<kernels::cpu::ExpertPool>(workers, affinity);

    if (!load_ram(reads, err)) return false;
    if (homes_.file > 0 && cover_ > 0) {
        if (!reads_.open(s_.path, s_.fd, err)) return false;
        uses_.assign((size_t) n_layers_ * n_expert_, 0);
    }
    if (s_.admit_uses > 0) uses_.assign((size_t) n_layers_ * n_expert_, 0);
    // what the start picked by itself, on the server's auto line; the RAM floor the tier left free
    std::printf("auto flags: resident %.1f GB, pool-workers %d\n",
                (double) one_each_ram * cover_ / 1e9, pool_->workers());   // the server keeps the words after the name
    std::fflush(stdout);
    basalt::log::info("basalt generate", "RAM floor %.1f GiB of %.1f GiB free", (double) ram_floor / 1073741824.0,
                      (double) avail / 1073741824.0);
    for (size_t c = 0; c < peers_.size(); ++c) {   // one card: the single second card's line; several: each with its block of layers
        const int nl = blocks_.hi[c] - blocks_.lo[c];
        if (peers_.size() == 1)
            basalt::log::info(s_.area.c_str(), "peer GPU %d: %lld experts", peers_[c]->device(), (long long) nl * homes_.peer);
        else
            basalt::log::info(s_.area.c_str(), "peer GPU %d: %lld experts (layers %d-%d)", peers_[c]->device(),
                              (long long) nl * homes_.peer, blocks_.lo[c], blocks_.hi[c] - 1);
    }
    basalt::log::info(s_.area.c_str(), "expert homes of %d per layer: main card %d, second card %d, RAM %d, file %d; main card copies %lld",
                 n_expert_, homes_.main, homes_.peer, homes_.ram, homes_.file, (long long) (per_layer_ + ext_per_layer_ - homes_.main));
    program::warn_spill(spill(), peers_.empty() ? ", or --gpu 0,1" : "");
    basalt::log::info(s_.area.c_str(), "VRAM expert tier %.2f GiB, %lld of %d experts per layer (%.0f%%); CPU pool %d workers",
                 cache_.gib(), (long long) per_layer_, n_expert_, 100.0 * (double) per_layer_ / n_expert_, pool_->workers());
    bool uniform = true;
    for (int l = 0; l < n_layers_; ++l) uniform = uniform && base_n(l) == per_layer_;
    if (!uniform) {
        std::string by;
        for (int l = 0; l < n_layers_; ++l) by += (l ? " " : "") + std::to_string(base_n(l));
        basalt::log::info(s_.area.c_str(), "VRAM expert tier copies by layer (the model's weights): %s", by.c_str());
    }
    // the RAM the tier took emptied the file cache; with no expert read from the file, it takes the token embedding back
    if (homes_.file == 0 && s_.host.ram_only) s_.host.ram_only();
    return true;
}

// Second card `card`'s prompt share takes its buffers before the card's slots are sized, and the prices of its layers.
bool ExpertTier::open_share(size_t card, std::string& err) {
    prefill::PeerShare::Dims d;
    d.n_embd = s_.n_embd;
    d.n_ff = s_.n_ff;
    d.group_bytes = (size_t) s_.share_stride * (size_t) d.launch_experts;
    d.act_bytes = prefill::mmq::q8_bytes(s_.share_max_tokens, s_.n_embd);
    d.max_tokens = s_.share_max_tokens;
    d.max_rows = (int64_t) s_.share_max_tokens * s_.top_k;
    d.n_expert = n_expert_;
    int main_dev = 0;
    cudaGetDevice(&main_dev);
    // the share runs on the card that holds the block's homes: start_peer hands it device pointers into that card's slots
    const int dev = peers_[card]->device();
    auto share = std::make_unique<prefill::PeerShare>();
    if (!share->open(main_dev, dev, d, err)) return false;
    shares_.push_back(std::move(share));
    return true;
}

// Each layer's prompt-share prices, from the main card and the card that holds the layer (blocks_ is set).
void ExpertTier::fill_share_rates() {
    int main_dev = 0;
    cudaGetDevice(&main_dev);
    const core::DeviceRates r0 = core::device_rates(main_dev);
    // a row of a layer's products is three matrices of n_embd x n_ff multiply-adds
    const double macs = 3.0 * (double) s_.n_embd * (double) s_.n_ff;
    rates_.assign((size_t) n_layers_, {});
    for (size_t card = 0; card < blocks_.lo.size() && card < peers_.size(); ++card) {
        const core::DeviceRates r1 = core::device_rates(peers_[card]->device());
        for (int l = blocks_.lo[card]; l < blocks_.hi[card]; ++l) {
            prefill::ShareRates& r = rates_[(size_t) l];
            r.link0 = r0.link_bytes;
            r.link1 = r1.link_bytes;
            r.rows0 = r0.sm_hz * prefill::kMmqMacsPerSmCycle / macs;
            r.rows1 = r1.sm_hz * prefill::kMmqMacsPerSmCycle / macs;
            r.sm_hz0 = r0.sm_hz;
        }
    }
}

program::ExpertSpill ExpertTier::spill() const {
    uint64_t one_each = 0;
    for (const ExpertTierSpec::Layer& e : s_.layers) one_each += e.blob_bytes;
    return {(int64_t) n_layers_ * homes_.file, (int64_t) n_layers_ * n_expert_, one_each * (uint64_t) homes_.file};
}

namespace {

// The second card's real path over the homes of the layer it holds most of, one token at a time: each share is armed
// as a layer arms it with its head, the card waits at its word, and the round trip runs from the plan to the rows.
class PeerPath final : public program::ProbePeerPath {
public:
    PeerPath(core::PeerExperts& peer, int top_k) : peer_(peer), k_(top_k) {}
    bool setup(const program::ProbeEnv& env, const kernels::cpu::NativeFmt& f, std::string& why) override {
        L_ = kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        for (int64_t e = 0; e < env.n_expert && (int) ids_.size() < 8; ++e)
            if (peer_.slot_ptr(0, e) != nullptr) ids_.push_back((int32_t) e);
        if (ids_.empty()) { why = "no_experts_on_peer"; return false; }
        if (!x_.alloc((size_t) f.n_embd * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped)) { why = "alloc_failed"; return false; }
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int64_t i = 0; i < f.n_embd; ++i) ((float*) x_.p)[i] = nd(rng);
        return true;
    }
    int max_entries() const override { return (int) ids_.size(); }
    int rings() const override { return 24; }
    int blocks() const override { return 0; }
    void fill(int n) override { n_ = n; }
    bool window(const kernels::cpu::NativeFmt&, const std::function<void(int)>& before, std::vector<double>& rt,
                std::string& err) override {
        int32_t ids[ExpertTier::kMaxK] = {}, kind[ExpertTier::kMaxK] = {};
        const int k = std::max(k_, n_);
        for (int i = 0; i < n_; ++i) { ids[i] = ids_[(size_t) i]; kind[i] = 2; }
        rt.clear();
        bool ok = true;
        for (int r = 0; r < rings() && ok; ++r) {
            before(r);
            ok = peer_.arm(L_, (const float*) x_.p, 1, k, core::policy::peer_groups(1), err);
            const auto ta = program::ProbeClock::now();
            while (ok && program::us_since(ta) < kHeadUs) {
            }
            const auto t0 = program::ProbeClock::now();
            ok = ok && peer_.raise(0, ids, kind, err) && peer_.finish(nullptr, err);
            rt.push_back(program::us_since(t0));
        }
        return ok;
    }

private:
    static constexpr double kHeadUs = 100;   // about a head's time: the arm's launches are on the card by then
    core::PeerExperts& peer_;
    program::HostBuf x_;
    int k_, n_ = 1;
    kernels::NativeExpertLayout L_;
    std::vector<int32_t> ids_;
};

}  // namespace

void ExpertTier::probe_env(program::ProbeEnv& env) {
    env.pool = pool_.get();
    env.ram_blob = [this](int64_t l, int64_t e) { return locate((int) l, (int) e).pinned; };
    env.fmt = fmt_;
    env.n_layers = n_layers_;
    env.n_expert = n_expert_;
    env.top_k = s_.top_k;
    if (!peers_.empty()) {   // the probe prices the first second card's path for every second card
        env.peer_device = peer_for(0)->device();
        env.peer_path = std::make_shared<PeerPath>(*peer_for(0), s_.top_k);
    }
}

bool ExpertTier::load_ram(core::LoadPlan& plan, std::string& err) {
    uint64_t max_blob = 0;
    for (const ExpertTierSpec::Layer& e : s_.layers) max_blob = std::max<uint64_t>(max_blob, e.blob_bytes);
    cover_ = homes_.ram;
    ram_.assign((size_t) n_layers_, nullptr);
    ram_idx_.assign((size_t) n_layers_ * n_expert_, -1);
    spare_.assign((size_t) n_layers_, -1);
    last_miss_.assign((size_t) n_layers_ * n_expert_, 0);
    std::vector<int32_t> device_home;   // [layer * n_expert + e] the slot of a home on either card, or -1
    if (homes_.main > 0 || homes_.peer > 0) device_home.assign((size_t) n_layers_ * n_expert_, -1);
    if (!peers_.empty()) {   // peer_owner_ keeps the global slot l * homes_.peer + n; each card numbers its own from 0
        peer_owner_.assign((size_t) n_layers_ * homes_.peer, -1);
        for (size_t c = 0; c < peers_.size(); ++c) {
            std::vector<int64_t> sizes;
            for (int l = blocks_.lo[c]; l < blocks_.hi[c]; ++l)
                for (int s = 0; s < homes_.peer; ++s) sizes.push_back((int64_t) s_.layers[(size_t) l].blob_bytes);
            if (!peers_[c]->open_slots(sizes, err)) return false;
            // scratch slots for the misses the card fetches over its own link, in what its homes leave above its reserve
            const uint64_t keep = (uint64_t) s_.peer_reserve_mib << 20, room = peers_[c]->room();
            const uint64_t slot = (max_blob + 255) / 256 * 256;
            const int n = room > keep ? (int) std::min<uint64_t>(core::policy::kPeerFetchSlots, (room - keep) / slot) : 0;
            std::string se;
            if (n > 0 && !peers_[c]->open_scratch(n, max_blob, se))
                basalt::log::warn(s_.area.c_str(), "second card GPU %d fetches no misses over its link: %s",
                                  peers_[c]->device(), se.c_str());
        }
    }
    for (int l = 0; l < n_layers_; ++l) {
        std::vector<int> held;
        int n_main = 0, n_peer = 0;
        for (int r = 0; r < n_expert_; ++r) {
            const int e = ranked(l, r);
            const size_t idx = (size_t) l * n_expert_ + e;
            switch (core::policy::place_of_rank(homes_, homes_.main > 0 ? 0 : (int) base_n(l), r)) {
                case core::policy::Place::Main: {
                    const int32_t slot = (int32_t) (base_lo_[(size_t) l] + n_main++);
                    res_[idx] = slot;
                    owner_[(size_t) slot] = e;
                    device_home[idx] = slot;
                    break;
                }
                case core::policy::Place::Peer: {
                    const int32_t slot = (int32_t) ((int64_t) l * homes_.peer + n_peer);   // global
                    const int32_t local = core::policy::local_slot(blocks_, homes_.peer, l, n_peer++);
                    peer_for(l)->admit(l, e, local);
                    peer_owner_[(size_t) slot] = e;
                    device_home[idx] = local;   // the load sink finds the card from the layer
                    break;
                }
                case core::policy::Place::Ram: held.push_back(e); break;
                case core::policy::Place::File: break;
            }
        }
        std::sort(held.begin(), held.end());
        for (size_t i = 0; i < held.size(); ++i) ram_idx_[(size_t) l * n_expert_ + held[i]] = (int32_t) i;
    }
    if (cover_ <= 0 && device_home.empty()) return true;
    if (cover_ > 0)                                     // one more RAM place per layer takes the experts on the move
        for (int l = 0; l < n_layers_; ++l) spare_[(size_t) l] = cover_;
    ram_t0_ = std::chrono::steady_clock::now();
    ram_queued_ = true;
    // the layout: one span per layer, each with its swap spare
    ram_allocations_.resize((size_t) n_layers_);
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertTierSpec::Layer& el = s_.layers[(size_t) l];
        if (cover_ <= 0) continue;
        const uint64_t span = el.blob_bytes * (uint64_t) (cover_ + 1);
        if (!ram_allocations_[(size_t) l].allocate(span)) {
            err = s_.name + ": cannot allocate the RAM expert tier";
            return false;
        }
        ram_[(size_t) l] = ram_allocations_[(size_t) l].data();
    }
    // every expert's blob: into its RAM place, and onto the card that is its home, in the start's pass (the file's
    // share is not read)
    const int experts_file = plan.file_fd(s_.fd);
    int main_device = 0;
    if (cudaGetDevice(&main_device) != cudaSuccess) main_device = 0;
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertTierSpec::Layer& el = s_.layers[(size_t) l];
        for (int e = 0; e < n_expert_; ++e) {
            const size_t idx = (size_t) l * n_expert_ + (size_t) e;
            core::LoadItem it;
            it.file = experts_file;
            it.off = s_.range.offset + el.offset + (uint64_t) e * el.blob_bytes;
            it.bytes = el.blob_bytes;
            if (ram_idx_[idx] >= 0) {
                it.host = ram_[(size_t) l] + (size_t) ram_idx_[idx] * (size_t) el.blob_bytes;
                plan.add(it);
                it.host = nullptr;
            }
            if (!device_home.empty() && device_home[idx] >= 0) {
                const int32_t home = device_home[idx];
                if (res_[idx] >= 0) {
                    it.dev = const_cast<uint8_t*>(slot_ptr(home));
                    it.device = main_device;
                } else {
                    it.dev = peer_for(l)->slot_ptr(home);
                    it.device = peer_for(l)->device();
                }
                plan.add(it);
            }
        }
    }
    return true;
}

bool ExpertTier::attach(std::string& err) {
    if (!land_ram(err)) return false;
    if (homes_.main == 0) seed_copies(0, /*base_shares=*/true);
    promoted_ = 0;
    return plan_ext(err) && open_route(err);   // after the RAM tier is pinned: its registration takes VRAM too
}

bool ExpertTier::held(int layer, int e) const {
    return res_[(size_t) layer * n_expert_ + e] >= 0 || (!peers_.empty() && peer_for(layer)->slot_ptr(layer, e) != nullptr);
}

// The lossy routing flags' table, counters and zero row, once the slots are placed; off when every expert is held.
bool ExpertTier::open_route(std::string& err) {
    if (s_.route_tail_skip <= 0 && s_.route_prior <= 0.0f) return true;
    bool all = true;
    for (int l = 0; all && l < n_layers_; ++l)
        for (int e = 0; all && e < n_expert_; ++e) all = held(l, e);
    if (all) {
        if (s_.route_tail_skip > 0) basalt::log::info(s_.area.c_str(), "route-tail-skip is off: every expert is held by a GPU");
        if (s_.route_prior > 0.0f) basalt::log::info(s_.area.c_str(), "route-prior is off: every expert is held by a GPU");
        return true;
    }
    const size_t n = (size_t) n_layers_ * n_expert_;
    if (cudaMalloc((void**) &held_dev_, n * 4) != cudaSuccess || cudaHostAlloc((void**) &held_pin_, n * 4, cudaHostAllocPortable) != cudaSuccess ||
        cudaMalloc((void**) &route_counts_, 8 * 8) != cudaSuccess || cudaMemset(route_counts_, 0, 8 * 8) != cudaSuccess ||
        cudaMalloc((void**) &zero_row_, (size_t) s_.n_embd * 4) != cudaSuccess || cudaMemset(zero_row_, 0, (size_t) s_.n_embd * 4) != cudaSuccess ||
        cudaMemset(held_dev_, 0xff, n * 4) != cudaSuccess || cudaEventCreateWithFlags(&held_ev_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        err = s_.name + ": cannot allocate the lossy routing flags' tables";
        return false;
    }
    held_pub_.assign(n, -1);
    route_tail_skip_ = s_.route_tail_skip;
    route_prior_ = s_.route_prior;
    return true;
}

void ExpertTier::publish_held(cudaStream_t stream) {
    if (held_dev_ == nullptr) return;
    cudaEventSynchronize(held_ev_);   // the last publish's copies have read the staging
    const size_t ne = (size_t) n_expert_;
    for (int l = 0; l < n_layers_;) {
        // a run of layers that changed goes out in one copy
        const auto changed = [&](int layer) {
            bool c = false;
            int32_t* pin = held_pin_ + (size_t) layer * ne;
            int32_t* pub = held_pub_.data() + (size_t) layer * ne;
            for (int e = 0; e < n_expert_; ++e) {
                const int32_t v = held(layer, e) ? 0 : -1;
                c |= v != pub[e];
                pin[e] = pub[e] = v;
            }
            return c;
        };
        if (!changed(l)) {
            ++l;
            continue;
        }
        int hi = l + 1;
        while (hi < n_layers_ && changed(hi)) ++hi;
        cudaMemcpyAsync(held_dev_ + (size_t) l * ne, held_pin_ + (size_t) l * ne, (size_t) (hi - l) * ne * 4, cudaMemcpyHostToDevice, stream);
        l = hi + 1;   // layer hi was checked (and is unchanged, or past the end)
    }
    cudaEventRecord(held_ev_, stream);
}

kernels::ResidentTabs ExpertTier::held_tabs(int layer) const {
    kernels::ResidentTabs t;
    if (held_dev_ != nullptr) t.add(held_dev_ + (size_t) layer * n_expert_);
    return t;
}

// The RAM tier's experts have landed (the pass's pinned staging is freed by then): its spans are pinned.
bool ExpertTier::land_ram(std::string& err) {
    if (!ram_queued_) return true;
    uint64_t per_layer = 0;
    for (const ExpertTierSpec::Layer& e : s_.layers) per_layer += e.blob_bytes;
    if (cover_ > 0) {
        uint64_t host_pinned = 0;
        for (const auto& a : ram_allocations_)
            if (a.allocation_pinned()) host_pinned += a.capacity();
        ram_pins_ = core::pin_ram_spans(ram_allocations_, /*mapped=*/false);
        if (host_pinned)
            basalt::log::info(s_.area.c_str(), "RAM expert tier: %.2f GiB CUDA-pinned through host VMM", host_pinned / 1073741824.0);
        int unpinned = 0;
        for (const core::RamPin& p : ram_pins_) unpinned += p.registered_bytes == 0;
        if (unpinned > 0)
            basalt::log::warn(s_.area.c_str(), "RAM expert tier: %d of %d layers not pinned, locked: %s", unpinned, n_layers_,
                              ram_pins_[(size_t) n_layers_ - 1].lock_note.c_str());
        // Pinned past what the driver backs, the runtime stops answering on every card, and returning the pages does not
        // bring it back: the start ends here rather than at the first request.
        std::vector<int> cards{-1};   // the main card is the current one
        for (const auto& p : peers_) cards.push_back(p->device());
        std::string dead;
        if (!core::devices_allocate(cards, dead)) {
            err = s_.name + ": the device cannot allocate after pinning the RAM tier (" + dead +
                  "); pin less with a smaller --resident-budget-gib";
            return false;
        }
    }
    // what the load left in the page cache goes
    platform::advise_dontneed(s_.fd, s_.range.offset, s_.range.size());
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ram_t0_).count();
    basalt::log::info(s_.area.c_str(), "RAM expert tier %.1f GiB (%d of %d experts per layer) loaded in %.1f s",
                 (double) per_layer * cover_ / 1073741824.0, cover_, n_expert_, s);
    return true;
}

int ExpertTier::ranked(int layer, int r) const {
    return rank_.empty() ? r : rank_[(size_t) layer * n_expert_ + r];
}

// The routing order from the profile file: per layer, the order its pairs come in.
bool ExpertTier::load_rank(std::string& err) {
    const std::string& path = s_.expert_profile;
    if (path.empty()) return true;
    uint64_t bytes = 0;
    if (!platform::file_size(path, bytes)) {
        err = "cannot read the expert profile " + path;
        return false;
    }
    std::vector<std::pair<int32_t, int32_t>> pairs;
    int64_t slots = 0;
    if (!core::read_expert_profile(core::FileRange{path, 0, bytes}, n_layers_, n_expert_, pairs, slots, err)) return false;
    if (pairs.size() != (size_t) n_layers_ * n_expert_) {
        err = "the expert profile " + path + " does not rank every expert";
        return false;
    }
    rank_.resize(pairs.size());
    std::vector<int32_t> next((size_t) n_layers_, 0);
    for (const auto& [layer, e] : pairs) rank_[(size_t) layer * n_expert_ + next[(size_t) layer]++] = e;
    return true;
}

// The base arena's layout: per_layer slots in every layer, or, for copies with layer weights, the same total spread by them.
void ExpertTier::set_base(int64_t per_layer) {
    const bool weighted = homes_.main == 0 && !s_.layer_weights.empty() && (int) s_.layer_weights.size() == n_layers_;
    const std::vector<int64_t> n = weighted ? core::policy::layer_slots(per_layer, s_.layer_weights, std::min<int64_t>(per_layer, s_.top_k), n_expert_)
                                            : std::vector<int64_t>((size_t) n_layers_, per_layer);
    base_lo_.assign((size_t) n_layers_ + 1, 0);
    for (int l = 0; l < n_layers_; ++l) base_lo_[(size_t) l + 1] = base_lo_[(size_t) l] + n[(size_t) l];
}

// Copies the most routed RAM tier experts the main card does not hold yet into `n` slots of each layer.
void ExpertTier::seed_copies(int64_t n_each, bool base_shares) {
    for (int l = 0; l < n_layers_; ++l) {
        const int64_t n = base_shares ? base_n(l) : n_each;
        for (int64_t r = 0, placed = 0; r < n_expert_ && placed < n; ++r) {
            const int e = ranked(l, (int) r);
            if (!pinned(l, e) || res_[(size_t) l * n_expert_ + e] >= 0) continue;
            ++tick_;
            if (claim(l, e, blob_of(l, e), copy_stream_) < 0) {
                if (const cudaError_t ce = cudaStreamSynchronize(copy_stream_); ce != cudaSuccess)
                    basalt::log::error(s_.area.c_str(), "tier seed copy sync: %s", cudaGetErrorString(ce));
                land_all();
                ++tick_;
                if (claim(l, e, blob_of(l, e), copy_stream_) < 0) return;
            }
            ++placed;
        }
    }
    if (const cudaError_t ce = cudaStreamSynchronize(copy_stream_); ce != cudaSuccess)
        basalt::log::error(s_.area.c_str(), "tier seed sync: %s", cudaGetErrorString(ce));
    land_all();
}

const uint8_t* ExpertTier::blob_of(int layer, int e) const {
    const ExpertTierSpec::Layer& el = s_.layers[(size_t) layer];
    const int32_t r = ram_idx_[(size_t) layer * n_expert_ + e];
    if (r >= 0) return ram_[(size_t) layer] + (size_t) r * el.blob_bytes;
    return file_.data() + s_.range.offset + el.offset + (uint64_t) e * el.blob_bytes;
}

const uint8_t* ExpertTier::slot_ptr(int32_t slot) const {
    const int64_t nb = base_slots();
    if (slot < nb) return cache_.device_slot(phys_.empty() ? slot : phys_[(size_t) slot]);
    return ext_.device_slot((int32_t) (slot - nb));
}

void ExpertTier::apply(const Pending& p) {
    const size_t in = (size_t) p.layer * n_expert_ + p.expert;
    res_[in] = p.slot;
    busy_[(size_t) p.slot] = 0;
    free_events_.push_back(p.event);
    if (p.out >= 0) {   // the home that left has its RAM place, and the one that came gives its own up
        ram_idx_[(size_t) p.layer * n_expert_ + p.out] = p.place;
        spare_[(size_t) p.layer] = ram_idx_[in];
        ram_idx_[in] = -1;
    }
}

void ExpertTier::apply(const PeerMove& m) {
    const size_t base = (size_t) m.layer * n_expert_;
    core::PeerExperts* p = peer_for(m.layer);   // m.slot is that card's local slot
    p->admit(m.layer, m.out, -1);
    p->admit(m.layer, m.in, m.slot);
    ram_idx_[base + m.out] = m.place;
    spare_[(size_t) m.layer] = ram_idx_[base + m.in];
    ram_idx_[base + m.in] = -1;
}

// A layer's copies were claimed after its last lookup and before anything else reads its places: they land together,
// in the order they were claimed, before its next lookup.
void ExpertTier::land(int layer) {
    issue(layer, 0);
    size_t keep = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
        const Pending p = pending_[i];
        if (p.layer != layer) {
            pending_[keep++] = p;
            continue;
        }
        if (const cudaError_t ce = cudaEventSynchronize(events_[(size_t) p.event]); ce != cudaSuccess)
            basalt::log::error(s_.area.c_str(), "tier land layer %d sync: %s", layer, cudaGetErrorString(ce));
        apply(p);
    }
    pending_.resize(keep);
    if (peer_moves_.empty()) return;
    std::string err;
    keep = 0;
    for (size_t i = 0; i < peer_moves_.size(); ++i) {
        const PeerMove m = peer_moves_[i];
        if (m.layer != layer) {
            peer_moves_[keep++] = m;
            continue;
        }
        peer_for(m.layer)->read_until(m.in_mark, err);
        apply(m);
    }
    peer_moves_.resize(keep);
}

void ExpertTier::land_all() {
    issue(-1, kAllPieces);
    for (const Pending& p : pending_) {
        if (const cudaError_t ce = cudaEventSynchronize(events_[(size_t) p.event]); ce != cudaSuccess)
            basalt::log::error(s_.area.c_str(), "tier land_all layer %d sync: %s", p.layer, cudaGetErrorString(ce));
        apply(p);
    }
    pending_.clear();
    claims_ = 0;
    if (peer_moves_.empty()) return;
    std::string err;
    for (const auto& p : peers_) p->read_wait(err);
    for (const PeerMove& m : peer_moves_) apply(m);
    peer_moves_.clear();
}

void ExpertTier::settle() {
    std::string err;
    if (!await_reads(err)) basalt::log::error(s_.area.c_str(), "%s", err.c_str());
    land_all();
    for (const auto& [name, st] : std::array<std::pair<const char*, cudaStream_t>, 3>{
             {{"copy", copy_stream_}, {"out", out_stream_}, {"fetch", fetch_stream_}}}) {
        if (const cudaError_t ce = cudaStreamSynchronize(st); ce != cudaSuccess)
            basalt::log::error(s_.area.c_str(), "tier settle %s sync: %s", name, cudaGetErrorString(ce));
    }
}

void ExpertTier::begin_token(int n_tokens) {
    if (!uses_.empty() && core::policy::use_half_life(token_, (uint64_t) n_tokens))
        for (uint16_t& u : uses_) u >>= 1;
    token_ += (uint64_t) n_tokens;
    claims_ = 0;
    claim_cap_ = s_.claims_per_row > 0 ? s_.claims_per_row * std::max(n_tokens, 1) : (int) kMaxPending;
}

bool ExpertTier::lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err) {
    land(layer);
    const uint64_t now = ++tick_;
    lk.k = k;
    if (reads_.is_open() && !read_misses(layer, ids, k, lk.reading, err)) return false;
    for (int i = 0; i < k; ++i) {
        const int e = ids[i];
        lk.experts[i] = e;
        lk.again[i] = false;
        lk.fetched[i] = false;
        lk.dropped[i] = e == -1 && route_tail_skip_ > 0;
        if (lk.dropped[i]) {
            lk.dev[i] = lk.peer[i] = lk.host[i] = nullptr;
            continue;
        }
        if (e < 0 || e >= n_expert_) {
            err = s_.name + ": router returned a bad expert id";
            return false;
        }
        const size_t idx = (size_t) layer * n_expert_ + e;
        stamp_[idx] = now;
        const int32_t s = res_[idx];
        lk.peer[i] = nullptr;
        if (s >= 0) {
            lk.dev[i] = slot_ptr(s);
            lk.host[i] = nullptr;
            ++hits_;
        } else if (!peers_.empty() && (lk.peer[i] = peer_for(layer)->slot_ptr(layer, e))) {
            lk.dev[i] = nullptr;
            lk.host[i] = nullptr;
            ++peer_hits_;
        } else {
            lk.dev[i] = nullptr;
            lk.host[i] = blob_of(layer, e);
            lk.again[i] = s_.admit_uses > 0 ? core::policy::count_use(uses_[idx], s_.admit_uses)
                                            : core::policy::repeat_miss(last_miss_[idx], token_, core::policy::kPromoteWindow);
            ++(pinned(layer, e) && !lk.reading[i] ? ram_hits_ : file_hits_);
            last_miss_[idx] = token_ + 1;
            ++cpu_;
        }
    }
    return true;
}

ExpertTier::Source ExpertTier::locate(int layer, int e) const {
    Source r;
    const int32_t s = res_[(size_t) layer * n_expert_ + e];
    if (s >= 0) {
        r.dev = slot_ptr(s);
        return r;
    }
    if (!peers_.empty()) r.peer = peer_for(layer)->slot_ptr(layer, e);
    if (!r.peer && pinned(layer, e)) r.pinned = blob_of(layer, e);
    return r;
}

void ExpertTier::start_cpu(int layer, const float* x, const Lookup& lk, float* rows, bool reading) {
    const kernels::cpu::NativeFmt& f = fmt_[(size_t) layer];
    int n = 0;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.peer[i] || lk.dropped[i] || lk.reading[i] != reading) continue;
        cpu_jobs_[n] = kernels::cpu::ExpertJobMulti{};
        cpu_jobs_[n].blob = lk.host[i];
        cpu_jobs_[n].nt = 1;
        cpu_jobs_[n].nact[0] = act_;
        cpu_jobs_[n].out[0] = rows + (size_t) i * (size_t) s_.n_embd;
        ++n;
    }
    if (!n) return;
    kernels::cpu::native_quant_act(f, x, act_);
    pool_->start_split_multi_native(f, cpu_jobs_, n);
}

void ExpertTier::start_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows, bool reading) {
    const kernels::cpu::NativeFmt& f = fmt_[(size_t) layer];
    const size_t n_embd = (size_t) s_.n_embd;
    bool quantized[kMaxWindow] = {};
    int n = 0;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.peer[i] || lk.dropped[i] || lk.reading[i] != reading) continue;
        cpu_jobs_[n] = kernels::cpu::ExpertJobMulti{};
        cpu_jobs_[n].blob = lk.host[i];
        cpu_jobs_[n].nt = use[i].nt;
        for (int j = 0; j < use[i].nt; ++j) {
            const int t = use[i].tok[j];
            if (!quantized[t]) {
                kernels::cpu::native_quant_act(f, x + (size_t) t * n_embd, act_ + (size_t) t * kernels::cpu::kNativeActBytes);
                quantized[t] = true;
            }
            cpu_jobs_[n].nact[j] = act_ + (size_t) t * kernels::cpu::kNativeActBytes;
            cpu_jobs_[n].out[j] = rows + (size_t) use[i].row[j] * n_embd;
        }
        ++n;
    }
    if (n) pool_->start_split_multi_native(f, cpu_jobs_, n);
}

void ExpertTier::finish_cpu() { pool_->finish_split_multi_native(); }

int32_t ExpertTier::oldest_slot(int layer, bool homes_too, uint64_t& best) const {
    int32_t slot = -1;
    best = ~0ull;
    auto scan = [&](int64_t lo, int64_t n) {
        for (int64_t s = lo; s < lo + n; ++s) {
            if (busy_[(size_t) s]) continue;
            const int32_t o = owner_[(size_t) s];
            const uint64_t st = o < 0 ? 0 : stamp_[(size_t) layer * n_expert_ + o];
            if (st < best) {
                best = st;
                slot = (int32_t) s;
            }
        }
    };
    if (homes_.main == 0 || (homes_too && spare_[(size_t) layer] >= 0))
        scan(base_lo_[(size_t) layer], live_n(layer));
    if (ext_per_layer_) scan(base_slots() + (int64_t) layer * ext_per_layer_, ext_per_layer_);
    return slot;
}

bool ExpertTier::move_to_peer(int layer, int e, const uint8_t* src) {
    if (homes_.peer == 0 || spare_[(size_t) layer] < 0) return false;
    const size_t base = (size_t) layer * n_expert_;
    int32_t slot = -1;
    uint64_t oldest = ~0ull;
    for (int64_t s = (int64_t) layer * homes_.peer; s < (int64_t) (layer + 1) * homes_.peer; ++s) {
        const uint64_t st = stamp_[base + peer_owner_[(size_t) s]];
        if (st != tick_ && st < oldest) {   // not in use by this very lookup
            oldest = st;
            slot = (int32_t) s;
        }
    }
    uint64_t on_main = ~0ull;
    oldest_slot(layer, /*homes_too=*/true, on_main);
    // the second card computes an expert ~3.6x slower than the main card: a promotion goes there only when what it
    // replaces has gone unused well longer than the main card's oldest
    if (slot < 0 || oldest + core::policy::kPeerMoveAge >= on_main) return false;
    const int32_t place = spare_[(size_t) layer];
    spare_[(size_t) layer] = -1;
    const uint64_t bytes = s_.layers[(size_t) layer].blob_bytes;
    // `slot` is global (peer_owner_'s); the card is told its own number for it
    const int32_t local = core::policy::local_slot(blocks_, homes_.peer, layer, (int) (slot - (int64_t) layer * homes_.peer));
    peer_moves_.push_back({layer, e, peer_owner_[(size_t) slot], local, place, ram_[(size_t) layer] + (size_t) place * bytes, src, bytes});
    peer_owner_[(size_t) slot] = e;
    ++promoted_;
    return true;
}

int32_t ExpertTier::claim(int layer, int e, const uint8_t* src, cudaStream_t stream, bool homes_too, bool whole, bool later) {
    if (claims_ >= claim_cap_ || free_events_.empty()) {
        ++dropped_;
        return -1;
    }
    uint64_t best = ~0ull;
    const int32_t slot = oldest_slot(layer, homes_too, best);
    if (slot < 0 || best == tick_) return -1;   // every slot is in use by this very lookup
    const int32_t o = owner_[(size_t) slot];
    if (o >= 0) res_[(size_t) layer * n_expert_ + o] = -1;
    owner_[(size_t) slot] = e;
    busy_[(size_t) slot] = 1;
    stamp_[(size_t) layer * n_expert_ + e] = tick_;
    const uint64_t bytes = s_.layers[(size_t) layer].blob_bytes;
    uint8_t* dst = const_cast<uint8_t*>(slot_ptr(slot));
    const bool home = homes_.main > 0 && slot < base_slots();
    int32_t place = -1;
    const uint64_t piece = whole ? bytes : core::policy::kPieceBytes;
    const auto pieces = [&](uint8_t* to, const uint8_t* from, cudaMemcpyKind kind) {
        for (uint64_t off = 0; off < bytes; off += piece)
            cudaMemcpyAsync(to + off, from + off, std::min(piece, bytes - off), kind, stream);
    };
    if (home) {   // the home's bytes first: the stream keeps the order
        place = spare_[(size_t) layer];
        spare_[(size_t) layer] = -1;
    }
    const int ev = free_events_.back();
    free_events_.pop_back();
    ++promoted_;
    ++claims_;
    if (later) {
        Queued q{{layer, e, slot, ev, home ? o : -1, place}, dst, src, home ? ram_[(size_t) layer] + (size_t) place * bytes : nullptr, bytes};
        queue_.push_back(q);
        return slot;
    }
    if (home) pieces(ram_[(size_t) layer] + (size_t) place * bytes, dst, cudaMemcpyDeviceToHost);
    pieces(dst, src, cudaMemcpyHostToDevice);
    cudaEventRecord(events_[(size_t) ev], stream);
    pending_.push_back({layer, e, slot, ev, home ? o : -1, place});
    return slot;
}

void ExpertTier::mark_gap(cudaStream_t main) {
    bool left = !queue_.empty();
    for (const PeerMove& m : peer_moves_) left = left || m.in_off < m.bytes;
    gap_marked_ = left && cudaEventRecord(gap_ev_, main) == cudaSuccess;
}

void ExpertTier::issue_peer(int layer, int pieces) {
    const uint64_t P = core::policy::kPieceBytes;
    // the slot's bytes out first, then the incoming expert's in: the stream keeps the order
    const auto piece = [&](PeerMove& m) {
        if (m.out_off < m.bytes) {
            peer_for(m.layer)->slot_out(m.slot, m.to, m.out_off, std::min(P, m.bytes - m.out_off));
            m.out_off += P;
            return;
        }
        peer_for(m.layer)->slot_in(m.slot, m.from, m.in_off, std::min(P, m.bytes - m.in_off));
        m.in_off += P;
        if (m.in_off >= m.bytes) m.in_mark = peer_for(m.layer)->mark_in();   // that card's own mark
    };
    for (PeerMove& m : peer_moves_)
        if (m.layer == layer)
            while (m.in_off < m.bytes) piece(m);
    for (size_t i = 0; i < peer_moves_.size() && pieces > 0; ++i)
        for (PeerMove& m = peer_moves_[i]; m.in_off < m.bytes && pieces > 0; --pieces) piece(m);
}

void ExpertTier::issue(int layer, int pieces) {
    if (gap_marked_) {
        cudaStreamWaitEvent(copy_stream_, gap_ev_, 0);
        cudaStreamWaitEvent(out_stream_, gap_ev_, 0);
        for (size_t c = 0; c < peers_.size(); ++c)   // every card with work queued waits for the gap (an event of any card is legal)
            for (const PeerMove& m : peer_moves_)
                if (layer_card_[(size_t) m.layer] == (int) c) {
                    peers_[c]->copies_after(gap_ev_);
                    break;
                }
    }
    gap_marked_ = false;
    // the second card's own bus: what one head leaves it (the rest at the window's end)
    if (!peer_moves_.empty()) issue_peer(layer, pieces == kAllPieces ? pieces : std::min(pieces, core::policy::kPeerGapPieces));
    if (queue_.empty()) return;
    const uint64_t P = core::policy::kPieceBytes;
    // A copy engine takes copies in the order they were queued, whatever their stream: a copy queued later (a layer's
    // bus fetch, a window's rows, the drafter's result) would wait for every piece queued before it. A small operation
    // between two pieces lets it go first, after one piece at most. The home's bytes go out on a stream of their own, so
    // the pieces in do not wait for them, nor they for the copies in.
    const auto out = [&](Queued& q) {   // every piece out of the home q replaces, and the event behind them
        if (!q.home_to || q.home_off >= q.bytes) return;
        for (; q.home_off < q.bytes; q.home_off += P) {
            cudaMemcpyAsync(q.home_to + q.home_off, q.dst + q.home_off, std::min(P, q.bytes - q.home_off), cudaMemcpyDeviceToHost,
                            out_stream_);
            cudaMemsetAsync(q.dst + q.home_off, 0, 4, out_stream_);   // out already; the pieces in overwrite it
        }
        cudaEventRecord(out_events_[(size_t) q.p.event], out_stream_);
    };
    const auto piece = [&](Queued& q) {   // one piece in, the first once the home's bytes are out
        if (q.home_to && q.off == 0) {
            out(q);
            cudaStreamWaitEvent(copy_stream_, out_events_[(size_t) q.p.event], 0);
        }
        cudaMemsetAsync(q.dst + q.off, 0, 4, copy_stream_);   // the piece overwrites it
        cudaMemcpyAsync(q.dst + q.off, q.src + q.off, std::min(P, q.bytes - q.off), cudaMemcpyHostToDevice, copy_stream_);
        q.off += P;
        if (q.off >= q.bytes) {
            cudaEventRecord(events_[(size_t) q.p.event], copy_stream_);
            pending_.push_back(q.p);
        }
    };
    // the layer's own copies whole: its lookup is next
    for (Queued& q : queue_)
        if (q.p.layer == layer)
            while (q.off < q.bytes) piece(q);
    for (size_t i = 0; i < queue_.size() && pieces > 0; ++i) {
        if (queue_[i].off < queue_[i].bytes)   // the next home goes out while this promotion comes in
            for (size_t j = i + 1; j < queue_.size(); ++j)
                if (queue_[j].home_to && queue_[j].off == 0) {
                    out(queue_[j]);
                    break;
                }
        for (Queued& q = queue_[i]; q.off < q.bytes && pieces > 0; --pieces) piece(q);
    }
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const Queued& q) { return q.off >= q.bytes; }), queue_.end());
}

void ExpertTier::pump(int layer, int pieces) {
    issue(layer, pieces);
    if (layer >= 0) land(layer);
}

int ExpertTier::fetch(int layer, Lookup& lk) {
    // with the RAM tier caching the file the bus stays with the reads and the promotions
    if (s_.diag_tier_frozen || reads_.is_open()) return 0;
    int np = 0, nf = 0;
    // with an admission threshold only an expert that earned its slot goes over the bus; the others stay with the CPU
    const auto eligible = [&](int i) { return !lk.dropped[i] && pinned(layer, lk.experts[i]) && (s_.admit_uses == 0 || lk.again[i]); };
    for (int i = 0; i < lk.k; ++i)
        if (!lk.dev[i] && !lk.peer[i] && !lk.dropped[i]) ++(eligible(i) ? np : nf);
    int left = core::policy::bus_count(np, nf);
    // A second card fetches some over its own link into a scratch slot and computes them there. Their main card slots are
    // claimed all the same, filled behind the window's work as a promotion is, so every expert's place is the one the
    // main card's link alone would have given it.
    core::PeerExperts* pc = peer_for(layer);   // this layer's second card: it computes the experts it fetches
    int far = pc ? std::min(core::policy::peer_bus_count(np, nf), pc->scratch_free()) : 0;
    const uint64_t bytes = s_.layers[(size_t) layer].blob_bytes;
    int started = 0;
    for (int i = lk.k - 1; i >= 0 && left > 0; --i) {
        if (lk.dev[i] || !eligible(i)) continue;   // pinned sources only
        --left;
        const int32_t slot = far > 0 ? claim(layer, lk.experts[i], lk.host[i], copy_stream_, false, false, /*later=*/true)
                                     : claim(layer, lk.experts[i], lk.host[i], fetch_stream_, false, /*whole=*/true);
        if (slot < 0) continue;
        if (far > 0) {
            lk.peer[i] = pc->fetch(lk.experts[i], lk.host[i], bytes);
            --far;
        } else {
            lk.dev[i] = slot_ptr(slot);
            lk.host[i] = nullptr;
            ++started;
        }
        lk.fetched[i] = true;
        --cpu_;
        ++fetched_;
    }
    if (started) cudaEventRecord(fetch_ev_, fetch_stream_);
    return started;
}

void ExpertTier::await_fetch(cudaStream_t main) { cudaStreamWaitEvent(main, fetch_ev_, 0); }

void ExpertTier::promote(int layer, const Lookup& lk) {
    if (s_.diag_tier_frozen) return;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.fetched[i] || lk.dropped[i] || !pinned(layer, lk.experts[i])) continue;   // pinned sources only
        // with the RAM tier caching the file, the experts the CPU keeps computing; else a repeat miss
        if (reads_.is_open() && s_.admit_uses == 0 ? !core::policy::count_use(uses_[(size_t) layer * n_expert_ + lk.experts[i]]) : !lk.again[i]) continue;
        if (!move_to_peer(layer, lk.experts[i], lk.host[i]))
            claim(layer, lk.experts[i], lk.host[i], copy_stream_, /*homes_too=*/true, /*whole=*/false, /*later=*/true);
    }
}

void ExpertTier::recall(int layer, int64_t lo, int64_t n) {
    if (recall_.empty()) recall_.resize((size_t) n_layers_);
    std::vector<int32_t>& r = recall_[(size_t) layer];
    for (int64_t s = lo; s < lo + n; ++s) {
        const int32_t o = owner_[(size_t) s];
        if (o >= 0 && std::find(r.begin(), r.end(), o) == r.end()) r.push_back(o);
    }
    const size_t base = (size_t) layer * n_expert_;
    std::stable_sort(r.begin(), r.end(), [&](int32_t a, int32_t b) { return stamp_[base + a] > stamp_[base + b]; });
}

bool ExpertTier::recalled() const {
    for (const auto& r : recall_)
        if (!r.empty()) return true;
    return false;
}

void ExpertTier::restore(bool ranked_too) {
    if (recall_.empty()) recall_.resize((size_t) n_layers_);
    cudaDeviceSynchronize();   // a slot arena opened or mapped again is zeroed off the copy stream: the copies go after it
    const int64_t nb = base_slots(), promoted = promoted_;
    for (int l = 0; l < n_layers_; ++l) {
        const int64_t lo = base_lo_[(size_t) l], xlo = nb + (int64_t) l * ext_per_layer_;
        int64_t empty = 0;
        for (int64_t s = lo; s < lo + live_n(l); ++s) empty += owner_[(size_t) s] < 0;
        for (int64_t s = xlo; ext_per_layer_ && s < xlo + ext_per_layer_; ++s) empty += owner_[(size_t) s] < 0;
        const auto held = [&](int e) {   // resident, or claimed and not landed yet
            if (res_[(size_t) l * n_expert_ + e] >= 0) return true;
            for (int64_t s = lo; s < lo + live_n(l); ++s)
                if (owner_[(size_t) s] == e) return true;
            for (int64_t s = xlo; ext_per_layer_ && s < xlo + ext_per_layer_; ++s)
                if (owner_[(size_t) s] == e) return true;
            return false;
        };
        const auto place = [&](int e) {
            if (!pinned(l, e) || held(e)) return true;
            ++tick_;
            if (claim(l, e, blob_of(l, e), copy_stream_) < 0) {
                if (const cudaError_t ce = cudaStreamSynchronize(copy_stream_); ce != cudaSuccess)
                    basalt::log::error(s_.area.c_str(), "tier restore sync: %s", cudaGetErrorString(ce));
                land_all();
                ++tick_;
                if (claim(l, e, blob_of(l, e), copy_stream_) < 0) return false;
            }
            --empty;
            return true;
        };
        std::vector<int32_t>& r = recall_[(size_t) l];
        size_t used = 0;
        for (; used < r.size() && empty > 0; ++used)
            if (!place(r[used])) break;
        r.erase(r.begin(), r.begin() + (ptrdiff_t) used);
        for (int k = 0; ranked_too && k < n_expert_ && empty > 0; ++k)
            if (!place(ranked(l, k))) break;
    }
    if (const cudaError_t ce = cudaStreamSynchronize(copy_stream_); ce != cudaSuccess)
        basalt::log::error(s_.area.c_str(), "tier restore sync: %s", cudaGetErrorString(ce));
    land_all();
    promoted_ = promoted;
}

void ExpertTier::shrink() {
    if (ext_per_layer_ == 0) return;
    settle();
    const int64_t nb = base_slots();
    for (int l = 0; l < n_layers_; ++l) recall(l, nb + (int64_t) l * ext_per_layer_, ext_per_layer_);
    for (int64_t s = nb; s < (int64_t) owner_.size(); ++s) {
        const int32_t o = owner_[(size_t) s];
        if (o >= 0) res_[(size_t) ((s - nb) / ext_per_layer_) * n_expert_ + o] = -1;
    }
    owner_.resize((size_t) nb);
    busy_.resize((size_t) nb);
    ext_.close();
    ext_per_layer_ = 0;
}

void ExpertTier::lend_all() {
    shrink();
    if (lent_per_layer_ > 0 || !cache_.valid() || homes_.main > 0) return;
    settle();
    for (int l = 0; l < n_layers_; ++l) recall(l, base_lo_[(size_t) l], live_n(l));
    std::fill(res_.begin(), res_.end(), -1);
    owner_.clear();
    busy_.clear();
    lent_per_layer_ = per_layer_;
    planned_per_layer_ = std::max(planned_per_layer_, per_layer_);
    per_layer_ = 0;
    set_base(0);
    phys_.clear();
    live_.clear();
    cache_.close();
}

bool ExpertTier::reopen_base(int64_t n, std::string& err) {
    std::string e;
    if (!open_base(n, e)) {
        set_base(0);
        phys_.clear();
        live_.clear();
        err = s_.name + ": the VRAM expert tier cannot take back its " + std::to_string(n) + " slots per layer: " + e;
        return false;
    }
    per_layer_ = n;
    lent_per_layer_ = 0;
    owner_.assign((size_t) base_slots(), -1);
    busy_.assign((size_t) base_slots(), 0);
    return true;
}

bool ExpertTier::open_base(int64_t per_layer, std::string& err) {
    set_base(per_layer);
    const int64_t nb = base_slots();
    std::vector<int32_t> order((size_t) nb);   // [place in the arena] its slot
    std::vector<int> layer_of((size_t) nb);
    std::vector<int64_t> k_of((size_t) nb);
    for (int l = 0; l < n_layers_; ++l)
        for (int64_t k = 0; k < base_n(l); ++k) {
            layer_of[(size_t) (base_lo_[(size_t) l] + k)] = l;
            k_of[(size_t) (base_lo_[(size_t) l] + k)] = k;
        }
    for (int64_t i = 0; i < nb; ++i) order[(size_t) i] = (int32_t) i;
    phys_.clear();
    live_.clear();
    const bool seg = s_.lend_segment > 0 && homes_.main == 0 && nb > 0 && core::vmm::supported();
    if (seg) {
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            const int la = layer_of[(size_t) a], lb = layer_of[(size_t) b];
            const int64_t x = (2 * k_of[(size_t) a] + 1) * base_n(lb), y = (2 * k_of[(size_t) b] + 1) * base_n(la);
            return x != y ? x < y : la < lb;
        });
        phys_.resize((size_t) nb);
        for (int64_t i = 0; i < nb; ++i) phys_[(size_t) order[(size_t) i]] = (int32_t) i;
    }
    std::vector<int64_t> slot_bytes;
    for (int32_t s : order) slot_bytes.push_back((int64_t) s_.layers[(size_t) layer_of[(size_t) s]].blob_bytes);
    cache_.set_segment_bytes(seg ? (int64_t) s_.lend_segment : 0);
    if (!cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) {
        phys_.clear();
        return false;
    }
    if (seg) count_live();
    return true;
}

// Each layer's base slots inside the mapped part of the arena: a prefix, since a layer's slots lie deeper the later they are.
void ExpertTier::count_live() {
    live_.assign((size_t) n_layers_, 0);
    const int64_t mapped = cache_.slots();
    for (int l = 0; l < n_layers_; ++l) {
        int64_t k = 0;
        while (k < base_n(l) && phys_[(size_t) (base_lo_[(size_t) l] + k)] < mapped) ++k;
        live_[(size_t) l] = k;
    }
}

int64_t ExpertTier::live_base() const {
    if (live_.empty()) return base_slots();
    int64_t n = 0;
    for (int64_t v : live_) n += v;
    return n;
}

void ExpertTier::keep_hottest(const std::vector<int64_t>& keep) {
    const int64_t nb = base_slots();
    std::vector<int32_t> held;
    std::vector<uint8_t> hot((size_t) n_expert_);
    for (int l = 0; l < n_layers_; ++l) {
        const size_t base = (size_t) l * n_expert_;
        const int64_t lo = base_lo_[(size_t) l], kept = std::min(keep[(size_t) l], live_n(l));
        // the layer's slots holding an expert, base then beside, the most recently used first
        held.clear();
        for (int64_t s = lo; s < lo + live_n(l); ++s)
            if (owner_[(size_t) s] >= 0) held.push_back((int32_t) s);
        for (int64_t s = nb + (int64_t) l * ext_per_layer_; ext_per_layer_ && s < nb + (int64_t) (l + 1) * ext_per_layer_; ++s)
            if (owner_[(size_t) s] >= 0) held.push_back((int32_t) s);
        std::stable_sort(held.begin(), held.end(), [&](int32_t a, int32_t b) {
            const int ea = owner_[(size_t) a], eb = owner_[(size_t) b];
            const uint64_t sa = stamp_[base + ea], sb = stamp_[base + eb];
            return sa != sb ? sa > sb : ea < eb;
        });
        const size_t nh = std::min<size_t>(held.size(), (size_t) kept);
        std::fill(hot.begin(), hot.end(), 0);
        for (size_t i = 0; i < nh; ++i) hot[(size_t) owner_[(size_t) held[i]]] = 1;
        // the hot ones outside the kept slots move into kept slots that hold none of them
        int64_t d = lo;
        for (size_t i = 0; i < nh; ++i) {
            const int32_t src = held[i];
            if (src >= lo && src < lo + kept) continue;
            while (owner_[(size_t) d] >= 0 && hot[(size_t) owner_[(size_t) d]]) ++d;
            const int e = owner_[(size_t) src];
            if (owner_[(size_t) d] >= 0) res_[base + owner_[(size_t) d]] = -1;
            cudaMemcpyAsync(const_cast<uint8_t*>(slot_ptr((int32_t) d)), slot_ptr(src), s_.layers[(size_t) l].blob_bytes, cudaMemcpyDeviceToDevice,
                            copy_stream_);
            owner_[(size_t) d] = e;
            res_[base + e] = (int32_t) d;
            owner_[(size_t) src] = -1;
            ++d;
        }
        // the rest past the kept slots leave the card
        recall(l, lo + kept, live_n(l) - kept);
        if (ext_per_layer_) recall(l, nb + (int64_t) l * ext_per_layer_, ext_per_layer_);
        auto drop = [&](int64_t s) {
            const int32_t o = owner_[(size_t) s];
            if (o >= 0 && res_[base + o] == (int32_t) s) res_[base + o] = -1;
            owner_[(size_t) s] = -1;
        };
        for (int64_t s = lo + kept; s < lo + live_n(l); ++s) drop(s);
        for (int64_t s = nb + (int64_t) l * ext_per_layer_; ext_per_layer_ && s < nb + (int64_t) (l + 1) * ext_per_layer_; ++s) drop(s);
    }
    if (const cudaError_t ce = cudaStreamSynchronize(copy_stream_); ce != cudaSuccess)
        basalt::log::error(s_.area.c_str(), "tier keep sync: %s", cudaGetErrorString(ce));
}

bool ExpertTier::lend(uint64_t bytes) {
    if (!partial_lend() || lent_per_layer_ > 0 || !cache_.valid()) return false;
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) return false;
    if (free_b >= bytes + free_floor_) return true;
    const uint64_t miss = bytes + free_floor_ - free_b;
    const uint64_t beside = ext_per_layer_ ? (uint64_t) ext_.bytes() : 0;
    const int64_t mapped = cache_.mapped_bytes();
    int64_t keep_b = mapped;
    if (miss > beside) {   // whole segments of the arena's tail
        const int64_t more = (int64_t) (miss - beside), seg = cache_.segment_bytes();
        if (more >= mapped || seg <= 0) return false;
        keep_b = (mapped - more) / seg * seg;
    }
    std::vector<int64_t> keep((size_t) n_layers_);
    const int64_t kept_slots = cache_.slots_within(keep_b);
    for (int l = 0; l < n_layers_; ++l) {
        int64_t k = 0;
        while (k < live_n(l) && phys_[(size_t) (base_lo_[(size_t) l] + k)] < kept_slots) ++k;
        keep[(size_t) l] = k;
    }
    settle();
    const int64_t had = slots();
    keep_hottest(keep);
    shrink();
    if (keep_b < mapped) {
        std::string e;
        const bool ok = cache_.shrink(keep_b, e);
        count_live();
        if (!ok) {
            basalt::log::warn(s_.area.c_str(), "VRAM expert tier could not lend its slots: %s", e.c_str());
            return false;
        }
    }
    basalt::log::info(s_.area.c_str(), "VRAM expert tier lends %.2f GiB: %lld of its %lld slots stay (their most recent experts)",
                      (double) (beside + (uint64_t) (mapped - cache_.mapped_bytes())) / 1073741824.0, (long long) slots(), (long long) had);
    return true;
}

// The slots a prompt borrowed come back at the size planned at the start, or as many as the card holds above the tier's
// room and the reserve when what the prompt left (caches that grew) takes some of theirs; a later reclaim opens the
// planned size again once it fits.
bool ExpertTier::reclaim(std::string& err, bool restore_lent) {
    if (partial_lend() && lent_per_layer_ == 0 && cache_.mapped_bytes() < cache_.full_bytes()) {   // a bounded lend: map the tail again
        uint64_t free_b = 0, total_b = 0;
        std::string why;
        if (!core::device_memory(-1, free_b, total_b, why)) {
            err = s_.name + ": cannot query the free device memory: " + why;
            return false;
        }
        const int64_t mapped = cache_.mapped_bytes(), lent = cache_.full_bytes() - mapped;
        const bool all = free_b >= free_floor_ + (uint64_t) lent;
        const int64_t room = all ? lent : free_b > reserve_ + free_floor_ ? (int64_t) (free_b - reserve_ - free_floor_) : 0;
        std::string e;
        const bool ok = cache_.grow(all ? INT64_MAX : mapped + room, e);
        count_live();
        if (!ok || !all)
            basalt::log::warn(s_.area.c_str(), "VRAM expert tier takes back %.2f of its %.2f GiB lent (%lld MiB free)%s%s",
                              (double) (cache_.mapped_bytes() - mapped) / 1073741824.0, (double) lent / 1073741824.0, (long long) (free_b >> 20),
                              ok ? "" : ": ", e.c_str());
        if (restore_lent) restore(false);
        return true;
    }
    if (lent_per_layer_ == 0) return true;
    const int64_t want = std::max(lent_per_layer_, planned_per_layer_);
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = s_.name + ": cannot query the free device memory: " + why;
        return false;
    }
    const int64_t n = core::policy::reclaim_slots(free_b, reserve_, free_floor_, one_each_, want);
    if (n < 1) {
        err = s_.name + ": the VRAM expert tier cannot take back any of its " + std::to_string(want) + " slots per layer (" +
              std::to_string(free_b >> 20) + " MiB free)";
        return false;
    }
    if (!reopen_base(n, err)) return false;
    if (restore_lent) restore(false);
    if (n < want)
        basalt::log::warn(s_.area.c_str(), "VRAM expert tier takes back %lld of its %lld slots per layer (%lld MiB free)", (long long) n,
                          (long long) want, (long long) (free_b >> 20));
    return true;
}

// The start's one plan of the slots beside the base arena: read once, after everything else the start allocates, then
// held for good (a prompt's shrink and grow open the same size again). An exact --expert-cache the card cannot hold
// stops the start.
bool ExpertTier::plan_ext(std::string& err) {
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = s_.name + ": cannot query the free device memory: " + why;
        return false;
    }
    ext_n_ = per_layer_ >= n_expert_ || !cache_.valid()
        ? 0 : core::policy::grow_slots(free_b, reserve_ + modules_, free_floor_, one_each_, per_layer_, n_expert_, cap_per_layer_);
    if (!core::policy::holds_exact(exact_per_layer_, per_layer_, ext_n_)) {
        err = core::exact_slots_refusal(s_.expert_cache, per_layer_ + ext_n_, " per layer", free_b,
                                        s_.vram_reserve_mib);
        return false;
    }
    grow();
    return true;
}

// An explicit new --vram-reserve-mib: the one place the size is planned again after the start.
bool ExpertTier::replan(int64_t mib, std::string& err) {
    set_free_floor(mib);
    lend_all();
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = s_.name + ": cannot query the free device memory: " + why;
        return false;
    }
    if (lent_per_layer_ > 0) {
        const int64_t n = core::policy::reclaim_slots(free_b, reserve_, free_floor_, one_each_, lent_per_layer_);
        if (n >= 1 && !reopen_base(n, err)) return false;   // nothing fits above the new reserve: the slots stay lent
        if (n >= 1) planned_per_layer_ = n;
        if (!core::device_memory(-1, free_b, total_b, why)) {
            err = s_.name + ": cannot query the free device memory: " + why;
            return false;
        }
    }
    ext_n_ = per_layer_ >= n_expert_ || !cache_.valid()
        ? 0 : core::policy::grow_slots(free_b, reserve_ + (token_ > 0 ? 0 : modules_), free_floor_, one_each_, per_layer_, n_expert_, cap_per_layer_);
    grow();
    return true;
}

void ExpertTier::grow() {
    if (ext_per_layer_ > 0 || ext_n_ < 1 || per_layer_ >= n_expert_ || !cache_.valid()) return;
    const int64_t n = ext_n_;
    std::vector<int64_t> slot_bytes;
    for (int l = 0; l < n_layers_; ++l)
        for (int64_t s = 0; s < n; ++s) slot_bytes.push_back((int64_t) s_.layers[(size_t) l].blob_bytes);
    std::string err;
    if (!ext_.open_sized(slot_bytes, n_layers_, n_expert_, err)) {
        basalt::log::warn(s_.area.c_str(), "VRAM expert tier cannot take back its %lld slots per layer beside the prompt workspaces: %s",
                          (long long) n, err.c_str());
        return;
    }
    ext_per_layer_ = n;
    owner_.resize((size_t) slot_total(), -1);
    busy_.resize((size_t) slot_total(), 0);
    basalt::log::info(s_.area.c_str(), "VRAM expert tier %lld of %d experts per layer (%lld beside the prompt workspaces)",
                 (long long) (per_layer_ + n), n_expert_, (long long) per_layer_);
    cudaDeviceSynchronize();   // the arena's zeroing is not ordered against the copy stream
    if (recalled()) {   // what a shrink or lend took off the card goes back first
        restore(true);
        return;
    }
    const int64_t promoted = promoted_;
    seed_copies(n);
    promoted_ = promoted;
}

bool ExpertTier::read_misses(int layer, const int32_t* ids, int k, bool* reading, std::string& err) {
    if (!await_reads(err)) return false;
    const ExpertTierSpec::Layer& el = s_.layers[(size_t) layer];
    const size_t base = (size_t) layer * n_expert_;
    blobs_.clear();
    for (int i = 0; i < k; ++i) {
        reading[i] = false;
        const int e = ids[i];
        if (e < 0 || e >= n_expert_ || res_[base + e] >= 0 || pinned(layer, e) || (!peers_.empty() && peer_for(layer)->slot_ptr(layer, e))) {
            for (int j = 0; j < i; ++j) reading[i] = reading[i] || (reading[j] && ids[j] == e);
            continue;
        }
        const int v = core::policy::pick_victim(&ram_idx_[base], &stamp_[base], n_expert_, ids, k);
        if (v < 0) continue;   // every place holds one of `ids`: the CPU reads this one from the mapped file
        ram_idx_[base + e] = ram_idx_[base + v];
        ram_idx_[base + v] = -1;
        blobs_.push_back({s_.range.offset + el.offset + (uint64_t) e * el.blob_bytes, el.blob_bytes,
                          ram_[(size_t) layer] + (size_t) ram_idx_[base + e] * el.blob_bytes});
        reading[i] = true;
    }
    if (blobs_.empty()) return true;
    read_ += (int64_t) blobs_.size();
    reads_.start(blobs_);
    return true;
}

bool ExpertTier::await_reads(std::string& err) { return reads_.wait(err); }

void ExpertTier::set_free_floor(int64_t mib) { free_floor_ = mib < 0 ? start_floor_ : (uint64_t) mib << 20; }

void ExpertTier::report(const char* tag) const {
    const int64_t n = hits_ + cpu_ + fetched_ + peer_hits_;
    if (n == 0) return;
    basalt::log::info(s_.area.c_str(), "TIER %s: %lld lookups: %.1f%% VRAM, %.1f%% second card, %.1f%% bus copy, %.1f%% CPU (file %.1f%%); promoted %lld, dropped %lld", tag,
                 (long long) n, 100.0 * (double) hits_ / (double) n, 100.0 * (double) peer_hits_ / (double) n, 100.0 * (double) fetched_ / (double) n,
                 100.0 * (double) cpu_ / (double) n, 100.0 * (double) file_hits_ / (double) n, (long long) promoted_, (long long) dropped_);
    if (reads_.is_open())
        basalt::log::info(s_.area.c_str(), "TIER %s: %lld experts read into the RAM tier (%s)", tag, (long long) read_,
                          reads_.unbuffered() ? "unbuffered" : "through the file cache");
    unsigned long long c[8] = {};
    if (route_counts_ != nullptr && cudaMemcpy(c, route_counts_, sizeof c, cudaMemcpyDeviceToHost) == cudaSuccess)
        for (const std::string& line : route_report(c, route_tail_skip_, route_prior_))
            basalt::log::info(s_.area.c_str(), "TIER %s: %s", tag, line.c_str());
    else
        cudaGetLastError();
}

}  // namespace basalt::core
