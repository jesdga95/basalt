// src/models/deepseek41/tier.cpp - the DeepSeek-V4.1 routed-expert tiers.
#include "basalt/log.hpp"
#include "models/deepseek41/tier.hpp"

#include "basalt/core/load_progress.hpp"
#include <optional>

#include "basalt/core/expert_source.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "models/deepseek41/model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {
namespace {

// Host memory left to the system beside the RAM tier: what serving takes after the start, and the page cache that
// file-held experts and the load read through.
constexpr uint64_t kRamHeadroom = 16ull << 30;
constexpr size_t kMaxPending = 128;              // promotions in flight
constexpr uint64_t kPrereadBytes = 12ull << 30;  // file-backed experts read ahead into the page cache

// The usual routing order of this model's experts: the RAM and VRAM tiers start from the head of it.
constexpr int kRankLayers = 40, kRankExperts = 384;
}  // namespace

ExpertTier::ExpertTier(const DsModel& model) : m_(model) {}

ExpertTier::~ExpertTier() {
    preread_quit_ = true;
    if (preread_.joinable()) preread_.join();
    if (worker_.joinable()) {
        {
            std::lock_guard<std::mutex> g(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }
    report("final");
    if (peer_ && !peer_moves_.empty()) {
        std::string e;
        peer_->read_wait(e);
    }
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (fetch_stream_) cudaStreamSynchronize(fetch_stream_);
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
    if (fetch_ev_) cudaEventDestroy(fetch_ev_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (fetch_stream_) cudaStreamDestroy(fetch_stream_);
    for (size_t l = 0; l < ram_.size(); ++l) {
        if (ram_[l] == nullptr) continue;
        if (l < ram_pins_.size()) core::unpin_ram(ram_[l], ram_pins_[l]);
        platform::free_aligned(ram_[l]);
    }
    platform::free_aligned(act_);
}

bool ExpertTier::open(const Reserve& reserve, float* h_rows, std::string& err) {
    const Geometry& g = m_.geo();
    n_layers_ = g.n_layers;
    n_expert_ = g.n_expert;
    uint64_t one_each = 0, one_each_ram = 0;   // bytes of one slot in every layer, of one RAM place in every layer
    for (const ExpertLayer& e : m_.pack().experts) {
        one_each += (e.blob_bytes + 255) / 256 * 256;
        one_each_ram += e.blob_bytes;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        err = "deepseek41: cannot query the free device memory";
        return false;
    }
    one_each_ = one_each;
    auto slots_in = [&](uint64_t room, uint64_t left_free) {
        return (int) std::min<uint64_t>(room > left_free ? (room - left_free) / one_each : 0, (uint64_t) n_expert_);
    };
    if (!load_rank(err)) return false;
    const uint64_t asked_free = (uint64_t) m_.tiers().vram_reserve_mib << 20;
    int copies = slots_in(free_b, reserve.copies + asked_free);
    int main_homes = slots_in(free_b, reserve.homes + asked_free);
    start_floor_ = free_floor_ = asked_free;
    if (m_.tiers().expert_cache > 0) cap_per_layer_ = m_.tiers().expert_cache / n_layers_;
    if (m_.diag_tier_vram() >= 0) cap_per_layer_ = cap_per_layer_ >= 0 ? std::min<int64_t>(cap_per_layer_, m_.diag_tier_vram()) : m_.diag_tier_vram();
    if (cap_per_layer_ >= 0) {
        copies = (int) std::min<int64_t>(copies, cap_per_layer_);
        main_homes = (int) std::min<int64_t>(main_homes, cap_per_layer_);
    }

    int peer_slots = 0;
    if (m_.peer_device() >= 0) {
        peer_ = std::make_unique<core::PeerExperts>();
        std::string pe;
        if (peer_->open_plain(m_.peer_device(), n_layers_, n_expert_, g.n_embd, g.n_ff_exp, h_rows, pe)) {
            peer_slots = slots_in(peer_->room(), (uint64_t) m_.tiers().peer_reserve_mib << 20);
        } else {
            basalt::log::warn("basalt deepseek41", "second card not used: %s", pe.c_str());
            peer_.reset();
        }
    }

    core::detail::HostMemory hm;
    uint64_t avail = 0;
    if (core::detail::host_available_memory(hm)) avail = hm.available;
    auto ram_in = [&](uint64_t left_free) {
        return (int) std::min<uint64_t>(avail > left_free ? (avail - left_free) / one_each_ram : 0, (uint64_t) n_expert_);
    };
    int ram_slots = ram_in(kRamHeadroom);
    if (m_.tiers().resident_budget > 0)
        ram_slots = std::min(ram_slots, (int) std::min<uint64_t>(m_.tiers().resident_budget / one_each_ram, (uint64_t) n_expert_));
    if (m_.diag_tier_ram() >= 0) ram_slots = std::min(m_.diag_tier_ram(), n_expert_);
    homes_ = policy::plan_homes(n_expert_, main_homes, peer_slots, ram_slots);
    if (homes_.peer == 0) peer_.reset();

    reserve_ = homes_.main > 0 ? reserve.beside_homes : reserve.copies;
    modules_ = homes_.main > 0 ? reserve.modules : 0;
    per_layer_ = homes_.main > 0 ? homes_.main : copies;
    if (homes_.main == 0 && per_layer_ < g.n_used) {
        err = "deepseek41: not enough device memory for the expert tier";
        return false;
    }
    std::vector<int64_t> slot_bytes;
    for (int l = 0; l < n_layers_; ++l)
        for (int64_t s = 0; s < per_layer_; ++s) slot_bytes.push_back((int64_t) m_.pack().experts[(size_t) l].blob_bytes);
    if (!cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) return false;
    res_.assign((size_t) n_layers_ * n_expert_, -1);
    stamp_.assign((size_t) n_layers_ * n_expert_, 0);
    owner_.assign(slot_bytes.size(), -1);
    busy_.assign(slot_bytes.size(), 0);
    int lo_prio = 0, hi_prio = 0;
    cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio);
    if (cudaEventCreateWithFlags(&fetch_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithPriority(&fetch_stream_, cudaStreamNonBlocking, hi_prio) != cudaSuccess) {
        err = "deepseek41: cannot create the promotion stream";
        return false;
    }
    for (size_t i = 0; i < kMaxPending; ++i) {
        cudaEvent_t e;
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "deepseek41: cannot create a promotion event";
            return false;
        }
        events_.push_back(e);
        free_events_.push_back((int) i);
    }

    // the CPU side: the file view, the per-layer formats, the pool
    if (homes_.file > 0) {
        std::optional<platform::MappedFile> mf = platform::MappedFile::open_ro(m_.path(), &err);
        if (!mf) return false;
        file_ = std::move(*mf);
    }
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        kernels::cpu::NativeFmt f;
        if (!kernels::cpu::native_fmt(el.gu_type, el.d_type, g.n_embd, g.n_ff_exp, f, err, g.clamp_exp[(size_t) l]))
            return false;
        fmt_.push_back(f);
    }
    act_ = (uint8_t*) platform::alloc_aligned(kernels::cpu::kNativeActBytes * kMaxWindow, 64);
    const int workers = m_.tiers().pool_workers > 0 ? m_.tiers().pool_workers : (int) kernels::cpu::physical_cores(true).size();
    pool_ = std::make_unique<kernels::cpu::ExpertPool>(workers, m_.tiers().pool_affinity_given ? m_.tiers().pool_affinity : kernels::cpu::PoolAffinity::All);

    if (!load_ram(err)) return false;
    if (homes_.main > 0) {
        grow();
    } else {
        seed_copies(0, per_layer_);
        promoted_ = 0;
    }
    // what the start picked by itself, on the server's auto line; the RAM floor the tier left free
    std::printf("auto flags: resident %.1f GB, pool-workers %d\n",
                (double) one_each_ram * cover_ / 1e9, pool_->workers());   // the server keeps the words after the name
    std::fflush(stdout);
    basalt::log::info("basalt generate", "RAM floor %.1f GiB of %.1f GiB free", (double) kRamHeadroom / 1073741824.0,
                      (double) avail / 1073741824.0);
    if (peer_) basalt::log::info("basalt deepseek41", "peer GPU %d: %lld experts", peer_->device(), (long long) n_layers_ * homes_.peer);
    basalt::log::info("basalt deepseek41", "expert homes of %d per layer: main card %d, second card %d, RAM %d, file %d; main card copies %lld",
                 n_expert_, homes_.main, homes_.peer, homes_.ram, homes_.file, (long long) (homes_.main > 0 ? ext_per_layer_ : per_layer_));
    program::warn_spill(spill(), peer_ ? "" : ", or --gpu 0,1");
    basalt::log::info("basalt deepseek41", "VRAM expert tier %.2f GiB, %lld of %d experts per layer (%.0f%%); CPU pool %d workers",
                 cache_.gib(), (long long) per_layer_, n_expert_, 100.0 * (double) per_layer_ / n_expert_, pool_->workers());
    return true;
}

program::ExpertSpill ExpertTier::spill() const {
    uint64_t one_each = 0;
    for (const ExpertLayer& e : m_.pack().experts) one_each += e.blob_bytes;
    return {(int64_t) n_layers_ * homes_.file, (int64_t) n_layers_ * n_expert_, one_each * (uint64_t) homes_.file};
}

namespace {

// The second card's real path: launches over the homes of the layer it holds most of, one token at a time.
class PeerPath final : public program::ProbePeerPath {
public:
    PeerPath(core::PeerExperts& peer, int top_k) : peer_(peer), k_(top_k) {}
    bool setup(const program::ProbeEnv& env, const kernels::cpu::NativeFmt& f, std::string& why) override {
        L_ = kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        for (int64_t e = 0; e < env.n_expert && (int) ids_.size() < 8; ++e)
            if (peer_.slot_ptr(0, e) != nullptr) ids_.push_back((int32_t) e);
        if (ids_.empty()) { why = "no_experts_on_peer"; return false; }
        if (!x_.alloc((size_t) f.n_embd * sizeof(float), cudaHostAllocPortable)) { why = "alloc_failed"; return false; }
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
            const auto t0 = program::ProbeClock::now();
            ok = peer_.launch_plain(0, L_, (const float*) x_.p, ids, k, kind, err) && peer_.finish(nullptr, err);
            rt.push_back(program::us_since(t0));
        }
        return ok;
    }

private:
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
    env.top_k = m_.geo().n_used;
    if (peer_) {
        env.peer_device = peer_->device();
        env.peer_path = std::make_shared<PeerPath>(*peer_, m_.geo().n_used);
    }
}

bool ExpertTier::load_ram(std::string& err) {
    uint64_t per_layer = 0, max_blob = 0;
    for (const ExpertLayer& e : m_.pack().experts) {   // one expert in every layer
        per_layer += e.blob_bytes;
        max_blob = std::max<uint64_t>(max_blob, e.blob_bytes);
    }
    cover_ = homes_.ram;
    ram_.assign((size_t) n_layers_, nullptr);
    ram_idx_.assign((size_t) n_layers_ * n_expert_, -1);
    spare_.assign((size_t) n_layers_, -1);
    spare_since_.assign((size_t) n_layers_, 0);
    last_miss_.assign((size_t) n_layers_ * n_expert_, 0);
    const int64_t copies = homes_.main > 0 ? 0 : per_layer_;
    std::vector<int32_t> device_home;   // [layer * n_expert + e] the slot of a home on either card, or -1
    if (homes_.main > 0 || homes_.peer > 0) device_home.assign((size_t) n_layers_ * n_expert_, -1);
    if (peer_) {
        peer_owner_.assign((size_t) n_layers_ * homes_.peer, -1);
        std::vector<int64_t> sizes;
        for (int l = 0; l < n_layers_; ++l)
            for (int s = 0; s < homes_.peer; ++s) sizes.push_back((int64_t) m_.pack().experts[(size_t) l].blob_bytes);
        if (!peer_->open_slots(sizes, err)) return false;
    }
    for (int l = 0; l < n_layers_; ++l) {
        std::vector<int> held;
        int n_main = 0, n_peer = 0;
        for (int r = 0; r < n_expert_; ++r) {
            const int e = ranked(l, r);
            const size_t idx = (size_t) l * n_expert_ + e;
            switch (policy::place_of_rank(homes_, (int) copies, r)) {
                case policy::Place::Main: {
                    const int32_t slot = (int32_t) ((int64_t) l * per_layer_ + n_main++);
                    res_[idx] = slot;
                    owner_[(size_t) slot] = e;
                    device_home[idx] = slot;
                    break;
                }
                case policy::Place::Peer: {
                    const int32_t slot = (int32_t) ((int64_t) l * homes_.peer + n_peer++);
                    peer_->admit(l, e, slot);
                    peer_owner_[(size_t) slot] = e;
                    device_home[idx] = slot;
                    break;
                }
                case policy::Place::Ram: held.push_back(e); break;
                case policy::Place::File: break;
            }
        }
        std::sort(held.begin(), held.end());
        for (size_t i = 0; i < held.size(); ++i) ram_idx_[(size_t) l * n_expert_ + held[i]] = (int32_t) i;
    }
    if (cover_ <= 0 && device_home.empty()) return true;
    const bool swaps = homes_.file > 0 && cover_ > 0;   // the file's experts swap into RAM from a thread
    if (cover_ > 0)                                     // one more RAM place per layer takes the experts on the move
        for (int l = 0; l < n_layers_; ++l) spare_[(size_t) l] = cover_;
    const auto t0 = std::chrono::steady_clock::now();
    // what the start places from here on: the file's share never loads, a second card and the RAM tier have their bars
    const uint64_t ram_bytes = per_layer * (uint64_t) cover_;
    const uint64_t main_bytes = per_layer * (uint64_t) homes_.main, peer_bytes = per_layer * (uint64_t) homes_.peer;
    core::LoadProgress::skipped(per_layer * (uint64_t) homes_.file);
    if (ram_bytes > 0) core::LoadProgress::plan("ram", ram_bytes);
    std::optional<core::LoadProgress> ram_load, main_load, peer_load;
    if (ram_bytes > 0) ram_load.emplace("ram", ram_bytes);
    if (main_bytes > 0) main_load.emplace(core::LoadProgress::cuda(0), main_bytes);
    if (peer_bytes > 0) {
        core::LoadProgress::plan(core::LoadProgress::cuda(peer_->device()), peer_bytes);
        peer_load.emplace(core::LoadProgress::cuda(peer_->device()), peer_bytes);
    }
    // the layout: one span per layer, each with its swap spare
    std::vector<uint64_t> span_bytes((size_t) n_layers_, 0);
    core::RamGrid grid;
    grid.n_layers = n_layers_;
    grid.n_expert = n_expert_;
    grid.file_base = m_.pack().experts_range.offset;
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        grid.layer_offset.push_back(el.offset);
        grid.blob_bytes.push_back(el.blob_bytes);
        if (cover_ <= 0) continue;
        span_bytes[(size_t) l] = el.blob_bytes * (uint64_t) (cover_ + 1);
        ram_[(size_t) l] = core::alloc_ram(span_bytes[(size_t) l]);
        if (ram_[(size_t) l] == nullptr) {
            err = "deepseek41: cannot allocate the RAM expert tier";
            return false;
        }
    }
    core::RamLoad load;
    load.fd = m_.experts_fd();
    load.stage_pinned = true;
    load.threads = 8;
    load.progress = ram_load ? &*ram_load : nullptr;
    load.groups = core::ram_layer_runs(grid, ram_, ram_idx_, device_home);
    load.sink = [&](int64_t l, const core::RamRun& r, const uint8_t* bytes, std::string& e) {
        const size_t idx = (size_t) l * n_expert_ + (size_t) r.first;
        const int32_t home = device_home[idx];
        const bool ok = res_[idx] >= 0 ? cudaMemcpy(cache_.device_slot(home), bytes, r.bytes, cudaMemcpyHostToDevice) == cudaSuccess
                                       : peer_->fill_slot(home, bytes, r.bytes);
        if (!ok) {
            e = "cannot place an expert on its card";
            return false;
        }
        (res_[idx] >= 0 ? main_load : peer_load)->add(r.bytes);
        return true;
    };
    core::RamLoadStats read;
    if (!core::read_ram_runs(load, read, err)) {
        err = "deepseek41: loading the expert tiers: " + err;
        return false;
    }
    if (cover_ > 0) {
        ram_pins_ = core::pin_ram_spans(ram_, span_bytes, /*mapped=*/false);
        int unpinned = 0;
        for (const core::RamPin& p : ram_pins_) unpinned += p.registered_bytes == 0;
        if (unpinned > 0)
            basalt::log::warn("basalt deepseek41", "RAM expert tier: %d of %d layers not pinned, locked: %s", unpinned, n_layers_,
                              ram_pins_[(size_t) n_layers_ - 1].lock_note.c_str());
    }
    if (ram_load) ram_load->finish();
    if (main_load) main_load->finish();
    if (peer_load) peer_load->finish();
    if (swaps) worker_ = std::thread([this] { swap_worker(); });
    // what the load left in the page cache goes, and the most routed of the experts left in the file take its place
    platform::advise_dontneed(m_.experts_fd(), m_.pack().experts_range.offset, m_.pack().experts_range.size());
    if (swaps) preread_ = std::thread([this] { preread(); });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    basalt::log::info("basalt deepseek41", "RAM expert tier %.1f GiB (%d of %d experts per layer) loaded in %.1f s",
                 (double) per_layer * cover_ / 1073741824.0, cover_, n_expert_, s);
    return true;
}

int ExpertTier::ranked(int layer, int r) const {
    return rank_.empty() ? r : rank_[(size_t) layer * n_expert_ + r];
}

// The routing order from the profile file: per layer, the order its pairs come in.
bool ExpertTier::load_rank(std::string& err) {
    const std::string& path = m_.tiers().expert_profile;
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

// Copies the most routed RAM tier experts from rank `first_rank` on into `n` slots of each layer.
void ExpertTier::seed_copies(int first_rank, int64_t n) {
    for (int l = 0; l < n_layers_; ++l) {
        for (int64_t r = first_rank, placed = 0; r < n_expert_ && placed < n; ++r) {
            const int e = ranked(l, (int) r);
            if (!pinned(l, e)) continue;
            ++tick_;
            if (claim(l, e, blob_of(l, e), copy_stream_) < 0) {
                cudaStreamSynchronize(copy_stream_);
                drain();
                ++tick_;
                if (claim(l, e, blob_of(l, e), copy_stream_) < 0) return;
            }
            ++placed;
        }
    }
    cudaStreamSynchronize(copy_stream_);
    drain();
}

const uint8_t* ExpertTier::blob_of(int layer, int e) const {
    const ExpertLayer& el = m_.pack().experts[(size_t) layer];
    const int32_t r = ram_idx_[(size_t) layer * n_expert_ + e];
    if (r >= 0) return ram_[(size_t) layer] + (size_t) r * el.blob_bytes;
    return file_.data() + m_.pack().experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
}

const uint8_t* ExpertTier::slot_ptr(int32_t slot) const {
    const int64_t nb = (int64_t) n_layers_ * per_layer_;
    return slot < nb ? cache_.device_slot(slot) : ext_.device_slot((int32_t) (slot - nb));
}

void ExpertTier::drain() {
    for (const Pending& p : pending_) {
        cudaEventSynchronize(events_[(size_t) p.event]);
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
    pending_.clear();
    if (peer_moves_.empty()) return;
    std::string err;
    peer_->read_wait(err);
    for (const PeerMove& m : peer_moves_) {
        const size_t base = (size_t) m.layer * n_expert_;
        peer_->admit(m.layer, m.out, -1);
        peer_->admit(m.layer, m.in, m.slot);
        ram_idx_[base + m.out] = m.place;
        spare_[(size_t) m.layer] = ram_idx_[base + m.in];
        ram_idx_[base + m.in] = -1;
    }
    peer_moves_.clear();
}

void ExpertTier::settle() {
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    drain();
}

void ExpertTier::begin_token(int n_tokens) {
    token_ += (uint64_t) n_tokens;
    drain();
    finish_swaps();
    start_swaps();
}

void ExpertTier::swap_worker() {
    std::unique_lock<std::mutex> lk(mu_);
    size_t next = 0;
    for (;;) {
        cv_.wait(lk, [&] { return quit_ || next < jobs_.size(); });
        if (quit_) return;
        const Swap sw = jobs_[next];
        lk.unlock();
        const ExpertLayer& el = m_.pack().experts[(size_t) sw.layer];
        uint8_t* dst = ram_[(size_t) sw.layer] + (size_t) spare_[(size_t) sw.layer] * el.blob_bytes;
        const bool ok = platform::pread_all(m_.experts_fd(), dst, el.blob_bytes,
                                            m_.pack().experts_range.offset + el.offset + (uint64_t) sw.expert * el.blob_bytes);
        lk.lock();
        if (!ok) jobs_[next].expert = -1;
        ++next;
        ++jobs_done_;
        cv_.notify_all();
    }
}

void ExpertTier::preread() {
    uint64_t left = kPrereadBytes;
    for (int r = n_expert_ - homes_.file; r < n_expert_; ++r) {
        for (int l = 0; l < n_layers_; ++l) {
            if (preread_quit_.load(std::memory_order_relaxed)) return;
            const ExpertLayer& el = m_.pack().experts[(size_t) l];
            if (left < el.blob_bytes) return;
            left -= el.blob_bytes;
            platform::advise_willneed(m_.experts_fd(), m_.pack().experts_range.offset + el.offset + (uint64_t) ranked(l, r) * el.blob_bytes,
                                      el.blob_bytes);
        }
    }
}

void ExpertTier::finish_swaps() {
    if (flight_.empty()) return;
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return jobs_done_ == jobs_.size(); });
        for (size_t i = 0; i < flight_.size(); ++i) flight_[i].expert = jobs_[jobs_.size() - flight_.size() + i].expert;
    }
    for (const Swap& sw : flight_) {
        if (sw.expert < 0) continue;
        const size_t base = (size_t) sw.layer * n_expert_;
        const int32_t v = ram_idx_[base + sw.victim];
        ram_idx_[base + sw.expert] = spare_[(size_t) sw.layer];
        ram_idx_[base + sw.victim] = -1;
        spare_[(size_t) sw.layer] = v;
        spare_since_[(size_t) sw.layer] = token_;
        ++swapped_;
    }
    flight_.clear();
}

void ExpertTier::start_swaps() {
    if (homes_.file == 0 || want_.empty() || m_.tiers().diag_tier_frozen) {
        want_.clear();
        return;
    }
    std::vector<Swap> batch;
    for (const auto& w : want_) {
        if ((int) batch.size() >= policy::kSwapsPerToken) break;
        const int l = w.first, e = w.second;
        if (spare_[(size_t) l] < 0 || token_ <= spare_since_[(size_t) l] || pinned(l, e)) continue;
        bool dup = false;
        for (const Swap& b : batch) dup = dup || b.layer == l;
        if (dup) continue;
        const int v = policy::pick_victim(&ram_idx_[(size_t) l * n_expert_], &stamp_[(size_t) l * n_expert_], n_expert_, nullptr, 0);
        // a victim used in the last 16 tokens stays
        if (v < 0 || stamp_[(size_t) l * n_expert_ + v] + 16ull * (uint64_t) n_layers_ >= tick_) continue;
        batch.push_back({l, e, v});
    }
    want_.clear();
    if (batch.empty()) return;
    {
        std::lock_guard<std::mutex> g(mu_);
        jobs_.insert(jobs_.end(), batch.begin(), batch.end());
    }
    cv_.notify_all();
    flight_ = std::move(batch);
}

bool ExpertTier::lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err) {
    const uint64_t now = ++tick_;
    lk.k = k;
    for (int i = 0; i < k; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert_) {
            err = "deepseek41: router returned a bad expert id";
            return false;
        }
        lk.experts[i] = e;
        lk.again[i] = false;
        lk.fetched[i] = false;
        const size_t idx = (size_t) layer * n_expert_ + e;
        stamp_[idx] = now;
        const int32_t s = res_[idx];
        lk.peer[i] = nullptr;
        if (s >= 0) {
            lk.dev[i] = slot_ptr(s);
            lk.host[i] = nullptr;
            ++hits_;
        } else if (peer_ && (lk.peer[i] = peer_->slot_ptr(layer, e))) {
            lk.dev[i] = nullptr;
            lk.host[i] = nullptr;
            ++peer_hits_;
        } else {
            lk.dev[i] = nullptr;
            lk.host[i] = blob_of(layer, e);
            lk.again[i] = policy::repeat_miss(last_miss_[idx], token_, policy::kPromoteWindow);
            if (pinned(layer, e)) {
                ++ram_hits_;
            } else {
                ++file_hits_;
                if (policy::repeat_miss(last_miss_[idx], token_, policy::kAdmitWindow) && want_.size() < 64) want_.emplace_back(layer, e);
            }
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
    if (peer_) r.peer = peer_->slot_ptr(layer, e);
    if (!r.peer && pinned(layer, e)) r.pinned = blob_of(layer, e);
    return r;
}

bool ExpertTier::run_cpu(int layer, const float* x, const Lookup& lk, float* rows, std::string& err) {
    const kernels::cpu::NativeFmt& f = fmt_[(size_t) layer];
    kernels::cpu::ExpertJobMulti jobs[kMaxK];
    int n = 0;
    bool any = false;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.peer[i]) continue;
        any = true;
        jobs[n] = kernels::cpu::ExpertJobMulti{};
        jobs[n].blob = lk.host[i];
        jobs[n].nt = 1;
        jobs[n].nact[0] = act_;
        jobs[n].out[0] = rows + (size_t) i * (size_t) m_.geo().n_embd;
        ++n;
    }
    if (!any) return true;
    kernels::cpu::native_quant_act(f, x, act_);
    pool_->run_split_multi_native(f, jobs, n);
    (void) err;
    return true;
}

bool ExpertTier::run_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows, std::string& err) {
    const kernels::cpu::NativeFmt& f = fmt_[(size_t) layer];
    const size_t n_embd = (size_t) m_.geo().n_embd;
    kernels::cpu::ExpertJobMulti jobs[kMaxK];
    bool quantized[kMaxWindow] = {};
    int n = 0;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.peer[i]) continue;
        jobs[n] = kernels::cpu::ExpertJobMulti{};
        jobs[n].blob = lk.host[i];
        jobs[n].nt = use[i].nt;
        for (int j = 0; j < use[i].nt; ++j) {
            const int t = use[i].tok[j];
            if (!quantized[t]) {
                kernels::cpu::native_quant_act(f, x + (size_t) t * n_embd, act_ + (size_t) t * kernels::cpu::kNativeActBytes);
                quantized[t] = true;
            }
            jobs[n].nact[j] = act_ + (size_t) t * kernels::cpu::kNativeActBytes;
            jobs[n].out[j] = rows + (size_t) use[i].row[j] * n_embd;
        }
        ++n;
    }
    if (n) pool_->run_split_multi_native(f, jobs, n);
    (void) err;
    return true;
}

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
    if (homes_.main == 0 || (homes_too && homes_.file == 0 && spare_[(size_t) layer] >= 0))
        scan((int64_t) layer * per_layer_, per_layer_);
    if (ext_per_layer_) scan((int64_t) n_layers_ * per_layer_ + (int64_t) layer * ext_per_layer_, ext_per_layer_);
    return slot;
}

bool ExpertTier::move_to_peer(int layer, int e, const uint8_t* src) {
    if (homes_.peer == 0 || homes_.file > 0 || spare_[(size_t) layer] < 0) return false;
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
    if (slot < 0 || oldest >= on_main) return false;
    const int32_t place = spare_[(size_t) layer];
    spare_[(size_t) layer] = -1;
    const uint64_t bytes = m_.pack().experts[(size_t) layer].blob_bytes;
    peer_->exchange_slot(slot, ram_[(size_t) layer] + (size_t) place * bytes, src, bytes);
    peer_moves_.push_back({layer, e, peer_owner_[(size_t) slot], slot, place});
    peer_owner_[(size_t) slot] = e;
    ++promoted_;
    return true;
}

int32_t ExpertTier::claim(int layer, int e, const uint8_t* src, cudaStream_t stream, bool homes_too) {
    if (pending_.size() >= kMaxPending || free_events_.empty()) {
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
    const uint64_t bytes = m_.pack().experts[(size_t) layer].blob_bytes;
    uint8_t* dst = const_cast<uint8_t*>(slot_ptr(slot));
    const bool home = homes_.main > 0 && slot < (int64_t) n_layers_ * per_layer_;
    int32_t place = -1;
    const auto pieces = [&](uint8_t* to, const uint8_t* from, cudaMemcpyKind kind) {
        for (uint64_t off = 0; off < bytes; off += policy::kPieceBytes)
            cudaMemcpyAsync(to + off, from + off, std::min(policy::kPieceBytes, bytes - off), kind, stream);
    };
    if (home) {   // the home's bytes first: the stream keeps the order
        place = spare_[(size_t) layer];
        spare_[(size_t) layer] = -1;
        pieces(ram_[(size_t) layer] + (size_t) place * bytes, dst, cudaMemcpyDeviceToHost);
    }
    pieces(dst, src, cudaMemcpyHostToDevice);
    const int ev = free_events_.back();
    free_events_.pop_back();
    cudaEventRecord(events_[(size_t) ev], stream);
    pending_.push_back({layer, e, slot, ev, home ? o : -1, place});
    ++promoted_;
    return slot;
}

int ExpertTier::fetch(int layer, Lookup& lk) {
    if (m_.tiers().diag_tier_frozen) return 0;
    int np = 0, nf = 0;
    for (int i = 0; i < lk.k; ++i)
        if (!lk.dev[i] && !lk.peer[i]) ++(pinned(layer, lk.experts[i]) ? np : nf);
    int left = policy::bus_count(np, nf);
    int started = 0;
    for (int i = lk.k - 1; i >= 0 && left > 0; --i) {
        if (lk.dev[i] || !pinned(layer, lk.experts[i])) continue;   // pinned sources only
        --left;
        const int32_t slot = claim(layer, lk.experts[i], lk.host[i], fetch_stream_);
        if (slot < 0) continue;
        lk.dev[i] = slot_ptr(slot);
        lk.host[i] = nullptr;
        lk.fetched[i] = true;
        --cpu_;
        ++fetched_;
        ++started;
    }
    if (started) cudaEventRecord(fetch_ev_, fetch_stream_);
    return started;
}

void ExpertTier::await_fetch(cudaStream_t main) { cudaStreamWaitEvent(main, fetch_ev_, 0); }

void ExpertTier::promote(int layer, const Lookup& lk) {
    if (m_.tiers().diag_tier_frozen) return;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || !lk.again[i] || !pinned(layer, lk.experts[i])) continue;   // pinned sources only
        if (!move_to_peer(layer, lk.experts[i], lk.host[i])) claim(layer, lk.experts[i], lk.host[i], copy_stream_, /*homes_too=*/true);
    }
}

void ExpertTier::shrink() {
    if (ext_per_layer_ == 0) return;
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    drain();
    const int64_t nb = (int64_t) n_layers_ * per_layer_;
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
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    drain();
    std::fill(res_.begin(), res_.end(), -1);
    owner_.clear();
    busy_.clear();
    lent_per_layer_ = per_layer_;
    per_layer_ = 0;
    cache_.close();
}

bool ExpertTier::reclaim(std::string& err) {
    if (lent_per_layer_ == 0) return true;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        err = "deepseek41: cannot query the free device memory";
        return false;
    }
    const int64_t n = policy::reclaim_slots(free_b, reserve_, free_floor_, one_each_, lent_per_layer_);
    if (n < 1) return true;   // nothing fits above the reserve: the slots stay lent
    std::vector<int64_t> slot_bytes;
    for (int l = 0; l < n_layers_; ++l)
        for (int64_t s = 0; s < n; ++s) slot_bytes.push_back((int64_t) m_.pack().experts[(size_t) l].blob_bytes);
    if (!cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) return false;
    per_layer_ = n;
    lent_per_layer_ = 0;
    owner_.assign(slot_bytes.size(), -1);
    busy_.assign(slot_bytes.size(), 0);
    return true;
}

void ExpertTier::grow() {
    if (ext_per_layer_ > 0 || per_layer_ >= n_expert_ || !cache_.valid()) return;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
    const int64_t n = policy::grow_slots(free_b, reserve_ + (token_ > 0 ? 0 : modules_), free_floor_, one_each_, per_layer_, n_expert_, cap_per_layer_);
    if (n < 1) return;
    std::vector<int64_t> slot_bytes;
    for (int l = 0; l < n_layers_; ++l)
        for (int64_t s = 0; s < n; ++s) slot_bytes.push_back((int64_t) m_.pack().experts[(size_t) l].blob_bytes);
    std::string err;
    if (!ext_.open_sized(slot_bytes, n_layers_, n_expert_, err)) {
        basalt::log::warn("basalt deepseek41", "VRAM expert tier cannot grow: %s", err.c_str());
        return;
    }
    ext_per_layer_ = n;
    owner_.resize((size_t) slot_total(), -1);
    busy_.resize((size_t) slot_total(), 0);
    basalt::log::info("basalt deepseek41", "VRAM expert tier %lld of %d experts per layer (%lld beside the prompt workspaces)",
                 (long long) (per_layer_ + n), n_expert_, (long long) per_layer_);
    if (homes_.main > 0) {
        cudaDeviceSynchronize();   // the arena's zeroing is not ordered against the copy stream
        const int64_t promoted = promoted_;
        seed_copies(homes_.main + homes_.peer, n);
        promoted_ = promoted;
    }
}

void ExpertTier::set_free_floor(int64_t mib) { free_floor_ = mib < 0 ? start_floor_ : (uint64_t) mib << 20; }

void ExpertTier::report(const char* tag) const {
    const int64_t n = hits_ + cpu_ + fetched_ + peer_hits_;
    if (n == 0) return;
    basalt::log::info("basalt deepseek41", "TIER %s: %lld lookups: %.1f%% VRAM, %.1f%% second card, %.1f%% bus copy, %.1f%% CPU (file %.1f%%); promoted %lld, dropped %lld, swapped %lld", tag,
                 (long long) n, 100.0 * (double) hits_ / (double) n, 100.0 * (double) peer_hits_ / (double) n, 100.0 * (double) fetched_ / (double) n,
                 100.0 * (double) cpu_ / (double) n, 100.0 * (double) file_hits_ / (double) n, (long long) promoted_, (long long) dropped_, (long long) swapped_);
}

}  // namespace basalt::models::deepseek41
