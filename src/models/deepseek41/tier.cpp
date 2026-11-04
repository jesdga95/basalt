// src/models/deepseek41/tier.cpp - the DeepSeek-V4.1 routed-expert tiers.
#include "basalt/log.hpp"
#include "models/deepseek41/tier.hpp"

#include "basalt/core/expert_source.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/peer_tier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {
namespace {

constexpr uint64_t kRamHeadroom = 20ull << 30;   // host memory left to the system beside the RAM tier
constexpr uint64_t kRamHeadroomNoFile = 16ull << 30;   // ... when no expert is read through the page cache
constexpr uint64_t kPeerReserve = 600ull << 20;  // second card memory left free beside its slots
constexpr size_t kMaxPending = 128;              // promotions in flight
constexpr uint64_t kPrereadBytes = 12ull << 30;  // file-backed experts read ahead into the page cache

// The usual routing order of this model's experts: the RAM and VRAM tiers start from the head of it.
constexpr int kRankLayers = 40, kRankExperts = 384;
constexpr uint16_t kExpertRank[kRankLayers][kRankExperts] =
#include "models/deepseek41/expert_rank.inc"
;

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
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (fetch_stream_) cudaStreamSynchronize(fetch_stream_);
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
    if (fetch_ev_) cudaEventDestroy(fetch_ev_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (fetch_stream_) cudaStreamDestroy(fetch_stream_);
    for (uint8_t* p : ram_)
        if (p) cudaFreeHost(p);
    platform::free_aligned(act_);
}

bool ExpertTier::open(const Reserve& reserve, const float* h_x, float* h_rows, std::string& err) {
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
    int copies = slots_in(free_b, reserve.copies);
    int main_homes = slots_in(free_b, reserve.homes);
    if (m_.diag_tier_vram() >= 0) {
        copies = std::min(copies, m_.diag_tier_vram());
        main_homes = std::min(main_homes, m_.diag_tier_vram());
    }

    int peer_slots = 0;
    if (m_.peer_device() >= 0) {
        peer_ = std::make_unique<PeerTier>();
        std::string pe;
        if (peer_->open(m_.peer_device(), n_layers_, n_expert_, g.n_embd, g.n_ff_exp, h_x, h_rows, pe)) {
            peer_slots = slots_in(peer_->room(), kPeerReserve);
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
    int ram_slots = ram_in(kRamHeadroom), ram_no_file = ram_in(kRamHeadroomNoFile);
    if (m_.diag_tier_ram() >= 0) ram_slots = ram_no_file = std::min(m_.diag_tier_ram(), n_expert_);
    homes_ = policy::plan_homes(n_expert_, main_homes, peer_slots, ram_slots, ram_no_file);
    if (homes_.peer == 0) peer_.reset();

    reserve_ = homes_.main > 0 ? reserve.beside_homes : reserve.copies;
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
    pool_ = std::make_unique<kernels::cpu::ExpertPool>((int) kernels::cpu::physical_cores(true).size());

    if (!load_ram(err)) return false;
    if (homes_.main > 0) {
        grow();
    } else {
        seed_copies(0, per_layer_);
        promoted_ = 0;
    }
    basalt::log::info("basalt deepseek41", "expert homes of %d per layer: main card %d, second card %d, RAM %d, file %d; main card copies %lld",
                 n_expert_, homes_.main, homes_.peer, homes_.ram, homes_.file, (long long) (homes_.main > 0 ? ext_per_layer_ : per_layer_));
    basalt::log::info("basalt deepseek41", "VRAM expert tier %.2f GiB, %lld of %d experts per layer (%.0f%%); CPU pool %d workers",
                 cache_.gib(), (long long) per_layer_, n_expert_, 100.0 * (double) per_layer_ / n_expert_, pool_->workers());
    return true;
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
        std::vector<int64_t> sizes;
        for (int l = 0; l < n_layers_; ++l)
            for (int s = 0; s < homes_.peer; ++s) sizes.push_back((int64_t) m_.pack().experts[(size_t) l].blob_bytes);
        if (!peer_->slots(sizes, err)) return false;
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
    const bool swaps = homes_.file > 0 && cover_ > 0;   // one more RAM place per layer takes the experts swapped in
    if (swaps)
        for (int l = 0; l < n_layers_; ++l) spare_[(size_t) l] = cover_;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::mutex mu;
    // reads that bypass the OS cache leave it to the system; ranges that are not block aligned go through a staging buffer
    constexpr uint64_t kAlign = 4096;
    const int direct = platform::open_unbuffered(m_.experts_fd());
    auto work = [&]() {
        uint8_t* stage = nullptr;
        if (cudaHostAlloc((void**) &stage, max_blob + 2 * kAlign, cudaHostAllocPortable) != cudaSuccess) {
            std::lock_guard<std::mutex> g(mu);
            err = "deepseek41: cannot pin the expert staging";
            failed = true;
            return;
        }
        // the blob at `off` in `stage`; the address it landed at
        auto read_blob = [&](uint64_t bytes, uint64_t off) -> const uint8_t* {
            const uint64_t a0 = off / kAlign * kAlign, a1 = (off + bytes + kAlign - 1) / kAlign * kAlign;
            if (direct >= 0 && platform::pread_all(direct, stage, a1 - a0, a0)) return stage + (off - a0);
            return platform::pread_all(m_.experts_fd(), stage, bytes, off) ? stage : nullptr;
        };
        for (;;) {
            const int l = next++;
            if (l >= n_layers_ || failed) break;
            const ExpertLayer& el = m_.pack().experts[(size_t) l];
            uint8_t* p = nullptr;
            if (cover_ > 0) {
                const uint64_t span = el.blob_bytes * (uint64_t) cover_;
                if (cudaHostAlloc((void**) &p, span + (swaps ? el.blob_bytes : 0), cudaHostAllocPortable) != cudaSuccess) {
                    std::lock_guard<std::mutex> g(mu);
                    err = "deepseek41: cannot pin the RAM expert tier";
                    failed = true;
                    break;
                }
                ram_[(size_t) l] = p;
            }
            for (int e = 0; e < n_expert_; ++e) {
                const size_t idx = (size_t) l * n_expert_ + e;
                const int32_t r = ram_idx_[idx];
                const int32_t home = device_home.empty() ? -1 : device_home[idx];
                if (r < 0 && home < 0) continue;
                const uint8_t* got = read_blob(el.blob_bytes, m_.pack().experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes);
                bool ok = got != nullptr;
                if (ok && r >= 0) std::memcpy(p + (uint64_t) r * el.blob_bytes, got, el.blob_bytes);
                if (ok && home >= 0)
                    ok = res_[idx] >= 0 ? cudaMemcpy(cache_.device_slot(home), got, el.blob_bytes, cudaMemcpyHostToDevice) == cudaSuccess
                                        : peer_->fill(home, got, el.blob_bytes);
                if (!ok) {
                    std::lock_guard<std::mutex> g(mu);
                    err = "deepseek41: short read loading the expert tiers";
                    failed = true;
                    break;
                }
            }
        }
        if (stage) cudaFreeHost(stage);
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i) th.emplace_back(work);
    for (std::thread& t : th) t.join();
    if (direct >= 0) platform::close_fd(direct);
    if (failed) return false;
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
    return n_layers_ == kRankLayers && n_expert_ == kRankExperts ? kExpertRank[layer][r] : r;
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
        res_[(size_t) p.layer * n_expert_ + p.expert] = p.slot;
        busy_[(size_t) p.slot] = 0;
        free_events_.push_back(p.event);
    }
    pending_.clear();
}

void ExpertTier::begin_token() {
    ++token_;
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
    if (homes_.file == 0 || want_.empty()) {
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
        const size_t idx = (size_t) layer * n_expert_ + e;
        stamp_[idx] = now;
        const int32_t s = res_[idx];
        lk.peer[i] = nullptr;
        if (s >= 0) {
            lk.dev[i] = slot_ptr(s);
            lk.host[i] = nullptr;
            ++hits_;
        } else if (peer_ && (lk.peer[i] = peer_->blob(layer, e))) {
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
    if (peer_) r.peer = peer_->blob(layer, e);
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


int32_t ExpertTier::claim(int layer, int e, const uint8_t* src, cudaStream_t stream) {
    if (pending_.size() >= kMaxPending || free_events_.empty()) {
        ++dropped_;
        return -1;
    }
    int32_t slot = -1;
    uint64_t best = ~0ull;
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
    if (homes_.main == 0) scan((int64_t) layer * per_layer_, per_layer_);   // homes do not move
    if (ext_per_layer_) scan((int64_t) n_layers_ * per_layer_ + (int64_t) layer * ext_per_layer_, ext_per_layer_);
    if (slot < 0 || best == tick_) return -1;   // every slot is in use by this very lookup
    const int32_t o = owner_[(size_t) slot];
    if (o >= 0) res_[(size_t) layer * n_expert_ + o] = -1;
    owner_[(size_t) slot] = e;
    busy_[(size_t) slot] = 1;
    stamp_[(size_t) layer * n_expert_ + e] = tick_;
    const uint64_t bytes = m_.pack().experts[(size_t) layer].blob_bytes;
    uint8_t* dst = const_cast<uint8_t*>(slot_ptr(slot));
    for (uint64_t off = 0; off < bytes; off += policy::kPieceBytes)
        cudaMemcpyAsync(dst + off, src + off, std::min(policy::kPieceBytes, bytes - off), cudaMemcpyHostToDevice, stream);
    const int ev = free_events_.back();
    free_events_.pop_back();
    cudaEventRecord(events_[(size_t) ev], stream);
    pending_.push_back({layer, e, slot, ev});
    ++promoted_;
    return slot;
}

void ExpertTier::fetch(int layer, Lookup& lk, cudaStream_t main) {
    int np = 0, nf = 0;
    for (int i = 0; i < lk.k; ++i)
        if (!lk.dev[i] && !lk.peer[i]) ++(pinned(layer, lk.experts[i]) ? np : nf);
    int left = policy::bus_count(np, nf);
    bool any = false;
    for (int i = lk.k - 1; i >= 0 && left > 0; --i) {
        if (lk.dev[i] || !pinned(layer, lk.experts[i])) continue;   // pinned sources only
        --left;
        const int32_t slot = claim(layer, lk.experts[i], lk.host[i], fetch_stream_);
        if (slot < 0) continue;
        lk.dev[i] = slot_ptr(slot);
        lk.host[i] = nullptr;
        --cpu_;
        ++fetched_;
        any = true;
    }
    if (any) {
        cudaEventRecord(fetch_ev_, fetch_stream_);
        cudaStreamWaitEvent(main, fetch_ev_, 0);
    }
}

void ExpertTier::promote(int layer, const Lookup& lk) {
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || !lk.again[i] || !pinned(layer, lk.experts[i])) continue;   // pinned sources only
        claim(layer, lk.experts[i], lk.host[i], copy_stream_);
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
    std::vector<int64_t> slot_bytes;
    for (int l = 0; l < n_layers_; ++l)
        for (int64_t s = 0; s < lent_per_layer_; ++s) slot_bytes.push_back((int64_t) m_.pack().experts[(size_t) l].blob_bytes);
    if (!cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) return false;
    per_layer_ = lent_per_layer_;
    lent_per_layer_ = 0;
    owner_.assign(slot_bytes.size(), -1);
    busy_.assign(slot_bytes.size(), 0);
    return true;
}

void ExpertTier::grow() {
    if (ext_per_layer_ > 0 || per_layer_ >= n_expert_ || !cache_.valid()) return;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b <= reserve_) return;
    int64_t n = std::min<int64_t>((int64_t) ((free_b - reserve_) / one_each_), n_expert_ - per_layer_);
    if (m_.diag_tier_vram() >= 0) n = std::min<int64_t>(n, m_.diag_tier_vram() - per_layer_);   // the total stays at the fixed size
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

void ExpertTier::report(const char* tag) const {
    const int64_t n = hits_ + cpu_ + fetched_ + peer_hits_;
    if (n == 0) return;
    basalt::log::info("basalt deepseek41", "TIER %s: %lld lookups: %.1f%% VRAM, %.1f%% second card, %.1f%% bus copy, %.1f%% CPU (file %.1f%%); promoted %lld, dropped %lld, swapped %lld", tag,
                 (long long) n, 100.0 * (double) hits_ / (double) n, 100.0 * (double) peer_hits_ / (double) n, 100.0 * (double) fetched_ / (double) n,
                 100.0 * (double) cpu_ / (double) n, 100.0 * (double) file_hits_ / (double) n, (long long) promoted_, (long long) dropped_, (long long) swapped_);
}

}  // namespace basalt::models::deepseek41
