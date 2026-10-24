// src/core/peer_experts.cpp - see include/basalt/core/peer_experts.hpp.
#include "basalt/core/load_progress.hpp"
#include "basalt/core/peer_experts.hpp"

#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "basalt/log.hpp"

namespace basalt::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }
static bool g_prompt_peer = false;
void set_prompt_peer(bool on) { g_prompt_peer = on; }
bool prompt_peer() { return g_prompt_peer; }
bool peers_access(int device) {
    int a = 0, b = 0;
    if (cudaDeviceCanAccessPeer(&a, 0, device) != cudaSuccess || cudaDeviceCanAccessPeer(&b, device, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return a && b;
}

namespace {

constexpr int64_t H = basalt::kernels::cpu::H;
constexpr int64_t CAP = basalt::kernels::cpu::MAXT * 10;   // entries per layer: MAXT tokens x top-10

struct Meta {                           // one block, uploaded with one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    if (device_ < 0) return;
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (evict_ev_) cudaEventDestroy(evict_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_q8_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = evict_ev_ = nullptr;
    fsrc_ = nullptr;
    evicting_.clear();
    pending_.clear();
    device_ = -1;
}

namespace {

size_t scratch_bytes() {
    const auto& lay = basalt::kernels::cpu::expert_layout();
    int64_t ff = basalt::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    return std::max<size_t>((size_t) basalt::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                            basalt::kernels::native_expert_scratch_bytes(CAP, ff));
}

}  // namespace

uint64_t PeerExperts::device_overhead_bytes() {
    return (uint64_t) scratch_bytes() + (uint64_t) CAP * H * sizeof(float) * 2 + sizeof(Meta) +
           (uint64_t) CAP * (H / 32) * 36;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err, bool exclusive) {
    if (!prepare(device, src, n_layers, n_expert, err)) return false;
    // the pairs the primary does not hold, in rank order, as many as fit
    On on(device);
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b256 = slot_bytes(pr.first);
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
    }
    return fill(pick, err, exclusive);
}

uint64_t PeerExperts::slot_bytes(int64_t layer) {
    const auto& lay = basalt::kernels::cpu::expert_layout();
    const uint64_t b = lay.blob_bytes(layer);
    return lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
}

bool PeerExperts::prepare(int device, ExpertSource& src, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    hold_.assign((size_t) (n_layers * n_expert), 0);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, 0, device);
        cudaDeviceCanAccessPeer(&b, device, 0);
        p2p_ = a && b;
        if (p2p_) {
            On on0(0);
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    const size_t scratch = scratch_bytes();
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaEventCreateWithFlags(&evict_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, sizeof(Meta)), "plan", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }
    return true;
}

bool PeerExperts::fill(const std::vector<std::pair<int32_t, int32_t>>& pick, std::string& err, bool exclusive) {
    const auto& lay = basalt::kernels::cpu::expert_layout();
    const int device = device_;
    const int64_t n_layers = n_layers_, n_expert = n_expert_;
    ExpertSource& src = *src_;
    On on(device);
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    std::vector<int64_t> sizes;
    for (const auto& pr : pick) sizes.push_back((int64_t) lay.blob_bytes(pr.first));
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    uint64_t fill_total = 0;
    for (const auto& pr : pick) fill_total += lay.blob_bytes(pr.first);
    LoadProgress::plan(LoadProgress::cuda(device), fill_total, !exclusive);
    LoadProgress progress(LoadProgress::cuda(device), fill_total, !exclusive);
    std::vector<int32_t> pick_slots(pick.size());
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        const int32_t slot = cache_.admit(pr.first, pr.second);
        if (slot < 0) {
            err = "peer experts: fill failed: no slot";
            close();
            return false;
        }
        pick_slots[i] = slot;
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!fill_cache_slots(src, cache_, device, pick.data(), pick_slots.data(), (int64_t) pick.size(),
                          [&](uint64_t bytes) { progress.add(bytes); }, err)) {
        err = "peer experts: fill failed: " + err;
        close();
        return false;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    progress.finish();
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    return true;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out) {
    const bool direct = out != nullptr;
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    Meta& m = *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
        m.ptr[groups] = (unsigned long long) cache_.device_slot(slot);
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    // direct: x is the verifier's pinned (portable) doorbell row block - copied from where it is
    if (!direct) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
    On on(device_);
    const cudaStream_t s = stream_;
    if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "input", err) ||
        !ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err))
        return false;
    Meta* dm = (Meta*) d_meta_;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    if (lay.native) {
        basalt::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        const auto& f = lay.fmt[(size_t) layer];
        const auto L = basalt::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        basalt::kernels::native_grouped_set_mma_hint(rows >= 2 * groups);   // several entries per expert
        basalt::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
        basalt::kernels::native_grouped_set_mma_hint(false);
    } else {
        err = "peer experts: only native packs are supported";
        return false;
    }
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            basalt::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        return true;
    }
    return ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
              "results", err);
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!evicting_.empty() || max_swaps <= 0) return true;
    // experts whose refill is still landing are not resident yet: they must not be picked again as candidates
    // (their slots belong to no resident expert, so no victim can collide with them)
    std::vector<uint8_t> inflight;
    if (!pending_.empty()) {
        inflight.assign((size_t) (n_layers_ * n_expert_), 0);
        for (const auto& p : pending_) inflight[(size_t) p.first] = 1;
    }
    const auto& lay = basalt::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0) {
                if (!hold_[(size_t) (l * n_expert_ + e)]) vict.emplace_back(u[e], e);
            }
            else if (r0[e] < 0 && u[e] >= 2.0f && (inflight.empty() || !inflight[(size_t) (l * n_expert_ + e)])) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    On on(device_);
    if (fsrc_ != nullptr) {
        // exclusive: a victim without a RAM place is copied into a free one first (it stays resident and computing
        // meanwhile); one that has a place already (a slot reassigned from the primary) needs no copy.  Either way
        // the slot is refilled at apply_pending, once the copy has landed
        for (const Swap& s : swaps) {
            const int32_t slot = res_[(size_t) (s.layer * n_expert_ + s.out)];
            if (slot < 0 || !fsrc_->has_resident(s.layer, s.in)) continue;
            uint64_t off = kHasPlace;
            if (!fsrc_->has_resident(s.layer, s.out)) {
                uint8_t* dst = fsrc_->spare_take(s.layer, off);
                if (dst == nullptr) continue;
                if (!ck(cudaMemcpyAsync(dst, cache_.device_slot(slot), (size_t) lay.blob_bytes(s.layer),
                                        cudaMemcpyDeviceToHost, refill_), "evict", err)) {
                    fsrc_->spare_return(s.layer, off);
                    return false;
                }
            }
            evicting_.push_back({s.layer, s.in, s.out, slot, off});
        }
        return evicting_.empty() || ck(cudaEventRecord(evict_ev_, refill_), "evict event", err);
    }
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                refill_), "refill", err))
            return false;
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    return ck(cudaEventRecord(refill_ev_, refill_), "refill event", err);
}

std::vector<size_t> PeerExperts::incoming() const {
    std::vector<size_t> v;
    v.reserve(pending_.size() + evicting_.size());
    for (const auto& p : pending_) v.push_back((size_t) p.first);
    for (const Evict& x : evicting_) v.push_back((size_t) (x.layer * n_expert_ + x.in));
    return v;
}

bool PeerExperts::reassign(int64_t layer, int32_t in, int32_t out, const uint8_t* blob, std::string& err) {
    const size_t i_in = (size_t) (layer * n_expert_ + in), i_out = (size_t) (layer * n_expert_ + out);
    const int32_t slot = res_[i_in];
    hold_[i_in] = 0;
    if (slot < 0) return true;
    res_[i_in] = kNotResident;                 // the primary computes it from now on
    if (res_[i_out] >= 0 || blob == nullptr) return true;
    for (const auto& p : pending_)
        if (p.second == slot) return true;
    for (const Evict& x : evicting_)
        if (x.slot == slot) return true;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    On on(device_);
    if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), blob, (size_t) lay.blob_bytes(layer), cudaMemcpyHostToDevice,
                            refill_), "reassign", err))
        return false;
    pending_.emplace_back((int32_t) i_out, slot);
    ++swaps_;
    return ck(cudaEventRecord(refill_ev_, refill_), "reassign event", err);
}

bool PeerExperts::copy_to(int64_t layer, int64_t expert, uint8_t* dst, cudaStream_t stream, std::string& err) {
    const uint8_t* p = slot_ptr(layer, expert);
    if (p == nullptr) { err = "peer experts: an expert to copy out is not resident"; return false; }
    return ck(cudaMemcpyPeerAsync(dst, 0, p, device_, (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer),
                                  stream), "copy to the primary", err);
}

bool PeerExperts::issue_pending(bool wait) {
    if (evicting_.empty()) return true;
    On on(device_);
    if (wait) cudaEventSynchronize(evict_ev_);
    else if (cudaEventQuery(evict_ev_) != cudaSuccess) return false;
    // the victims are in RAM now: they are computed from there while their slots take the incoming experts
    const auto& lay = basalt::kernels::cpu::expert_layout();
    for (const Evict& x : evicting_) {
        const bool placed = x.off == kHasPlace;
        // the slot may have been reassigned or its victim promoted since the round was planned
        if (res_[(size_t) (x.layer * n_expert_ + x.out)] != x.slot || (placed && !fsrc_->has_resident(x.layer, x.out))) {
            if (!placed) fsrc_->spare_return(x.layer, x.off);
            continue;
        }
        const uint8_t* b = src_->blob(x.layer, x.in);
        if (!placed && !fsrc_->spare_land(x.layer, x.out, x.off)) {
            fsrc_->spare_return(x.layer, x.off);   // the victim got a place meanwhile: try again next round
            continue;
        }
        if (b == nullptr ||
            cudaMemcpyAsync(cache_.device_slot(x.slot), b, (size_t) lay.blob_bytes(x.layer), cudaMemcpyHostToDevice,
                            refill_) != cudaSuccess) {
            basalt::log::warn("basalt", "peer experts: an exclusive swap failed (layer %d, expert %d)", (int) x.layer,
                         (int) x.in);
            continue;
        }
        res_[(size_t) (x.layer * n_expert_ + x.out)] = kNotResident;
        pending_.emplace_back((int32_t) (x.layer * n_expert_ + x.in), x.slot);
        ++swaps_;
    }
    evicting_.clear();
    return pending_.empty() || cudaEventRecord(refill_ev_, refill_) == cudaSuccess;
}

void PeerExperts::apply_pending(bool wait) {
    if (!issue_pending(wait)) return;
    if (pending_.empty()) return;
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) {
        res_[(size_t) i] = slot;
        if (fsrc_ != nullptr) fsrc_->spare_release((int64_t) i / n_expert_, (int64_t) i % n_expert_);   // peer-only now
    }
    pending_.clear();
}

}  // namespace basalt::core
