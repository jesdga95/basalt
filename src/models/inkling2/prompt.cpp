// src/models/inkling2/prompt.cpp - the prompt path of an Inkling 2 session. A read of more rows than a decode window runs in
// chunks of up to kPromptChunk rows (kWidePromptChunk for a long read with room for it), each one forward through the visits of
// walk.hpp over chunk-wide row buffers: the Q8_0 projections and the experts are MMQ products on the int8 tensor cores (the
// experts grouped by expert and read in place), the router's scores and the loop controllers' products run on the tf32 tensor
// cores, the attention flash style over the caches (kernels/attn_prompt.cu), the rest are the decode forward's kernels over all
// the rows. A chunk leaves the caches as the decode forward leaves them: a full visit's cells, a sliding-window visit's ring
// (the chunk's keys are staged, then its last positions moved in) and the convolution rings' last kConvRing positions.
#include "models/inkling2/prompt.hpp"

#include "basalt/core/progress.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/expert_plan.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/models/inkling2/prompt_kernels.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/moe_mxfp8.hpp"
#include "basalt/prefill/q8_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/inkling2/dspark.hpp"
#include "models/inkling2/walk.hpp"
#include "models/inkling2/wide_chunk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;
namespace mmq = basalt::prefill::mmq;
namespace fp4 = basalt::prefill::fp4;
namespace mxfp8 = basalt::prefill::mxfp8;
namespace q8 = basalt::prefill::q8;

constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr int kTypeMxfp8 = 46;
constexpr int kOut = kHeads * kHeadDim;    // the attention output's width
constexpr uint64_t kMmqSpare = 32ull << 20;   // MMQ's stream-k fixup pool
constexpr int kRoutedRows = 1024;          // a chunk of at most this many rows stages only the RAM experts it routes to

// One allocation carved into the chunk's buffers; with no base it only counts, so the sizing and the allocation agree.
struct Carve {
    uint8_t* base = nullptr;
    size_t at = 0;
    template <class T> T* take(size_t n) {
        at = (at + 255) & ~(size_t) 255;
        T* p = base ? reinterpret_cast<T*>(base + at) : nullptr;
        at += n * sizeof(T);
        return p;
    }
};

// The chunk-wide buffers. The arena holds the attention's buffers, then (over them, once they are spent) the experts':
// msg, mix, pair, hid and nrm keep their places into the MLP's input, the experts' rows start at the arena's base.
struct Bufs {
    float* role[kBufs] = {};
    float* resid = nullptr;
    float *msg = nullptr, *mix = nullptr, *hid = nullptr, *nrm = nullptr;
    float *yq = nullptr, *yk = nullptr, *yv = nullptr, *q = nullptr, *r = nullptr, *sink = nullptr, *attn = nullptr, *o = nullptr;
    void* xq = nullptr;          // the experts' q8_1 rows (their input, then H)
    void *a4h = nullptr, *a4x = nullptr;   // NVFP4 experts (inside xq): H's FP4 tiles, the input's FP4 token planes
    float *gsh = nullptr, *gsx = nullptr;  // their token scales
    int32_t *tile_off = nullptr, *blk_off = nullptr;
    float *gu = nullptr, *h = nullptr, *moe = nullptr;   // gu also holds the experts' outputs, h the sum
    void* xqd = nullptr;         // a dense product's q8_1 rows
    float* rlog = nullptr;
    float* rw = nullptr;
    int32_t *ids = nullptr, *bounds = nullptr, *tok = nullptr, *dst = nullptr, *exp = nullptr;
    void* sort_scr = nullptr;
    void *k_stage = nullptr, *v_stage = nullptr;
    int32_t *pos = nullptr, *tokd = nullptr, *ident = nullptr, *dbounds = nullptr;
};

void carve(Carve& c, Bufs& b, const Geometry& g, int chunk) {
    const size_t E = (size_t) g.n_embd, R = (size_t) chunk, P = R * (size_t) g.n_used;
    for (float*& p : b.role) p = c.take<float>(R * E);
    b.resid = c.take<float>(R * E);
    // the attention's buffers
    const size_t arena = c.at;
    b.msg = c.take<float>(4 * R * E);
    b.mix = c.take<float>(3 * R * E);
    b.hid = c.take<float>(R * (size_t) g.gate_rank);
    b.nrm = c.take<float>(3 * R * E);
    b.yq = c.take<float>(R * kProjQ);
    b.yk = c.take<float>(R * kProjKV);
    b.yv = c.take<float>(R * kProjKV);
    b.q = c.take<float>(R * kOut);
    b.r = c.take<float>(R * kHeads * kRel);
    b.sink = c.take<float>(R * kHeads);
    b.attn = c.take<float>(R * kOut);
    b.o = c.take<float>(R * E);
    const size_t end_a = c.at;
    // the experts' buffers, from the arena's base: the MLP's input (mix, its controller, nrm) is read before any is written
    Carve x = c;
    x.at = arena;
    b.xq = x.take<uint8_t>(mmq::q8_bytes((int64_t) P, (int64_t) E));
    b.gu = x.take<float>(P * 2 * (size_t) g.n_ff_exp);
    b.h = x.take<float>(P * (size_t) g.n_ff_exp);
    b.moe = b.h;
    // the NVFP4 experts' FP4 planes take xq's place (an NVFP4 layer has no q8_1 rows)
    const size_t a4h = fp4::a4_bytes((int64_t) P, 2 * g.n_ff_exp, (int) g.n_expert), a4x = fp4::token_bytes((int64_t) R, 2 * (int64_t) E);
    if (a4h + 256 + a4x > mmq::q8_bytes((int64_t) P, (int64_t) E)) throw std::logic_error("inkling2 prompt: the FP4 planes outgrow xq");
    b.a4h = b.xq;
    b.a4x = b.xq ? (uint8_t*) b.xq + ((a4h + 255) & ~(size_t) 255) : nullptr;
    // the experts' input rows are rounded from nrm while it is read, so they end before it
    if (c.base && (uint8_t*) b.xq + mmq::q8_bytes((int64_t) P, (int64_t) E) > (uint8_t*) b.nrm)
        throw std::logic_error("inkling2 prompt: the experts' input rows overlap the MLP's normed rows");
    c.at = std::max(end_a, x.at);
    b.xqd = c.take<uint8_t>(std::max(mmq::q8_bytes((int64_t) R, std::max<int64_t>((int64_t) E, kOut)), q8::act_bytes((int64_t) R, std::max<int64_t>((int64_t) E, kOut))));
    b.rlog = c.take<float>(R * (size_t) g.n_expert);
    b.rw = c.take<float>(P);
    b.ids = c.take<int32_t>(P);
    b.bounds = c.take<int32_t>((size_t) g.n_expert + 1);
    b.tok = c.take<int32_t>(P);
    b.dst = c.take<int32_t>(P);
    b.exp = c.take<int32_t>(P);
    b.sort_scr = c.take<uint8_t>(bk::moe_sort_scratch((int) P, g.n_expert));
    const size_t row_bytes = (size_t) std::max(kv_cache_row_bytes(false, false),
                                               std::max(kv_cache_row_bytes(true, false), kv_cache_row_bytes(false, true)));
    const size_t stage = (size_t) kKvHeads * R * row_bytes;   // staging must fit the widest cache format
    b.k_stage = c.take<uint8_t>(stage);
    b.v_stage = c.take<uint8_t>(stage);
    b.pos = c.take<int32_t>(R);
    b.tokd = c.take<int32_t>(R);
    b.ident = c.take<int32_t>(P);
    b.dbounds = c.take<int32_t>(2);
    b.gsh = c.take<float>(fp4::scale_count((int64_t) P, (int) g.n_expert));
    b.gsx = c.take<float>(R);
    b.tile_off = c.take<int32_t>((size_t) g.n_expert + 1);
    b.blk_off = c.take<int32_t>((size_t) g.n_expert + 1);
}

// One card's side of the prompt path: its chunk buffers, products and stream (the first card's: the reading session's).
struct Lane {
    int device = 0;
    Bufs b;
    void* mem = nullptr;
    mmq::Context mmq;
    cudaStream_t stream = nullptr;
    cudaEvent_t hand = nullptr;   // behind the card's half of a crossing
    // a spill's card: the staging's copies run on `copy` while the visit before computes; `freed` follows the last spilled
    // visit's products over the staging, `staged` the next one's copies
    cudaStream_t copy = nullptr;
    cudaEvent_t freed = nullptr, staged = nullptr;
};

class Prompt final : public PromptRunner {
public:
    Prompt(InklingModel& m, cudaStream_t s) : m_(m), g_(m.geo()) { lane_[0].stream = s; }
    ~Prompt() override {   // every read synced its streams before it returned
        // the copies decode keeps in the arena go back to RAM; an upload into it that may still run keeps the arena
        if (m_.has_spill() && m_.cards() == 1 && !m_.set_prompt_arena(nullptr, 0)) return;
        for (cudaEvent_t& e : done_)
            if (e) cudaEventDestroy(e);
        for (int i = lanes() - 1; i >= 0; --i) {
            Lane& k = lane_[i];
            cudaSetDevice(k.device);
            if (k.hand) cudaEventDestroy(k.hand);
            if (k.freed) cudaEventDestroy(k.freed);
            if (k.staged) cudaEventDestroy(k.staged);
            if (k.copy) cudaStreamDestroy(k.copy);
            if (i > 0 && k.stream) cudaStreamDestroy(k.stream);
            if (i == 0 && wide_va_) {
                if (wide_h_) core::vmm::unmap(wide_va_ + narrow_bytes_, wide_bytes_ - narrow_bytes_, wide_h_);
                if (narrow_h_) core::vmm::unmap(wide_va_, narrow_bytes_, narrow_h_);
                core::vmm::release_range(wide_va_, wide_bytes_);
            } else if (k.mem) {
                cudaFree(k.mem);
            }
        }
        if (hand_) cudaFreeHost(hand_);
        for (void* q : {(void*) px_nrm_, (void*) px_ids_, (void*) px_rw_, (void*) px_out_, (void*) px_cnt_})
            if (q) cudaFreeHost(q);
        if (row_ix_) cudaFree(row_ix_);
        if (routed_ids_) cudaFreeHost(routed_ids_);
    }

    bool init(std::string& err) {
        if (!walk(g_, walk_, err)) return false;
        {   // the runs of visits on each card (walk.hpp)
            std::vector<int> card_of((size_t) g_.n_layers);
            for (int l = 0; l < g_.n_layers; ++l) card_of[(size_t) l] = m_.card_of(l);
            segs_ = split_walk(g_, walk_, card_of);
        }
        for (cudaEvent_t& e : done_)
            if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                err = "cannot create the prompt path's events";
                return false;
            }
        for (int i = lanes() - 1; i >= 0; --i) {   // the first card's last: it stays current
            cur_ = i;
            lane_[i].device = lane_device(i);
            cudaSetDevice(lane_[i].device);
            if (i > 0 && cudaStreamCreateWithFlags(&lane_[i].stream, cudaStreamNonBlocking) != cudaSuccess) {
                err = "cannot create the prompt path's second stream";
                return false;
            }
            if (cudaEventCreateWithFlags(&lane_[i].hand, cudaEventDisableTiming) != cudaSuccess) {
                err = "cannot create the prompt path's events";
                return false;
            }
            if (!init_lane(err)) return false;
        }
        if (m_.peer_device() >= 0) {   // gpu-mode peer: a layer's rows to the second card and its partial sums back
            const size_t R = (size_t) chunk_, E = (size_t) g_.n_embd, P = R * (size_t) g_.n_used;
            constexpr unsigned flags = cudaHostAllocPortable | cudaHostAllocMapped;   // the main card's kernels write the rows there
            if (cudaHostAlloc((void**) &px_nrm_, R * E * 4, flags) != cudaSuccess ||
                cudaHostAlloc((void**) &px_ids_, P * 4, flags) != cudaSuccess ||
                cudaHostAlloc((void**) &px_rw_, P * 4, flags) != cudaSuccess ||
                cudaHostAlloc((void**) &px_cnt_, 64, flags) != cudaSuccess ||
                cudaHostAlloc((void**) &px_out_, R * E * 4, cudaHostAllocPortable) != cudaSuccess ||
                cudaMalloc((void**) &row_ix_, R * 4) != cudaSuccess) {
                cudaGetLastError();
                err = "cannot pin the rows the second card's experts take";
                return false;
            }
        }
        if (m_.spill().any()) {
            if (cudaHostAlloc((void**) &routed_ids_, (size_t) kRoutedRows * g_.n_used * 4, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                err = "cannot pin the prompt path's routed slots";
                return false;
            }
            routed_.assign((size_t) g_.n_expert, 0);
        }
        // the next spilled visit after each step (a spill runs on the first card only)
        next_spill_.assign(walk_.steps.size(), -1);
        for (int si = (int) walk_.steps.size() - 1; si >= 0; --si) {
            next_spill_[(size_t) si] = first_spill_;
            if (m_.layer(g_.visits[(size_t) walk_.steps[(size_t) si].visit].layer).slot_map) first_spill_ = si;
        }
        if (first_spill_ >= 0) {
            Lane& k = lane_[0];
            if (cudaStreamCreateWithFlags(&k.copy, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&k.freed, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&k.staged, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                err = "cannot create the prompt staging's stream";
                return false;
            }
        }
        if (m_.cards() > 1) {   // the crossings' page-locked slots, then one per drafter tap
            size_t pulls = 0;
            for (const Segment& sg : segs_) pulls = std::max(pulls, sg.pull.size());
            tap_slot0_ = (int) pulls;
            const size_t bytes = (pulls + kDsTargets) * (size_t) chunk_ * g_.n_embd * 4;
            if (cudaHostAlloc((void**) &hand_, bytes, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                err = "cannot pin " + std::to_string(bytes >> 20) + " MiB for the layer split's prompt hand-offs";
                return false;
            }
        }
        return true;
    }

    // A long read's wider chunk: on the first card alone, with the card's copies to make room from.
    bool wide_possible() const {
        return m_.copies() && m_.cards() == 1 && m_.peer_device() < 0 && core::vmm::supported();
    }

    // The current card's buffers carved over its arena at chunk_ rows, and their fixed contents.
    void carve_lane() {
        Carve c;
        c.base = static_cast<uint8_t*>(lane().mem);
        carve(c, b(), g_, chunk_);
        mmq::iota(b().ident, (int64_t) chunk_ * g_.n_used, strm());
        // the sorted entries past the ones a chunk fills (pairs another card computes are left out) read as row 0 of expert 0
        const size_t pair_bytes = (size_t) chunk_ * g_.n_used * 4;
        cudaMemsetAsync(b().tok, 0, pair_bytes, strm());
        cudaMemsetAsync(b().dst, 0, pair_bytes, strm());
        cudaMemsetAsync(b().exp, 0, pair_bytes, strm());
    }

    // A read of more than kPromptChunk rows takes kWidePromptChunk-row chunks when the card's copies make room for the rest
    // of the arena and the drafter's wider tap (the counted room, so the choice follows the requests). Before the first
    // chunk, after the prompt loan came back; false leaves the read at kPromptChunk rows.
    struct WideOps {
        Prompt& p;
        bool short_ = false;   // the card had less free memory than counted
        Take take(uint64_t bytes) { return p.m_.take_room(bytes); }
        bool give(uint64_t bytes) { return p.m_.give_room(bytes); }
        WideStep map() {
            bool out_of_memory = false;
            p.wide_h_ = core::vmm::map(p.wide_va_ + p.narrow_bytes_, p.wide_bytes_ - p.narrow_bytes_, &out_of_memory);
            if (p.wide_h_) return WideStep::Done;
            cudaGetLastError();
            short_ = out_of_memory;
            return out_of_memory ? WideStep::Short : WideStep::Broken;
        }
        bool unmap() {
            if (!core::vmm::unmap(p.wide_va_ + p.narrow_bytes_, p.wide_bytes_ - p.narrow_bytes_, p.wide_h_)) return false;
            p.wide_h_ = 0;
            return true;
        }
        WideStep tap_on() {
            const cudaError_t e = cudaMalloc((void**) &p.wide_tap_, dspark_tap_bytes(kWidePromptChunk));
            if (e != cudaSuccess) {
                cudaGetLastError();
                p.wide_tap_ = nullptr;
                short_ = e == cudaErrorMemoryAllocation;
                return short_ ? WideStep::Short : WideStep::Broken;
            }
            p.narrow_tap_ = p.ds_->tap_base();
            p.narrow_tap_rows_ = p.ds_->tap_rows();
            p.ds_->use_tap(p.wide_tap_, kWidePromptChunk);
            return WideStep::Done;
        }
        bool tap_off() {   // the drafter reads its own tap again; the wide one stays owned when its release fails
            p.ds_->use_tap(p.narrow_tap_, p.narrow_tap_rows_);
            if (cudaFree(p.wide_tap_) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            p.wide_tap_ = nullptr;
            return true;
        }
    };
    WideOpen widen(int n) {
        if (!wide_va_ || n <= kPromptChunk || chunk_ != kPromptChunk || m_.loaned() > 0) return WideOpen::Narrow;
        const bool tap = ds_->on();
        const uint64_t extra = wide_bytes_ - narrow_bytes_ + (tap ? dspark_tap_bytes(kWidePromptChunk) : 0);
        wide_ops_.short_ = false;
        const WideOpen w = wide_.open(extra, tap);
        if (wide_.stuck()) {
            m_.wedge();
            basalt::log::error("basalt inkling2", "the wide prompt chunk's memory could not be released; restart the engine");
        }
        if (w == WideOpen::Narrow && wide_ops_.short_ && !std::exchange(wide_short_, true))
            basalt::log::warn("basalt inkling2", "a long prompt read found no memory for its wider chunk: %d rows at a time",
                              kPromptChunk);
        if (w != WideOpen::Wide) return w;
        chunk_ = kWidePromptChunk;
        carve_lane();
        return w;
    }

    // After a wide read: the narrow chunk's buffers again, the extra memory back to the copies. `drained`: the read's
    // streams are confirmed idle (when not, nothing is released or touched). False when the copies did not all come back
    // or the memory stays taken. Never throws.
    bool narrow(bool drained) noexcept {
        if (!wide_.wide()) return !wide_.stuck();
        const WideClose c = wide_.close(drained);
        if (c == WideClose::Stuck) {
            m_.wedge();
            quiet_error("the wide prompt chunk's memory could not be released; restart the engine");
            return false;
        }
        chunk_ = kPromptChunk;
        if (c == WideClose::RestoreFailed) {   // the read fails first; the next read lays the narrow buffers out
            recarve_ = true;
            quiet_error("the copies a long prompt read lent did not all come back");
            return false;
        }
        carve_lane();
        return true;
    }
    static void quiet_error(const char* what) noexcept {
        try {
            basalt::log::error("basalt inkling2", "%s", what);
        } catch (...) {
        }
    }

    // The current card's chunk buffers and products.
    bool init_lane(std::string& err) {
        Carve c;
        carve(c, b(), g_, chunk_);
        if (cur_ == 0 && wide_possible()) {
            // the arena in a range of the wide chunk's size, backed as far as the chunk of kPromptChunk rows needs; a long
            // read backs the rest while it runs
            Carve w;
            Bufs wb;
            carve(w, wb, g_, kWidePromptChunk);
            const uint64_t gran = core::vmm::granularity();
            narrow_bytes_ = (c.at + gran - 1) / gran * gran;
            wide_bytes_ = (w.at + gran - 1) / gran * gran;
            if (!(wide_va_ = core::vmm::reserve(wide_bytes_)) || !(narrow_h_ = core::vmm::map(wide_va_, narrow_bytes_))) {
                cudaGetLastError();
                err = "out of device memory for the prompt path (" + std::to_string(c.at >> 20) + " MiB)";
                return false;
            }
            lane().mem = wide_va_;
        } else if (cudaMalloc(&lane().mem, c.at) != cudaSuccess) {
            cudaGetLastError();
            err = "out of device memory for the prompt path (" + std::to_string(c.at >> 20) + " MiB)";
            return false;
        }
        carve_lane();
        // the arena before `ident` (the one buffer set once) is scratch outside a prompt: decode may keep expert copies there
        // (InklingModel::set_prompt_arena); not under gpu-mode peer, which reads the zeros above
        if (cur_ == 0 && m_.has_spill() && m_.cards() == 1 && m_.peer_device() < 0)
            m_.set_prompt_arena(static_cast<uint8_t*>(lane().mem), (uint64_t) (reinterpret_cast<uint8_t*>(b().ident) - static_cast<uint8_t*>(lane().mem)));
        // the products' tiles
        const int E = g_.n_embd, F = g_.n_ff_exp;
        const auto ready = [&](int type, int64_t rows) {
            if (!mmq::fits(type, rows)) {
                err = "the prompt path's MMQ has no tile for type " + std::to_string(type) + " at " + std::to_string(rows) + " rows";
                return false;
            }
            mmq::load(type, rows);
            return true;
        };
        for (int l = 0; l < g_.n_layers; ++l) {
            const LayerWeights& w = m_.layer(l);
            if (!lane_runs(l)) continue;
            if (w.gu_type == kTypeMxfp8 && w.d_type == kTypeMxfp8 && mxfp8_shape()) continue;
            if (!ready(w.gu_type, 2 * (int64_t) F) || !ready(w.d_type, E)) return false;
        }
        if (cudaStreamSynchronize(strm()) != cudaSuccess) {
            err = "the prompt path's setup failed";
            return false;
        }
        return true;
    }

    model::Result read(const model::PrefillRequest& r, std::vector<int32_t>& hist, const std::vector<VisitCache>& caches,
                       DsparkHook& dspark, void* stream, const Prepare& prepare) override {
        bind(caches, dspark, stream);
        return run(r.tokens, nullptr, r.n_tokens, r.pos, r, hist, 0, prepare);
    }

    model::Result read_rows(const float* rows, int n, int64_t pos, int32_t pad, std::vector<int32_t>& hist,
                            const std::vector<VisitCache>& caches, DsparkHook& dspark, void* stream,
                            const Prepare& prepare) override {
        bind(caches, dspark, stream);
        model::PrefillRequest r;
        r.n_tokens = n;
        r.pos = pos;
        return run(nullptr, rows, n, pos, r, hist, pad, prepare);
    }

private:
    // The second lane: the layer split's card, or gpu-mode peer's, which runs its experts' pairs of every layer that has some.
    int lanes() const { return m_.cards() > 1 || m_.peer_device() >= 0 ? 2 : 1; }
    int lane_device(int i) const { return i == 0 ? m_.card_device(0) : m_.cards() > 1 ? m_.card_device(1) : m_.peer_device(); }
    bool lane_runs(int l) const { return m_.cards() > 1 ? m_.card_of(l) == cur_ : cur_ == 0 || m_.peer_n(l) > 0; }
    Lane& lane() { return lane_[cur_]; }
    Bufs& b() { return lane_[cur_].b; }
    cudaStream_t strm() const { return lane_[cur_].stream; }
    float* buf(Buf x) { return b().role[(int) x]; }
    void bind(const std::vector<VisitCache>& caches, DsparkHook& dspark, void* stream) {
        vc_ = &caches;
        ds_ = &dspark;
        cur_ = 0;
        lane_[0].stream = (cudaStream_t) stream;
        if (m_.cards() > 1) {   // the second card after whatever the session queued (its windows' second-card work)
            cudaEventRecord(lane_[0].hand, lane_[0].stream);
            cudaSetDevice(lane_[1].device);
            cudaStreamWaitEvent(lane_[1].stream, lane_[0].hand, 0);
            cudaSetDevice(lane_[0].device);
        }
    }
    // The layer split: the chunk crosses from card `from` to segment `sg`'s card, which waits for it and takes the row
    // buffers it reads that were last written on `from`, through the page-locked slots.
    void cross(int from, const Segment& sg, int rows) {
        Lane& a = lane_[from];
        Lane& k = lane_[sg.card];
        const size_t n = (size_t) rows * g_.n_embd, slot = (size_t) chunk_ * g_.n_embd;
        cudaSetDevice(a.device);
        for (size_t i = 0; i < sg.pull.size(); ++i)
            cudaMemcpyAsync(hand_ + i * slot, a.b.role[(int) sg.pull[i]], n * 4, cudaMemcpyDeviceToHost, a.stream);
        cudaEventRecord(a.hand, a.stream);
        cudaSetDevice(k.device);
        cudaStreamWaitEvent(k.stream, a.hand, 0);
        for (size_t i = 0; i < sg.pull.size(); ++i)
            cudaMemcpyAsync(k.b.role[(int) sg.pull[i]], hand_ + i * slot, n * 4, cudaMemcpyHostToDevice, k.stream);
        cur_ = sg.card;
    }
    // DSPARK HOOK: a second-card visit's input rows to the drafter's tap on the first card
    void tap_across(int t, const float* rows_dev, int rows) {
        Lane& a = lane_[1];
        const size_t n = (size_t) rows * g_.n_embd;
        float* slot = hand_ + (size_t) (tap_slot0_ + t) * chunk_ * g_.n_embd;
        cudaMemcpyAsync(slot, rows_dev, n * 4, cudaMemcpyDeviceToHost, a.stream);
        cudaEventRecord(a.hand, a.stream);
        cudaSetDevice(lane_[0].device);
        cudaStreamWaitEvent(lane_[0].stream, a.hand, 0);
        cudaMemcpyAsync(ds_->tap(t), slot, n * 4, cudaMemcpyHostToDevice, lane_[0].stream);
        cudaSetDevice(a.device);
    }

    // `n` rows (tokens, or host embedding rows) in chunks of about equal size; one chunk stays queued ahead of the host,
    // which reports a chunk once the card has run it. A read that ends any other way than its final wait (an error, an
    // exception) runs out its streams first (drain), then a wide read gives its memory back (narrow).
    model::Result run(const int32_t* tokens, const float* emb, int n, int64_t pos, const model::PrefillRequest& r,
                      std::vector<int32_t>& hist, int32_t pad, const Prepare& prepare) {
        struct Narrow {   // a read that ends any other way: after the drain, and only what the drain confirmed
            Prompt* p;
            ~Narrow() { (void) p->narrow(p->drained_); }
        } narrow_after{this};
        struct Guard {
            Prompt* p;
            bool armed = true;
            ~Guard() {
                if (armed) p->drained_ = p->drain();
            }
        } guard{this};
        drained_ = false;
        std::string err;
        int reported = 0, k = 0;
        prestaged_ = false;
        const bool go = !(r.stop && r.stop());
        if (go && prepare && (!prepare() || m_.wedged() || m_.loaned() > 0))
            return {model::Status::Error, "inkling2: the prompt path's memory could not be taken back"};
        if (go && std::exchange(recarve_, false)) carve_lane();
        if (go && widen(n) == WideOpen::Failed)
            return {model::Status::Error, "inkling2: the copies a long prompt read borrowed from could not be put back"};
        const int chunks = (n + chunk_ - 1) / chunk_;
        const int per = (n + chunks - 1) / chunks;
        for (int t0 = 0; go && t0 < n; t0 += per, ++k) {
            if (k > 0 && r.stop && r.stop()) break;
            const int rows = std::min(per, n - t0);
            next_long_ = t0 + per < n && std::min(per, n - t0 - per) > kRoutedRows;
            if (!chunk(tokens ? tokens + t0 : nullptr, emb ? emb + (size_t) t0 * g_.n_embd : nullptr, rows, pos + t0, err))
                return {model::Status::Error, "inkling2: " + err};
            ds_->note(pos + t0, rows);
            if (tokens) hist.insert(hist.end(), tokens + t0, tokens + t0 + rows);
            else hist.insert(hist.end(), (size_t) rows, pad);
            if (!ok(cudaEventRecord(done_[k % 2], strm())) || (k > 0 && !ok(cudaEventSynchronize(done_[(k - 1) % 2]))))
                return {model::Status::Error, std::string("inkling2: a prompt chunk's completion could not be waited for: ") +
                                                  cudaGetErrorString(fail_)};
            if (k > 0 && r.progress) r.progress(reported);   // the chunk before this one has run
            reported = t0 + rows;
        }
        if (cudaStreamSynchronize(strm()) != cudaSuccess) return {model::Status::Error, "inkling2: the prompt failed"};
        // a stopped read may leave the next chunk's copies queued (staging_read)
        if (lane_[0].copy && cudaStreamSynchronize(lane_[0].copy) != cudaSuccess)
            return {model::Status::Error, "inkling2: the prompt staging failed"};
        if (!narrow(true)) return {model::Status::Error, "inkling2: a long prompt read's borrowed memory did not all come back"};
        guard.armed = false;
        if (r.progress && reported > 0) r.progress(reported);
        return {};
    }
    // A failed read: every stream it queued work on runs out before the staging, the chunk buffers or a card copy it reads
    // can be used again. When one cannot, the engine ends without the CUDA teardown, which would wait for that work
    // (core::gpu_left_running).
    bool drain() noexcept {
        bool out = true;
        for (int i = m_.cards() - 1; i >= 0; --i) {
            Lane& k = lane_[i];
            if (cudaSetDevice(k.device) != cudaSuccess) {
                out = false;
                continue;
            }
            if (k.copy && cudaStreamSynchronize(k.copy) != cudaSuccess) out = false;
            if (cudaStreamSynchronize(k.stream) != cudaSuccess) out = false;
        }
        cur_ = 0;
        if (cudaSetDevice(lane_[0].device) != cudaSuccess) out = false;
        cudaGetLastError();
        if (!out) {
            basalt::core::gpu_left_running().store(true);
            quiet_error("a failed prompt read's work could not be run out");
        }
        return out;
    }
    // The first failing call's status of the read (staging_failed), for its error.
    bool ok(cudaError_t e) {
        if (e != cudaSuccess && fail_ == cudaSuccess) fail_ = e;
        return e == cudaSuccess;
    }

    // One chunk of `rows` rows at pos0..: the embedding, the visits, the caches.
    bool chunk(const int32_t* tokens, const float* emb, int rows, int64_t pos0, std::string& err) {
        const int E = g_.n_embd;
        for (int i = m_.cards() - 1; i >= 0; --i) {   // each card its rows' positions, the first card's last: it starts
            cur_ = i;
            cudaSetDevice(lane().device);
            prompt_rows(b().pos, pos0, rows, b().dbounds, strm());
        }
        // a long chunk stages every RAM expert of a spilled layer: the copies need no routing, so they run ahead on the copy
        // stream, each layer's once the spilled visit before it has read the staging
        ahead_ = rows > kRoutedRows && first_spill_ >= 0;
        fail_ = cudaSuccess;
        const bool prestaged = prestaged_;
        prestaged_ = false;
        if (ahead_ && !prestaged && !stage_ahead(first_spill_)) {
            err = staging_failed();
            return false;
        }
        float* x0 = buf(Buf::X0);
        if (emb) {
            cudaMemcpyAsync(x0, emb, (size_t) rows * E * 4, cudaMemcpyHostToDevice, strm());
        } else {
            cudaMemcpyAsync(b().tokd, tokens, (size_t) rows * 4, cudaMemcpyHostToDevice, strm());
            bk::iq_embed_rows(kTypeBF16, m_.embedding(), (size_t) E * 2, b().tokd, rows, E, b().mix, strm());
            bk::rms_norm_rows(b().mix, m_.embed_norm(), rows, E, g_.rms_eps, x0, strm());
        }
        for (size_t gi = 0; gi < segs_.size(); ++gi) {   // one segment on one card; the layer split's crossings between
            const Segment& sg = segs_[gi];
            if (gi > 0) cross(segs_[gi - 1].card, sg, rows);
            for (size_t si = sg.s0; si < sg.s1; ++si) {
                const Step& s = walk_.steps[si];
                if (const int k = ds_->slot_of_visit(s.visit); k >= 0) {   // the drafter's features: the visit's input rows
                    if (cur_ == 0) cudaMemcpyAsync(ds_->tap(k), buf(s.hid), (size_t) rows * E * 4, cudaMemcpyDeviceToDevice, strm());
                    else tap_across(k, buf(s.hid), rows);
                }
                step_ = (int) si;
                if (!layer(s, rows, pos0)) {
                    err = staging_failed();
                    return false;
                }
            }
        }
        if (ds_->on()) {   // and every row's final normed state, then the rows' context keys and values
            MixStates st;
            for (int i = 0; i < walk_.n_blocks; ++i) st.s[st.n++] = buf(walk_.blocks[i]);
            st.s[st.n++] = buf(walk_.final_hid);
            attnres_mix(st, m_.output_query(), 1, rows, 1e-6f, b().mix, (int64_t) rows * E, strm());
            bk::rms_norm_rows(b().mix, m_.output_norm(), rows, E, g_.rms_eps, ds_->tap(kDsTargets - 1), strm());
            DsparkHook::PromptScratch ps;
            ps.mix = b().mix;
            ps.kb = b().yk;
            ps.vb = b().yv;
            ps.xq = b().xqd;
            ps.pos3 = reinterpret_cast<int32_t*>(b().rlog);
            ps.bounds = b().dbounds;
            ps.ident = b().ident;
            ds_->commit_prompt(b().pos, rows, lane().mmq, ps, strm());
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    // x (rows of `cols` values) as MMQ's q8_1 rows for products of `type`
    void quant(const float* x, int cols, int rows, int type) { mmq::quantize(x, nullptr, b().xqd, type, cols, cols, rows, strm()); }

    // A loop controller over the chunk: out = blend(dest, msg) of stream `st` (out may be dest).
    void controller(const LayerWeights& w, int st, const float* dest, const float* msg, int rows, float* out) {
        const int rank = g_.gate_rank;
        ctrl_down_prompt(dest, msg, w.ctrl_down[st], rows, rank, b().hid, strm());
        ctrl_up_blend_prompt(dest, msg, b().hid, w.ctrl_gf[st], rows, rank, out, strm());
    }

    // All four streams' messages of the previous pass into b().msg[0..3] in one read of bank[0..1].
    void messages(const Step& s, const LayerWeights& w, int rows) {
        MixStates st;
        st.s[st.n++] = buf(s.bank[0]);
        st.s[st.n++] = buf(s.bank[1]);
        attnres_mix(st, w.msg_query, 4, rows, 1e-6f, b().msg, (int64_t) rows * g_.n_embd, strm());
    }

    // False when the experts' staging failed (nothing of the visit's experts is queued then).
    bool layer(const Step& s, int rows, int64_t pos0) {
        const Visit& v = g_.visits[(size_t) s.visit];
        const int l = v.layer;
        const LayerWeights& w = m_.layer(l);
        const bool full = !g_.swa[(size_t) l];
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const VisitCache& c = (*vc_)[(size_t) v.slot];
        const float* hid = buf(s.hid);
        float* out = buf(s.out);
        const float* blocks[kMaxMix];
        for (int i = 0; i < s.n_blocks; ++i) blocks[i] = buf(s.blocks[i]);
        // the attention's inputs
        const float* in[3] = {hid, hid, hid};
        if (w.attn_query) {
            MixStates st;
            for (int i = 0; i < s.n_blocks; ++i) st.s[st.n++] = blocks[i];
            st.s[st.n++] = hid;
            attnres_mix(st, w.attn_query, full ? 3 : 1, rows, 1e-6f, b().mix, (int64_t) RE, strm());
            for (int j = 0; j < 3; ++j) in[j] = b().mix + (full ? j : 0) * RE;
        }
        if (s.msgs) {
            messages(s, w, rows);
            // controller j writes mix slot j; a sliding-window layer's one input feeds all three, so slot 0 goes last
            for (const int j : {1, 2, 0}) controller(w, j, in[j], b().msg + (size_t) j * RE, rows, b().mix + (size_t) j * RE);
            for (int j = 0; j < 3; ++j) in[j] = b().mix + (size_t) j * RE;
        }
        const bool one = !full && in[0] == in[1] && in[1] == in[2];
        // the projections: fused RMSNorm + Q8 planes quantization into b().xqd, then 2-CTA TMA prefill::q8 GEMM
        const int8_t* proj_c = reinterpret_cast<const int8_t*>(w.proj);
        if (one) {
            rms_norm_q8_planes_prompt(in[0], w.attn_norm, rows, g_.rms_eps, b().xqd, strm());
            const q8::Segment segs[3] = {
                {proj_c, kProjQ, w.proj_q_sc_t, b().yq, kProj},
                {proj_c + (size_t) kRowK * (size_t) E, kProjKV, w.proj_k_sc_t, b().yq + kProjQ, kProj},
                {proj_c + (size_t) kRowV * (size_t) E, kProjKV, w.proj_v_sc_t, b().yq + kProjQ + kProjKV, kProj},
            };
            q8::gemm_segmented(segs, 3, E, b().xqd, rows, q8::Epilogue::Store, strm());
        } else {
            const float* wn[3] = {full ? w.q_norm : w.attn_norm, full ? w.k_norm : w.attn_norm, full ? w.v_norm : w.attn_norm};
            rms_norm_q8_planes_prompt(in[0], wn[0], rows, g_.rms_eps, b().xqd, strm());
            q8::gemm(proj_c, w.proj_q_sc_t, kProjQ, E, b().xqd, rows, b().yq, kProjQ, q8::Epilogue::Store, strm());
            rms_norm_q8_planes_prompt(in[1], wn[1], rows, g_.rms_eps, b().xqd, strm());
            q8::gemm(proj_c + (size_t) kRowK * (size_t) E, w.proj_k_sc_t, kProjKV, E, b().xqd, rows, b().yk, kProjKV,
                     q8::Epilogue::Store, strm());
            rms_norm_q8_planes_prompt(in[2], wn[2], rows, g_.rms_eps, b().xqd, strm());
            q8::gemm(proj_c + (size_t) kRowV * (size_t) E, w.proj_v_sc_t, kProjKV, E, b().xqd, rows, b().yv, kProjKV,
                     q8::Epilogue::Store, strm());
        }
        // convolutions, clips and the keys: a sliding-window visit stages the chunk's own, a full visit writes its cells
        const bool fp8 = m_.cache_fp8();
        const bool int8 = m_.cache_int8();
        AttnPrep ap;
        ap.yq = b().yq;
        ap.yk = one ? b().yq + kProjQ : b().yk;
        ap.yv = one ? b().yq + kProjQ + kProjKV : b().yv;
        if (one) ap.ldq = ap.ldk = ap.ldv = kProj;
        ap.pos = b().pos;
        ap.full = full;
        ap.k_conv = w.k_conv;
        ap.v_conv = w.v_conv;
        ap.q_scale = w.q_clip;
        ap.r_scale = w.r_clip;
        ap.qk_clip = g_.qk_clip;
        ap.rel_clip = g_.rel_clip;
        ap.eps = g_.rms_eps;
        ap.log_floor = g_.log_floor;
        ap.log_alpha = g_.log_alpha;
        ap.k_ring = c.k_ring;
        ap.v_ring = c.v_ring;
        ap.k_cache = c.ring ? b().k_stage : c.k;
        ap.v_cache = c.ring ? b().v_stage : c.v;
        ap.cap = c.ring ? chunk_ : c.cap;
        ap.fp8 = fp8;
        ap.int8 = int8;
        ap.q = b().q;
        ap.r = b().r;
        ap.sink = b().sink;
        attn_prepare(ap, rows, strm());
        AttnPromptArgs aa;
        aa.q = b().q;
        aa.r = b().r;
        aa.sink = b().sink;
        aa.rel_table = w.rel_table;
        aa.pos0 = pos0;
        aa.k_cache = c.k;
        aa.v_cache = c.v;
        aa.cap = c.cap;
        aa.k_chunk = ap.k_cache;
        aa.v_chunk = ap.v_cache;
        aa.chunk_cap = ap.cap;
        aa.fp8 = fp8;
        aa.int8 = int8;
        aa.window = c.ring ? g_.window : 0;
        aa.scale = 1.0f / std::sqrt((float) kHeadDim);
        attention_prompt(aa, rows, b().attn, strm());
        if (c.ring) stage_to_ring(b().k_stage, b().v_stage, chunk_, c.k, c.v, c.cap, pos0, rows, g_.window,
                                  kv_cache_row_bytes(fp8, int8), strm());
        attn_gate_q8_planes_prompt(b().attn, b().yq, one ? kProj : kProjQ, rows, b().xqd, strm());
        q8::gemm(reinterpret_cast<const int8_t*>(w.out), w.out_sc_t, E, kOut, b().xqd, rows, b().o, E,
                 q8::Epilogue::Store, strm());
        // the attention's convolution and the residual: a block's first layer starts a new sum
        const bool boundary = g_.boundary(l);
        post_conv(b().o, b().pos, w.attn_post_conv, c.a_ring, boundary ? nullptr : hid, rows, b().resid, strm());
        // the MLP's input
        MixStates st;
        for (int i = 0; i < s.n_blocks; ++i) st.s[st.n++] = blocks[i];
        if (boundary) st.s[st.n++] = hid;
        st.s[st.n++] = b().resid;
        attnres_mix(st, w.ffn_query, 1, rows, 1e-6f, b().mix, (int64_t) RE, strm());
        if (s.msgs) controller(w, 3, b().mix, b().msg + 3 * RE, rows, b().mix);
        bk::rms_norm_rows(b().mix, w.ffn_norm, rows, E, g_.rms_eps, b().nrm, strm());
        if (!experts(w, l, rows)) return false;
        post_conv(b().moe, b().pos, w.ffn_post_conv, c.m_ring, b().resid, rows, out, strm());
        return true;
    }

    // The routed experts of the chunk's normed MLP inputs (nrm) into moe: the router's scores as one product, the pairs
    // sorted by expert, the gate/up and down products per expert in two MMQ launches over the experts in place.
    bool experts(const LayerWeights& w, int l, int rows) {
        const int E = g_.n_embd, k = g_.n_used, pairs = rows * k;
        bk::f32_rows_dot(b().nrm, w.router, b().rlog, rows, g_.n_expert, E, strm());
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = k;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router_select(b().rlog, w.router_bias, rows, ra, b().ids, b().rw, strm());
        const bool peer = m_.peer_runs(l);
        const int on_peer = peer ? peer_send(w, l, rows) : 0;   // the second card's rows run beside the main card's
        if (w.slot_map) {   // a layer with experts in RAM (spill.hpp): its pairs by slot (the second card's left out), its RAM experts copied into the staging
            remap_ids(b().ids, pairs, peer ? w.slot_map : w.slot_full, strm());
            if (!ahead_) {   // else the copy stream stages it (stage_ahead) and the first product waits (staged_ready)
                const uint8_t* want = nullptr;
                if (rows <= kRoutedRows && !(want = routed(pairs))) return false;
                if (!m_.stage(l, want, strm(), !peer) ||
                    !ok(cudaMemsetAsync(w.experts + (uint64_t) g_.n_expert * w.expert_stride, 0, kSlotGuard, strm())))
                    return false;   // the guard after the last slot: spill.hpp
            }
        }
        if (!local_experts(w, g_.n_expert, rows)) return false;
        if (on_peer > 0) peer_join(on_peer);
        return true;
    }

    // The rows with a choice on the second card, their normed rows, choices and weights to it (the main card's kernel writes them
    // to page-locked memory), which runs its experts' pairs over the same kernels into its own partial sums (the choices it has
    // no expert for are left out) and sends them back (peer_join). The rows are counted first: the plan is one small launch and
    // the host waits for it. Returns the rows sent.
    int peer_send(const LayerWeights& w, int l, int rows) {
        const int k = g_.n_used;
        const Lane& m = lane_[0];
        peer_rows_plan(b().ids, rows, k, w.slot_map, row_ix_, px_cnt_, strm());
        cudaStreamSynchronize(strm());
        const int n = *static_cast<volatile int32_t*>(px_cnt_);
        if (n == 0) return 0;
        peer_rows_gather(b().nrm, b().ids, b().rw, k, row_ix_, n, px_nrm_, px_ids_, px_rw_, strm());
        cudaEventRecord(m.hand, strm());
        const size_t pairs = (size_t) n * k;
        cur_ = 1;
        cudaSetDevice(lane().device);
        cudaStreamWaitEvent(strm(), m.hand, 0);
        cudaMemcpyAsync(b().nrm, px_nrm_, (size_t) n * g_.n_embd * 4, cudaMemcpyHostToDevice, strm());
        cudaMemcpyAsync(b().ids, px_ids_, pairs * 4, cudaMemcpyHostToDevice, strm());
        cudaMemcpyAsync(b().rw, px_rw_, pairs * 4, cudaMemcpyHostToDevice, strm());
        LayerWeights pw = w;   // the layer's experts on this card: its own run of blobs, no staging
        pw.experts = m_.peer_base(l);
        pw.expert_stride = m_.peer_stride(l);
        pw.slot_map = nullptr;
        pw.n_spill = 0;
        remap_ids(b().ids, pairs, m_.peer_map(l), strm());
        local_experts(pw, m_.peer_n(l), n);   // no staging on this card (slot_map null): always true
        cudaMemcpyAsync(px_out_, b().moe, (size_t) n * g_.n_embd * 4, cudaMemcpyDeviceToHost, strm());
        cudaEventRecord(lane().hand, strm());
        cur_ = 0;
        cudaSetDevice(lane().device);
        return n;
    }
    void peer_join(int n) {
        cudaStreamWaitEvent(strm(), lane_[1].hand, 0);
        cudaMemcpyAsync(b().gu, px_out_, (size_t) n * g_.n_embd * 4, cudaMemcpyHostToDevice, strm());   // gu is spent
        peer_rows_add(b().gu, row_ix_, n, b().moe, strm());
    }

    // The pairs of b().ids (expert slots 0..n-1, or -1 for a pair another card computes) over the experts of `w`, combined into moe.
    // False when the next spilled visit's staging failed (staging_read).
    bool local_experts(const LayerWeights& w, int n, int rows) {
        const int E = g_.n_embd, F = g_.n_ff_exp, k = g_.n_used, pairs = rows * k;
        bk::moe_sort(b().ids, rows, k, n, b().sort_scr, b().bounds, b().tok, b().dst, b().exp, strm());
        if (w.gu_type == kTypeNvfp4 && w.d_type == kTypeNvfp4 && fp4_shape()) {
            return experts_fp4(w, n, rows);
        }
        if (w.gu_type == kTypeMxfp8 && w.d_type == kTypeMxfp8 && mxfp8_shape()) {
            return experts_mxfp8(w, n, rows);
        }
        mmq::quantize(b().nrm, b().tok, b().xq, w.gu_type, E, E, pairs, strm());
        mmq::Product p;
        p.w = w.experts;
        p.type = w.gu_type;
        p.w_rows = 2 * F;
        p.w_cols = E;
        p.expert_bytes = w.expert_stride;
        p.n = n;
        p.xq = b().xq;
        p.bounds = b().bounds;
        p.ids = b().ident;
        p.total_rows = pairs;
        p.max_rows = rows;
        p.opt_rows = (pairs + n - 1) / n;
        p.dst = b().gu;
        p.ld_dst = 2 * F;
        if (!staged_ready(w)) return false;
        lane().mmq.run(p, strm());
        // an NVFP4 expert's {s_gate, s_up, s_down, 0} tail is read in its blob
        const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, F);
        mmq::SwigluForm form;
        form.gate_softcap = g_.gate_softcap;
        form.up_softcap = g_.up_softcap;
        form.row_expert = b().exp;
        form.tail_stride = w.expert_stride;
        const bool nvfp4 = w.gu_type == kTypeNvfp4;
        mmq::swiglu(b().gu, b().h, pairs, F, false, strm(), nvfp4 ? (const float*) (w.experts + L.bytes - 16) : nullptr, nullptr, 0,
                    form);
        mmq::quantize(b().h, nullptr, b().xq, w.d_type, F, F, pairs, strm());
        mmq::Product d = p;
        d.w = w.experts + L.down_off;
        d.type = w.d_type;
        d.w_rows = E;
        d.w_cols = F;
        d.ids = b().dst;
        d.dst = b().gu;
        d.ld_dst = E;
        lane().mmq.run(d, strm());
        if (w.slot_map && !staging_read()) return false;
        moe_combine(b().gu, b().rw, rows, k, b().moe, strm(), b().ids);
        return true;
    }

    // Step si's RAM experts into the staging on the copy stream, after the products queued so far on the compute stream:
    // the copy stream waits for them, `staged` follows the copies and the guard after the layer's last slot (spill.hpp).
    // False when any of it could not be queued; `staged` then publishes nothing and the chunk fails.
    bool stage_ahead(int si) {
        Lane& k = lane();
        const int l = g_.visits[(size_t) walk_.steps[(size_t) si].visit].layer;
        const LayerWeights& w = m_.layer(l);
        return ok(cudaEventRecord(k.freed, k.stream)) && ok(cudaStreamWaitEvent(k.copy, k.freed, 0)) && m_.stage(l, nullptr, k.copy, !m_.peer_runs(l)) &&
               ok(cudaMemsetAsync(w.experts + (uint64_t) g_.n_expert * w.expert_stride, 0, kSlotGuard, k.copy)) &&
               ok(cudaEventRecord(k.staged, k.copy));
    }
    // Before a spilled visit's first product over the staging: on the copy stream's chunk, it waits for the copies, so the
    // sort and the activations' rounding before it run under them.
    bool staged_ready(const LayerWeights& w) { return !ahead_ || !w.slot_map || ok(cudaStreamWaitEvent(strm(), lane().staged, 0)); }
    // A spilled visit's products are queued over the staging: the chunk's next spilled visit's copies may start, or after the
    // chunk's last one, the next chunk's first when that chunk stages ahead too.
    bool staging_read() {
        if (!ahead_) return true;
        if (const int next = next_spill_[(size_t) step_]; next >= 0) return stage_ahead(next);
        if (!next_long_) return true;
        prestaged_ = true;
        return stage_ahead(first_spill_);
    }
    std::string staging_failed() {
        const cudaError_t e = fail_ != cudaSuccess ? fail_ : cudaGetLastError();
        return std::string("staging a layer's RAM experts failed") + (e != cudaSuccess ? std::string(": ") + cudaGetErrorString(e) : "");
    }

    // The slots a short chunk's pairs route to, for the staging: the host waits for the router, as a short chunk reads only a
    // part of a layer's RAM experts. Null when the readback failed.
    const uint8_t* routed(int pairs) {
        if (cudaMemcpyAsync(routed_ids_, b().ids, (size_t) pairs * 4, cudaMemcpyDeviceToHost, strm()) != cudaSuccess ||
            cudaStreamSynchronize(strm()) != cudaSuccess)
            return nullptr;
        std::fill(routed_.begin(), routed_.end(), (uint8_t) 0);
        for (int i = 0; i < pairs; ++i)
            if (routed_ids_[i] >= 0 && routed_ids_[i] < g_.n_expert) routed_[(size_t) routed_ids_[i]] = 1;
        return routed_.data();
    }

    // The NVFP4 experts on the FP4 block-scaled tensor cores, the weights read once a tile in place: each activation row
    // as two FP4 terms (its rounding and the rounding of the rest, E4M3 scales per 16 under one token scale, about 8 bits;
    // one FP4 term costs KLD 0.0025 -> 0.0041), both through each weight fragment on one fp32 chain. Gate/up from the token
    // planes into H as the softcapped SwiGLU, H rounded per expert tile, then down into the pairs' rows.
    bool fp4_shape() const { return g_.n_embd == 2304 && g_.n_ff_exp == 1152 && g_.n_expert <= 1024; }
    bool mxfp8_shape() const { return g_.n_embd % 128 == 0 && g_.n_ff_exp % 128 == 0 && g_.n_expert <= 1024; }
    bool experts_mxfp8(const LayerWeights& w, int n, int rows) {
        const int E = g_.n_embd, F = g_.n_ff_exp, pairs = rows * g_.n_used;
        const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, F);
        mxfp8::plan_blocks(b().bounds, n, b().blk_off, strm());
        mxfp8::quantize_tokens(b().nrm, rows, E, E, b().xq, strm());
        mxfp8::Product gu{};
        gu.base = w.experts;
        gu.stride = w.expert_stride;
        gu.w_off = 0;
        gu.w_rows = 2 * F;
        gu.w_cols = E;
        gu.n = n;
        gu.act = b().xq;
        gu.act_rows = rows;
        gu.bounds = b().bounds;
        gu.blk_off = b().blk_off;
        gu.blocks = mxfp8::max_blocks(pairs, n);
        gu.ids = b().tok;
        auto* h16 = reinterpret_cast<uint16_t*>(b().h);
        gu.dst_f16 = h16;
        gu.ld_dst = F;
        gu.swiglu = true;
        gu.gate_softcap = g_.gate_softcap;
        gu.up_softcap = g_.up_softcap;
        if (!staged_ready(w)) return false;
        mxfp8::run(gu, strm());
        mxfp8::quantize_f16(h16, pairs, F, F, b().xq, strm());
        mxfp8::Product d = gu;
        d.swiglu = false;
        d.ids = nullptr;
        d.w_off = L.down_off;
        d.w_rows = E;
        d.w_cols = F;
        d.act = b().xq;
        d.act_rows = pairs;
        auto* gu16 = reinterpret_cast<uint16_t*>(b().gu);
        d.dst = nullptr;
        d.dst_f16 = gu16;
        d.dst_ids = b().dst;
        d.ld_dst = E;
        mxfp8::run(d, strm());
        if (w.slot_map && !staging_read()) return false;
        moe_combine_f16(gu16, b().rw, rows, g_.n_used, b().moe, strm(), b().ids);
        return true;
    }
    bool experts_fp4(const LayerWeights& w, int n, int rows) {
        const int E = g_.n_embd, F = g_.n_ff_exp, pairs = rows * g_.n_used;
        const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, F);
        fp4::plan_blocks(b().bounds, n, b().tile_off, b().blk_off, strm());
        fp4::quantize_tokens(b().nrm, rows, E, E, b().a4x, b().gsx, strm(), true);
        fp4::Product gu;
        gu.base = w.experts;
        gu.stride = w.expert_stride;
        gu.blk_off = b().blk_off;
        gu.blocks = fp4::max_blocks(pairs, n);
        gu.n = n;
        gu.w_rows = 2 * F;
        gu.w_cols = E;
        gu.a4 = b().a4x;
        gu.gs = b().gsx;
        gu.bounds = b().bounds;
        gu.ids = b().tok;
        gu.tokens = rows;
        gu.swiglu = true;
        gu.tail_off = L.bytes - 16;
        gu.gate_softcap = g_.gate_softcap;
        gu.up_softcap = g_.up_softcap;
        auto* h16 = reinterpret_cast<uint16_t*>(b().h);
        gu.dst_f16 = h16;
        gu.ld_dst = F;
        gu.two = true;
        if (!staged_ready(w)) return false;
        fp4::run(gu, strm());
        fp4::quantize_f16(h16, b().bounds, b().tile_off, n, b().a4h, b().gsh, F, F, pairs, strm());
        fp4::Product d = gu;
        d.swiglu = false;
        d.ids = nullptr;
        d.w_off = L.down_off;
        d.w_rows = E;
        d.w_cols = F;
        d.a4 = b().a4h;
        d.gs = b().gsh;
        d.tile_off = b().tile_off;
        auto* gu16 = reinterpret_cast<uint16_t*>(b().gu);
        d.dst = nullptr;
        d.dst_f16 = gu16;
        d.dst_ids = b().dst;
        d.two = true;
        d.ld_dst = E;
        fp4::run(d, strm());
        if (w.slot_map && !staging_read()) return false;
        moe_combine_f16(gu16, b().rw, rows, g_.n_used, b().moe, strm(), b().ids);
        return true;
    }

    InklingModel& m_;
    const Geometry& g_;
    int chunk_ = kPromptChunk;   // rows of one chunk forward
    // the first card's arena when a long read may widen its chunk: the range, its two backings, the wide read's tap
    uint8_t* wide_va_ = nullptr;
    uint64_t narrow_bytes_ = 0, wide_bytes_ = 0;
    unsigned long long narrow_h_ = 0, wide_h_ = 0;
    float *wide_tap_ = nullptr, *narrow_tap_ = nullptr;
    int narrow_tap_rows_ = 0;
    bool wide_short_ = false;
    bool drained_ = false;   // the last failed read's streams were run out (drain)
    bool recarve_ = false;   // a wide read whose copies did not all return left its buffers laid out wide
    WideOps wide_ops_{*this};
    WideChunk<WideOps> wide_{wide_ops_};
    const std::vector<VisitCache>* vc_ = nullptr;   // the reading session's caches, drafter and stream
    DsparkHook* ds_ = nullptr;
    Walk walk_;
    std::vector<Segment> segs_;   // the walk's runs of visits on each card (walk.hpp split_walk)
    Lane lane_[2];                // each card's buffers, products and stream
    int cur_ = 0;                 // the card the launches go to
    float* px_nrm_ = nullptr, *px_rw_ = nullptr, *px_out_ = nullptr;   // gpu-mode peer: page-locked rows, weights and sums crossing
    int32_t* px_ids_ = nullptr, *px_cnt_ = nullptr, *row_ix_ = nullptr;   // and the rows sent: their count (page-locked), their rows
    float* hand_ = nullptr;       // the layer split: page-locked slots of a chunk's rows crossing between the cards
    int tap_slot0_ = 0;           // the first drafter tap's slot
    cudaEvent_t done_[2] = {};
    int32_t* routed_ids_ = nullptr;   // page-locked: a short chunk's pairs by slot (routed)
    std::vector<uint8_t> routed_;     // [slot] whether a pair of the chunk routes to it
    std::vector<int> next_spill_;     // [step] the next step whose layer has RAM experts (-1: none)
    int first_spill_ = -1;            // the first such step
    int step_ = 0;                    // the step being queued
    bool ahead_ = false;              // this chunk stages on the copy stream (stage_ahead)
    bool next_long_ = false;          // the read's next chunk stages on the copy stream too
    bool prestaged_ = false;          // the chunk before staged this chunk's first spilled visit
    cudaError_t fail_ = cudaSuccess;  // the read's first failing call (ok)
};

}  // namespace

std::unique_ptr<PromptRunner> make_prompt(InklingModel& m, void* stream, std::string& err) {
    auto p = std::make_unique<Prompt>(m, (cudaStream_t) stream);
    if (!p->init(err)) return nullptr;
    return p;
}

uint64_t prompt_bytes(const Geometry& g) {
    Carve c;
    Bufs b;
    carve(c, b, g, kPromptChunk);
    return (uint64_t) c.at + kMmqSpare;
}

}  // namespace basalt::models::inkling2
