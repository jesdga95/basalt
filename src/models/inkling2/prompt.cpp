// src/models/inkling2/prompt.cpp - the prompt path of an Inkling 2 session. A read of more rows than a decode window runs in
// chunks of up to kChunk rows, each one forward through the visits of walk.hpp over chunk-wide row buffers: the Q8_0
// projections and the experts are MMQ products on the int8 tensor cores (the experts grouped by expert and read in place),
// the router's scores and the loop controllers' products run on the tf32 tensor cores, the attention flash style over the
// caches (kernels/attn_prompt.cu), the rest are the decode forward's kernels over all the rows. A chunk leaves the caches as
// the decode forward leaves them: a full visit's cells, a sliding-window visit's ring (the chunk's keys are staged, then
// its last positions moved in) and the convolution rings' last kConvRing positions.
#include "models/inkling2/prompt.hpp"

#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/models/inkling2/prompt_kernels.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/inkling2/dspark.hpp"
#include "models/inkling2/walk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;
namespace mmq = basalt::prefill::mmq;
namespace fp4 = basalt::prefill::fp4;

constexpr int kChunk = kPromptChunk;       // rows of one chunk forward
constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr int kOut = kHeads * kHeadDim;    // the attention output's width
constexpr uint64_t kMmqSpare = 32ull << 20;   // MMQ's stream-k fixup pool

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
    float *msg = nullptr, *mix = nullptr, *pair = nullptr, *hid = nullptr, *nrm = nullptr;
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

void carve(Carve& c, Bufs& b, const Geometry& g) {
    const size_t E = (size_t) g.n_embd, R = kChunk, P = R * (size_t) g.n_used;
    for (float*& p : b.role) p = c.take<float>(R * E);
    b.resid = c.take<float>(R * E);
    // the attention's buffers
    const size_t arena = c.at;
    b.msg = c.take<float>(R * E);
    b.mix = c.take<float>(3 * R * E);
    b.pair = c.take<float>(2 * R * E);
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
    b.xqd = c.take<uint8_t>(mmq::q8_bytes((int64_t) R, std::max<int64_t>((int64_t) E, kOut)));
    b.rlog = c.take<float>(R * (size_t) g.n_expert);
    b.rw = c.take<float>(P);
    b.ids = c.take<int32_t>(P);
    b.bounds = c.take<int32_t>((size_t) g.n_expert + 1);
    b.tok = c.take<int32_t>(P);
    b.dst = c.take<int32_t>(P);
    b.exp = c.take<int32_t>(P);
    b.sort_scr = c.take<uint8_t>(moe_sort_scratch((int) P, g.n_expert));
    const size_t stage = (size_t) kKvHeads * R * kHeadDim * 2;   // fp16 at most
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

class Prompt final : public PromptRunner {
public:
    Prompt(InklingModel& m, const std::vector<VisitCache>& vc, DsparkHook& ds, cudaStream_t s)
        : m_(m), g_(m.geo()), vc_(vc), ds_(ds), stream_(s) {}
    ~Prompt() override {
        if (stream_) cudaStreamSynchronize(stream_);
        for (cudaEvent_t& e : done_)
            if (e) cudaEventDestroy(e);
        if (mem_) cudaFree(mem_);
    }

    bool init(std::string& err) {
        if (!walk(g_, walk_, err)) return false;
        Carve c;
        carve(c, b_, g_);
        if (cudaMalloc(&mem_, c.at) != cudaSuccess) {
            cudaGetLastError();
            err = "out of device memory for the prompt path (" + std::to_string(c.at >> 20) + " MiB)";
            return false;
        }
        c = Carve{};
        c.base = static_cast<uint8_t*>(mem_);
        carve(c, b_, g_);
        for (cudaEvent_t& e : done_)
            if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                err = "cannot create the prompt path's events";
                return false;
            }
        mmq::iota(b_.ident, (int64_t) kChunk * g_.n_used, stream_);
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
        if (!ready(8, kProjQ) || !ready(8, kProjKV) || !ready(8, E)) return false;
        for (int l = 0; l < g_.n_layers; ++l) {
            const LayerWeights& w = m_.layer(l);
            if (!ready(w.gu_type, 2 * (int64_t) F) || !ready(w.d_type, E)) return false;
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the prompt path's setup failed";
            return false;
        }
        return true;
    }

    model::Result read(const model::PrefillRequest& r, std::vector<int32_t>& hist) override {
        return run(r.tokens, nullptr, r.n_tokens, r.pos, r, hist, 0);
    }

    model::Result read_rows(const float* rows, int n, int64_t pos, int32_t pad, std::vector<int32_t>& hist) override {
        model::PrefillRequest r;
        r.n_tokens = n;
        r.pos = pos;
        return run(nullptr, rows, n, pos, r, hist, pad);
    }

private:
    float* buf(Buf x) const { return b_.role[(int) x]; }

    // `n` rows (tokens, or host embedding rows) in chunks of about equal size; one chunk stays queued ahead of the host,
    // which reports a chunk once the card has run it.
    model::Result run(const int32_t* tokens, const float* emb, int n, int64_t pos, const model::PrefillRequest& r,
                      std::vector<int32_t>& hist, int32_t pad) {
        const int chunks = (n + kChunk - 1) / kChunk;
        const int per = (n + chunks - 1) / chunks;
        std::string err;
        int reported = 0, k = 0;
        for (int t0 = 0; t0 < n; t0 += per, ++k) {
            if (r.stop && r.stop()) break;
            const int rows = std::min(per, n - t0);
            if (!chunk(tokens ? tokens + t0 : nullptr, emb ? emb + (size_t) t0 * g_.n_embd : nullptr, rows, pos + t0, err))
                return {model::Status::Error, "inkling2: " + err};
            ds_.note(pos + t0, rows);
            if (tokens) hist.insert(hist.end(), tokens + t0, tokens + t0 + rows);
            else hist.insert(hist.end(), (size_t) rows, pad);
            cudaEventRecord(done_[k % 2], stream_);
            if (k > 0) {   // the chunk before this one has run
                cudaEventSynchronize(done_[(k - 1) % 2]);
                if (r.progress) r.progress(reported);
            }
            reported = t0 + rows;
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2: the prompt failed"};
        if (r.progress && reported > 0) r.progress(reported);
        return {};
    }

    // One chunk of `rows` rows at pos0..: the embedding, the visits, the caches.
    bool chunk(const int32_t* tokens, const float* emb, int rows, int64_t pos0, std::string& err) {
        const int E = g_.n_embd;
        prompt_rows(b_.pos, pos0, rows, b_.dbounds, stream_);
        float* x0 = buf(Buf::X0);
        if (emb) {
            cudaMemcpyAsync(x0, emb, (size_t) rows * E * 4, cudaMemcpyHostToDevice, stream_);
        } else {
            cudaMemcpyAsync(b_.tokd, tokens, (size_t) rows * 4, cudaMemcpyHostToDevice, stream_);
            bk::iq_embed_rows(kTypeBF16, m_.embedding(), (size_t) E * 2, b_.tokd, rows, E, b_.mix, stream_);
            bk::rms_norm_rows(b_.mix, m_.embed_norm(), rows, E, g_.rms_eps, x0, stream_);
        }
        for (const Step& s : walk_.steps) {
            if (const int k = ds_.slot_of_visit(s.visit); k >= 0)   // the drafter's features: the visit's input rows
                cudaMemcpyAsync(ds_.tap(k), buf(s.hid), (size_t) rows * E * 4, cudaMemcpyDeviceToDevice, stream_);
            layer(s, rows, pos0);
        }
        if (ds_.on()) {   // and every row's final normed state, then the rows' context keys and values
            MixStates st;
            for (int i = 0; i < walk_.n_blocks; ++i) st.s[st.n++] = buf(walk_.blocks[i]);
            st.s[st.n++] = buf(walk_.final_hid);
            attnres_mix(st, m_.output_query(), 1, rows, 1e-6f, b_.mix, (int64_t) rows * E, stream_);
            bk::rms_norm_rows(b_.mix, m_.output_norm(), rows, E, g_.rms_eps, ds_.tap(kDsTargets - 1), stream_);
            ds_.commit(b_.pos, rows, stream_);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    // x (rows of `cols` values) as MMQ's q8_1 rows for products of `type`
    void quant(const float* x, int cols, int rows, int type) { mmq::quantize(x, nullptr, b_.xqd, type, cols, cols, rows, stream_); }
    // y[rows][n_out] = W x for the rows quant() rounded: one Q8_0 matrix [n_out][cols] at w
    void product(const uint8_t* w, int n_out, int cols, int rows, float* y) {
        mmq::Product p;
        p.w = w;
        p.type = 8;
        p.w_rows = n_out;
        p.w_cols = cols;
        p.expert_bytes = mmq::matrix_bytes(8, n_out, cols);
        p.n = 1;
        p.xq = b_.xqd;
        p.bounds = b_.dbounds;
        p.ids = b_.ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = y;
        p.ld_dst = n_out;
        mmq_.run(p, stream_);
    }

    // A loop controller over the chunk: out = blend(dest, msg) of stream `st` (out may be dest).
    void controller(const LayerWeights& w, int st, const float* dest, const float* msg, int rows, float* out) {
        const int E = g_.n_embd, rank = g_.gate_rank;
        ctrl_pair_norm(dest, msg, rows, b_.pair, stream_);
        bk::bf16_rows_dot(b_.pair, w.ctrl_down[st], b_.hid, rows, rank, 2 * E, stream_);
        bk::bf16_rows_dot(b_.hid, w.ctrl_gf[st], b_.pair, rows, 2 * E, rank, stream_);
        ctrl_blend(dest, msg, b_.pair, rows, out, stream_);
    }

    // The previous pass's message of stream `j` into msg (the same bits as the decode forward's four at once).
    void message(const Step& s, const LayerWeights& w, int j, int rows) {
        MixStates st;
        st.s[st.n++] = buf(s.bank[0]);
        st.s[st.n++] = buf(s.bank[1]);
        attnres_mix(st, w.msg_query + (size_t) j * g_.n_embd, 1, rows, 1e-6f, b_.msg, (int64_t) rows * g_.n_embd, stream_);
    }

    void layer(const Step& s, int rows, int64_t pos0) {
        const Visit& v = g_.visits[(size_t) s.visit];
        const int l = v.layer;
        const LayerWeights& w = m_.layer(l);
        const bool full = !g_.swa[(size_t) l];
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const VisitCache& c = vc_[(size_t) v.slot];
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
            attnres_mix(st, w.attn_query, full ? 3 : 1, rows, 1e-6f, b_.mix, (int64_t) RE, stream_);
            for (int j = 0; j < 3; ++j) in[j] = b_.mix + (full ? j : 0) * RE;
        }
        if (s.msgs) {
            // controller j writes mix slot j; a sliding-window layer's one input feeds all three, so slot 0 goes last
            for (const int j : {1, 2, 0}) {
                message(s, w, j, rows);
                controller(w, j, in[j], b_.msg, rows, b_.mix + j * RE);
            }
            for (int j = 0; j < 3; ++j) in[j] = b_.mix + j * RE;
        }
        const bool one = !full && in[0] == in[1] && in[1] == in[2];
        if (one) {
            bk::rms_norm_rows(in[0], w.attn_norm, rows, E, g_.rms_eps, b_.nrm, stream_);
        } else {
            const float* wn[3] = {full ? w.q_norm : w.attn_norm, full ? w.k_norm : w.attn_norm, full ? w.v_norm : w.attn_norm};
            for (int j = 0; j < 3; ++j) bk::rms_norm_rows(in[j], wn[j], rows, E, g_.rms_eps, b_.nrm + j * RE, stream_);
        }
        // the projections: the query input's rows, then the key's, then the value's
        const size_t row_bytes = (size_t) E / 32 * 34;
        quant(b_.nrm, E, rows, 8);
        product(w.proj, kProjQ, E, rows, b_.yq);
        if (!one) quant(b_.nrm + RE, E, rows, 8);
        product(w.proj + (size_t) kRowK * row_bytes, kProjKV, E, rows, b_.yk);
        if (!one) quant(b_.nrm + 2 * RE, E, rows, 8);
        product(w.proj + (size_t) kRowV * row_bytes, kProjKV, E, rows, b_.yv);
        // convolutions, clips and the keys: a sliding-window visit stages the chunk's own, a full visit writes its cells
        const bool fp8 = m_.cache_fp8();
        AttnPrep ap;
        ap.yq = b_.yq;
        ap.yk = b_.yk;
        ap.yv = b_.yv;
        ap.pos = b_.pos;
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
        ap.k_cache = c.ring ? b_.k_stage : c.k;
        ap.v_cache = c.ring ? b_.v_stage : c.v;
        ap.cap = c.ring ? kChunk : c.cap;
        ap.fp8 = fp8;
        ap.q = b_.q;
        ap.r = b_.r;
        ap.sink = b_.sink;
        attn_prepare(ap, rows, stream_);
        AttnPromptArgs aa;
        aa.q = b_.q;
        aa.r = b_.r;
        aa.sink = b_.sink;
        aa.rel_table = w.rel_table;
        aa.pos0 = pos0;
        aa.k_cache = c.k;
        aa.v_cache = c.v;
        aa.cap = c.cap;
        aa.k_chunk = ap.k_cache;
        aa.v_chunk = ap.v_cache;
        aa.chunk_cap = ap.cap;
        aa.fp8 = fp8;
        aa.window = c.ring ? g_.window : 0;
        aa.scale = 1.0f / std::sqrt((float) kHeadDim);
        attention_prompt(aa, rows, b_.attn, stream_);
        if (c.ring) stage_to_ring(b_.k_stage, b_.v_stage, kChunk, c.k, c.v, c.cap, pos0, rows, g_.window, fp8 ? 1 : 2, stream_);
        attn_gate(b_.attn, b_.yq, rows, b_.q, stream_);   // the queries are spent: the gated rows go there
        quant(b_.q, kOut, rows, 8);
        product(w.out, E, kOut, rows, b_.o);
        // the attention's convolution and the residual: a block's first layer starts a new sum
        const bool boundary = g_.boundary(l);
        post_conv(b_.o, b_.pos, w.attn_post_conv, c.a_ring, boundary ? nullptr : hid, rows, b_.resid, stream_);
        // the MLP's input
        MixStates st;
        for (int i = 0; i < s.n_blocks; ++i) st.s[st.n++] = blocks[i];
        if (boundary) st.s[st.n++] = hid;
        st.s[st.n++] = b_.resid;
        attnres_mix(st, w.ffn_query, 1, rows, 1e-6f, b_.mix, (int64_t) RE, stream_);
        if (s.msgs) {
            message(s, w, 3, rows);
            controller(w, 3, b_.mix, b_.msg, rows, b_.mix);
        }
        bk::rms_norm_rows(b_.mix, w.ffn_norm, rows, E, g_.rms_eps, b_.nrm, stream_);
        experts(w, l, rows);
        post_conv(b_.moe, b_.pos, w.ffn_post_conv, c.m_ring, b_.resid, rows, out, stream_);
    }

    // The routed experts of the chunk's normed MLP inputs (nrm) into moe: the router's scores as one product, the pairs
    // sorted by expert, the gate/up and down products per expert in two MMQ launches over the experts in place.
    void experts(const LayerWeights& w, int l, int rows) {
        const int E = g_.n_embd, F = g_.n_ff_exp, k = g_.n_used, pairs = rows * k;
        bk::f32_rows_dot(b_.nrm, w.router, b_.rlog, rows, g_.n_expert, E, stream_);
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = k;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router_select(b_.rlog, w.router_bias, rows, ra, b_.ids, b_.rw, stream_);
        if (w.slot_map) {   // a layer with experts in RAM (spill.hpp): its pairs by slot, its RAM experts copied into the staging
            remap_ids(b_.ids, pairs, w.slot_map, stream_);
            const uint64_t kept = (uint64_t) (g_.n_expert - w.n_spill) * w.expert_stride;
            if (!m_.stage_from_card(l, stream_))   // the elastic spill's card copies when the layer has them all
                cudaMemcpyAsync(w.experts + kept, w.spill_host, (size_t) w.n_spill * w.expert_stride, cudaMemcpyHostToDevice, stream_);
            cudaMemsetAsync(w.experts + (uint64_t) g_.n_expert * w.expert_stride, 0, kSlotGuard, stream_);   // spill.hpp
        }
        moe_sort(b_.ids, rows, k, g_.n_expert, b_.sort_scr, b_.bounds, b_.tok, b_.dst, b_.exp, stream_);
        if (w.gu_type == kTypeNvfp4 && w.d_type == kTypeNvfp4 && fp4_shape()) {
            experts_fp4(w, rows);
            return;
        }
        mmq::quantize(b_.nrm, b_.tok, b_.xq, w.gu_type, E, E, pairs, stream_);
        mmq::Product p;
        p.w = w.experts;
        p.type = w.gu_type;
        p.w_rows = 2 * F;
        p.w_cols = E;
        p.expert_bytes = w.expert_stride;
        p.n = g_.n_expert;
        p.xq = b_.xq;
        p.bounds = b_.bounds;
        p.ids = b_.ident;
        p.total_rows = pairs;
        p.max_rows = rows;
        p.opt_rows = (pairs + g_.n_expert - 1) / g_.n_expert;
        p.dst = b_.gu;
        p.ld_dst = 2 * F;
        mmq_.run(p, stream_);
        // an NVFP4 expert's {s_gate, s_up, s_down, 0} tail is read in its blob
        const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, F);
        mmq::SwigluForm form;
        form.gate_softcap = g_.gate_softcap;
        form.up_softcap = g_.up_softcap;
        form.row_expert = b_.exp;
        form.tail_stride = w.expert_stride;
        const bool nvfp4 = w.gu_type == kTypeNvfp4;
        mmq::swiglu(b_.gu, b_.h, pairs, F, false, stream_, nvfp4 ? (const float*) (w.experts + L.bytes - 16) : nullptr, nullptr, 0,
                    form);
        mmq::quantize(b_.h, nullptr, b_.xq, w.d_type, F, F, pairs, stream_);
        mmq::Product d = p;
        d.w = w.experts + L.down_off;
        d.type = w.d_type;
        d.w_rows = E;
        d.w_cols = F;
        d.ids = b_.dst;
        d.dst = b_.gu;
        d.ld_dst = E;
        mmq_.run(d, stream_);
        moe_combine(b_.gu, b_.rw, rows, k, b_.moe, stream_);
    }

    // The NVFP4 experts on the FP4 block-scaled tensor cores, the weights read once a tile in place: each activation row
    // as two FP4 terms (its rounding and the rounding of the rest, E4M3 scales per 16 under one token scale, about 8 bits;
    // one FP4 term costs KLD 0.0025 -> 0.0041), both through each weight fragment on one fp32 chain. Gate/up from the token
    // planes into H as the softcapped SwiGLU, H rounded per expert tile, then down into the pairs' rows.
    bool fp4_shape() const { return g_.n_embd == 2304 && g_.n_ff_exp == 1152 && g_.n_expert <= 1024; }
    void experts_fp4(const LayerWeights& w, int rows) {
        const int E = g_.n_embd, F = g_.n_ff_exp, n = g_.n_expert, pairs = rows * g_.n_used;
        const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, F);
        fp4::plan_blocks(b_.bounds, n, b_.tile_off, b_.blk_off, stream_);
        fp4::quantize_tokens(b_.nrm, rows, E, E, b_.a4x, b_.gsx, stream_, true);
        fp4::Product gu;
        gu.base = w.experts;
        gu.stride = w.expert_stride;
        gu.blk_off = b_.blk_off;
        gu.blocks = fp4::max_blocks(pairs, n);
        gu.n = n;
        gu.w_rows = 2 * F;
        gu.w_cols = E;
        gu.a4 = b_.a4x;
        gu.gs = b_.gsx;
        gu.bounds = b_.bounds;
        gu.ids = b_.tok;
        gu.tokens = rows;
        gu.swiglu = true;
        gu.tail_off = L.bytes - 16;
        gu.gate_softcap = g_.gate_softcap;
        gu.up_softcap = g_.up_softcap;
        gu.dst = b_.h;
        gu.ld_dst = F;
        gu.two = true;
        fp4::run(gu, stream_);
        fp4::quantize(b_.h, nullptr, b_.bounds, b_.tile_off, n, b_.a4h, b_.gsh, F, F, pairs, stream_, true);
        fp4::Product d = gu;
        d.swiglu = false;
        d.ids = nullptr;
        d.w_off = L.down_off;
        d.w_rows = E;
        d.w_cols = F;
        d.a4 = b_.a4h;
        d.gs = b_.gsh;
        d.tile_off = b_.tile_off;
        d.dst = b_.gu;
        d.dst_ids = b_.dst;
        d.two = true;
        d.ld_dst = E;
        fp4::run(d, stream_);
        moe_combine(b_.gu, b_.rw, rows, g_.n_used, b_.moe, stream_);
    }

    InklingModel& m_;
    const Geometry& g_;
    const std::vector<VisitCache>& vc_;
    DsparkHook& ds_;
    cudaStream_t stream_ = nullptr;
    Walk walk_;
    Bufs b_;
    void* mem_ = nullptr;
    cudaEvent_t done_[2] = {};
    mmq::Context mmq_;
};

}  // namespace

std::unique_ptr<PromptRunner> make_prompt(InklingModel& m, const std::vector<VisitCache>& caches, DsparkHook& dspark, void* stream,
                                          std::string& err) {
    auto p = std::make_unique<Prompt>(m, caches, dspark, (cudaStream_t) stream);
    if (!p->init(err)) return nullptr;
    return p;
}

uint64_t prompt_bytes(const Geometry& g) {
    Carve c;
    Bufs b;
    carve(c, b, g);
    return (uint64_t) c.at + kMmqSpare;
}

}  // namespace basalt::models::inkling2
