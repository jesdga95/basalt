// src/dense/model.cpp - the dense stack's shared scratch and geometry checks.
#include "internal.hpp"

#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"

#include <algorithm>
#include <cstdlib>

namespace basalt::dense {

Arena::~Arena() {
    if (base_) cudaFree(base_);
}

bool Arena::reserve(size_t bytes) {
    if (base_) cudaFree(base_);
    base_ = nullptr;
    size_ = used_ = 0;
    if (bytes == 0) return true;
    if (cudaMalloc(&base_, bytes) != cudaSuccess) {
        cudaGetLastError();
        base_ = nullptr;
        return false;
    }
    size_ = bytes;
    return true;
}

void* Arena::take_bytes(size_t n) {
    const size_t at = (used_ + 255) & ~(size_t) 255;
    if (at + n > size_) return nullptr;
    used_ = at + n;
    return base_ + at;
}

int Geometry::gdn_layers() const { return (int) std::count(kinds.begin(), kinds.end(), Mixer::Gdn); }
int Geometry::attn_layers() const { return (int) std::count(kinds.begin(), kinds.end(), Mixer::Attention); }

std::string Geometry::check() const {
    if ((int) kinds.size() != n_layers || n_layers <= 0) return "the layer kinds do not cover the layers";
    if (n_embd % 256 || n_ff % 256) return "widths must be whole 256-value blocks";
    if (attn_layers() > 0) {
        if (head_dim != 256) return "attention heads must be 256 wide";
        if (n_kv <= 0 || n_head % n_kv || n_head / n_kv > 8) return "attention head grouping not supported";
        if (rope.rot <= 0 || rope.rot > head_dim || rope.rot % 2 || rope.rot / 2 > 64) return "rotary width not supported";
    }
    if (gdn_layers() > 0) {
        if (!kernels::gdn_prompt_supported(gdn_state, gdn_k_heads, gdn_v_heads, gdn_conv)) return "GDN geometry not supported";
        if (!kernels::dec_gdn_rec_supported(n_embd)) return "GDN window kernel not built for this width";
    }
    return {};
}

namespace {

// bytes of the decode window's scratch
// the block drafter's rows in the window scratch
size_t block_scratch_floats(const Geometry& g, const BlockDrafter& bd) {
    if (!bd.present) return 0;
    const size_t W = kBatch, AQ = (size_t) bd.n_head * bd.head_dim, KV = (size_t) bd.n_kv * bd.head_dim;
    return W * (3 * AQ + 2 * KV + bd.layers[0].conv_a.rows + g.n_embd + kernels::kLogprobStride + bd.rank + 3) +
           kernels::gqa_decode_scratch_floats(bd.window + 256, kWindow, bd.n_head, bd.head_dim) + 64 * 64;
}

int64_t widest_input(const Geometry& g) {
    const size_t AQ = (size_t) g.n_head * g.head_dim;
    return (int64_t) std::max<size_t>({(size_t) g.n_embd, (size_t) g.n_ff, (size_t) g.value_width(), AQ, 2 * (size_t) g.n_embd});
}
// a window's activation buffers for the widest input: the window gemm's bf16 rows, MMQ's q8_1 rows
size_t act_slot(const Geometry& g) { return (kernels::wgemm::act_bytes(widest_input(g)) + 255) / 256 * 256; }
size_t q8_slot(const Geometry& g) { return (prefill::mmq::q8_bytes(kBatch, widest_input(g)) + 255) / 256 * 256; }

// The window gemm's split-K workspace for every call a window makes (forward.cpp): the groups and each matrix alone, in
// either format (a superset is harmless).
void wgemm_need(const Geometry& g, size_t& partial, size_t& tickets) {
    namespace wg = kernels::wgemm;
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    const int64_t C = g.conv_channels(), VD = g.value_width(), F = g.n_ff;
    struct Shape {
        int64_t rows[3];
        int n;
        int64_t K;
        wg::Epilogue epi;
    };
    std::vector<Shape> shapes = {{{2 * AQ, KV, KV}, 3, H, wg::Epilogue::Store}, {{2 * AQ}, 1, H, wg::Epilogue::Store},
                                 {{KV}, 1, H, wg::Epilogue::Store},          {{H}, 1, AQ, wg::Epilogue::Add},
                                 {{F, F}, 2, H, wg::Epilogue::SwiGLU},       {{F}, 1, H, wg::Epilogue::Store},
                                 {{H}, 1, F, wg::Epilogue::Add},             {{g.n_vocab}, 1, H, wg::Epilogue::Store},
                                 {{256}, 1, H, wg::Epilogue::Store},   // the block drafter's selector (its rank)
                                 {{kDraftVocab}, 1, H, wg::Epilogue::Store}};   // and its head's rows
    if (g.gdn_layers() > 0) {
        shapes.push_back({{C, VD}, 2, H, wg::Epilogue::Store});
        shapes.push_back({{C}, 1, H, wg::Epilogue::Store});
        shapes.push_back({{VD}, 1, H, wg::Epilogue::Store});
        shapes.push_back({{H}, 1, VD, wg::Epilogue::Add});
    }
    partial = tickets = 0;
    for (const Shape& sh : shapes)
        for (const wg::Format f : {wg::Format::F8, wg::Format::Nvfp4, wg::Format::Bf16}) {
            if (sh.K <= 0 || sh.K % 256 != 0) continue;
            wg::Call c;
            c.fmt = f;
            c.K = sh.K;
            c.n = sh.n;
            c.epi = sh.epi;
            bool ok = true;
            for (int i = 0; i < sh.n; ++i) {
                c.m[i].rows = sh.rows[i];
                ok = ok && sh.rows[i] > 0 && (f == wg::Format::F8 || sh.rows[i] % 128 == 0);
            }
            if (!ok || (f != wg::Format::Nvfp4 && sh.epi == wg::Epilogue::SwiGLU)) continue;
            size_t p, t;
            wg::workspace_need(c, p, t);
            partial = std::max(partial, p);
            tickets = std::max(tickets, t);
        }
}

size_t window_bytes(const Geometry& g, const BlockDrafter& bd, int64_t max_context) {
    const size_t W = kBatch, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim;
    size_t f = 4 * W * H + W * g.value_width() + W * (2 * AQ + 2 * (size_t) g.n_kv * g.head_dim + 3 * AQ) + 3 * W * g.n_ff;
    if (g.attn_layers() > 0) f += kernels::gqa_decode_scratch_floats(max_context, kWindow, g.n_head);
    f += W * (2 * (size_t) g.conv_channels() + 2 * (size_t) g.gdn_v_heads) + (sizeof(BatchTable) + 3) / 4;
    f += W * 2 * H + kernels::row_logprobs_scratch_floats(kWindow, g.n_vocab) + W * g.value_width() + W * (size_t) g.n_vocab;
    f += block_scratch_floats(g, bd);
    size_t wp, wt;
    wgemm_need(g, wp, wt);
    return f * 4 + 2 * act_slot(g) + q8_slot(g) + (wp + wt + 1) * 4 + W * 4 * 6 + 76 * 256;
}

size_t chunk_bytes(const Geometry& g) {
    const size_t T = kChunk, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim, C = (size_t) g.conv_channels();
    const size_t VD = (size_t) g.value_width(), HV = (size_t) g.gdn_v_heads;
    const size_t widest = std::max<size_t>({H, (size_t) g.n_ff, VD, AQ});
    const size_t keep = 6 * T * H * 4 + H * 4 +
                        std::max(prefill::mmq::q8_bytes(T, (int64_t) widest), prefill::fp8::act_bytes(T, (int64_t) widest)) +
                        prefill::fp4_dense::act_bytes((int64_t) T, (int64_t) std::max<size_t>(H, g.n_ff)) + 2 * T * 4 + 5 * 256;
    const size_t gdn = g.gdn_layers() ? (2 * T * C + 3 * T * VD + 4 * T * HV) * 4 + kernels::gdn_work_bytes(T) : 0;
    const size_t attn = g.attn_layers() ? T * (2 * AQ + 2 * (size_t) g.n_kv * g.head_dim + 3 * AQ) * 4 +
                                              kernels::gqa_prompt_split_floats((int64_t) T, g.n_head, kAttnSplits) * 4
                                        : 0;
    const size_t ffn = 3 * T * g.n_ff * 4;
    return keep + std::max({gdn, attn, ffn}) + T * 4 * 6 + 32 * 256;
}

}  // namespace

namespace {
// The block drafter as decode windows run it, on the window gemm: q, o, gate, up and down rounded to NVFP4 planes in place
// and the convolutions repacked to Q8 planes in place (prompt chunks never read them); k and v (NVFP4) and the feature
// projection (Q8 planes), which prompt chunks read as blocks, get copies.  Then the window gemm's workspace grows to its
// calls if they need more.
bool decode_drafter(Model& m, std::string& err) {
    m.bd_dec = m.w.dflash;
    BlockDrafter& bd = m.bd_dec;
    if (!bd.present || m.o.drafter != Drafter::Block) return true;
    void* tmp = nullptr;
    size_t tmp_bytes = 0;
    auto q8 = [](const Linear& l) { return l.type == 8 && l.cols % 256 == 0 && l.rows > 0; };
    auto bytes = [](const Linear& l) { return (size_t) l.rows * (size_t) l.cols / 32 * 34; };
    auto in_place = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        const size_t n = bytes(l);
        if (n > tmp_bytes) {
            if (tmp) cudaFree(tmp);
            tmp = nullptr;
            if (cudaMalloc(&tmp, n) != cudaSuccess) return false;
            tmp_bytes = n;
        }
        if (cudaMemcpy(tmp, l.w, n, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
        kernels::wgemm::q8_planes(tmp, l.rows, l.cols, const_cast<void*>(l.w), nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        l.type = kTypeQ8P;
        return true;
    };
    auto copy = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes(l)) != cudaSuccess) return false;
        m.owned.push_back(p);
        kernels::wgemm::q8_planes(l.w, l.rows, l.cols, p, nullptr);
        l.w = p;
        l.type = kTypeQ8P;
        return true;
    };
    // The drafter's big products (gate, up, down, q, o) in NVFP4 planes, rounded from its Q8_0 at load: the drafts' precision
    // only (every draft is verified exactly, so the output does not change); about half the bytes a pass reads.
    auto to_fp4 = [&](Linear& l) -> bool {
        if (!q8(l) || !kernels::nvfp4p::fits(l.rows, l.cols)) return in_place(l);
        const size_t n = bytes(l);
        if (n > tmp_bytes) {
            if (tmp) cudaFree(tmp);
            tmp = nullptr;
            if (cudaMalloc(&tmp, n) != cudaSuccess) return false;
            tmp_bytes = n;
        }
        void* sc = nullptr;
        if (cudaMemcpy(tmp, l.w, n, cudaMemcpyDeviceToDevice) != cudaSuccess || cudaMalloc(&sc, 16) != cudaSuccess) return false;
        m.owned.push_back(sc);
        kernels::nvfp4p::from_q8(tmp, l.rows, l.cols, const_cast<void*>(l.w), (float*) sc, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        l.scale = (const float*) sc;
        l.type = kTypeNvfp4Planes;
        return true;
    };
    bool ok = true;
    for (BlockDrafter::Layer& D : bd.layers) {
        for (Linear* l : {&D.gate, &D.up, &D.down, &D.q, &D.o}) ok = ok && to_fp4(*l);
        for (Linear* l : {&D.conv_a, &D.conv_f}) ok = ok && in_place(*l);
        if (kernels::nvfp4p::fits(D.k.rows, D.k.cols) && kernels::nvfp4p::fits(D.v.rows, D.v.cols)) {
            // k and v as NVFP4 copies (prompt chunks read their blocks), so q, k and v stay one call
            for (Linear* l : {&D.k, &D.v}) {
                void* p = nullptr;
                void* sc = nullptr;
                if (!q8(*l) || cudaMalloc(&p, kernels::nvfp4p::bytes(l->rows, l->cols)) != cudaSuccess ||
                    cudaMalloc(&sc, 16) != cudaSuccess) {
                    ok = false;
                    break;
                }
                m.owned.push_back(p);
                m.owned.push_back(sc);
                kernels::nvfp4p::from_q8(l->w, l->rows, l->cols, p, (float*) sc, nullptr);
                l->w = p;
                l->scale = (const float*) sc;
                l->type = kTypeNvfp4Planes;
            }
        } else {
            ok = ok && copy(D.k) && copy(D.v);
        }
    }
    for (Linear& l : bd.fc) ok = ok && copy(l);
    if (tmp) cudaFree(tmp);
    if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
        err = m.o.arch + ": laying out the block drafter for decode windows failed";
        return false;
    }
    // the workspace for the drafter's calls (each matrix, q + k + v, k + v)
    namespace wg = kernels::wgemm;
    size_t need_p = m.ws.w8.partial_floats, need_t = m.ws.w8.tickets_n;
    auto need = [&](std::initializer_list<const Linear*> ls) {
        wg::Call c;
        c.fmt = wg::Format::Q8;
        c.n = 0;
        for (const Linear* l : ls) {
            if (l->type != kTypeQ8P && l->type != kTypeNvfp4Planes) return;
            c.fmt = l->type == kTypeQ8P ? wg::Format::Q8 : wg::Format::Nvfp4;
            c.K = l->cols;
            c.m[c.n++].rows = l->rows;
        }
        if (ls.size() == 2 && c.fmt == wg::Format::Nvfp4) c.epi = wg::Epilogue::SwiGLU;   // the FFN pair
        size_t p, t;
        wg::workspace_need(c, p, t);
        need_p = std::max(need_p, p);
        need_t = std::max(need_t, t);
    };
    for (const BlockDrafter::Layer& D : bd.layers) {
        for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) need({l});
        need({&D.q, &D.k, &D.v});
        need({&D.k, &D.v});
        need({&D.gate, &D.up});
    }
    for (const Linear& l : bd.fc) need({&l});
    if (need_p > m.ws.w8.partial_floats || need_t > m.ws.w8.tickets_n) {
        void *p = nullptr, *t = nullptr;
        if (cudaMalloc(&p, std::max<size_t>(need_p, 1) * 4) != cudaSuccess || cudaMalloc(&t, need_t * 4) != cudaSuccess ||
            cudaMemset(t, 0, need_t * 4) != cudaSuccess) {
            err = m.o.arch + ": out of device memory for the drafter's workspace";
            return false;
        }
        m.owned.push_back(p);
        m.owned.push_back(t);
        m.ws.w8.partial = (float*) p;
        m.ws.w8.tickets = (int32_t*) t;
        m.ws.w8.partial_floats = need_p;
        m.ws.w8.tickets_n = need_t;
    }
    return true;
}

}  // namespace

uint64_t shared_bytes(const Geometry& g, int64_t max_context) { return window_bytes(g, BlockDrafter{}, max_context) + chunk_bytes(g) + (512ull << 20); }

uint64_t session_bytes(const Geometry& g, int64_t max_context, bool mtp, bool kv_int8) {
    const uint64_t C = (uint64_t) g.conv_channels();
    const uint64_t gdn = (uint64_t) g.gdn_layers() *
                         (2 * ((uint64_t) g.gdn_state * g.gdn_v_heads * g.gdn_state + C * 3) +
                          kWindow * (2 * C + 2 * (uint64_t) g.gdn_v_heads)) * 4;
    const uint64_t cell16 = (uint64_t) g.n_kv * g.head_dim * 2 * 2;   // a cell of one layer, K and V, fp16
    const uint64_t cell8 = (uint64_t) g.n_kv * (g.head_dim + g.head_dim / 64 * 2) * 2;
    const uint64_t kv = (uint64_t) max_context * ((uint64_t) g.attn_layers() * (kv_int8 ? cell8 : cell16) + (mtp ? cell16 : 0)) +
                        (uint64_t) g.attn_layers() * 4 * 256;
    const uint64_t drafter = mtp ? ((uint64_t) (kWindow + 3) * g.n_embd + g.n_vocab + (uint64_t) kWindow * 64) * 4 : 0;
    return gdn + kv + drafter + (uint64_t) kWindow * g.n_vocab * 4 + 64 * 256;
}

Model::~Model() {
    for (BatchGraph& g : batch_graphs) cudaGraphExecDestroy(g.exec);
    for (void* p : owned) cudaFree(p);
    vision_detach();
}

bool Model::vision_attach(int tokens, std::string& err) {
    if (!encoder) {
        err = o.arch + ": started without --vision";
        return false;
    }
    const size_t need = encoder->arena_bytes(tokens);
    if (enc_arena && enc_bytes >= need) return true;
    vision_detach();
    if (cudaMalloc(&enc_arena, need) != cudaSuccess) {
        cudaGetLastError();
        enc_arena = nullptr;
        err = o.arch + ": no device memory for the image encoder (" + std::to_string((unsigned long long) (need >> 20)) + " MiB)";
        return false;
    }
    enc_bytes = need;
    if (!encoder->attach(enc_arena, enc_bytes, tokens, err)) {
        vision_detach();
        return false;
    }
    return true;
}

void Model::vision_detach() {
    if (!enc_arena) return;
    if (encoder) encoder->detach();
    cudaFree(enc_arena);
    enc_arena = nullptr;
    enc_bytes = 0;
}

bool Model::ready_products(std::string& err) {
    mmq = std::make_unique<prefill::mmq::Context>();
    std::vector<const Layer*> all;
    for (const Layer& L : w.layers) all.push_back(&L);
    if (o.drafter == Drafter::Mtp) all.push_back(&w.mtp.layer);
    std::vector<const Linear*> ls = {&w.head};
    if (o.drafter == Drafter::Mtp) ls.push_back(&w.mtp.fc);
    if (o.drafter == Drafter::Block) {
        for (const Linear& l : w.dflash.fc) ls.push_back(&l);
        for (const BlockDrafter::Layer& D : w.dflash.layers)
            for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) ls.push_back(l);
    }
    for (const Layer* Lp : all)
        for (const Linear* l : {&Lp->qkv, &Lp->z, &Lp->out, &Lp->q, &Lp->k, &Lp->v, &Lp->o, &Lp->gate, &Lp->up, &Lp->down})
            if (l->w) ls.push_back(l);
    for (const Linear* l : ls) {
        if (l->type == kTypeF8 || l->type == kTypeNvfp4Planes) continue;   // the window gemm's, not MMQ's
        if (!prefill::mmq::fits(l->type, l->rows)) {
            err = o.arch + ": the products have no tile for a " + std::to_string(l->rows) + "-row matrix";
            return false;
        }
        prefill::mmq::load(l->type, l->rows);
    }
    return true;
}

bool Model::ensure_chunk(std::string& err) {
    if (chunk_ready) return true;
    const Geometry& gg = g;
    if (!pre_arena.reserve(chunk_bytes(gg))) {
        err = o.arch + ": out of device memory for the prompt workspaces";
        return false;
    }
    const size_t T = kChunk, H = (size_t) gg.n_embd, AQ = (size_t) gg.n_head * gg.head_dim, C = (size_t) gg.conv_channels();
    const size_t VD = (size_t) gg.value_width(), HV = (size_t) gg.gdn_v_heads, KV = (size_t) gg.n_kv * gg.head_dim;
    const size_t widest = std::max<size_t>({H, (size_t) gg.n_ff, VD, AQ});
    ChunkScratch& b = cs;
    b.x = pre_arena.take<float>(T * H);
    b.xn = pre_arena.take<float>(T * H);
    b.tmp = pre_arena.take<float>(T * H);
    b.xq = pre_arena.take_bytes(std::max(prefill::mmq::q8_bytes(T, (int64_t) widest), prefill::fp8::act_bytes(T, (int64_t) widest)));
    b.tok = pre_arena.take<int32_t>(T);
    b.pos = pre_arena.take<int32_t>(3 * T);
    b.iota = pre_arena.take<int32_t>(T);
    b.bounds = pre_arena.take<int32_t>(2);
    if (o.drafter == Drafter::Mtp) {
        b.cat = pre_arena.take<float>(T * H);
        b.hn = pre_arena.take<float>((T + 1) * H);
    }
    if (o.drafter == Drafter::Block) b.dacc = pre_arena.take<float>(T * H);
    // the MLP's FP4 activations (fp4_dense) and their token scales
    b.a4 = pre_arena.take_bytes(prefill::fp4_dense::act_bytes((int64_t) T, (int64_t) std::max<size_t>(H, gg.n_ff)));
    b.gs = pre_arena.take<float>(T);
    b.hmax = pre_arena.take<float>(T);
    const size_t phase = pre_arena.mark();
    if (gg.gdn_layers()) {
        b.qkv = pre_arena.take<float>(T * C);
        b.hc = pre_arena.take<float>(T * C);
        b.z = pre_arena.take<float>(T * VD);
        b.y = pre_arena.take<float>(T * VD);
        b.yn = pre_arena.take<float>(T * VD);
        b.gate = pre_arena.take<float>(T * HV);
        b.beta = pre_arena.take<float>(T * HV);
        b.ab = pre_arena.take<float>(T * 2 * HV);
        b.work = pre_arena.take_bytes(kernels::gdn_work_bytes(T));
        pre_arena.rewind(phase);
    }
    if (gg.attn_layers()) {
        b.qfull = pre_arena.take<float>(T * 2 * AQ);
        b.k = pre_arena.take<float>(T * KV);
        b.v = pre_arena.take<float>(T * KV);
        b.qo = pre_arena.take<float>(T * AQ);
        b.ao = pre_arena.take<float>(T * AQ);
        b.ag = pre_arena.take<float>(T * AQ);
        b.attn_split = pre_arena.take<float>(kernels::gqa_prompt_split_floats((int64_t) T, gg.n_head, kAttnSplits));
        pre_arena.rewind(phase);
    }
    b.g = pre_arena.take<float>(T * gg.n_ff);
    b.u = pre_arena.take<float>(T * gg.n_ff);
    b.h = pre_arena.take<float>(T * gg.n_ff);
    if (!b.h || !b.x) {
        err = o.arch + ": the prompt workspaces do not fit their arena";
        return false;
    }
    prefill::mmq::iota(b.iota, (int64_t) T, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = o.arch + ": preparing the prompt kernels failed";
        return false;
    }
    chunk_ready = true;
    return true;
}

ModelPtr make_model(const Geometry& g, const Weights& w, const Options& o, std::string& err) {
    ModelPtr m(new Model());
    m->g = g;
    m->w = w;
    m->o = o;
    if (const std::string e = g.check(); !e.empty()) {
        err = o.arch + ": " + e;
        return nullptr;
    }
    int gi = 0, ai = 0;
    for (Mixer k : g.kinds) {
        m->gdn_index.push_back(k == Mixer::Gdn ? gi++ : -1);
        m->attn_index.push_back(k == Mixer::Attention ? ai++ : -1);
    }
    if (o.vision) {   // the projector stays in page-locked RAM; each request's pictures take it to the card
        m->mmproj = std::make_unique<vision::Mmproj>();
        m->encoder = std::make_unique<vision::CudaEncoder>();
        if (o.mmproj.empty()) {
            err = o.arch + ": --vision needs the pack's vision/mmproj section";
            return nullptr;
        }
        if (!m->mmproj->open(o.mmproj, err) || !m->encoder->stage(*m->mmproj, true, err) || !m->encoder->create(err)) {
            err = o.arch + ": vision: " + err;
            return nullptr;
        }
        if (m->mmproj->hp().proj != g.n_embd) {
            err = o.arch + ": the vision projector's width differs from the model's";
            return nullptr;
        }
        m->mmproj->release();
        basalt::log::info("basalt dense", "image encoder: %.2f GiB staged in page-locked RAM, on the card per request",
                          (double) m->encoder->weights_bytes() / 1073741824.0);
    }
    if (!m->dec_arena.reserve(window_bytes(g, o.drafter == Drafter::Block ? w.dflash : BlockDrafter{}, o.max_context))) {
        err = o.arch + ": out of device memory for the decode scratch";
        return nullptr;
    }
    const size_t W = kBatch, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim, KV = (size_t) g.n_kv * g.head_dim;
    WindowScratch& b = m->ws;
    Arena& a = m->dec_arena;
    b.x = a.take<uint16_t>(W * H);
    b.xn = a.take<float>(W * H);
    b.tmp = a.take<float>(W * H);
    b.dacc = a.take<float>(W * H);
    b.xq = a.take_bytes(act_slot(g));
    b.hq = a.take_bytes(act_slot(g));
    b.x8 = a.take_bytes(q8_slot(g));
    wgemm_need(g, b.w8.partial_floats, b.w8.tickets_n);
    b.w8.partial = a.take<float>(std::max<size_t>(b.w8.partial_floats, 1));
    b.w8.tickets = a.take<int32_t>(std::max<size_t>(b.w8.tickets_n, 1));   // zero: the arena is cleared below
    b.z = a.take<float>(W * g.value_width());
    b.qfull = a.take<float>(W * 2 * AQ);
    b.k = a.take<float>(W * KV);
    b.v = a.take<float>(W * KV);
    b.qo = a.take<float>(W * AQ);
    b.ao = a.take<float>(W * AQ);
    b.ag = a.take<float>(W * AQ);
    b.g = a.take<float>(W * g.n_ff);
    b.u = a.take<float>(W * g.n_ff);
    b.h = a.take<float>(W * g.n_ff);
    if (g.attn_layers() > 0) b.attn = a.take<float>(kernels::gqa_decode_scratch_floats(o.max_context, kWindow, g.n_head));
    b.tok = a.take<int32_t>(W);
    b.pos = a.take<int32_t>(W * 3);
    b.cat = a.take<float>(W * 2 * H);
    b.lp_scratch = a.take<float>(kernels::row_logprobs_scratch_floats(kWindow, g.n_vocab));
    if (o.drafter == Drafter::Block) {
        const BlockDrafter& bd = w.dflash;
        const size_t AQd = (size_t) bd.n_head * bd.head_dim, KVd = (size_t) bd.n_kv * bd.head_dim;
        b.bq = a.take<float>(W * AQd);
        b.bk = a.take<float>(W * KVd);
        b.bv = a.take<float>(W * KVd);
        b.bqo = a.take<float>(W * AQd);
        b.bao = a.take<float>(W * AQd);
        b.dyn = a.take<float>(W * bd.layers[0].conv_a.rows);
        b.bh = a.take<float>(W * H);
        b.blp = a.take<float>(W * kernels::kLogprobStride);
        b.bsel = a.take<float>(W * bd.rank);
        b.bpos = a.take<int32_t>(W * 3);
        b.battn = a.take<float>(kernels::gqa_decode_scratch_floats(bd.window + 256, kWindow, bd.n_head, bd.head_dim));
    }
    b.y = a.take<float>(W * g.value_width());
    b.hl = a.take<float>(W * (size_t) g.n_vocab);
    b.iota = a.take<int32_t>(W);
    b.bounds = a.take<int32_t>(2);
    b.pos0 = a.take<int32_t>(2);
    if (g.gdn_layers() > 0) {
        b.rq = a.take<float>(W * (size_t) g.conv_channels());
    }
    b.btab = a.take<BatchTable>(1);
    if (!b.bounds) {
        err = o.arch + ": the decode scratch does not fit its arena";
        return nullptr;
    }
    // rows past a window's last stay finite: they are quantized with it
    if (cudaMemset(b.x, 0, a.mark()) != cudaSuccess) {   // b.x is the arena's first buffer
        err = o.arch + ": clearing the decode scratch failed";
        return nullptr;
    }
    prefill::mmq::iota(b.iota, kBatch, nullptr);
    if (!decode_drafter(*m, err)) return nullptr;
    if (!m->ready_products(err) || cudaDeviceSynchronize() != cudaSuccess) {
        if (err.empty()) err = o.arch + ": readying the products failed";
        return nullptr;
    }
    return m;
}

void destroy(Model* m) { delete m; }

}  // namespace basalt::dense
