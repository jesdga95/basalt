// src/dense/session.cpp - a conversation of the dense stack as a model::ModelSession.
#include "internal.hpp"

#include "basalt/core/conversation_cache.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace basalt::dense {
namespace {

using model::Result;
using model::Status;

Result fail(const std::string& m) { return {Status::Error, m}; }

class Session final : public model::ModelSession {
public:
    explicit Session(Model& m) : m_(m), g_(m.g) {}
    ~Session() override {
        for (cudaGraphExec_t& e : st_.wg)
            if (e) cudaGraphExecDestroy(e);
        if (st_.stream) cudaStreamDestroy(st_.stream);
        if (h_picks_) cudaFreeHost(h_picks_);
    }

    bool init(std::string& err) {
        const Geometry& g = g_;
        if (cudaStreamCreateWithFlags(&st_.stream, cudaStreamNonBlocking) != cudaSuccess) {
            err = "cannot create a CUDA stream";
            return false;
        }
        drafter_ = m_.o.drafter == Drafter::Mtp;
        block_ = m_.o.drafter == Drafter::Block;
        size_t block_bytes = 0;
        if (block_) {
            const BlockDrafter& bd = m_.w.dflash;
            block_bytes = bd.layers.size() * 2 * (size_t) (bd.window + 256) * bd.n_kv * bd.head_dim * 2 +
                          (size_t) kWindow * (g.n_embd * 4 + 16) + 64 * 256;
        }
        if (!st_.arena.reserve(session_bytes(g, m_.o.max_context, drafter_, m_.o.kv_int8) + block_bytes)) {
            err = m_.o.arch + ": out of device memory for a session of " + std::to_string((long long) m_.o.max_context) +
                  " cells (" + std::to_string((unsigned long long) (session_bytes(g, m_.o.max_context, drafter_, m_.o.kv_int8) >> 20)) +
                  " MiB); lower --max-context or --parallel";
            return false;
        }
        const size_t C = (size_t) g.conv_channels(), HV = (size_t) g.gdn_v_heads;
        st_.gdn_floats = (size_t) g.gdn_state * HV * g.gdn_state + C * 3;
        const size_t ng = (size_t) g.gdn_layers();
        if (ng) {
            st_.gdn = st_.arena.take<float>(ng * st_.gdn_floats);
            st_.rec_qkv = st_.arena.take<float>(ng * kWindow * C);
            st_.rec_h = st_.arena.take<float>(ng * kWindow * C);
            st_.rec_gate = st_.arena.take<float>(ng * kWindow * HV);
            st_.rec_beta = st_.arena.take<float>(ng * kWindow * HV);
        }
        const size_t cells = (size_t) m_.o.max_context * g.n_kv * g.head_dim;
        for (int i = 0; i < g.attn_layers(); ++i) {
            kernels::KvInt8 q8;
            if (m_.o.kv_int8) {
                q8.k = st_.arena.take<int8_t>(cells);
                q8.v = st_.arena.take<int8_t>(cells);
                q8.ks = st_.arena.take<uint16_t>(cells / 64);
                q8.vs = st_.arena.take<uint16_t>(cells / 64);
                st_.kc.push_back(nullptr);
                st_.vc.push_back(nullptr);
                if (!q8.vs) {
                    err = m_.o.arch + ": the KV cache does not fit the session's arena";
                    return false;
                }
            } else {
                st_.kc.push_back(st_.arena.take<uint16_t>(cells));
                st_.vc.push_back(st_.arena.take<uint16_t>(cells));
            }
            st_.kq.push_back(q8);
        }
        if (drafter_) {
            st_.mtp_kc = st_.arena.take<uint16_t>(cells);
            st_.mtp_vc = st_.arena.take<uint16_t>(cells);
            st_.hid = st_.arena.take<float>((size_t) kWindow * g.n_embd);
            st_.tail = st_.arena.take<float>(g.n_embd);
            st_.chain = st_.arena.take<float>(2 * (size_t) g.n_embd);
            st_.dlogits = st_.arena.take<float>(g.n_vocab);
            st_.lp = st_.arena.take<float>((size_t) kWindow * kernels::kLogprobStride);
            st_.win_tok = st_.arena.take<int32_t>(kWindow);
            if (!st_.win_tok) {
                err = m_.o.arch + ": the drafter's buffers do not fit the session's arena";
                return false;
            }
        }
        if (block_) {
            const BlockDrafter& bd = m_.w.dflash;
            st_.dctx_cap = bd.window + 256;
            for (size_t l = 0; l < bd.layers.size(); ++l) {
                st_.dk.push_back(st_.arena.take<uint16_t>((size_t) st_.dctx_cap * bd.n_kv * bd.head_dim));
                st_.dv.push_back(st_.arena.take<uint16_t>((size_t) st_.dctx_cap * bd.n_kv * bd.head_dim));
            }
            st_.dacc = st_.arena.take<float>((size_t) kWindow * g.n_embd);
            st_.dids = st_.arena.take<int32_t>(kWindow);
            st_.dconf = st_.arena.take<float>(kWindow);
            st_.win_tok = st_.arena.take<int32_t>(kWindow);
            if (!st_.win_tok) {
                err = m_.o.arch + ": the block drafter's buffers do not fit the session's arena";
                return false;
            }
        }
        st_.logits = st_.arena.take<float>((size_t) kWindow * g.n_vocab);
        st_.picks = st_.arena.take<int32_t>(kWindow);
        st_.n_keep = st_.arena.take<int32_t>(4);
        if (!st_.n_keep || (ng && !st_.gdn) || (!st_.kc.empty() && !m_.o.kv_int8 && !st_.vc.back())) {
            err = m_.o.arch + ": a session's buffers do not fit its arena";
            return false;
        }
        if (ng && cudaMemsetAsync(st_.gdn, 0, ng * st_.gdn_floats * 4, st_.stream) != cudaSuccess) {
            err = "clearing the recurrent state failed";
            return false;
        }
        if (cudaHostAlloc((void**) &h_picks_, kWindow * sizeof(int32_t), cudaHostAllocPortable) != cudaSuccess) {
            err = "out of page-locked memory";
            return false;
        }
        lp_scratch_.resize(1);
        return cudaStreamSynchronize(st_.stream) == cudaSuccess;
    }

    int max_window() const override {
        if (block_) return std::clamp(std::min(m_.o.spec, m_.w.dflash.block), 1, kWindow);
        return drafter_ ? std::clamp(m_.o.spec, 1, kWindow) : 1;
    }

    // The drafter's chain after `anchor` (the token at `pos`): step 0 pairs main's last normed row (cell pos - 1) with
    // the anchor, each later step its own output with the draft before it; all on the stream, read by draft_end.
    Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (block_) {
            if (Result s = settle(pos); !s.ok()) return s;
            std::string err;
            if (pos + m_.w.dflash.block > m_.o.max_context) return fail(m_.o.arch + ": no room for a draft block");
            if (!block_draft(m_, st_, anchor, pos, err)) return fail(err);
            n_draft_ = std::clamp(n, 1, std::min(max_window() - 1, m_.w.dflash.block - 1));
            return {};
        }
        if (!drafter_) return {Status::NotImplemented, m_.o.arch + ": no drafter is loaded"};
        if (Result s = settle(pos); !s.ok()) return s;
        if (st_.tail_cell != pos - 1) return fail(m_.o.arch + ": the drafter has no row before position " + std::to_string((long long) pos));
        n = std::clamp(n, 1, max_window() - 1);
        if (pos + n > m_.o.max_context) n = (int) std::max<int64_t>(1, m_.o.max_context - pos);
        std::string err;
        cudaMemcpyAsync(st_.win_tok, &anchor, sizeof(int32_t), cudaMemcpyHostToDevice, st_.stream);
        const float* hid = st_.tail;
        const int32_t* tok = st_.win_tok;
        for (int k = 0; k < n; ++k) {
            float* out = st_.chain + (size_t) (k & 1) * g_.n_embd;
            float* lp = st_.lp + (size_t) k * kernels::kLogprobStride;
            if (!mtp_window(m_, st_, hid, tok, 1, pos - 1 + k, out, lp, err)) return fail(err);
            hid = out;
            tok = reinterpret_cast<const int32_t*>(lp + 2);
        }
        st_.tail_cell = -1;   // cell pos - 1 holds the right pair now
        n_draft_ = n;
        return {};
    }

    Result draft_end(int32_t* ids, float* conf, int max_n, int& n) override {
        if (block_) {
            n = std::min(n_draft_, max_n);
            if (cudaMemcpyAsync(ids, st_.dids, (size_t) n * sizeof(int32_t), cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess ||
                cudaMemcpyAsync(conf, st_.dconf, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess ||
                cudaStreamSynchronize(st_.stream) != cudaSuccess)
                return fail(m_.o.arch + ": the block drafter failed");
            return {};
        }
        if (!drafter_) return {Status::NotImplemented, m_.o.arch + ": no drafter is loaded"};
        float h[kWindow * kernels::kLogprobStride];
        n = std::min(n_draft_, max_n);
        if (cudaMemcpyAsync(h, st_.lp, (size_t) n_draft_ * kernels::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost,
                            st_.stream) != cudaSuccess || cudaStreamSynchronize(st_.stream) != cudaSuccess)
            return fail(m_.o.arch + ": the drafter failed");
        for (int k = 0; k < n; ++k) {
            const float* r = h + (size_t) k * kernels::kLogprobStride;
            int32_t id;
            std::memcpy(&id, r + 2, sizeof id);
            ids[k] = id;
            conf[k] = std::exp(r[3]);
        }
        return {};
    }

    Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1 || r.n_tokens > kWindow) return fail(m_.o.arch + ": a decode window of " + std::to_string(r.n_tokens) + " rows");
        if (r.pos + r.n_tokens > m_.o.max_context) return fail(m_.o.arch + ": the window passes the context");
        if (Result s = settle(r.pos); !s.ok()) return s;
        const int T = r.n_tokens;
        int32_t pos3[kWindow * 3];
        for (int t = 0; t < T; ++t) pos3[t * 3] = pos3[t * 3 + 1] = pos3[t * 3 + 2] = (int32_t) (r.pos + t + st_.rope_delta);
        std::string err;
        if (!prepare_draw(r, err)) return fail(err);
        if (drafter_) cudaMemcpyAsync(st_.win_tok, r.tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, st_.stream);
        if (!forward_window(m_, st_, r.tokens, T, r.pos, pos3, err)) return fail(err);
        draw(T);
        pend_pos_ = r.pos;
        pend_T_ = T;
        if (r.logits) cudaMemcpyAsync(r.logits, st_.logits + (size_t) (T - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4,
                                      cudaMemcpyDeviceToHost, st_.stream);
        cudaMemcpyAsync(h_picks_, st_.picks, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, st_.stream);
        if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
        if (r.row_tokens) std::copy(h_picks_, h_picks_ + T, r.row_tokens);
        if (r.next_token) *r.next_token = h_picks_[T - 1];
        return {};
    }

    Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > m_.o.max_context) return fail(m_.o.arch + ": the prompt passes the context");
        if (Result s = settle(r.pos); !s.ok()) return s;
        std::vector<int32_t> pos3;
        for (int at = 0; at < r.n_tokens;) {
            const int n = std::min(kChunk, r.n_tokens - at);
            const int64_t p = r.pos + at;
            pos3.resize((size_t) n * 3);
            for (int t = 0; t < n; ++t) pos3[(size_t) t * 3] = pos3[(size_t) t * 3 + 1] = pos3[(size_t) t * 3 + 2] = (int32_t) (p + t + st_.rope_delta);
            std::string err;
            if (!forward_chunk(m_, st_, r.tokens + at, nullptr, n, p, pos3.data(), err)) return fail(err);
            if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
            at += n;
            if (r.progress) r.progress(at);
            if (at < r.n_tokens && r.stop && r.stop()) break;
        }
        return {};
    }

    bool reads_patches() const override { return m_.encoder != nullptr; }

    Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        if (!m_.encoder) return {Status::NotImplemented, m_.o.arch + ": started without --vision"};
        vision::Image img;
        std::string err;
        if (!vision::decode_image(data, bytes, img, err)) return fail(err);
        const vision::Geometry geo;
        const vision::Size sz = vision::fit_size(img.w, img.h, geo);
        std::vector<float> planar, patches;
        vision::normalize_planar(vision::to_canvas(img, sz), planar);
        vision::patchify(planar, sz, geo, patches);
        if (Result r = encode_patches(patches, sz.w, sz.h, rows); !r.ok()) return r;
        n_rows = (int) (rows.size() / (size_t) g_.n_embd);
        grid = {sz.w / (geo.patch * geo.merge), sz.h / (geo.patch * geo.merge)};
        return {};
    }

    Result encode_patches(const std::vector<float>& patches, int w, int h, std::vector<float>& rows) override {
        if (!m_.encoder) return {Status::NotImplemented, m_.o.arch + ": started without --vision"};
        std::string err;
        if (!m_.vision_attach((w / 16) * (h / 16) / 4, err)) return fail(err);
        if (!m_.encoder->encode(patches, vision::Size{w, h}, rows)) return fail("the vision encoder failed: " + m_.encoder->last_error());
        return {};
    }

    void images_done() override { m_.vision_detach(); }

    Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid image_grid) override {
        if (pos + n_rows > m_.o.max_context) return fail(m_.o.arch + ": the image passes the context");
        if (Result s = settle(pos); !s.ok()) return s;
        // one grid of merged rows: row j at (t, h, w) = (p, p + j / nx, p + j % nx), the text after it at p + max(nx, ny)
        const int64_t p = pos + st_.rope_delta;
        const int nx = image_grid.nx > 0 ? image_grid.nx : 0, ny = image_grid.ny > 0 ? image_grid.ny : 0;
        const bool grid = nx > 0 && ny > 0 && (int64_t) nx * ny == n_rows;
        std::vector<int32_t> pos3;
        for (int at = 0; at < n_rows;) {
            const int n = std::min(kChunk, n_rows - at);
            pos3.resize((size_t) n * 3);
            for (int t = 0; t < n; ++t) {
                const int j = at + t;
                pos3[(size_t) t * 3] = (int32_t) (grid ? p : p + j);
                pos3[(size_t) t * 3 + 1] = (int32_t) (grid ? p + j / nx : p + j);
                pos3[(size_t) t * 3 + 2] = (int32_t) (grid ? p + j % nx : p + j);
            }
            std::string err;
            if (!forward_chunk(m_, st_, nullptr, rows + (size_t) at * g_.n_embd, n, pos + at, pos3.data(), err)) return fail(err);
            if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
            at += n;
        }
        if (grid) st_.rope_delta += std::max(nx, ny) - n_rows;
        return {};
    }

    size_t running_bytes() const override {
        return (size_t) g_.gdn_layers() * st_.gdn_floats * 4 + 24 + (drafter_ ? (size_t) g_.n_embd * 4 : 0);
    }

    Result save_running(uint8_t* host, int64_t pos) override {
        if (Result s = settle(pos); !s.ok()) return s;
        const size_t n = (size_t) g_.gdn_layers() * st_.gdn_floats * 4;
        if (n && cudaMemcpyAsync(host, st_.gdn, n, cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess) return fail("saving the state failed");
        if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail("saving the state failed");
        const int64_t tail[3] = {st_.rope_delta, pos, st_.tail_cell};
        if (drafter_ && cudaMemcpy(host + n + sizeof tail, st_.tail, (size_t) g_.n_embd * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("saving the state failed");
        std::memcpy(host + n, tail, sizeof tail);
        return {};
    }

    Result restore_running(const uint8_t* host, int64_t pos) override {
        pend_T_ = 0;
        st_.dctx_from = pos;   // the block drafter's context below `pos` is not this conversation's
        const size_t n = (size_t) g_.gdn_layers() * st_.gdn_floats * 4;
        if (n && cudaMemcpyAsync(st_.gdn, host, n, cudaMemcpyHostToDevice, st_.stream) != cudaSuccess) return fail("restoring the state failed");
        if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail("restoring the state failed");
        int64_t tail[3];
        std::memcpy(tail, host + n, sizeof tail);
        st_.rope_delta = tail[0];
        st_.tail_cell = tail[2];
        if (drafter_ && cudaMemcpy(st_.tail, host + n + sizeof tail, (size_t) g_.n_embd * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("restoring the state failed");
        return {};
    }

    // a layer's cells: fp16 K and V (format 1), or int8 codes with fp16 scales per 64 values (format 2); the MTP
    // drafter's own layer is fp16 and comes last
    size_t row_bytes(bool int8) const { return (size_t) g_.n_kv * g_.head_dim * (int8 ? 1 : 2); }
    size_t cells_bytes(int64_t positions) const override {
        const size_t main = m_.o.kv_int8 ? 2 * (row_bytes(true) + row_bytes(true) / 32) : 2 * row_bytes(false);
        return (size_t) positions * ((size_t) g_.attn_layers() * main + (drafter_ ? 2 * row_bytes(false) : 0));
    }

    Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        cudaStreamSynchronize(st_.stream);
        kv.clear();
        const auto down = [&](const void* dev, core::ConversationBuffer& to, size_t bytes) {
            to.resize_for_overwrite(bytes);
            return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                return cudaMemcpy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
            });
        };
        for (size_t i = 0; i < st_.kc.size() + (drafter_ ? 1 : 0); ++i) {
            const bool mtp = i == st_.kc.size();
            const bool int8 = !mtp && m_.o.kv_int8;
            core::ConversationKv k;
            k.format = int8 ? 2 : 1;
            k.cells = positions;
            k.heads = g_.n_kv;
            k.head_dim = g_.head_dim;
            k.page_size = 1;
            const size_t rows = (size_t) positions * row_bytes(int8);
            bool ok;
            if (int8) {
                const kernels::KvInt8& q = st_.kq[i];
                ok = down(q.k, k.k, rows) && down(q.v, k.v, rows) && down(q.ks, k.k_scale, rows / 32) && down(q.vs, k.v_scale, rows / 32);
            } else {
                ok = down(mtp ? st_.mtp_kc : st_.kc[i], k.k, rows) && down(mtp ? st_.mtp_vc : st_.vc[i], k.v, rows);
            }
            if (!ok) return fail("reading the KV cells back failed");
            kv.push_back(std::move(k));
        }
        return {};
    }

    Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        cudaStreamSynchronize(st_.stream);
        if (kv.size() != st_.kc.size() + (drafter_ ? 1 : 0)) return fail(m_.o.arch + ": a parked conversation has other caches");
        if (positions > m_.o.max_context) return fail(m_.o.arch + ": a parked conversation is longer than the context");
        const auto up = [&](void* dev, const core::ConversationBuffer& from, size_t bytes) {
            return from.size() >= bytes && from.visit(0, bytes, [&](const uint8_t* p, size_t n, size_t at) {
                return cudaMemcpy((uint8_t*) dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
            });
        };
        for (size_t i = 0; i < kv.size(); ++i) {
            const bool mtp = i == st_.kc.size();
            const bool int8 = !mtp && m_.o.kv_int8;
            const core::ConversationKv& k = kv[i];
            if (k.format != (int8 ? 2 : 1) || k.heads != g_.n_kv || k.head_dim != g_.head_dim)
                return fail(m_.o.arch + ": a parked conversation has other caches");
            const size_t rows = (size_t) positions * row_bytes(int8);
            bool ok;
            if (int8) {
                const kernels::KvInt8& q = st_.kq[i];
                ok = up(q.k, k.k, rows) && up(q.v, k.v, rows) && up(q.ks, k.k_scale, rows / 32) && up(q.vs, k.v_scale, rows / 32);
            } else {
                ok = up(mtp ? st_.mtp_kc : st_.kc[i], k.k, rows) && up(mtp ? st_.mtp_vc : st_.vc[i], k.v, rows);
            }
            if (!ok) return fail(m_.o.arch + ": a parked conversation has other caches");
        }
        pend_T_ = 0;
        return {};
    }

    Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return fail(e);
        return {};
    }

    Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (!device_bias(d_mask_, bias, e)) return fail(e);
        kernels::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        kernels::sample_tokens(st_.logits + (size_t) row * g_.n_vocab, 1, g_.n_vocab,
                               history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr, history_len_, sp,
                               st_.picks, st_.stream);
        if (cudaMemcpyAsync(h_picks_, st_.picks, sizeof(int32_t), cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess ||
            cudaStreamSynchronize(st_.stream) != cudaSuccess)
            return fail("the masked draw failed");
        token = h_picks_[0];
        return {};
    }

    Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (n <= 0 || row0 < 0 || row0 + n > kWindow) return fail("row_logprobs: rows out of range");
        std::string e;
        if (!lp_buffers(e)) return fail(e);
        float* d_out = lp_buf_ + kernels::row_logprobs_scratch_floats(kWindow, g_.n_vocab);
        int32_t* d_picks = reinterpret_cast<int32_t*>(d_out + (size_t) kWindow * kernels::kLogprobStride);
        if (cudaMemcpyAsync(d_picks, picks, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, st_.stream) != cudaSuccess)
            return fail("row_logprobs: the picks copy failed");
        kernels::row_logprobs(st_.logits + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_picks, top, d_out, lp_buf_, st_.stream);
        if (cudaMemcpyAsync(out, d_out, (size_t) n * kernels::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost,
                            st_.stream) != cudaSuccess || cudaStreamSynchronize(st_.stream) != cudaSuccess)
            return fail("row_logprobs: the kernel failed");
        return {};
    }

    Result copy_logits(int row, float* out) override {
        cudaStreamSynchronize(st_.stream);
        if (cudaMemcpy(out, st_.logits + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("the logits copy failed");
        return {};
    }

    std::string info_words() const override {
        const std::string kv = m_.o.kv_int8 ? "kv=int8" : "kv=fp16";
        return kv + (block_ ? " drafter=block" : drafter_ ? " drafter=mtp" : "");
    }
    int64_t disk_min_tokens() const override { return 4096; }

private:
    // The last window's accepted rows go into the recurrent state once the next call says where it starts.
    Result settle(int64_t pos) {
        if (pos == 0) {   // a new conversation: the running state starts empty
            pend_T_ = 0;
            st_.rope_delta = 0;
            st_.tail_cell = -1;
            st_.dctx_from = 0;
            if (g_.gdn_layers() &&
                cudaMemsetAsync(st_.gdn, 0, (size_t) g_.gdn_layers() * st_.gdn_floats * 4, st_.stream) != cudaSuccess)
                return fail("clearing the recurrent state failed");
            return {};
        }
        if (pend_T_ == 0) return {};
        const int64_t n = pos - pend_pos_;
        pend_T_ = std::max(pend_T_, 0);
        const int T = pend_T_;
        pend_T_ = 0;
        if (n < 0 || n > T) return fail(m_.o.arch + ": position " + std::to_string((long long) pos) + " is not reachable from the last window");
        commit_window(m_, st_, (int) n);
        if (block_ && n > 0) {   // the kept rows become the block drafter's context
            std::string err;
            if (!block_ctx_window(m_, st_, (int) n, pend_pos_, err)) return fail(err);
        }
        if (drafter_ && n > 0) {
            // the drafter's history: the waiting row with the window's first token, then each kept row with the next
            std::string err;
            if (st_.tail_cell == pend_pos_ - 1 && !mtp_window(m_, st_, st_.tail, st_.win_tok, 1, pend_pos_ - 1, nullptr, nullptr, err))
                return fail(err);
            if (n > 1 && !mtp_window(m_, st_, st_.hid, st_.win_tok + 1, (int) n - 1, pend_pos_, nullptr, nullptr, err)) return fail(err);
            cudaMemcpyAsync(st_.tail, st_.hid + (size_t) (n - 1) * g_.n_embd, (size_t) g_.n_embd * sizeof(float),
                            cudaMemcpyDeviceToDevice, st_.stream);
            st_.tail_cell = pend_pos_ + n - 1;
        }
        return {};
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            void* v = nullptr;
            if (cudaMalloc(&v, bytes) != cudaSuccess) { err = "out of device memory for a logit bias"; return false; }
            owned_.emplace_back(v);
            dev = (float*) v;
        }
        if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "the logit bias upload failed"; return false; }
        return true;
    }

    bool lp_buffers(std::string& err) {
        if (lp_buf_) return true;
        const size_t f = kernels::row_logprobs_scratch_floats(kWindow, g_.n_vocab) + (size_t) kWindow * (kernels::kLogprobStride + 1);
        void* v = nullptr;
        if (cudaMalloc(&v, f * sizeof(float)) != cudaSuccess) { err = "out of device memory for logprobs"; return false; }
        owned_.emplace_back(v);
        lp_buf_ = (float*) v;
        return true;
    }

    // How the window's picks are drawn: the shared sampler (greedy when the request does not sample).
    bool prepare_draw(const model::DecodeRequest& r, std::string& err) {
        history_len_ = 0;
        sampling_ = kernels::SamplerParams{};
        sampling_.greedy = true;
        if (r.sampling != nullptr) sampling_ = *r.sampling;
        sampling_.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (!sampling_.greedy && kernels::sampler_list_len(sampling_.top_k, g_.n_vocab) > kernels::kSamplerNarrowMax &&
            !kernels::sampler_wide_reserve(g_.n_vocab, st_.stream)) {
            err = "out of device memory for the sampler";
            return false;
        }
        if (history_len_ > 0) {
            if (history_len_ > kHistoryCap) history_len_ = kHistoryCap;
            if (d_hist_ == nullptr) {
                void* v = nullptr;
                if (cudaMalloc(&v, (size_t) kHistoryCap * kWindow * sizeof(int32_t)) != cudaSuccess) { err = "out of device memory for the penalties"; return false; }
                owned_.emplace_back(v);
                d_hist_ = (int32_t*) v;
            }
            if (cudaMemcpy(d_hist_, r.history, (size_t) r.n_tokens * r.history_len * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "the penalty history upload failed";
                return false;
            }
            history_len_ = r.history_len;
        }
        return true;
    }

    void draw(int T) {
        kernels::sample_tokens(st_.logits, T, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sampling_,
                               st_.picks, st_.stream);
    }

    struct DevFree {
        void operator()(void* p) const { cudaFree(p); }
    };

    static constexpr int kHistoryCap = 4096;
    Model& m_;
    const Geometry& g_;
    SessionState st_;
    int32_t* h_picks_ = nullptr;
    bool drafter_ = false, block_ = false;
    int n_draft_ = 0;
    int64_t pend_pos_ = 0;
    int pend_T_ = 0;
    kernels::SamplerParams sampling_{};
    bool bias_on_ = false;
    int history_len_ = 0;
    float *d_bias_ = nullptr, *d_mask_ = nullptr, *lp_buf_ = nullptr;
    int32_t* d_hist_ = nullptr;
    std::vector<int> lp_scratch_;
    std::vector<std::unique_ptr<void, DevFree>> owned_;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Model& m, std::string& err) {
    auto s = std::make_unique<Session>(m);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::dense
