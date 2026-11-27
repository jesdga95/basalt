#include "models/deepseek4/prefill_ffn.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/deepseek4/model_weights.hpp"
#include "models/deepseek4/routed_expert.hpp"
#include "models/deepseek4/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace basalt::models::deepseek4 {
namespace {

namespace mmq = basalt::prefill::mmq;
constexpr int kRoutes = 6;
constexpr int kGroup = mmq::kGatherGroupMax;
constexpr int kBf16 = 30;

bool dense_supported(int type, int64_t rows) {
    return type == kBf16 || mmq::fits(type, rows);
}

class Impl final : public PrefillFfn {
public:
    Impl(ModelWeights& weights, core::ExpertTier& tier, int cap)
        : weights_(weights), tier_(tier), g_(weights.geo()), cap_(cap), route_cap_((size_t) cap * kRoutes) {}

    ~Impl() override {
        if (last_stream_) cudaStreamSynchronize(last_stream_);
        for (void* p : dev_) cudaFree(p);
        if (stage_host_) cudaFreeHost(stage_host_);
    }

    bool init(std::string& err) {
        size_t blob_cap = 0, gu_cap = 0, down_cap = 0;
        for (int layer = 0; layer < g_.n_layers; ++layer) {
            const ExpertLayer& e = weights_.pack().experts[(size_t) layer];
            if (!mmq::fits(e.gate_up_type, 2 * (int64_t) g_.n_ff_exp) ||
                !mmq::fits(e.down_type, g_.n_embd)) {
                err = "deepseek4: MMQ does not cover the routed expert formats in this pack";
                return false;
            }
            const auto L = kernels::native_expert_layout(e.gate_up_type, e.down_type, g_.n_embd, g_.n_ff_exp,
                                                          g_.swiglu_limit);
            if (L.bytes != e.bytes || L.up_type != e.gate_up_type) {
                err = "deepseek4: routed expert blob does not match the MMQ layout";
                return false;
            }
            blob_cap = std::max(blob_cap, (size_t) e.bytes);
            gu_cap = std::max(gu_cap, mmq::gu_slot_bytes(e.gate_up_type, e.gate_up_type,
                                                          g_.n_ff_exp, g_.n_embd));
            down_cap = std::max(down_cap, mmq::matrix_bytes(e.down_type, g_.n_embd, g_.n_ff_exp));
        }
        const size_t expert_cap = (size_t) g_.n_expert;
        if (expert_cap == 0 || blob_cap > std::numeric_limits<size_t>::max() / expert_cap ||
            gu_cap > std::numeric_limits<size_t>::max() / expert_cap ||
            down_cap > std::numeric_limits<size_t>::max() / expert_cap) {
            err = "deepseek4: prompt expert staging size overflows";
            return false;
        }
        stage_stride_ = (blob_cap + 15) & ~size_t(15);
        gu_stride_ = gu_cap;
        down_stride_ = down_cap;
        bool ok = device(xq_, mmq::q8_bytes((int64_t) route_cap_, g_.n_embd)) &&
                  device(hq_, mmq::q8_bytes((int64_t) route_cap_, g_.n_ff_exp)) &&
                  device(gu_, route_cap_ * 2 * (size_t) g_.n_ff_exp * sizeof(float)) &&
                  device(h_, route_cap_ * (size_t) g_.n_ff_exp * sizeof(float)) &&
                  device(rows_, route_cap_ * (size_t) g_.n_embd * sizeof(float)) &&
                  device(token_ids_, route_cap_ * sizeof(int32_t)) &&
                  device(output_ids_, route_cap_ * sizeof(int32_t)) &&
                  device(bounds_, ((size_t) g_.n_expert + 1) * sizeof(int32_t)) &&
                  device(group_bounds_, (expert_cap + 1) * sizeof(int32_t)) &&
                  device(stage_, stage_stride_ * kGroup) && device(group_gu_, gu_stride_ * expert_cap) &&
                  device(group_down_, down_stride_ * expert_cap) &&
                  device(shared_xq_, mmq::q8_bytes(cap_, g_.n_embd)) &&
                  device(shared_hq_, mmq::q8_bytes(cap_, g_.n_ff_exp)) &&
                  device(shared_gate_, (size_t) cap_ * g_.n_ff_exp * sizeof(float)) &&
                  device(shared_up_, (size_t) cap_ * g_.n_ff_exp * sizeof(float)) &&
                  device(shared_hidden_, (size_t) cap_ * g_.n_ff_exp * sizeof(float)) &&
                  device(shared_bounds_, 2 * sizeof(int32_t)) && device(identity_, (size_t) cap_ * sizeof(int32_t));
        if (!ok || cudaHostAlloc(reinterpret_cast<void**>(&stage_host_), stage_stride_ * expert_cap,
                                 cudaHostAllocPortable) != cudaSuccess) {
            err = "deepseek4: out of memory for the prompt MMQ buffers";
            return false;
        }
        rows_host_.resize(route_cap_);
        token_host_.resize(route_cap_);
        bounds_host_.resize((size_t) g_.n_expert + 1);
        mmq::iota(identity_, cap_, nullptr);
        const int32_t b[2] = {0, 0};
        if (cudaMemcpy(shared_bounds_, b, sizeof(b), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "deepseek4: cannot initialize prompt MMQ bounds";
            return false;
        }
        cudaDeviceSynchronize();
        std::vector<std::pair<int, int64_t>> tiles;
        auto add = [&](int type, int64_t rows) {
            const auto item = std::make_pair(type, rows);
            if (std::find(tiles.begin(), tiles.end(), item) == tiles.end()) tiles.push_back(item);
        };
        for (int layer = 0; layer < g_.n_layers; ++layer) {
            const auto& e = weights_.pack().experts[(size_t) layer];
            add(e.gate_up_type, 2 * (int64_t) g_.n_ff_exp);
            add(e.down_type, g_.n_embd);
            const auto names = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
            for (const char* name : names)
                if (const DeviceTensor* t = weights_.tensor(layer, name); t && mmq::fits(t->type, g_.n_ff_exp))
                    add(t->type, g_.n_ff_exp);
        }
        for (const auto& [type, rows] : tiles) mmq::load(type, rows);
        return true;
    }

    bool routed(int layer, int T, const float* x, const int32_t* ids, const float* weights,
                float* out, void* stream, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || T < 1 || T > cap_ || !x || !ids || !weights || !out || !stream) {
            err = "deepseek4: invalid chunk routed-expert request";
            return false;
        }
        last_stream_ = (cudaStream_t) stream;
        const size_t nr_cap = (size_t) T * kRoutes;
        std::vector<int32_t> host_ids(nr_cap);
        if (cudaMemcpyAsync(host_ids.data(), ids, nr_cap * sizeof(int32_t), cudaMemcpyDeviceToHost,
                            last_stream_) != cudaSuccess || cudaStreamSynchronize(last_stream_) != cudaSuccess) {
            err = "deepseek4: cannot read prompt routes for expert grouping";
            return false;
        }
        std::vector<std::vector<int32_t>> by((size_t) g_.n_expert);
        for (size_t r = 0; r < nr_cap; ++r) {
            const int e = host_ids[r];
            if (e == -1) continue;
            if (e < 0 || e >= g_.n_expert) {
                err = "deepseek4: router returned a bad expert id in the prompt chunk";
                return false;
            }
            by[(size_t) e].push_back((int32_t) r);
        }
        std::vector<int> used;
        used.reserve((size_t) g_.n_expert);
        int total = 0, max_rows = 0;
        bounds_host_[0] = 0;
        for (int e = 0; e < g_.n_expert; ++e) {
            if (!by[(size_t) e].empty()) used.push_back(e);
            for (int32_t route : by[(size_t) e]) {
                rows_host_[(size_t) total] = route;
                token_host_[(size_t) total] = route / kRoutes;
                ++total;
            }
            max_rows = std::max(max_rows, (int) by[(size_t) e].size());
            bounds_host_[(size_t) e + 1] = total;
        }
        if (cudaMemsetAsync(rows_, 0, nr_cap * (size_t) g_.n_embd * sizeof(float), last_stream_) != cudaSuccess ||
            (total && (cudaMemcpyAsync(token_ids_, token_host_.data(), (size_t) total * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, last_stream_) != cudaSuccess ||
                       cudaMemcpyAsync(output_ids_, rows_host_.data(), (size_t) total * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, last_stream_) != cudaSuccess)) ||
            cudaMemcpyAsync(bounds_, bounds_host_.data(), ((size_t) g_.n_expert + 1) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
            err = "deepseek4: cannot upload prompt MMQ route groups";
            return false;
        }
        if (!total) {
            weighted_route_sum(rows_, weights, T, g_.n_embd, out, stream);
            return true;
        }

        std::vector<int32_t> all_bounds(used.size() + 1);
        for (size_t q = 0; q < used.size(); ++q)
            all_bounds[q] = bounds_host_[(size_t) used[q]];
        all_bounds.back() = total;
        if (cudaMemcpyAsync(group_bounds_, all_bounds.data(), all_bounds.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
            err = "deepseek4: cannot upload prompt MMQ expert bounds";
            return false;
        }

        const ExpertLayer& el = weights_.pack().experts[(size_t) layer];
        const auto L = kernels::native_expert_layout(el.gate_up_type, el.down_type, g_.n_embd, g_.n_ff_exp,
                                                      g_.swiglu_limit);
        mmq::quantize(x, token_ids_, xq_, el.gate_up_type, g_.n_embd, g_.n_embd, total, stream);
        for (size_t first = 0; first < used.size(); first += (size_t) kGroup) {
            const int n = (int) std::min((size_t) kGroup, used.size() - first);
            mmq::GatherGroup gather;
            for (int q = 0; q < n; ++q) {
                const int e = used[first + (size_t) q];
                core::ExpertTier::Source src = tier_.locate(layer, e);
                if (src.dev) gather.blob[q] = src.dev;
                else {
                    uint8_t* const dst = stage_host_ + stage_stride_ * (first + (size_t) q);
                    if (src.pinned) {
                        if (cudaMemcpyAsync(stage_ + stage_stride_ * (size_t) q, src.pinned, (size_t) el.bytes,
                                            cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
                            err = "deepseek4: cannot stage a pinned prompt expert";
                            return false;
                        }
                    } else {
                        if (!weights_.read_expert_blob(layer, e, dst, (size_t) el.bytes, err)) return false;
                        if (cudaMemcpyAsync(stage_ + stage_stride_ * (size_t) q, dst, (size_t) el.bytes,
                                            cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
                            err = "deepseek4: cannot upload a prompt expert blob";
                            return false;
                        }
                    }
                    gather.blob[q] = stage_ + stage_stride_ * (size_t) q;
                }
            }
            gather.first = 0;
            gather.n = n;
            const size_t gate_bytes = L.gu_row * (size_t) g_.n_ff_exp;
            const size_t up_bytes = L.up_row * (size_t) g_.n_ff_exp;
            const size_t down_bytes = L.d_row * (size_t) g_.n_embd;
            if (!mmq::gather_native_group(gather, L.up_off, gate_bytes, L.down_off, down_bytes,
                                          reinterpret_cast<uint8_t*>(group_gu_) + first * gu_stride_, gu_stride_,
                                          reinterpret_cast<uint8_t*>(group_down_) + first * down_stride_, down_stride_,
                                          stream, 0, nullptr, up_bytes)) {
                for (int q = 0; q < n; ++q) {
                    const uint8_t* b = gather.blob[q];
                    mmq::gather_native(b, b + L.up_off, gate_bytes, b + L.down_off, down_bytes,
                                       reinterpret_cast<uint8_t*>(group_gu_) + (first + (size_t) q) * gu_stride_,
                                       reinterpret_cast<uint8_t*>(group_down_) + (first + (size_t) q) * down_stride_,
                                       stream, nullptr, nullptr, up_bytes);
                }
            }
        }
        mmq::Product gu;
        gu.type = el.gate_up_type;
        gu.w_rows = 2 * (int64_t) g_.n_ff_exp;
        gu.w_cols = g_.n_embd;
        gu.expert_bytes = gu_stride_;
        gu.xq = xq_;
        gu.ids = output_ids_;
        gu.total_rows = total;
        gu.max_rows = max_rows;
        gu.dst = gu_;
        gu.ld_dst = gu.w_rows;
        for (size_t first = 0; first < used.size(); first += (size_t) kGroup) {
            gu.w = reinterpret_cast<uint8_t*>(group_gu_) + first * gu_stride_;
            gu.n = (int) std::min((size_t) kGroup, used.size() - first);
            gu.bounds = group_bounds_ + first;
            mmq_.run(gu, stream);
            if (cudaStreamSynchronize(last_stream_) != cudaSuccess) {
                err = "deepseek4: prompt gate/up MMQ failed for expert group " + std::to_string(first);
                return false;
            }
        }
        swiglu_split_clamped(gu_, total, g_.n_ff_exp, g_.swiglu_limit, h_, stream);
        mmq::quantize(h_, output_ids_, hq_, el.down_type, g_.n_ff_exp, g_.n_ff_exp, total, stream);
        mmq::Product down;
        down.type = el.down_type;
        down.w_rows = g_.n_embd;
        down.w_cols = g_.n_ff_exp;
        down.expert_bytes = down_stride_;
        down.xq = hq_;
        down.ids = output_ids_;
        down.total_rows = total;
        down.max_rows = max_rows;
        down.dst = rows_;
        down.ld_dst = g_.n_embd;
        for (size_t first = 0; first < used.size(); first += (size_t) kGroup) {
            down.w = reinterpret_cast<uint8_t*>(group_down_) + first * down_stride_;
            down.n = (int) std::min((size_t) kGroup, used.size() - first);
            down.bounds = group_bounds_ + first;
            mmq_.run(down, stream);
        }
        weighted_route_sum(rows_, weights, T, g_.n_embd, out, stream);
        return true;
    }

    bool dense_mmq(const float* x, int ld_x, int w_type, const void* w, int n_in, int n_out,
                   int T, float* out, int ld_out, void* stream, std::string& err) override {
        if (!x || !w || !out || !stream || T < 1 || T > cap_ || n_in < 32 || n_out < 1 ||
            ld_x < n_in || ld_out < n_out || !mmq::fits(w_type, n_out)) {
            err = "deepseek4: invalid dense MMQ request";
            return false;
        }
        if (mmq::q8_bytes(T, n_in) > mmq::q8_bytes((int64_t) route_cap_, g_.n_embd)) {
            err = "deepseek4: dense MMQ activation scratch overflow";
            return false;
        }
        last_stream_ = (cudaStream_t) stream;
        const int32_t b[2] = {0, T};
        if (cudaMemcpyAsync(shared_bounds_, b, sizeof(b), cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
            err = "deepseek4: cannot upload dense MMQ bounds";
            return false;
        }
        try {
            mmq::quantize(x, identity_, xq_, w_type, n_in, ld_x, T, last_stream_);
            mmq::Product p;
            p.w = w;
            p.type = w_type;
            p.w_rows = n_out;
            p.w_cols = n_in;
            p.n = 1;
            p.xq = xq_;
            p.bounds = shared_bounds_;
            p.ids = identity_;
            p.total_rows = T;
            p.max_rows = T;
            p.dst = out;
            p.ld_dst = ld_out;
            mmq_.run(p, last_stream_);
        } catch (const std::exception& e) {
            err = std::string("deepseek4 dense MMQ: ") + e.what();
            return false;
        }
        return true;
    }

    bool shared(int layer, int T, const float* x, float* out, void* stream, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || T < 1 || T > cap_ || !x || !out || !stream) {
            err = "deepseek4: invalid chunk shared-expert request";
            return false;
        }
        last_stream_ = (cudaStream_t) stream;
        const DeviceTensor *gate = weights_.tensor(layer, "ffn_gate_shexp.weight"),
                          *up = weights_.tensor(layer, "ffn_up_shexp.weight"),
                          *down = weights_.tensor(layer, "ffn_down_shexp.weight");
        if (!gate || !up || !down || !dense_supported(gate->type, g_.n_ff_exp) ||
            !dense_supported(up->type, g_.n_ff_exp) || !dense_supported(down->type, g_.n_embd)) {
            err = "deepseek4: shared-expert format is not covered by prompt MMQ";
            return false;
        }
        const int32_t b[2] = {0, T};
        if (cudaMemcpyAsync(shared_bounds_, b, sizeof(b), cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
            err = "deepseek4: cannot upload shared-expert MMQ bounds";
            return false;
        }
        if (!dense(*gate, g_.n_ff_exp, g_.n_embd, x, shared_xq_, T, shared_gate_, stream) ||
            !dense(*up, g_.n_ff_exp, g_.n_embd, x, shared_xq_, T, shared_up_, stream)) {
            err = "deepseek4: shared-expert MMQ launch failed";
            return false;
        }
        try {
            swiglu_clamped(shared_gate_, shared_up_, g_.n_ff_exp, g_.swiglu_limit, shared_hidden_, T, stream);
        } catch (const std::exception& e) {
            err = std::string("deepseek4: shared-expert SwiGLU failed: ") + e.what();
            return false;
        }
        if (!dense(*down, g_.n_embd, g_.n_ff_exp, shared_hidden_, shared_hq_, T, out, stream)) {
            err = "deepseek4: shared down-projection MMQ launch failed";
            return false;
        }
        return true;
    }

private:
    bool alloc(void*& p, size_t bytes) {
        if (!bytes || cudaMalloc(&p, bytes) != cudaSuccess) return false;
        dev_.push_back(p);
        return true;
    }
    template<class T> bool device(T*& p, size_t bytes) { return alloc(reinterpret_cast<void*&>(p), bytes); }
    bool dense(const DeviceTensor& w, int rows, int cols, const float* x, void* xq, int T, float* out, void* stream) {
        if (w.type == kBf16) {
            kernels::bf16_rows_dot(x, static_cast<const uint16_t*>(w.ptr), out, T, rows, cols, stream);
            return true;
        }
        if (!mmq::fits(w.type, rows)) return false;
        mmq::quantize(x, nullptr, xq, w.type, cols, cols, T, stream);
        mmq::Product p;
        p.w = w.ptr;
        p.type = w.type;
        p.w_rows = rows;
        p.w_cols = cols;
        p.expert_bytes = mmq::matrix_bytes(w.type, rows, cols);
        p.n = 1;
        p.xq = xq;
        p.bounds = shared_bounds_;
        p.ids = identity_;
        p.total_rows = T;
        p.max_rows = T;
        p.dst = out;
        p.ld_dst = rows;
        mmq_.run(p, stream);
        return true;
    }
    ModelWeights& weights_;
    core::ExpertTier& tier_;
    const Geometry& g_;
    int cap_ = 0;
    size_t route_cap_ = 0, stage_stride_ = 0, gu_stride_ = 0, down_stride_ = 0;
    std::vector<void*> dev_;
    uint8_t *stage_ = nullptr, *stage_host_ = nullptr, *xq_ = nullptr, *hq_ = nullptr,
            *shared_xq_ = nullptr, *shared_hq_ = nullptr;
    float *gu_ = nullptr, *h_ = nullptr, *rows_ = nullptr, *group_gu_ = nullptr, *group_down_ = nullptr,
          *shared_gate_ = nullptr, *shared_up_ = nullptr, *shared_hidden_ = nullptr;
    int32_t *token_ids_ = nullptr, *output_ids_ = nullptr, *bounds_ = nullptr, *group_bounds_ = nullptr, *shared_bounds_ = nullptr,
            *identity_ = nullptr;
    std::vector<int32_t> rows_host_, token_host_, bounds_host_;
    mmq::Context mmq_;
    cudaStream_t last_stream_ = nullptr;
};

}  // namespace

std::unique_ptr<PrefillFfn> make_prefill_ffn(ModelWeights& weights, core::ExpertTier& tier, int row_cap,
                                              std::string& err) {
    if (row_cap < 1 || row_cap > 8192) {
        err = "deepseek4: prompt MMQ row capacity must be in [1, 8192]";
        return nullptr;
    }
    auto result = std::make_unique<Impl>(weights, tier, row_cap);
    if (!result->init(err)) return nullptr;
    err.clear();
    return result;
}

}  // namespace basalt::models::deepseek4
