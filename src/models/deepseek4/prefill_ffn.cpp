#include "models/deepseek4/prefill_ffn.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
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
#include <thread>
#include <vector>

namespace basalt::models::deepseek4 {
namespace {

namespace mmq = basalt::prefill::mmq;
constexpr int kRoutes = 6;
constexpr int kGroup = mmq::kGatherGroupMax;
constexpr int kBf16 = 30;
constexpr int kCpuRows = 8;
constexpr int kShareTokens = 512;
constexpr int kShareRows = 1024;
constexpr int64_t kAheadPairs = 12288;

bool dense_supported(int type, int64_t rows) {
    return type == kBf16 || mmq::fits(type, rows);
}

class Impl final : public PrefillFfn {
public:
    struct Stage {
        uint8_t* blobs = nullptr;
        uint8_t* host = nullptr;
        uint8_t* gu = nullptr;
        uint8_t* down = nullptr;
        int32_t* bounds = nullptr;
        cudaEvent_t copied = nullptr;
        cudaEvent_t done = nullptr;
    };

    Impl(ModelWeights& weights, core::ExpertTier& tier, int cap)
        : weights_(weights), tier_(tier), g_(weights.geo()), cap_(cap), route_cap_((size_t) cap * kRoutes) {}

    ~Impl() override {
        if (last_stream_) cudaStreamSynchronize(last_stream_);
        if (copy_) cudaStreamSynchronize(copy_);
        for (Stage& s : stage_) {
            if (s.copied) cudaEventDestroy(s.copied);
            if (s.done) cudaEventDestroy(s.done);
            if (s.host) cudaFreeHost(s.host);
        }
        if (copy_) cudaStreamDestroy(copy_);
        if (hx_) cudaFreeHost(hx_);
        if (hrows_) cudaFreeHost(hrows_);
        if (act_) cudaFreeHost(act_);
        for (void* p : dev_) cudaFree(p);
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
                  device(shared_xq_, mmq::q8_bytes(cap_, g_.n_embd)) &&
                  device(shared_hq_, mmq::q8_bytes(cap_, g_.n_ff_exp)) &&
                  device(shared_gate_, (size_t) cap_ * g_.n_ff_exp * sizeof(float)) &&
                  device(shared_up_, (size_t) cap_ * g_.n_ff_exp * sizeof(float)) &&
                  device(shared_hidden_, (size_t) cap_ * g_.n_ff_exp * sizeof(float)) &&
                  device(shared_bounds_, 2 * sizeof(int32_t)) && device(identity_, route_cap_ * sizeof(int32_t)) &&
                  cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) == cudaSuccess;
        cudaError_t pinned = cudaSuccess;
        for (int i = 0; i < 2 && ok; ++i) {
            Stage& s = stage_[i];
            ok = device(s.blobs, stage_stride_ * kGroup) &&
                 device(s.gu, gu_stride_ * kGroup) &&
                 device(s.down, down_stride_ * kGroup) &&
                 device(s.bounds, ((size_t) kGroup + 1) * sizeof(int32_t)) &&
                 cudaEventCreateWithFlags(&s.copied, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) == cudaSuccess;
            if (ok) pinned = cudaHostAlloc(reinterpret_cast<void**>(&s.host), stage_stride_ * kGroup, cudaHostAllocPortable);
            if (pinned != cudaSuccess) break;
        }
        if (!ok || pinned != cudaSuccess) {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            cudaGetLastError();
            err = "deepseek4: out of memory for the prompt MMQ buffers (" + std::to_string(free_b >> 20) + " MiB free" +
                  (ok ? std::string(", pinned host: ") + cudaGetErrorString(pinned) : ", device buffers") + ")";
            return false;
        }
        rows_host_.resize(route_cap_);
        token_host_.resize(route_cap_);
        bounds_host_.resize((size_t) g_.n_expert + 1);
        mmq::iota(identity_, (int64_t) route_cap_, nullptr);
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
        for (int e = 0; e < g_.n_expert; ++e)
            if (!by[(size_t) e].empty()) used.push_back(e);

        struct Share {
            std::vector<kernels::cpu::ExpertJobMulti> jobs;
            std::vector<int32_t> act_tokens, pair;
            int cpu_rows = 0;
            std::thread cpu_thread;
            void join() {
                if (cpu_thread.joinable()) cpu_thread.join();
            }
            ~Share() { join(); }
        } share;

        if ((int64_t) T * kRoutes < kAheadPairs && T <= kShareTokens && tier_.pool()) {
            int pinned = 0;
            for (int e : used) pinned += tier_.locate(layer, e).pinned != nullptr;
            if (pinned > 0 && ensure_share_buffers()) {
                const size_t n = (size_t) g_.n_embd;
                std::vector<uint8_t> stays((size_t) g_.n_expert, 0);
                std::vector<int32_t> tok_row((size_t) T, -1);
                int rows = 0;
                for (int c = 1; c <= kCpuRows; ++c) {
                    for (int e : used) {
                        const core::ExpertTier::Source src = tier_.locate(layer, e);
                        if (!src.pinned || (int) by[(size_t) e].size() != c || stays[(size_t) e]) continue;
                        if (share.cpu_rows + c > pinned - (int) share.jobs.size() - 1 || rows + c > kShareRows) break;
                        kernels::cpu::ExpertJobMulti j;
                        j.blob = src.pinned;
                        j.nt = c;
                        for (int i = 0; i < c; ++i) {
                            const int32_t p = by[(size_t) e][(size_t) i], t = p / kRoutes;
                            if (tok_row[(size_t) t] < 0) {
                                tok_row[(size_t) t] = (int32_t) share.act_tokens.size();
                                share.act_tokens.push_back(t);
                            }
                            j.nact[i] = act_ + (size_t) tok_row[(size_t) t] * kernels::cpu::kNativeActBytes;
                            j.out[i] = hrows_ + (size_t) rows * n;
                            share.pair.push_back(p);
                            ++rows;
                        }
                        share.jobs.push_back(j);
                        stays[(size_t) e] = 1;
                        share.cpu_rows += c;
                    }
                }
                if (rows > 0) {
                    used.erase(std::remove_if(used.begin(), used.end(), [&](int e) { return stays[(size_t) e] != 0; }),
                               used.end());
                    if (cudaMemcpyAsync(hx_, x, (size_t) T * n * sizeof(float), cudaMemcpyDeviceToHost, last_stream_) != cudaSuccess ||
                        cudaStreamSynchronize(last_stream_) != cudaSuccess) {
                        err = "deepseek4: reading short-prompt activations for CPU offload failed";
                        return false;
                    }
                    share.cpu_thread = std::thread([this, layer, &share] {
                        const kernels::cpu::NativeFmt& f = tier_.fmt(layer);
                        for (size_t i = 0; i < share.act_tokens.size(); ++i)
                            kernels::cpu::native_quant_act(f, hx_ + (size_t) share.act_tokens[i] * g_.n_embd,
                                                           act_ + i * kernels::cpu::kNativeActBytes);
                        tier_.pool()->run_split_multi_native(f, share.jobs.data(), (int) share.jobs.size());
                    });
                }
            }
        }

        int total = 0, max_rows = 0;
        for (int e : used) {
            bounds_host_[(size_t) e] = total;
            for (int32_t route : by[(size_t) e]) {
                rows_host_[(size_t) total] = route;
                token_host_[(size_t) total] = route / kRoutes;
                ++total;
            }
            max_rows = std::max(max_rows, (int) by[(size_t) e].size());
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
            share.join();
            for (size_t i = 0; i < share.pair.size(); ++i) {
                if (cudaMemcpyAsync(rows_ + (size_t) share.pair[i] * (size_t) g_.n_embd,
                                    hrows_ + i * (size_t) g_.n_embd, (size_t) g_.n_embd * sizeof(float),
                                    cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
                    err = "deepseek4: cannot upload CPU short-prompt expert rows";
                    return false;
                }
            }
            weighted_route_sum(rows_, weights, T, g_.n_embd, out, stream);
            return true;
        }

        const ExpertLayer& el = weights_.pack().experts[(size_t) layer];
        const auto L = kernels::native_expert_layout(el.gate_up_type, el.down_type, g_.n_embd, g_.n_ff_exp,
                                                      g_.swiglu_limit);
        const size_t gate_bytes = L.gu_row * (size_t) g_.n_ff_exp;
        const size_t up_bytes = L.up_row * (size_t) g_.n_ff_exp;
        const size_t down_bytes = L.d_row * (size_t) g_.n_embd;

        for (size_t first = 0, gidx = 0; first < used.size(); first += (size_t) kGroup, ++gidx) {
            Stage& B = stage_[gidx & 1];
            const int n = (int) std::min((size_t) kGroup, used.size() - first);
            if (cudaStreamWaitEvent(copy_, B.done, 0) != cudaSuccess) {
                err = "deepseek4: copy stream wait failed";
                return false;
            }
            int32_t gb[kGroup + 1]{};
            int group_rows = 0, group_max = 0;
            for (int q = 0; q < n; ++q) {
                const int e = used[first + (size_t) q];
                const int cnt = (int) by[(size_t) e].size();
                gb[q] = group_rows;
                group_rows += cnt;
                group_max = std::max(group_max, cnt);
            }
            gb[n] = group_rows;
            mmq::GatherGroup gather;
            bool host_synced = false;
            for (int q = 0; q < n; ++q) {
                const int e = used[first + (size_t) q];
                core::ExpertTier::Source src = tier_.locate(layer, e);
                if (src.dev) gather.blob[q] = src.dev;
                else {
                    uint8_t* const dev_dst = B.blobs + stage_stride_ * (size_t) q;
                    if (src.pinned) {
                        if (cudaMemcpyAsync(dev_dst, src.pinned, (size_t) el.bytes, cudaMemcpyHostToDevice, copy_) != cudaSuccess) {
                            err = "deepseek4: cannot stage a pinned prompt expert";
                            return false;
                        }
                    } else {
                        if (!host_synced) {
                            if (cudaEventSynchronize(B.copied) != cudaSuccess) {
                                err = "deepseek4: host staging sync failed";
                                return false;
                            }
                            host_synced = true;
                        }
                        uint8_t* const host_dst = B.host + stage_stride_ * (size_t) q;
                        if (!weights_.read_expert_blob(layer, e, host_dst, (size_t) el.bytes, err)) return false;
                        if (cudaMemcpyAsync(dev_dst, host_dst, (size_t) el.bytes, cudaMemcpyHostToDevice, copy_) != cudaSuccess) {
                            err = "deepseek4: cannot upload a prompt expert blob";
                            return false;
                        }
                    }
                    gather.blob[q] = dev_dst;
                }
            }
            gather.first = 0;
            gather.n = n;
            if (!mmq::gather_native_group(gather, L.up_off, gate_bytes, L.down_off, down_bytes,
                                          B.gu, gu_stride_, B.down, down_stride_, copy_, 0, nullptr, up_bytes)) {
                for (int q = 0; q < n; ++q) {
                    const uint8_t* b = gather.blob[q];
                    mmq::gather_native(b, b + L.up_off, gate_bytes, b + L.down_off, down_bytes,
                                       B.gu + (size_t) q * gu_stride_, B.down + (size_t) q * down_stride_,
                                       copy_, nullptr, nullptr, up_bytes);
                }
            }
            if (cudaEventRecord(B.copied, copy_) != cudaSuccess ||
                cudaStreamWaitEvent(last_stream_, B.copied, 0) != cudaSuccess) {
                err = "deepseek4: compute stream wait failed";
                return false;
            }
            if (cudaMemcpyAsync(B.bounds, gb, (size_t) (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
                err = "deepseek4: cannot upload prompt MMQ group bounds";
                return false;
            }
            const int row0 = bounds_host_[(size_t) used[first]];
            mmq::quantize(x, token_ids_ + row0, xq_, el.gate_up_type, g_.n_embd, g_.n_embd, group_rows, stream);
            mmq::Product gu;
            gu.type = el.gate_up_type;
            gu.w_rows = 2 * (int64_t) g_.n_ff_exp;
            gu.w_cols = g_.n_embd;
            gu.expert_bytes = gu_stride_;
            gu.xq = xq_;
            gu.ids = identity_;
            gu.total_rows = group_rows;
            gu.max_rows = group_max;
            gu.dst = gu_;
            gu.ld_dst = gu.w_rows;
            gu.w = B.gu;
            gu.n = n;
            gu.bounds = B.bounds;
            mmq_.run(gu, stream);

            swiglu_split_clamped(gu_, group_rows, g_.n_ff_exp, g_.swiglu_limit, h_, stream);
            mmq::quantize(h_, nullptr, hq_, el.down_type, g_.n_ff_exp, g_.n_ff_exp, group_rows, stream);

            mmq::Product down;
            down.type = el.down_type;
            down.w_rows = g_.n_embd;
            down.w_cols = g_.n_ff_exp;
            down.expert_bytes = down_stride_;
            down.xq = hq_;
            down.ids = output_ids_ + row0;
            down.total_rows = group_rows;
            down.max_rows = group_max;
            down.dst = rows_;
            down.ld_dst = g_.n_embd;
            down.w = B.down;
            down.n = n;
            down.bounds = B.bounds;
            mmq_.run(down, stream);
            if (cudaEventRecord(B.done, last_stream_) != cudaSuccess) {
                err = "deepseek4: compute stream event record failed";
                return false;
            }
        }
        share.join();
        for (size_t i = 0; i < share.pair.size(); ++i) {
            if (cudaMemcpyAsync(rows_ + (size_t) share.pair[i] * (size_t) g_.n_embd,
                                hrows_ + i * (size_t) g_.n_embd, (size_t) g_.n_embd * sizeof(float),
                                cudaMemcpyHostToDevice, last_stream_) != cudaSuccess) {
                err = "deepseek4: cannot upload CPU short-prompt expert rows";
                return false;
            }
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
    bool ensure_share_buffers() {
        if (hrows_) return true;
        return cudaHostAlloc(reinterpret_cast<void**>(&hx_), (size_t) kShareTokens * (size_t) g_.n_embd * sizeof(float),
                             cudaHostAllocPortable) == cudaSuccess &&
               cudaHostAlloc(reinterpret_cast<void**>(&hrows_), (size_t) kShareRows * (size_t) g_.n_embd * sizeof(float),
                             cudaHostAllocPortable) == cudaSuccess &&
               cudaHostAlloc(reinterpret_cast<void**>(&act_), (size_t) kShareTokens * kernels::cpu::kNativeActBytes,
                             cudaHostAllocPortable) == cudaSuccess;
    }
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
    Stage stage_[2];
    cudaStream_t copy_ = nullptr;
    uint8_t *xq_ = nullptr, *hq_ = nullptr,
            *shared_xq_ = nullptr, *shared_hq_ = nullptr;
    float *gu_ = nullptr, *h_ = nullptr, *rows_ = nullptr,
          *shared_gate_ = nullptr, *shared_up_ = nullptr, *shared_hidden_ = nullptr;
    int32_t *token_ids_ = nullptr, *output_ids_ = nullptr, *bounds_ = nullptr, *shared_bounds_ = nullptr,
            *identity_ = nullptr;
    std::vector<int32_t> rows_host_, token_host_, bounds_host_;
    mmq::Context mmq_;
    cudaStream_t last_stream_ = nullptr;
    float *hx_ = nullptr, *hrows_ = nullptr;
    uint8_t* act_ = nullptr;
};

}  // namespace

size_t prefill_ffn_bytes(const ModelWeights& weights, int row_cap) {
    const Geometry& g = weights.geo();
    const size_t cap = (size_t) std::max(row_cap, 1);
    size_t blob_cap = 0, gu_cap = 0, down_cap = 0;
    for (const ExpertLayer& e : weights.pack().experts) {
        blob_cap = std::max(blob_cap, (size_t) e.bytes);
        gu_cap = std::max(gu_cap, mmq::gu_slot_bytes(e.gate_up_type, e.gate_up_type, g.n_ff_exp, g.n_embd));
        down_cap = std::max(down_cap, mmq::matrix_bytes(e.down_type, g.n_embd, g.n_ff_exp));
    }
    const size_t route_cap = cap * 6;
    const size_t stage_stride = (blob_cap + 15) & ~size_t(15);
    return 2 * (stage_stride * kGroup + gu_cap * kGroup + down_cap * kGroup + (kGroup + 1) * sizeof(int32_t)) +
           mmq::q8_bytes((int64_t) route_cap, g.n_embd) + mmq::q8_bytes((int64_t) route_cap, g.n_ff_exp) +
           route_cap * 2 * (size_t) g.n_ff_exp * sizeof(float) + route_cap * (size_t) g.n_ff_exp * sizeof(float) +
           route_cap * (size_t) g.n_embd * sizeof(float) + route_cap * sizeof(int32_t) * 2 +
           ((size_t) g.n_expert + 1) * sizeof(int32_t) +
           mmq::q8_bytes((int64_t) cap, g.n_embd) + mmq::q8_bytes((int64_t) cap, g.n_ff_exp) +
           cap * (size_t) g.n_ff_exp * sizeof(float) * 3 + 2 * sizeof(int32_t) + route_cap * sizeof(int32_t);
}

uint64_t prompt_room_bytes(const ModelWeights& weights, int64_t context, int cap) {
    const Geometry& g = weights.geo();
    const uint64_t n = (uint64_t) g.n_embd, c = (uint64_t) std::max(cap, 1);
    const uint64_t tiles = 2 * 128 * 64 * 512 * 8 + 2 * 128 * 1024 * 4 + 2 * 128 * 256 * 4 +
                           128 * 1024 * 4 + 2 * 128 * 64 * 128 * 4 + 128 * 64 * 4 * 2 + 128 * 32 * 4 +
                           128 * 512 * 4 + 128 * 8192 * 4;
    const uint64_t rows = c * (8 * n + (uint64_t) g.n_expert + 4 * n + 2 * (uint64_t) g.n_used + 1024 +
                               2 * 512 + 16 + 16 * 4) * sizeof(float) +
                          c * 6 * sizeof(int32_t) + c * sizeof(uint32_t) * 4;
    const uint64_t hca = 8 * ((uint64_t) std::max<int64_t>(1, context / 128 + 1)) * sizeof(int32_t);
    return prefill_ffn_bytes(weights, cap) + rows + tiles + hca;
}

std::unique_ptr<PrefillFfn> make_prefill_ffn(ModelWeights& weights, core::ExpertTier& tier, int row_cap,
                                              std::string& err) {
    if (row_cap < 1 || row_cap > 16384) {
        err = "deepseek4: prompt MMQ row capacity must be in [1, 16384]";
        return nullptr;
    }
    auto result = std::make_unique<Impl>(weights, tier, row_cap);
    if (!result->init(err)) return nullptr;
    err.clear();
    return result;
}

}  // namespace basalt::models::deepseek4
