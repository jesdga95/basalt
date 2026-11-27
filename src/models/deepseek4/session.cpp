#include "models/deepseek4/session.hpp"

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/sparse_mqa.hpp"
#include "basalt/core/elastic_array.hpp"
#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/models/deepseek4/compress.hpp"
#include "basalt/models/deepseek4/indexer.hpp"
#include "basalt/models/deepseek4/layer.hpp"
#include "basalt/models/deepseek4/rope.hpp"
#include "basalt/models/deepseek4/vision.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"
#include "basalt/vision/transformer.hpp"
#include "basalt/models/deepseek4/router.hpp"
#include "models/deepseek4/dspark.hpp"
#include "models/deepseek4/routed_expert.hpp"
#include "models/deepseek4/prefill_ffn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace basalt::models::deepseek4 {
namespace {

constexpr int kBootstrapWindow = 128;
constexpr int kExpertWindow = 8;
constexpr int kRoutesPerToken = 6;
constexpr int kExpertWindowRoutes = kExpertWindow * kRoutesPerToken;

struct ExpertPlan {
    unsigned long long ptr[kExpertWindowRoutes]{};
    int32_t start[kExpertWindowRoutes + 1]{};
    int32_t groups[4]{};
    int32_t dst[kExpertWindowRoutes]{};
    int32_t tok[kExpertWindowRoutes]{};
};

struct ExpertPlanSingle {
    unsigned long long ptr[6]{};
    int32_t start[8]{};
    int32_t groups[4]{};
    int32_t dst[6]{};
    int32_t tok[6]{};
};

struct RouteMeta {
    int32_t fixed_row[kExpertWindow]{};
    uint8_t fixed_mask[kExpertWindow]{};
    uint8_t alternate_mask[kExpertWindow]{};
};

class Session final : public model::BatchSession {
    struct LayerCache {
        std::unique_ptr<core::ElasticArray> attention;
        std::unique_ptr<core::ElasticArray> indexer;
        float* attention_state_kv = nullptr;
        float* attention_state_score = nullptr;
        float* indexer_state_kv = nullptr;
        float* indexer_state_score = nullptr;
    };

public:
    explicit Session(DsModel& model) : model_(model), geo_(model.weights().geo()), batch_shared_(model.batch_shared()) {}

    ~Session() override {
        images_done();
        for (void* p : device_) if (p) cudaFree(p);
        if (stream_) cudaStreamDestroy(stream_);
        if (h_expert_input_) cudaFreeHost(h_expert_input_);
        if (h_expert_rows_) cudaFreeHost(h_expert_rows_);
        if (h_expert_ids_) cudaFreeHost(h_expert_ids_);
        if (h_expert_seq_) cudaFreeHost(h_expert_seq_);
        if (h_route_meta_) cudaFreeHost(h_route_meta_);
        if (h_initial_) cudaFreeHost(h_initial_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
            err = "cannot create a CUDA stream";
            return false;
        }
        const size_t n = (size_t) geo_.n_embd;
        const size_t nf = (size_t) geo_.n_ff_exp;
        const int max_index_rows = (int) ((model_.context() + 3) / 4);
        const int max_hca_rows = (int) ((model_.context() + 127) / 128);
        const int max_selected = std::max(kernels::kSparseMqaTopK, max_hca_rows);
        const auto dev = [&](auto*& p, size_t count) { return allocate(p, count, err); };
        const size_t rows = kExpertWindow;
        bool ok = dev(streams_, 4 * n * rows) && dev(next_streams_, 4 * n * rows) && dev(attention_output_, n * rows) &&
                  dev(ffn_output_, n * rows) && dev(shared_output_, n * rows) && dev(routed_output_, n * rows) &&
                  dev(final_logits_, (size_t) geo_.n_vocab * rows) && dev(retained_logits_, (size_t) geo_.n_vocab * rows) && dev(normalized_, n * rows) &&
                  dev(hc_workspace_, kernels::hc_mixes_workspace_floats(kExpertWindow)) &&
                  dev(attn_coef_, (size_t) kernels::kHcCoef * rows) && dev(ffn_coef_, (size_t) kernels::kHcCoef * rows) &&
                  dev(collapsed_, n * rows) && dev(attn_normed_, n * rows) && dev(ffn_normed_, n * rows) &&
                  dev(attn_q8_, kernels::native_q8_1_bytes(geo_.n_embd, kExpertWindow)) &&
                  dev(q_latent_, (size_t) geo_.q_lora * rows) &&
                  dev(q_latent_q8_, kernels::native_q8_1_bytes(geo_.q_lora, kExpertWindow)) &&
                  dev(q_heads_, (size_t) kernels::kSparseMqaHeads * kernels::kSparseMqaDim * rows) &&
                  dev(kv_projected_, 512 * rows) && dev(kv_row_, 512 * rows) &&
                  dev(mqa_output_, (size_t) kernels::kSparseMqaHeads * kernels::kSparseMqaDim * rows) &&
                   dev(mqa_workspace_, kernels::sparse_mqa_workspace_floats(kExpertWindow, max_selected)) &&
                  dev(attn_out_q8_, kernels::sparse_mqa_output_scratch_bytes(kExpertWindow)) &&
                  dev(attn_out_latent_, kernels::sparse_mqa_output_workspace_floats(kExpertWindow)) &&
                  dev(shared_in_q8_, kernels::native_q8_1_bytes(geo_.n_embd, kExpertWindow)) &&
                  dev(shared_hidden_q8_, kernels::native_q8_1_bytes((int64_t) nf, kExpertWindow)) &&
                  dev(shared_gate_, nf * rows) && dev(shared_up_, nf * rows) && dev(shared_hidden_, nf * rows) &&
                  dev(router_logits_, (size_t) geo_.n_expert * rows) && dev(route_ids_, (size_t) geo_.n_used * rows) &&
                  dev(route_weights_, (size_t) geo_.n_used * rows) &&
                    dev(position_meta_, 3 * rows) &&
                   dev(comp_row_, rows) && dev(selected_ids_, (size_t) max_selected * rows) && dev(indexer_q_, 64 * 128 * rows) &&
                   dev(indexer_head_weights_, 64 * rows) && dev(indexer_kv_, 256 * rows) && dev(indexer_score_, 256 * rows) &&
                   dev(attn_comp_kv_, 1024 * rows) && dev(attn_comp_score_, 1024 * rows) &&
                   dev(indexer_project_workspace_, indexer_project_workspace_bytes(kExpertWindow)) &&
                   dev(indexer_select_workspace_, indexer_select_workspace_bytes(std::max(max_index_rows, 1))) &&
                    dev(indexer_select_count_, rows) && dev(comp_limit_, rows) && dev(indexer_rope_angles_, 32 * rows) &&
                    dev(compress_rope_freq_, 32) &&
                  dev(sampled_, 1) && dev(expert_input_q8_, kernels::native_q8_1_bytes(geo_.n_embd, kExpertWindow)) &&
                  dev(expert_scratch_, kernels::native_expert_scratch_bytes(kExpertWindowRoutes, geo_.n_ff_exp)) &&
                   dev(expert_plan_, sizeof(ExpertPlan)) && dev(logit_bias_, (size_t) geo_.n_vocab);
        if (!ok) return false;
        positions_ = position_meta_;
        win_start_ = position_meta_ + rows;
        win_n_ = position_meta_ + 2 * rows;
        if (!dev(head_hc_workspace_, kernels::hc_plain_workspace_floats(kExpertWindow, 4)) ||
            !dev(head_coef_, (size_t) kernels::hc_plain_coef_stride(4) * rows) || !dev(head_collapsed_, n * rows) ||
            !dev(head_normed_, n * rows) || !dev(window_rings_, (size_t) geo_.n_layers * kBootstrapWindow * 512)) {
            err = "out of device memory for DeepSeek4 session state";
            return false;
        }

        const uint64_t hash_count = (uint64_t) geo_.hash_layers * (uint64_t) geo_.n_vocab * 6;
        if (hash_count > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
            !allocate(hash_routes_, (size_t) hash_count, err)) return false;
        for (int layer = 0; layer < geo_.hash_layers; ++layer) {
            std::vector<int32_t> ids;
            if (!model_.weights().hash_route_table(layer, ids, err)) return false;
            const size_t offset = (size_t) layer * geo_.n_vocab * 6;
            if (cudaMemcpyAsync(hash_routes_ + offset, ids.data(), ids.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
                err = "cannot upload the checkpoint hash route table";
                return false;
            }
        }

        constexpr unsigned host_flags = cudaHostAllocPortable | cudaHostAllocMapped;
        if (cudaHostAlloc(reinterpret_cast<void**>(&h_initial_), 4 * n * rows * sizeof(float), cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&h_expert_input_), kExpertWindow * n * sizeof(float), host_flags) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&h_expert_rows_), (size_t) kExpertWindowRoutes * n * sizeof(float), host_flags) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&h_expert_ids_), (size_t) kExpertWindowRoutes * sizeof(int32_t), host_flags) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&h_expert_seq_), 64, host_flags) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&h_route_meta_), sizeof(RouteMeta), host_flags) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_expert_input_), h_expert_input_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_expert_ids_), h_expert_ids_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_expert_seq_), h_expert_seq_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_expert_rows_), h_expert_rows_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_route_meta_), h_route_meta_, 0) != cudaSuccess) {
            err = "cannot map the expert staging buffers";
            return false;
        }
        *h_expert_seq_ = 0;
        route_fixed_row_ = d_route_meta_->fixed_row;
        route_fixed_mask_ = d_route_meta_->fixed_mask;
        route_alt_mask_ = d_route_meta_->alternate_mask;

        ropes_.resize((size_t) geo_.n_layers);
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            RopeParams params{geo_.rope_dim, geo_.rope_base, geo_.comp_rope_base, geo_.yarn_factor,
                              geo_.yarn_orig_ctx, geo_.yarn_beta_fast, geo_.yarn_beta_slow};
            if (!make_rope_tables(params, geo_.compress_ratio[(size_t) layer], ropes_[(size_t) layer], err)) return false;
        }
        if (cudaMemcpyAsync(compress_rope_freq_, ropes_[2].frequency.data(), 32 * sizeof(float),
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            err = "cannot upload DeepSeek4 compression RoPE frequencies";
            return false;
        }
        layer_cache_.resize((size_t) geo_.n_layers);
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            LayerCache& cache = layer_cache_[(size_t) layer];
            const uint64_t rows = (uint64_t) model_.context() / (uint64_t) ratio + 1;
            cache.attention = std::make_unique<core::ElasticArray>();
            if (!cache.attention->open(rows * 512 * sizeof(float), err)) return false;
            const int state_width = ratio == 4 ? 1024 : 512;
            const int state_rows = ratio == 4 ? 2 * ratio : ratio;
            if (!dev(cache.attention_state_kv, (size_t) state_width * state_rows) ||
                !dev(cache.attention_state_score, (size_t) state_width * state_rows)) return false;
            reset_compress_state(ratio == 4 ? CompressKind::Csa : CompressKind::Hca, ratio,
                                 cache.attention_state_kv, cache.attention_state_score, stream_);
            if (ratio == 4) {
                cache.indexer = std::make_unique<core::ElasticArray>();
                if (!cache.indexer->open(rows * 128 * sizeof(float), err) ||
                    !dev(cache.indexer_state_kv, 8 * 256) || !dev(cache.indexer_state_score, 8 * 256)) return false;
                reset_indexer_compress_state(cache.indexer_state_kv, cache.indexer_state_score, stream_);
            }
        }
        if (model::Result result = model_.ensure_tier(); !result.ok()) {
            err = result.message;
            return false;
        }
        if (model_.dspark()) {
            drafter_ = model_.dspark();
            dwin_ = drafter_->open_window();
            if (dwin_ < 0) {
                err = "deepseek4: more sessions than dspark windows";
                return false;
            }
            if (!dev(d_feat_, (size_t) kExpertWindow * drafter_->feature_floats())) {
                err = "deepseek4: cannot allocate dspark feature tap buffer";
                return false;
            }
        }
        err.clear();
        return true;
    }

    model::Result prepare() override { return {}; }

    int max_window() const override {
        if (!drafter_ || model_.spec() < 2) return 1;
        return std::clamp(model_.spec(), 1, std::min(kExpertWindow, drafter_->block_size() + 1));
    }
    int batch_rows() const override { return 8; }

    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (!drafter_) return {model::Status::NotImplemented, "deepseek4: no dspark drafter loaded"};
        if (feat_rows_ > 0 && feat_pos0_ >= 0) {
            const int take = std::min(feat_rows_, (int) std::max<int64_t>(0, pos - feat_pos0_));
            if (take > 0 && !drafter_->inject_device(dwin_, feat_pos0_, take, d_feat_, stream_, err_))
                return {model::Status::Error, err_};
            feat_rows_ = 0;
        }
        drafter_->forget_from(dwin_, pos);
        const int lim = std::min(max_window() - 1, drafter_->block_size());
        if (lim < 1) {
            last_draft_ = {};
            return {};
        }
        const int want = std::clamp(n, 1, lim);
        if (!drafter_->draft(dwin_, anchor, pos, want, stream_, last_draft_, err_))
            return {model::Status::Error, err_};
        return {};
    }

    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed&) override {
        if (!drafter_) return {model::Status::NotImplemented, "deepseek4: no dspark drafter loaded"};
        n = std::min(max_n, last_draft_.n);
        std::copy(last_draft_.token, last_draft_.token + n, ids);
        std::copy(last_draft_.conf, last_draft_.conf + n, conf);
        return {};
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.pos == 0 && !tokens_.empty() && !reset_sequence()) return {model::Status::Error, "deepseek4: " + err_};
        if (!r.tokens || r.n_tokens < 1 || r.n_tokens > 8 || r.pos < 0)
            return {model::Status::Error, "deepseek4: invalid decode window"};
        if (r.pos > (int64_t) tokens_.size())
            return {model::Status::Refused, "deepseek4: decode position is beyond the processed history"};
        if (r.pos < (int64_t) tokens_.size()) tokens_.resize((size_t) r.pos);
        if (r.pos + r.n_tokens > model_.context())
            return {model::Status::Refused, "deepseek4: request exceeds the loaded context"};

        retained_logits_rows_ = r.retain_logits_rows ? r.n_tokens : 0;
        if (!forward_rows(r.tokens, r.n_tokens, r.pos)) return {model::Status::Error, err_};
        if (r.retain_logits_rows && cudaMemcpyAsync(retained_logits_, final_logits_,
                (size_t) r.n_tokens * geo_.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
            return {model::Status::Error, "deepseek4: cannot retain decode logits rows"};
        tokens_.insert(tokens_.end(), r.tokens, r.tokens + r.n_tokens);
        if (r.row_tokens) {
            for (int t = 0; t < r.n_tokens; ++t) {
                kernels::SamplerParams params = r.sampling ? *r.sampling : kernels::SamplerParams{.temperature = 0.f, .greedy = true};
                params.counter += (uint64_t) t;
                const int* history = r.history ? r.history + (size_t) t * r.history_len : nullptr;
                kernels::sample_tokens(final_logits_ + (size_t) t * geo_.n_vocab, 1, geo_.n_vocab,
                                       history, r.history_len, params, sampled_, stream_);
                int32_t picked = -1;
                if (!copy_to_host(&picked, sampled_, sizeof(picked))) return {model::Status::Error, err_};
                r.row_tokens[t] = picked;
            }
        }
        const float* last_logits = final_logits_ + (size_t) (r.n_tokens - 1) * geo_.n_vocab;
        if (r.logits && !copy_to_host(r.logits, last_logits, (size_t) geo_.n_vocab * sizeof(float)))
            return {model::Status::Error, err_};
        if (r.next_token) {
            kernels::SamplerParams params = r.sampling ? *r.sampling : kernels::SamplerParams{.temperature = 0.f, .greedy = true};
            const int* history = r.history ? r.history + (size_t) (r.n_tokens - 1) * r.history_len : nullptr;
            kernels::sample_tokens(last_logits, 1, geo_.n_vocab, history, r.history_len, params, sampled_, stream_);
            if (!copy_to_host(r.next_token, sampled_, sizeof(*r.next_token))) return {model::Status::Error, err_};
        } else if (!sync()) return {model::Status::Error, err_};
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos == 0 && !tokens_.empty() && !reset_sequence()) return {model::Status::Error, "deepseek4: " + err_};
        if (!r.tokens || r.n_tokens < 1 || r.pos != (int64_t) tokens_.size())
            return {model::Status::Error, "deepseek4: invalid or nonconsecutive prefill chunk (pos=" +
                    std::to_string(r.pos) + ", history=" + std::to_string(tokens_.size()) + ")"};
        if (r.pos + r.n_tokens > model_.context())
            return {model::Status::Refused, "deepseek4: prompt exceeds the loaded context"};
        if (r.n_tokens <= kExpertWindow && model_.diag_rows() <= 0) {
            if (!forward_rows(r.tokens, r.n_tokens, r.pos)) return {model::Status::Error, err_};
            tokens_.insert(tokens_.end(), r.tokens, r.tokens + r.n_tokens);
            if (r.progress) r.progress(r.n_tokens);
            if (!sync()) return {model::Status::Error, err_};
            return {};
        }
        if (!prefill_rows(r.tokens, r.n_tokens, r.pos, nullptr, r.stop, r.progress))
            return {model::Status::Error, err_};
        return {};
    }

    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows,
                               model::ImageGrid& grid) override {
        n_rows = 0;
        grid = {};
        if (!model_.vision_enabled())
            return {model::Status::Refused, "deepseek4: image input requires --vision dynamic"};
        if (!data || bytes == 0) return {model::Status::Error, "deepseek4: image input is empty"};

        basalt::vision::Image image;
        if (!basalt::vision::decode_image(data, bytes, image, err_)) return fail_image(err_);
        VisionPlan plan;
        if (!plan_image(image.w, image.h, plan, err_)) return fail_image(err_);
        std::vector<float> patches;
        if (!image_patches(image, plan, patches, err_)) return fail_image(err_);

        if (!vision_mmproj_) {
            core::ExpertTier* tier = model_.tier();
            vision_borrowed_ = tier != nullptr;
            if (tier) tier->shrink();
            vision_mmproj_ = std::make_unique<basalt::vision::Mmproj>();
            if (!vision_mmproj_->open(model_.weights().pack().vision, err_, "deepseek4v")) return fail_image(err_);
            vision_blas_ = std::make_unique<basalt::vision::TransformerBlas>();
            vision_encoder_ = std::make_unique<basalt::vision::TransformerEncoder>();
            if (!vision_encoder_->init(*vision_mmproj_, *vision_blas_, err_)) {
                vision_encoder_.reset();
                if (!lend_vision(tier, model_.weights().pack().vision.size())) return fail_image(err_);
                err_.clear();
                vision_blas_ = std::make_unique<basalt::vision::TransformerBlas>();
                vision_encoder_ = std::make_unique<basalt::vision::TransformerEncoder>();
                if (!vision_encoder_->init(*vision_mmproj_, *vision_blas_, err_)) return fail_image(err_);
            }
        }

        const basalt::vision::TransformerGrid tg{plan.n_vit_w, plan.n_vit_h, plan.n_llm_w, plan.n_llm_h};
        core::ExpertTier* tier = model_.tier();
        bool room = vision_encoder_->reserve(tg, err_) && vision_room_above_floor(tier);
        if (!room && lend_vision(tier, vision_encoder_->work_bytes(tg))) {
            err_.clear();
            room = vision_encoder_->reserve(tg, err_) && vision_room_above_floor(tier);
        }
        if (!room) {
            if (err_.empty()) err_ = "not enough free device memory for the image encoder above the VRAM reserve";
            return fail_image(err_);
        }
        std::vector<float> projected;
        if (!vision_encoder_->encode_grid(patches, tg, projected, err_)) return fail_image(err_);
        ImageBlock block;
        if (!build_image_block(plan, 0, block, err_)) return fail_image(err_);
        const int width = vision_encoder_->width();
        const auto special = [&](const char* name) -> const float* {
            const auto* tensor = vision_mmproj_->find(name);
            return tensor && tensor->type == 0 && tensor->elems() == width
                ? static_cast<const float*>(tensor->data) : nullptr;
        };
        if (!assemble_image_embeddings(block, projected, width, special("v.token_embd.img_start"),
                                       special("v.token_embd.img_end"), special("v.token_embd.img_pad"),
                                       special("v.image_newline"), rows, err_)) return fail_image(err_);
        if (width != geo_.n_embd || rows.empty() || rows.size() % (size_t) geo_.n_embd != 0 ||
            rows.size() / (size_t) geo_.n_embd > (size_t) std::numeric_limits<int>::max()) {
            return fail_image("projected image rows do not match the text model width");
        }
        n_rows = (int) (rows.size() / (size_t) geo_.n_embd);
        grid = {plan.n_llm_w, plan.n_llm_h};
        err_.clear();
        return {};
    }

    void images_done() override {
        vision_encoder_.reset();
        vision_blas_.reset();
        vision_mmproj_.reset();
        if (!vision_borrowed_) return;
        vision_borrowed_ = false;
        if (core::ExpertTier* tier = model_.tier()) {
            std::string err;
            if (vision_lent_ && !tier->reclaim(err))
                basalt::log::warn("basalt deepseek4", "VRAM expert tier stays closed: %s", err.c_str());
            tier->grow();
        }
        vision_lent_ = false;
    }

    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid grid) override {
        (void) grid;
        if (pos == 0 && !tokens_.empty() && !reset_sequence()) return {model::Status::Error, "deepseek4: " + err_};
        if (!rows || n_rows < 1 || pos != (int64_t) tokens_.size())
            return {model::Status::Error, "deepseek4: invalid or nonconsecutive image rows"};
        if (pos + n_rows > model_.context())
            return {model::Status::Refused, "deepseek4: image prompt exceeds the loaded context"};
        std::vector<int32_t> tokens((size_t) n_rows, geo_.n_vocab);
        if (!prefill_rows(tokens.data(), n_rows, pos, rows))
            return {model::Status::Error, err_};
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        if (!bias) {
            has_bias_ = false;
            return {};
        }
        if (cudaMemcpyAsync(logit_bias_, bias, (size_t) geo_.n_vocab * sizeof(float),
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess || !sync())
            return {model::Status::Error, err_};
        has_bias_ = true;
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        if (row != 0 || tokens_.empty()) return {model::Status::Refused, "deepseek4: no decoded row to resample"};
        if (bias && cudaMemcpyAsync(logit_bias_, bias, (size_t) geo_.n_vocab * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return {model::Status::Error, "deepseek4: cannot upload the sampling bias"};
        kernels::SamplerParams params;
        params.temperature = 0.f;
        params.greedy = true;
        params.logit_bias = bias ? logit_bias_ : (has_bias_ ? logit_bias_ : nullptr);
        kernels::sample_tokens(final_logits_, 1, geo_.n_vocab, nullptr, 0, params, sampled_, stream_);
        if (!copy_to_host(&token, sampled_, sizeof(token))) return {model::Status::Error, err_};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= 8 || !out || (row > 0 && row >= retained_logits_rows_))
            return {model::Status::Error, "deepseek4: logits row is outside the retained decode window"};
        const float* source = retained_logits_rows_ > 0
            ? retained_logits_ + (size_t) row * geo_.n_vocab
            : final_logits_;
        return copy_to_host(out, source, (size_t) geo_.n_vocab * sizeof(float))
            ? model::Result{}
            : model::Result{model::Status::Error, "deepseek4: " + err_};
    }

    // Checkpoints keep the state which cannot be reconstructed by replaying the positional compressed rows:
    // the sliding-window rings and each compressor's rolling overlap state. Compressed rows themselves are
    // serialized by save_cells(). The checkpoint is taken only at the session's current token boundary.
    size_t running_bytes() const override {
        size_t bytes = (size_t) geo_.n_layers * kBootstrapWindow * 512 * sizeof(float);
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            const LayerCache& c = layer_cache_[(size_t) layer];
            const int width = ratio == 4 ? 1024 : 512;
            const int rows = ratio == 4 ? 8 : ratio;
            bytes += (size_t) width * rows * sizeof(float) * 2;
            if (ratio == 4) bytes += (size_t) 8 * 256 * sizeof(float) * 2;
            (void) c;
        }
        return bytes;
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& state) override {
        if (pos < 0 || pos > model_.context() || (save && pos != (int64_t) tokens_.size()))
            return {model::Status::Error, "deepseek4: checkpoint position is not a valid session boundary"};
        if (save && !sync()) return {model::Status::Error, "deepseek4: " + err_};
        state.regions.push_back({window_rings_, (size_t) geo_.n_layers * kBootstrapWindow * 512 * sizeof(float)});
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            const LayerCache& c = layer_cache_[(size_t) layer];
            const size_t width = ratio == 4 ? 1024 : 512;
            const size_t rows = ratio == 4 ? 8 : (size_t) ratio;
            const size_t bytes = width * rows * sizeof(float);
            state.regions.push_back({c.attention_state_kv, bytes});
            state.regions.push_back({c.attention_state_score, bytes});
            if (ratio == 4) {
                state.regions.push_back({c.indexer_state_kv, (size_t) 8 * 256 * sizeof(float)});
                state.regions.push_back({c.indexer_state_score, (size_t) 8 * 256 * sizeof(float)});
            }
        }
        state.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if (pos < 0 || pos > (int64_t) tokens_.size())
            return {model::Status::Error, "deepseek4: restored checkpoint position is outside the session history"};
        tokens_.resize((size_t) pos);
        has_bias_ = false;
        return {};
    }

    size_t cells_bytes(int64_t positions) const override {
        if (positions < 0) return 0;
        size_t bytes = 0;
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            const size_t rows = (size_t) (positions / ratio);
            bytes += rows * 512 * sizeof(float);
            if (ratio == 4) bytes += rows * 128 * sizeof(float);
        }
        return bytes;
    }

    int64_t disk_row_cells() const override { return 4; }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions < 0 || positions > (int64_t) tokens_.size())
            return {model::Status::Error, "deepseek4: cannot save positions that were not processed"};
        if (!sync()) return {model::Status::Error, "deepseek4: " + err_};
        kv.clear();
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            const size_t rows = (size_t) (positions / ratio);
            const size_t attn_bytes = rows * 512 * sizeof(float);
            const LayerCache& c = layer_cache_[(size_t) layer];
            if (attn_bytes > c.attention->backed() ||
                (ratio == 4 && rows * 128 * sizeof(float) > c.indexer->backed()))
                return {model::Status::Error, "deepseek4: a compressed cache is not backed through the checkpoint position"};
            core::ConversationKv item;
            item.format = 400 + ratio;
            item.cells = positions;
            item.page_size = ratio;
            item.heads = 1;
            item.head_dim = 512;
            item.pooled_rows = ratio == 4 ? (int64_t) rows : 0;
            item.idx_dim = ratio == 4 ? 128 : 0;
            item.k.resize_for_overwrite(attn_bytes);
            const auto down = [&](const uint8_t* src, core::ConversationBuffer& dst) {
                return dst.visit(0, dst.size(), [&](uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpyAsync(p, src + at, n, cudaMemcpyDeviceToHost, stream_) == cudaSuccess;
                });
            };
            if (!down(c.attention->data(), item.k))
                return {model::Status::Error, "deepseek4: compressed attention checkpoint copy failed"};
            if (ratio == 4) {
                item.pooled.resize_for_overwrite(rows * 128 * sizeof(float));
                if (!down(c.indexer->data(), item.pooled))
                    return {model::Status::Error, "deepseek4: indexer checkpoint copy failed"};
            }
            kv.push_back(std::move(item));
        }
        if (!sync()) return {model::Status::Error, "deepseek4: " + err_};
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions < 0 || positions > model_.context())
            return {model::Status::Error, "deepseek4: restored cache position is outside the model context"};
        size_t index = 0;
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            if (index >= kv.size()) return {model::Status::Error, "deepseek4: parked conversation is missing a compressed layer"};
            const core::ConversationKv& item = kv[index++];
            const size_t rows = (size_t) (positions / ratio);
            const size_t attn_bytes = rows * 512 * sizeof(float);
            if (item.format != 400 + ratio || item.cells != positions || item.page_size != ratio || item.heads != 1 ||
                item.head_dim != 512 || item.k.size() != attn_bytes ||
                (ratio == 4 && (item.pooled_rows != (int64_t) rows || item.idx_dim != 128 ||
                                item.pooled.size() != rows * 128 * sizeof(float))) ||
                (ratio != 4 && (!item.pooled.empty() || item.pooled_rows != 0 || item.idx_dim != 0)))
                return {model::Status::Error, "deepseek4: parked compressed layer geometry does not match"};
            LayerCache& c = layer_cache_[(size_t) layer];
            const uint64_t keep_free = model_.tier() ? model_.tier()->free_floor() : 0;
            std::string why;
            if (!c.attention->ensure(attn_bytes, keep_free, why)) {
                if (model_.tier()) {
                    model_.tier()->shrink();
                    if (!c.attention->ensure(attn_bytes, keep_free, why)) return {model::Status::Error, "deepseek4: " + why};
                } else return {model::Status::Error, "deepseek4: " + why};
            }
            if (ratio == 4 && !c.indexer->ensure(rows * 128 * sizeof(float), keep_free, why))
                return {model::Status::Error, "deepseek4: " + why};
            const auto up = [&](const core::ConversationBuffer& src, uint8_t* dst) {
                return src.visit(0, src.size(), [&](const uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpyAsync(dst + at, p, n, cudaMemcpyHostToDevice, stream_) == cudaSuccess;
                });
            };
            if (!up(item.k, c.attention->data()) || (ratio == 4 && !up(item.pooled, c.indexer->data())))
                return {model::Status::Error, "deepseek4: compressed cache restore copy failed"};
        }
        if (index != kv.size()) return {model::Status::Error, "deepseek4: parked conversation has unexpected cache layers"};
        return sync() ? model::Result{} : model::Result{model::Status::Error, "deepseek4: " + err_};
    }

    model::Result set_history(const int32_t* tokens, int64_t n) override {
        if (n < 0 || (n > 0 && !tokens) || n > model_.context())
            return {model::Status::Error, "deepseek4: invalid restored token history"};
        if (n == 0) tokens_.clear();
        else tokens_.assign(tokens, tokens + n);
        return {};
    }

    model::Result set_low_memory(bool low, int64_t& expert_slots) override {
        core::ExpertTier* tier = model_.tier();
        if (!tier) { expert_slots = -1; return {}; }
        if (cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "deepseek4: cannot synchronize before SLEEP/WAKE"};
        if (low) {
            tier->lend_all();
        } else {
            std::string why;
            if (!tier->reclaim(why)) return {model::Status::Error, "deepseek4: " + why};
            tier->grow();
        }
        expert_slots = tier->gpu_slots();
        return {};
    }

    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        core::ExpertTier* tier = model_.tier();
        if (!tier) { expert_slots = -1; return {}; }
        if (cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "deepseek4: cannot synchronize before changing the VRAM reserve"};
        std::string why;
        if (!tier->replan(mib, why)) return {model::Status::Error, "deepseek4: " + why};
        expert_slots = tier->gpu_slots();
        return {};
    }

private:
    const char* batch_name() const override { return "deepseek4"; }
    int64_t batch_context() const override { return model_.context(); }
    model::BatchShared& batch_shared() override { return batch_shared_; }
    cudaStream_t batch_stream() const override { return stream_; }
    bool batch_reserve(const model::DecodeRequest&, std::string&) override { return false; }
    bool batch_stage(const model::BatchView&, model::GraphLru<core::CapturedGraph>::Key&, std::string& err) override {
        err = "the V4 token path still steers expert routing on the host";
        return false;
    }
    bool batch_forward(const model::BatchView&, std::string& err) override {
        err = "the V4 token path has no shared batch forward yet";
        return false;
    }
    bool batch_ready() const override { return false; }
    void batch_take(const model::BatchView&, int) override {}
    const int32_t* batch_draw(const model::BatchView&, int, std::string& err) override {
        err = "the V4 token path has no shared batch draw";
        return nullptr;
    }

    bool reset_sequence() {
        if (!sync()) return false;
        if (cudaMemsetAsync(window_rings_, 0, (size_t) geo_.n_layers * kBootstrapWindow * 512 * sizeof(float), stream_) != cudaSuccess) {
            err_ = "cannot clear the attention window rings";
            return false;
        }
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int ratio = geo_.compress_ratio[(size_t) layer];
            if (!ratio) continue;
            LayerCache& c = layer_cache_[(size_t) layer];
            reset_compress_state(ratio == 4 ? CompressKind::Csa : CompressKind::Hca, ratio,
                                 c.attention_state_kv, c.attention_state_score, stream_);
            if (ratio == 4) reset_indexer_compress_state(c.indexer_state_kv, c.indexer_state_score, stream_);
        }
        tokens_.clear();
        feat_rows_ = 0;
        feat_pos0_ = -1;
        if (drafter_) drafter_->forget_from(dwin_, 0);
        return sync();
    }

    bool vision_room_above_floor(core::ExpertTier* tier) const {
        if (!tier) return true;
        size_t free_bytes = 0, total_bytes = 0;
        return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes >= tier->free_floor();
    }

    bool lend_vision(core::ExpertTier* tier, uint64_t bytes) {
        if (!tier || vision_lent_) return false;
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return false;
        if (tier->partial_lend()) {
            if (!tier->lend(bytes)) return false;
        } else {
            tier->lend_all();
        }
        vision_lent_ = true;
        return true;
    }

    model::Result fail_image(const std::string& message) {
        images_done();
        return {model::Status::Error, "deepseek4: " + message};
    }

    template<class T> bool allocate(T*& pointer, size_t count, std::string& err) {
        constexpr size_t item_bytes = [] {
            if constexpr (std::is_void_v<T>) return size_t{1};
            else return sizeof(T);
        }();
        if (count > std::numeric_limits<size_t>::max() / item_bytes) {
            err = "DeepSeek4 session allocation size overflows";
            return false;
        }
        void* allocation = nullptr;
        if (cudaMalloc(&allocation, count * item_bytes) != cudaSuccess) {
            err = "out of CUDA memory for DeepSeek4 session state";
            return false;
        }
        device_.push_back(allocation);
        pointer = static_cast<T*>(allocation);
        return true;
    }

    bool sync() {
        const cudaError_t error = cudaStreamSynchronize(stream_);
        if (error == cudaSuccess) return true;
        err_ = cudaGetErrorString(error);
        return false;
    }

    bool copy_to_host(void* dst, const void* src, size_t bytes) {
        if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
            err_ = "CUDA device-to-host copy failed";
            return false;
        }
        return sync();
    }

    bool forward_rows(const int32_t* tokens, int n_tokens, int64_t pos, const float* supplied_embeddings = nullptr,
                      bool attention_only = false, int layer_only = -1, bool begin_tier = true);
    bool prefill_rows(const int32_t* tokens, int n_tokens, int64_t pos, const float* supplied_embeddings = nullptr,
                      const std::function<bool()>& stop = {}, const std::function<void(int)>& progress = {});
    bool routed_step(int layer, const float* normalized, int n_tokens, const FfnRouteWeights& route,
                     const float* weights, float* output);
    bool fail_cuda(const char* operation);

    DsModel& model_;
    const Geometry& geo_;
    model::BatchShared& batch_shared_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> device_;
    std::vector<RopeTables> ropes_;
    std::vector<LayerCache> layer_cache_;
    std::vector<int32_t> tokens_;
    std::string err_;
    bool has_bias_ = false;
    std::unique_ptr<basalt::vision::Mmproj> vision_mmproj_;
    std::unique_ptr<basalt::vision::TransformerBlas> vision_blas_;
    std::unique_ptr<basalt::vision::TransformerEncoder> vision_encoder_;
    bool vision_borrowed_ = false;
    bool vision_lent_ = false;

    float *streams_ = nullptr, *next_streams_ = nullptr, *attention_output_ = nullptr;
    float *ffn_output_ = nullptr, *shared_output_ = nullptr, *routed_output_ = nullptr, *final_logits_ = nullptr;
    DsparkDrafter* drafter_ = nullptr;
    int dwin_ = -1;
    DsparkDraft last_draft_{};
    float* d_feat_ = nullptr;
    int64_t feat_pos0_ = -1;
    int feat_rows_ = 0;
    float *normalized_ = nullptr, *hc_workspace_ = nullptr, *attn_coef_ = nullptr, *ffn_coef_ = nullptr;
    float *collapsed_ = nullptr, *attn_normed_ = nullptr, *ffn_normed_ = nullptr;
    void *attn_q8_ = nullptr, *q_latent_q8_ = nullptr, *attn_out_q8_ = nullptr;
    float *q_latent_ = nullptr, *q_heads_ = nullptr, *kv_projected_ = nullptr, *kv_row_ = nullptr;
    float *mqa_output_ = nullptr, *mqa_workspace_ = nullptr, *attn_out_latent_ = nullptr;
    float *attn_comp_kv_ = nullptr, *attn_comp_score_ = nullptr;
    float *indexer_kv_ = nullptr, *indexer_score_ = nullptr, *indexer_q_ = nullptr, *indexer_head_weights_ = nullptr;
    float *compress_rope_freq_ = nullptr, *indexer_rope_angles_ = nullptr;
    void *indexer_project_workspace_ = nullptr, *indexer_select_workspace_ = nullptr;
    int32_t *comp_row_ = nullptr, *selected_ids_ = nullptr, *indexer_select_count_ = nullptr, *comp_limit_ = nullptr;
    void *shared_in_q8_ = nullptr, *shared_hidden_q8_ = nullptr;
    float *shared_gate_ = nullptr, *shared_up_ = nullptr, *shared_hidden_ = nullptr;
    float *router_logits_ = nullptr, *route_weights_ = nullptr;
    int32_t *route_ids_ = nullptr, *route_fixed_row_ = nullptr;
    uint8_t *route_fixed_mask_ = nullptr, *route_alt_mask_ = nullptr;
    int32_t *position_meta_ = nullptr, *positions_ = nullptr, *win_start_ = nullptr, *win_n_ = nullptr, *sampled_ = nullptr;
    int32_t* hash_routes_ = nullptr;
    float* window_rings_ = nullptr;
    void *expert_input_q8_ = nullptr, *expert_scratch_ = nullptr, *expert_plan_ = nullptr;
    float* logit_bias_ = nullptr;
    float *head_hc_workspace_ = nullptr, *head_coef_ = nullptr, *head_collapsed_ = nullptr, *head_normed_ = nullptr;
    float *retained_logits_ = nullptr;
    int retained_logits_rows_ = 0;
    float* h_initial_ = nullptr;
    float *h_expert_input_ = nullptr, *h_expert_rows_ = nullptr;
    float* d_expert_rows_ = nullptr;
    float* d_expert_input_ = nullptr;
    int32_t *h_expert_ids_ = nullptr;
    uint32_t *h_expert_seq_ = nullptr, *d_expert_seq_ = nullptr;
    int32_t* d_expert_ids_ = nullptr;
    RouteMeta *h_route_meta_ = nullptr, *d_route_meta_ = nullptr;
};

bool Session::forward_rows(const int32_t* tokens, int n_tokens, int64_t pos, const float* supplied_embeddings,
                           bool attention_only, int layer_only, bool begin_tier) {
    if (!tokens || n_tokens < 1 || n_tokens > kExpertWindow || pos < 0 || pos + n_tokens > model_.context()) {
        err_ = "deepseek4: invalid token window";
        return false;
    }
    const size_t n = (size_t) geo_.n_embd;
    std::vector<float> embedding;
    for (int t = 0; !attention_only && t < n_tokens; ++t) {
        const int32_t token = tokens[t];
        const float* source = supplied_embeddings ? supplied_embeddings + (size_t) t * n : nullptr;
        if (token < 0 || (token >= geo_.n_vocab && !source)) {
            err_ = "deepseek4: token id is outside the checkpoint vocabulary";
            return false;
        }
        if (!source) {
            if (!model_.weights().embedding_row(token, embedding, err_)) {
                err_ = "deepseek4: " + err_;
                return false;
            }
            source = embedding.data();
        }
        for (int stream = 0; stream < 4; ++stream)
            std::copy(source, source + n, h_initial_ + ((size_t) t * 4 + stream) * n);
    }
    if (!attention_only && cudaMemcpyAsync(streams_, h_initial_, (size_t) n_tokens * 4 * n * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_) != cudaSuccess)
        return fail_cuda("embedding upload");

    int32_t attention_meta[kExpertWindow * 3]{};
    for (int t = 0; t < n_tokens; ++t) {
        const int32_t position = (int32_t) (pos + t);
        const int32_t start = (int32_t) std::max<int64_t>(0, pos + t - kBootstrapWindow + 1);
        attention_meta[t] = position;
        attention_meta[kExpertWindow + t] = start;
        attention_meta[2 * kExpertWindow + t] = position - start + 1;
    }
    if (cudaMemcpyAsync(position_meta_, attention_meta, sizeof(attention_meta), cudaMemcpyHostToDevice, stream_) != cudaSuccess)
        return fail_cuda("attention metadata upload");

    if (begin_tier) if (core::ExpertTier* tier = model_.tier()) tier->begin_token(n_tokens);
    const int first_layer = layer_only >= 0 ? layer_only : 0;
    const int end_layer = layer_only >= 0 ? layer_only + 1 : geo_.n_layers;
    for (int layer = first_layer; layer < end_layer; ++layer) {
        AttentionLayerWeights attention;
        FfnRouteWeights route;
        SharedExpertWeights shared;
        if (!model_.weights().attention_layer_weights(layer, attention, err_) ||
            !model_.weights().ffn_layer_weights(layer, route, shared, err_)) {
            err_ = "deepseek4: " + err_;
            return false;
        }
        AttentionLayerScratch attention_scratch;
        attention_scratch.hc_workspace = hc_workspace_;
        attention_scratch.coefficients = attn_coef_;
        attention_scratch.collapsed = collapsed_;
        attention_scratch.normalized = attn_normed_;
        attention_scratch.attention = {attn_q8_, q_latent_, q_latent_q8_, q_heads_, kv_projected_, kv_row_,
                                       mqa_output_, mqa_workspace_, attn_out_q8_, attn_out_latent_, attention_output_};

        kernels::SparseMqaArgs selection;
        selection.n_tokens = n_tokens;
        selection.win = window_rings_ + (size_t) layer * kBootstrapWindow * 512;
        selection.win_type = kernels::MqaRowType::F32;
        selection.win_cap = kBootstrapWindow;
        selection.win_start = win_start_;
        selection.win_n = win_n_;
        selection.pos = positions_;
        selection.rope = ropes_[(size_t) layer].attention;

        try {
            prepare_attention_layer(streams_, n_tokens, attention, attention_scratch, stream_);
            kernels::native_quantize_q8_1(attn_normed_, attn_q8_, geo_.n_embd, n_tokens, stream_);
            kernels::sparse_mqa_q_latent(attn_q8_, attention.attention.q, n_tokens, q_latent_, q_latent_q8_, stream_);

            const int ratio = geo_.compress_ratio[(size_t) layer];
            const int visible = ratio ? (int) ((pos + n_tokens) / ratio) : 0;
            int32_t comp_limits[kExpertWindow]{};
            for (int t = 0; t < n_tokens; ++t)
                comp_limits[t] = ratio ? (int32_t) ((pos + t + 1) / ratio) : 0;
            if (ratio && cudaMemcpyAsync(comp_limit_, comp_limits, (size_t) n_tokens * sizeof(int32_t),
                                         cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                return fail_cuda("compressed attention limits upload");
        selection.n_sel = 0;
            selection.comp = nullptr;
            selection.sel = nullptr;
            selection.comp_limit = nullptr;
            if (ratio) {
                LayerCache& cache = layer_cache_[(size_t) layer];
                const uint64_t cache_bytes = (uint64_t) visible * 512 * sizeof(float);
                const uint64_t keep_free = model_.tier() ? model_.tier()->free_floor() : 0;
                std::string cache_error;
                const auto ensure_cache = [&](core::ElasticArray& array, uint64_t bytes) {
                    if (array.ensure(bytes, keep_free, cache_error)) return true;
                    if (model_.tier()) {
                        model_.tier()->shrink();
                        if (array.ensure(bytes, keep_free, cache_error)) return true;
                    }
                    err_ = "deepseek4: cannot grow compressed cache at position " + std::to_string(pos) + ": " + cache_error;
                    return false;
                };

                const DeviceTensor* attn_ape = model_.weights().tensor(layer, "attn_compressor_ape.weight");
                const DeviceTensor* attn_norm = model_.weights().tensor(layer, "attn_compressor_norm.weight");
                const DeviceTensor* attn_gate = model_.weights().tensor(layer, "attn_compressor_gate.weight");
                const DeviceTensor* attn_kv = model_.weights().tensor(layer, "attn_compressor_kv.weight");
                if (!attn_ape || !attn_norm || !attn_gate || !attn_kv) {
                    err_ = "deepseek4: compressed-attention layer is missing validated compressor tensors";
                    return false;
                }
                if (!ensure_cache(*cache.attention, cache_bytes)) return false;

                if (ratio == 4) {
                    const DeviceTensor* index_ape = model_.weights().tensor(layer, "indexer_compressor_ape.weight");
                    const DeviceTensor* index_norm = model_.weights().tensor(layer, "indexer_compressor_norm.weight");
                    const DeviceTensor* index_gate = model_.weights().tensor(layer, "indexer_compressor_gate.weight");
                    const DeviceTensor* index_kv = model_.weights().tensor(layer, "indexer_compressor_kv.weight");
                    const DeviceTensor* index_q = model_.weights().tensor(layer, "indexer.attn_q_b.weight");
                    const DeviceTensor* index_proj = model_.weights().tensor(layer, "indexer.proj.weight");
                    if (!index_ape || !index_norm || !index_gate || !index_kv || !index_q || !index_proj) {
                        err_ = "deepseek4: CSA layer is missing validated indexer tensors";
                        return false;
                    }
                    if (!ensure_cache(*cache.indexer, (uint64_t) visible * 128 * sizeof(float))) return false;

                    kernels::bf16_gemv_fp32_mmvf_multi(attn_normed_, geo_.n_embd,
                                                       static_cast<const uint16_t*>(index_kv->ptr), indexer_kv_,
                                                       256, geo_.n_embd, 256, n_tokens, stream_);
                    kernels::bf16_gemv_fp32_mmvf_multi(attn_normed_, geo_.n_embd,
                                                       static_cast<const uint16_t*>(index_gate->ptr), indexer_score_,
                                                       256, geo_.n_embd, 256, n_tokens, stream_);
                    indexer_compress_projected(n_tokens, indexer_kv_, indexer_score_, static_cast<const float*>(index_ape->ptr),
                        positions_, static_cast<const float*>(index_norm->ptr), geo_.rms_eps, compress_rope_freq_,
                        cache.indexer_state_kv, cache.indexer_state_score,
                        reinterpret_cast<float*>(cache.indexer->data()), comp_row_, stream_);

                    float angles[kExpertWindow * 32]{};
                    for (int t = 0; t < n_tokens; ++t)
                        for (int i = 0; i < 32; ++i)
                            angles[(size_t) t * 32 + i] = (float) (pos + t) * ropes_[(size_t) layer].frequency[(size_t) i];
                    if (cudaMemcpyAsync(indexer_rope_angles_, angles, (size_t) n_tokens * 32 * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                        return fail_cuda("indexer rope angles upload");
                    indexer_project_q8_0(index_q->ptr, static_cast<const uint16_t*>(index_proj->ptr), q_latent_,
                        attn_normed_, indexer_rope_angles_, n_tokens, indexer_q_, indexer_head_weights_,
                        indexer_project_workspace_, indexer_project_workspace_bytes(n_tokens), stream_);
                    if (visible > 0) {
                        indexer_select(indexer_q_, indexer_head_weights_, reinterpret_cast<const float*>(cache.indexer->data()),
                            visible, (int) pos, n_tokens, selected_ids_, indexer_select_count_, indexer_select_workspace_,
                            indexer_select_workspace_bytes(visible), stream_);
                        selection.n_sel = kernels::kSparseMqaTopK;
                        selection.sel = selected_ids_;
                        selection.comp = cache.attention->data();
                        selection.comp_type = kernels::MqaRowType::F32;
                        selection.comp_limit = comp_limit_;
                    }
                } else {
                    selection.n_sel = visible;
                    selection.comp = cache.attention->data();
                    selection.comp_type = kernels::MqaRowType::F32;
                    if (visible > 0) {
                        std::vector<int32_t> ids((size_t) n_tokens * visible);
                        for (int t = 0; t < n_tokens; ++t)
                            for (int i = 0; i < visible; ++i) ids[(size_t) t * visible + i] = i;
                        if (cudaMemcpyAsync(selected_ids_, ids.data(), ids.size() * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                            return fail_cuda("compressed attention ids upload");
                        selection.sel = selected_ids_;
                        selection.comp_limit = comp_limit_;
                    }
                }
                const CompressKind kind = ratio == 4 ? CompressKind::Csa : CompressKind::Hca;
                project_and_compress(kind, ratio, geo_.n_embd, n_tokens, attn_normed_,
                    static_cast<const uint16_t*>(attn_kv->ptr), static_cast<const uint16_t*>(attn_gate->ptr),
                    static_cast<const float*>(attn_ape->ptr), positions_, static_cast<const float*>(attn_norm->ptr),
                    geo_.rms_eps, compress_rope_freq_, attn_comp_kv_, attn_comp_score_,
                    cache.attention_state_kv, cache.attention_state_score,
                    reinterpret_cast<float*>(cache.attention->data()), comp_row_, stream_);
            }

            kernels::sparse_mqa_q_heads(attention.attention.q, n_tokens, positions_, ropes_[(size_t) layer].attention,
                                        q_latent_q8_, q_heads_, stream_);
            project_attention_kv(attn_normed_, attention.attention.kv_type, attention.attention.kv,
                attention.attention.kv_norm, geo_.rms_eps, positions_, ropes_[(size_t) layer].attention, n_tokens,
                attn_q8_, kv_projected_, kv_row_, stream_);
            cache_attention_window(window_rings_ + (size_t) layer * kBootstrapWindow * 512,
                                   kv_row_, positions_, n_tokens, stream_);
            selection.q = q_heads_;
            selection.sinks = attention.attention.sinks;
            selection.round_output_bf16 = true;
            selection.out = mqa_output_;
            kernels::sparse_mqa(selection, mqa_workspace_, stream_);
            kernels::sparse_mqa_output(mqa_output_, attention.attention.output_a_type, attention.attention.output_a,
                attention.attention.output_b_type, attention.attention.output_b, geo_.n_embd, n_tokens,
                attn_out_q8_, attn_out_latent_, attention_output_, stream_);
            finish_attention_layer(streams_, attn_coef_, attention_output_, n_tokens, geo_.n_embd, next_streams_, stream_);
        } catch (const std::exception& e) {
            err_ = std::string("deepseek4 attention layer ") + std::to_string(layer) + ": " + e.what();
            return false;
        }

        if (attention_only) return true;

        RouteRows rows;
        if (!build_route_rows(tokens, n_tokens, geo_.n_vocab, layer, geo_.n_layers, geo_.hash_layers, rows, err_)) return false;
        std::copy_n(rows.fixed_rows.data(), n_tokens, h_route_meta_->fixed_row);
        std::copy_n(rows.fixed_mask.data(), n_tokens, h_route_meta_->fixed_mask);
        std::copy_n(rows.alternate_mask.data(), n_tokens, h_route_meta_->alternate_mask);
        route.routing = router_mixed_args(geo_.n_expert, geo_.n_vocab,
            layer < geo_.hash_layers ? hash_routes_ + (size_t) layer * geo_.n_vocab * 6 : nullptr,
            route_fixed_row_, route_fixed_mask_, route_alt_mask_, route.routing.bias, route.routing.alternate_bias);

        FfnRouteScratch route_scratch{hc_workspace_, ffn_coef_, collapsed_, ffn_normed_, router_logits_, route_ids_, route_weights_};
        try {
            SharedExpertScratch shared_scratch{shared_in_q8_, shared_hidden_q8_, shared_gate_, shared_up_, shared_hidden_};
            prepare_ffn_routes(next_streams_, n_tokens, route, route_scratch, stream_);
            shared_expert_step(ffn_normed_, n_tokens, shared, shared_scratch, shared_output_, stream_);
        } catch (const std::exception& e) {
            err_ = std::string("deepseek4 FFN layer ") + std::to_string(layer) + ": " + e.what();
            return false;
        }
        if (!routed_step(layer, ffn_normed_, n_tokens, route, route_weights_, routed_output_)) return false;
        try {
            finish_ffn_layer(next_streams_, ffn_coef_, shared_output_, routed_output_, n_tokens, geo_.n_embd,
                             streams_, stream_);
        } catch (const std::exception& e) {
            err_ = std::string("deepseek4 FFN finish ") + std::to_string(layer) + ": " + e.what();
            return false;
        }
        if (drafter_) {
            const int slot = drafter_->feature_slot(layer);
            if (slot >= 0) {
                const size_t fstride = (size_t) drafter_->feature_floats();
                for (int t = 0; t < n_tokens; ++t) {
                    if (cudaMemcpyAsync(d_feat_ + (size_t) t * fstride + (size_t) slot * n,
                                        collapsed_ + (size_t) t * n, n * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                        return fail_cuda("dspark feature tap");
                }
                feat_pos0_ = pos;
                feat_rows_ = n_tokens;
            }
        }
    }

    HeadWeights head;
    if (!model_.weights().head_weights(head, err_)) {
        err_ = "deepseek4: " + err_;
        return false;
    }
    HeadScratch head_scratch{head_hc_workspace_, head_coef_, head_collapsed_, head_normed_};
    try {
        model_head_step(streams_, n_tokens, head, head_scratch, final_logits_, stream_);
    } catch (const std::exception& e) {
        err_ = std::string("deepseek4 output head: ") + e.what();
        return false;
    }
    return true;
}

bool Session::prefill_rows(const int32_t* tokens, int total_tokens, int64_t pos, const float* supplied_embeddings,
                           const std::function<bool()>& stop, const std::function<void(int)>& progress) {
    if (!tokens || total_tokens < 1 || pos < 0 || pos + total_tokens > model_.context()) {
        err_ = "deepseek4: invalid prompt row chunk";
        return false;
    }
    core::ExpertTier* tier = model_.tier();
    if (!tier) {
        err_ = "deepseek4: prompt expert tier is unavailable";
        return false;
    }

    // The prompt workspaces temporarily use the VRAM occupied by expert copies.  Destroy them before
    // reclaiming those slots so the tier can restore its planned placement after the full prompt.
    tier->settle();
    tier->shrink();
    tier->lend_all();
    struct TierRestore {
        core::ExpertTier* tier;
        ~TierRestore() {
            std::string err;
            if (!tier->reclaim(err, true))
                basalt::log::warn("basalt deepseek4", "VRAM expert tier stays reduced after prompt: %s", err.c_str());
            tier->grow();
        }
    } restore{tier};
    struct DeviceBuffers {
        std::vector<void*> allocations;
        void clear() {
            for (void* p : allocations) cudaFree(p);
            allocations.clear();
        }
        ~DeviceBuffers() { clear(); }
    } buffers;
    auto alloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!bytes || cudaMalloc(&p, bytes) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        buffers.allocations.push_back(p);
        return p;
    };

    const size_t n = (size_t) geo_.n_embd;
    const size_t stream_row = 4 * n;
    constexpr int kAttnTile = 128;
    int cap = std::min(model_.diag_rows() > 0 ? model_.diag_rows() : 8192, total_tokens);
    float *current = nullptr, *attended = nullptr, *coefficients = nullptr, *collapsed = nullptr,
          *normalized = nullptr, *hc_workspace = nullptr, *logits = nullptr, *route_weights = nullptr,
          *shared = nullptr, *routed = nullptr, *chunk_q_latent = nullptr, *chunk_kv_proj = nullptr,
          *chunk_kv_row = nullptr, *chunk_out_latent = nullptr, *tile_q_heads = nullptr, *tile_mqa_out = nullptr,
          *tile_comp_kv = nullptr, *tile_comp_score = nullptr, *tile_idx_kv = nullptr, *tile_idx_score = nullptr,
          *tile_idx_qr_bf16 = nullptr, *tile_idx_q_raw = nullptr, *tile_idx_q_out = nullptr,
          *tile_idx_w_raw = nullptr, *tile_idx_hw = nullptr, *tile_idx_angles = nullptr;
    int32_t *route_ids = nullptr, *fixed_rows = nullptr, *chunk_pos = nullptr, *chunk_win_start = nullptr,
            *chunk_win_n = nullptr, *chunk_comp_limit = nullptr, *tile_comp_row = nullptr, *hca_ids = nullptr,
            *tile_sel_ids = nullptr, *tile_sel_cnt = nullptr;
    uint8_t *fixed_mask = nullptr, *alternate_mask = nullptr;
    std::unique_ptr<PrefillFfn> prompt_ffn;
    const int hca_max_visible = std::max(1, (int) ((pos + total_tokens) / 16));

    for (;;) {
        buffers.clear();
        prompt_ffn.reset();
        const size_t state_bytes = (size_t) cap * stream_row * sizeof(float);
        const int tile_cap = std::min(kAttnTile, cap);
        current = static_cast<float*>(alloc(state_bytes));
        attended = static_cast<float*>(alloc(state_bytes));
        coefficients = static_cast<float*>(alloc((size_t) cap * kernels::kHcCoef * sizeof(float)));
        collapsed = static_cast<float*>(alloc((size_t) cap * n * sizeof(float)));
        normalized = static_cast<float*>(alloc((size_t) cap * n * sizeof(float)));
        hc_workspace = static_cast<float*>(alloc(kernels::hc_mixes_workspace_floats(cap) * sizeof(float)));
        logits = static_cast<float*>(alloc((size_t) cap * geo_.n_expert * sizeof(float)));
        route_ids = static_cast<int32_t*>(alloc((size_t) cap * geo_.n_used * sizeof(int32_t)));
        route_weights = static_cast<float*>(alloc((size_t) cap * geo_.n_used * sizeof(float)));
        shared = static_cast<float*>(alloc((size_t) cap * n * sizeof(float)));
        routed = static_cast<float*>(alloc((size_t) cap * n * sizeof(float)));
        fixed_rows = static_cast<int32_t*>(alloc((size_t) cap * sizeof(int32_t)));
        fixed_mask = static_cast<uint8_t*>(alloc((size_t) cap));
        alternate_mask = static_cast<uint8_t*>(alloc((size_t) cap));
        chunk_q_latent = static_cast<float*>(alloc((size_t) cap * 1024 * sizeof(float)));
        chunk_kv_proj = static_cast<float*>(alloc((size_t) cap * 512 * sizeof(float)));
        chunk_kv_row = static_cast<float*>(alloc((size_t) cap * 512 * sizeof(float)));
        chunk_out_latent = static_cast<float*>(alloc((size_t) cap * 1024 * sizeof(float)));
        tile_q_heads = static_cast<float*>(alloc((size_t) tile_cap * 64 * 512 * sizeof(float)));
        tile_mqa_out = static_cast<float*>(alloc((size_t) tile_cap * 64 * 512 * sizeof(float)));
        tile_comp_kv = static_cast<float*>(alloc((size_t) tile_cap * 1024 * sizeof(float)));
        tile_comp_score = static_cast<float*>(alloc((size_t) tile_cap * 1024 * sizeof(float)));
        tile_idx_kv = static_cast<float*>(alloc((size_t) tile_cap * 256 * sizeof(float)));
        tile_idx_score = static_cast<float*>(alloc((size_t) tile_cap * 256 * sizeof(float)));
        tile_idx_qr_bf16 = static_cast<float*>(alloc((size_t) tile_cap * 1024 * sizeof(float)));
        tile_idx_q_raw = static_cast<float*>(alloc((size_t) tile_cap * 64 * 128 * sizeof(float)));
        tile_idx_q_out = static_cast<float*>(alloc((size_t) tile_cap * 64 * 128 * sizeof(float)));
        tile_idx_w_raw = static_cast<float*>(alloc((size_t) tile_cap * 64 * sizeof(float)));
        tile_idx_hw = static_cast<float*>(alloc((size_t) tile_cap * 64 * sizeof(float)));
        tile_idx_angles = static_cast<float*>(alloc((size_t) tile_cap * 32 * sizeof(float)));
        tile_sel_ids = static_cast<int32_t*>(alloc((size_t) tile_cap * kernels::kSparseMqaTopK * sizeof(int32_t)));
        tile_sel_cnt = static_cast<int32_t*>(alloc((size_t) tile_cap * sizeof(int32_t)));
        tile_comp_row = static_cast<int32_t*>(alloc((size_t) tile_cap * sizeof(int32_t)));
        hca_ids = static_cast<int32_t*>(alloc((size_t) kExpertWindow * hca_max_visible * sizeof(int32_t)));
        chunk_pos = static_cast<int32_t*>(alloc((size_t) cap * sizeof(int32_t)));
        chunk_win_start = static_cast<int32_t*>(alloc((size_t) cap * sizeof(int32_t)));
        chunk_win_n = static_cast<int32_t*>(alloc((size_t) cap * sizeof(int32_t)));
        chunk_comp_limit = static_cast<int32_t*>(alloc((size_t) cap * sizeof(int32_t)));
        const bool bufs_ok = current && attended && coefficients && collapsed && normalized && hc_workspace &&
                             logits && route_ids && route_weights && shared && routed && fixed_rows &&
                             fixed_mask && alternate_mask && chunk_q_latent && chunk_kv_proj && chunk_kv_row &&
                             chunk_out_latent && tile_q_heads && tile_mqa_out && tile_comp_kv && tile_comp_score &&
                             tile_idx_kv && tile_idx_score && tile_idx_qr_bf16 && tile_idx_q_raw && tile_idx_q_out &&
                             tile_idx_w_raw && tile_idx_hw && tile_idx_angles && tile_sel_ids && tile_sel_cnt &&
                             tile_comp_row && hca_ids && chunk_pos && chunk_win_start && chunk_win_n &&
                             chunk_comp_limit;
        if (bufs_ok) {
            std::string ffn_err;
            prompt_ffn = make_prefill_ffn(model_.weights(), *tier, cap, ffn_err);
            if (prompt_ffn) break;
            cudaGetLastError();
            err_ = ffn_err;
        }
        if (model_.diag_rows() > 0 || cap <= 1024) {
            if (err_.empty()) err_ = "deepseek4: out of device memory for the chunked prompt workspaces";
            return false;
        }
        cap = std::max(1024, cap / 2);
    }

    {
        std::vector<int32_t> h_hca((size_t) kExpertWindow * hca_max_visible);
        for (int t = 0; t < kExpertWindow; ++t)
            for (int i = 0; i < hca_max_visible; ++i)
                h_hca[(size_t) t * hca_max_visible + i] = i;
        if (cudaMemcpyAsync(hca_ids, h_hca.data(), h_hca.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return fail_cuda("prompt HCA ids upload");
    }

    std::vector<float> host((size_t) cap * stream_row);
    std::vector<float> embedding;
    for (int offset = 0; offset < total_tokens;) {
        if (stop && stop()) {
            err_ = "deepseek4: prompt read was stopped";
            return false;
        }
        const int T = std::min(cap, total_tokens - offset);
        const int32_t* chunk_tokens = tokens + offset;
        const size_t state_bytes = (size_t) T * stream_row * sizeof(float);
        for (int t = 0; t < T; ++t) {
            const float* src = supplied_embeddings ? supplied_embeddings + (size_t) (offset + t) * n : nullptr;
            if (!src) {
                if (chunk_tokens[t] < 0 || chunk_tokens[t] >= geo_.n_vocab ||
                    !model_.weights().embedding_row(chunk_tokens[t], embedding, err_)) {
                    if (err_.empty()) err_ = "deepseek4: token id is outside the checkpoint vocabulary";
                    return false;
                }
                src = embedding.data();
            }
            for (int s = 0; s < 4; ++s)
                std::copy_n(src, n, host.data() + (size_t) t * stream_row + (size_t) s * n);
        }
        if (cudaMemcpyAsync(current, host.data(), state_bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return fail_cuda("prompt embedding upload");

        std::vector<int32_t> h_pos((size_t) T), h_wstart((size_t) T), h_wn((size_t) T), h_clim((size_t) T);
        for (int t = 0; t < T; ++t) {
            const int32_t p = (int32_t) (pos + offset + t);
            const int32_t s = (int32_t) std::max<int64_t>(0, (int64_t) p - kBootstrapWindow + 1);
            h_pos[(size_t) t] = p;
            h_wstart[(size_t) t] = s;
            h_wn[(size_t) t] = p - s + 1;
        }
        if (cudaMemcpyAsync(chunk_pos, h_pos.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(chunk_win_start, h_wstart.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(chunk_win_n, h_wn.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return fail_cuda("prompt attention positions upload");

        tier->begin_token(T);
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            AttentionLayerWeights attention;
            if (!model_.weights().attention_layer_weights(layer, attention, err_)) {
                err_ = "deepseek4: " + err_;
                return false;
            }
            AttentionLayerScratch attn_scratch;
            attn_scratch.hc_workspace = hc_workspace;
            attn_scratch.coefficients = coefficients;
            attn_scratch.collapsed = collapsed;
            attn_scratch.normalized = normalized;
            try {
                prepare_attention_layer(current, T, attention, attn_scratch, stream_);
            } catch (const std::exception& e) {
                err_ = std::string("deepseek4 prompt attention prep: ") + e.what();
                return false;
            }
            if (!prompt_ffn->dense_mmq(normalized, geo_.n_embd, attention.attention.q.wq_a_type,
                                       attention.attention.q.wq_a, geo_.n_embd, 1024, T,
                                       chunk_q_latent, 1024, stream_, err_)) return false;
            kernels::sparse_mqa_q_norm(chunk_q_latent, attention.attention.q.q_norm, 1024, geo_.rms_eps, T, stream_);
            if (!prompt_ffn->dense_mmq(normalized, geo_.n_embd, attention.attention.kv_type,
                                       attention.attention.kv, geo_.n_embd, 512, T,
                                       chunk_kv_proj, 512, stream_, err_)) return false;
            prepare_attention_kv(chunk_kv_proj, attention.attention.kv_norm, geo_.rms_eps, chunk_pos,
                                 ropes_[(size_t) layer].attention, T, chunk_kv_row, stream_);

            const int ratio = geo_.compress_ratio[(size_t) layer];
            const DeviceTensor *attn_ape = nullptr, *attn_norm = nullptr, *attn_gate = nullptr, *attn_kv = nullptr;
            const DeviceTensor *index_ape = nullptr, *index_norm = nullptr, *index_gate = nullptr, *index_kv = nullptr,
                               *index_q = nullptr, *index_proj = nullptr;
            if (ratio) {
                for (int t = 0; t < T; ++t) h_clim[(size_t) t] = (int32_t) ((pos + offset + t + 1) / ratio);
                if (cudaMemcpyAsync(chunk_comp_limit, h_clim.data(), (size_t) T * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                    return fail_cuda("prompt compressed limits upload");
                LayerCache& cache = layer_cache_[(size_t) layer];
                const int visible_end = (int) ((pos + offset + T) / ratio);
                const uint64_t keep_free = tier->free_floor();
                std::string cache_error;
                if (!cache.attention->ensure((uint64_t) visible_end * 512 * sizeof(float), keep_free, cache_error)) {
                    err_ = "deepseek4: cannot grow compressed cache: " + cache_error;
                    return false;
                }
                attn_ape = model_.weights().tensor(layer, "attn_compressor_ape.weight");
                attn_norm = model_.weights().tensor(layer, "attn_compressor_norm.weight");
                attn_gate = model_.weights().tensor(layer, "attn_compressor_gate.weight");
                attn_kv = model_.weights().tensor(layer, "attn_compressor_kv.weight");
                if (!attn_ape || !attn_norm || !attn_gate || !attn_kv) {
                    err_ = "deepseek4: compressed-attention layer is missing validated compressor tensors";
                    return false;
                }
                if (ratio == 4) {
                    if (!cache.indexer->ensure((uint64_t) visible_end * 128 * sizeof(float), keep_free, cache_error)) {
                        err_ = "deepseek4: cannot grow indexer cache: " + cache_error;
                        return false;
                    }
                    index_ape = model_.weights().tensor(layer, "indexer_compressor_ape.weight");
                    index_norm = model_.weights().tensor(layer, "indexer_compressor_norm.weight");
                    index_gate = model_.weights().tensor(layer, "indexer_compressor_gate.weight");
                    index_kv = model_.weights().tensor(layer, "indexer_compressor_kv.weight");
                    index_q = model_.weights().tensor(layer, "indexer.attn_q_b.weight");
                    index_proj = model_.weights().tensor(layer, "indexer.proj.weight");
                    if (!index_ape || !index_norm || !index_gate || !index_kv || !index_q || !index_proj) {
                        err_ = "deepseek4: CSA layer is missing validated indexer tensors";
                        return false;
                    }
                }
            }

            const size_t per_group_w = kernels::native_mmvq_weight_bytes(
                attention.attention.output_a_type, kernels::kSparseMqaGroupInput, kernels::kSparseMqaOutputRank);
            for (int row = 0; row < T; row += kAttnTile) {
                const int b = std::min(kAttnTile, T - row);
                if (!prompt_ffn->dense_mmq(chunk_q_latent + (size_t) row * 1024, 1024,
                                           attention.attention.q.wq_b_type, attention.attention.q.wq_b,
                                           1024, 64 * 512, b, tile_q_heads, 64 * 512, stream_, err_)) return false;
                kernels::sparse_mqa_q_head_norm_rope(tile_q_heads, chunk_pos + row, ropes_[(size_t) layer].attention,
                                                     attention.attention.q.head_rms_eps, b, stream_);

                const int tile_visible = ratio ? (int) ((pos + offset + row + b) / ratio) : 0;
                if (ratio) {
                    LayerCache& cache = layer_cache_[(size_t) layer];
                    if (ratio == 4) {
                        kernels::bf16_rows_dot(normalized + (size_t) row * n,
                                               static_cast<const uint16_t*>(index_kv->ptr), tile_idx_kv,
                                               b, 256, geo_.n_embd, stream_);
                        kernels::bf16_rows_dot(normalized + (size_t) row * n,
                                               static_cast<const uint16_t*>(index_gate->ptr), tile_idx_score,
                                               b, 256, geo_.n_embd, stream_);
                        indexer_compress_projected(b, tile_idx_kv, tile_idx_score,
                            static_cast<const float*>(index_ape->ptr), chunk_pos + row,
                            static_cast<const float*>(index_norm->ptr), geo_.rms_eps, compress_rope_freq_,
                            cache.indexer_state_kv, cache.indexer_state_score,
                            reinterpret_cast<float*>(cache.indexer->data()), tile_comp_row, stream_);
                        if (tile_visible > 0) {
                            indexer_round_bf16(chunk_q_latent + (size_t) row * 1024, b, tile_idx_qr_bf16, stream_);
                            if (!prompt_ffn->dense_mmq(tile_idx_qr_bf16, 1024, index_q->type, index_q->ptr,
                                                       1024, 64 * 128, b, tile_idx_q_raw, 64 * 128, stream_, err_))
                                return false;
                            std::vector<float> angles((size_t) b * 32);
                            for (int t = 0; t < b; ++t)
                                for (int i = 0; i < 32; ++i)
                                    angles[(size_t) t * 32 + i] =
                                        (float) (pos + offset + row + t) * ropes_[(size_t) layer].frequency[(size_t) i];
                            if (cudaMemcpyAsync(tile_idx_angles, angles.data(), angles.size() * sizeof(float),
                                                cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                                return fail_cuda("prompt indexer rope angles upload");
                            indexer_project_post(static_cast<const uint16_t*>(index_proj->ptr), tile_idx_q_raw,
                                normalized + (size_t) row * n, tile_idx_angles, b, tile_idx_w_raw,
                                tile_idx_q_out, tile_idx_hw, stream_);
                            indexer_select(tile_idx_q_out, tile_idx_hw,
                                reinterpret_cast<const float*>(cache.indexer->data()),
                                tile_visible, (int) (pos + offset + row), b, tile_sel_ids, tile_sel_cnt,
                                indexer_select_workspace_, indexer_select_workspace_bytes(tile_visible), stream_);
                        }
                    }
                    const CompressKind kind = ratio == 4 ? CompressKind::Csa : CompressKind::Hca;
                    project_and_compress(kind, ratio, geo_.n_embd, b, normalized + (size_t) row * n,
                        static_cast<const uint16_t*>(attn_kv->ptr), static_cast<const uint16_t*>(attn_gate->ptr),
                        static_cast<const float*>(attn_ape->ptr), chunk_pos + row,
                        static_cast<const float*>(attn_norm->ptr), geo_.rms_eps, compress_rope_freq_,
                        tile_comp_kv, tile_comp_score, cache.attention_state_kv, cache.attention_state_score,
                        reinterpret_cast<float*>(cache.attention->data()), tile_comp_row, stream_);
                }

                for (int sub = 0; sub < b; sub += kExpertWindow) {
                    const int count = std::min(kExpertWindow, b - sub);
                    const int g_row = row + sub;
                    const int64_t sub_pos = pos + offset + g_row;
                    kernels::SparseMqaArgs sel;
                    sel.n_tokens = count;
                    sel.win = window_rings_ + (size_t) layer * kBootstrapWindow * 512;
                    sel.win_type = kernels::MqaRowType::F32;
                    sel.win_cap = kBootstrapWindow;
                    sel.win_start = chunk_win_start + g_row;
                    sel.win_n = chunk_win_n + g_row;
                    sel.pos = chunk_pos + g_row;
                    sel.rope = ropes_[(size_t) layer].attention;
                    if (ratio) {
                        LayerCache& cache = layer_cache_[(size_t) layer];
                        const int visible = (int) ((sub_pos + count) / ratio);
                        if (ratio == 4 && visible > 0) {
                            sel.n_sel = kernels::kSparseMqaTopK;
                            sel.sel = tile_sel_ids + (size_t) sub * kernels::kSparseMqaTopK;
                            sel.comp = cache.attention->data();
                            sel.comp_type = kernels::MqaRowType::F32;
                            sel.comp_limit = chunk_comp_limit + g_row;
                        } else if (ratio != 4 && visible > 0) {
                            sel.n_sel = hca_max_visible;
                            sel.comp = cache.attention->data();
                            sel.comp_type = kernels::MqaRowType::F32;
                            sel.sel = hca_ids;
                            sel.comp_limit = chunk_comp_limit + g_row;
                        }
                    }
                    cache_attention_window(window_rings_ + (size_t) layer * kBootstrapWindow * 512,
                                           chunk_kv_row + (size_t) g_row * 512, chunk_pos + g_row, count, stream_);
                    sel.q = tile_q_heads + (size_t) sub * 64 * 512;
                    sel.sinks = attention.attention.sinks;
                    sel.round_output_bf16 = true;
                    sel.out = tile_mqa_out + (size_t) sub * 64 * 512;
                    kernels::sparse_mqa(sel, mqa_workspace_, stream_);
                }

                for (int g = 0; g < kernels::kSparseMqaOutputGroups; ++g) {
                    const auto* ww = static_cast<const uint8_t*>(attention.attention.output_a) + (size_t) g * per_group_w;
                    if (!prompt_ffn->dense_mmq(tile_mqa_out + (size_t) g * kernels::kSparseMqaGroupInput, 64 * 512,
                                               attention.attention.output_a_type, ww,
                                               kernels::kSparseMqaGroupInput, kernels::kSparseMqaOutputRank, b,
                                               chunk_out_latent + (size_t) row * 1024 + (size_t) g * kernels::kSparseMqaOutputRank,
                                               1024, stream_, err_)) return false;
                }
            }
            if (!prompt_ffn->dense_mmq(chunk_out_latent, 1024, attention.attention.output_b_type,
                                       attention.attention.output_b, 1024, geo_.n_embd, T,
                                       shared, geo_.n_embd, stream_, err_)) return false;
            try {
                finish_attention_layer(current, coefficients, shared, T, geo_.n_embd, attended, stream_);
            } catch (const std::exception& e) {
                err_ = std::string("deepseek4 prompt attention finish: ") + e.what();
                return false;
            }

            FfnRouteWeights route;
            SharedExpertWeights ignored_shared;
            if (!model_.weights().ffn_layer_weights(layer, route, ignored_shared, err_)) {
                err_ = "deepseek4: " + err_;
                return false;
            }
            RouteRows rows;
            if (!build_route_rows(chunk_tokens, T, geo_.n_vocab, layer, geo_.n_layers, geo_.hash_layers, rows, err_))
                return false;
            if (rows.fixed_rows.size() != (size_t) T || rows.fixed_mask.size() != (size_t) T ||
                rows.alternate_mask.size() != (size_t) T ||
                cudaMemcpyAsync(fixed_rows, rows.fixed_rows.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                cudaMemcpyAsync(fixed_mask, rows.fixed_mask.data(), (size_t) T, cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                cudaMemcpyAsync(alternate_mask, rows.alternate_mask.data(), (size_t) T, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
                err_ = "deepseek4: cannot upload chunked prompt route metadata";
                return false;
            }
            route.routing = router_mixed_args(geo_.n_expert, geo_.n_vocab,
                layer < geo_.hash_layers ? hash_routes_ + (size_t) layer * geo_.n_vocab * 6 : nullptr,
                fixed_rows, fixed_mask, alternate_mask, route.routing.bias, route.routing.alternate_bias);
            FfnRouteScratch scratch{hc_workspace, coefficients, collapsed, normalized, logits, route_ids, route_weights};
            try {
                prepare_ffn_routes(attended, T, route, scratch, stream_);
            } catch (const std::exception& e) {
                err_ = std::string("deepseek4 prompt FFN routes: ") + e.what();
                return false;
            }
            if (!prompt_ffn->shared(layer, T, normalized, shared, stream_, err_) ||
                !prompt_ffn->routed(layer, T, normalized, route_ids, route_weights, routed, stream_, err_)) return false;
            try {
                finish_ffn_layer(attended, coefficients, shared, routed, T, geo_.n_embd, current, stream_);
            } catch (const std::exception& e) {
                err_ = std::string("deepseek4 prompt FFN finish: ") + e.what();
                return false;
            }
            std::swap(current, attended);
        }

        if (offset + T == total_tokens) {
            HeadWeights head;
            if (!model_.weights().head_weights(head, err_)) {
                err_ = "deepseek4: " + err_;
                return false;
            }
            HeadScratch head_scratch{head_hc_workspace_, head_coef_, head_collapsed_, head_normed_};
            try {
                model_head_step(current + (size_t) (T - 1) * stream_row, 1, head, head_scratch, final_logits_, stream_);
            } catch (const std::exception& e) {
                err_ = std::string("deepseek4 prompt output head: ") + e.what();
                return false;
            }
        }
        if (!sync()) return false;
        tokens_.insert(tokens_.end(), chunk_tokens, chunk_tokens + T);
        offset += T;
        if (progress) progress(offset);
    }
    return true;
}

bool Session::routed_step(int layer, const float* normalized, int n_tokens, const FfnRouteWeights&,
                          const float* weights, float* output) {
    core::ExpertTier* tier = model_.tier();
    if (!tier || !normalized || !weights || !output || n_tokens < 1 || n_tokens > kExpertWindow) {
        err_ = "deepseek4: expert tier or routed-expert buffers are unavailable";
        return false;
    }

    const int n_ids = n_tokens * geo_.n_used;
    const uint32_t sequence = *h_expert_seq_;
    kernels::doorbell_publish(normalized, route_ids_, weights, (int64_t) n_tokens * geo_.n_embd, n_ids, d_expert_input_, d_expert_ids_,
                              nullptr, d_expert_seq_, stream_);

    // The single publish kernel writes both payloads before advancing the mapped sequence word.
    bool ids_ready = false;
    const volatile uint32_t* staged_sequence = h_expert_seq_;
    for (int spin = 0; spin < (1 << 22); ++spin) {
        if (*staged_sequence != sequence) {
            std::atomic_thread_fence(std::memory_order_acquire);
            ids_ready = true;
            break;
        }
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    if (!ids_ready && !sync()) return false;

    tier->pump(layer, 3);
    int32_t unique[kExpertWindowRoutes]{};
    core::ExpertTier::Use use[kExpertWindowRoutes]{};
    int n_unique = 0;
    for (int r = 0; r < n_ids; ++r) {
        int u = 0;
        while (u < n_unique && unique[u] != h_expert_ids_[r]) ++u;
        if (u == n_unique) unique[n_unique++] = h_expert_ids_[r];
        if (h_expert_ids_[r] >= 0) {
            core::ExpertTier::Use& row_use = use[u];
            const int nt = row_use.nt++;
            row_use.tok[nt] = r / geo_.n_used;
            row_use.row[nt] = r;
        }
    }
    core::ExpertTier::Lookup lookup;
    if (!tier->lookup(layer, unique, n_unique, lookup, err_)) {
        err_ = "deepseek4: " + err_;
        return false;
    }
    const int fetched = tier->fetch(layer, lookup);
    bool cpu = false, reading = false;
    for (int k = 0; k < lookup.k; ++k) {
        cpu = cpu || (!lookup.dev[k] && !lookup.peer[k] && !lookup.reading[k] && !lookup.dropped[k]);
        reading = reading || lookup.reading[k];
        if (lookup.dropped[k]) {
            for (int j = 0; j < use[k].nt; ++j) {
                const size_t row = (size_t) use[k].row[j];
                std::memset(h_expert_rows_ + row * geo_.n_embd, 0, (size_t) geo_.n_embd * sizeof(float));
            }
        }
    }
    if (cpu) tier->start_cpu_window(layer, h_expert_input_, lookup, use, h_expert_rows_);

    if (fetched) tier->await_fetch(stream_);
    ExpertPlan plan;
    int n_gpu = 0;
    int entries = 0;
    for (int k = 0; k < lookup.k; ++k) {
        if (!lookup.dev[k]) continue;
        plan.ptr[n_gpu] = (unsigned long long) lookup.dev[k];
        plan.start[n_gpu] = entries;
        for (int j = 0; j < use[k].nt; ++j) {
            plan.dst[entries] = use[k].row[j];
            plan.tok[entries] = use[k].tok[j];
            ++entries;
        }
        ++n_gpu;
    }
    plan.start[n_gpu] = entries;
    plan.groups[0] = n_gpu;
    if (n_gpu > 0) {
        const ExpertLayer& expert = model_.weights().pack().experts[(size_t) layer];
        const kernels::NativeExpertLayout layout = kernels::native_expert_layout(
            expert.gate_up_type, expert.down_type, geo_.n_embd, geo_.n_ff_exp, geo_.swiglu_limit);
        kernels::native_quantize_q8_1(normalized, expert_input_q8_, geo_.n_embd, n_tokens, stream_);
        ExpertPlanSingle single_plan;
        const void* host_plan = &plan;
        size_t plan_bytes = sizeof(plan);
        size_t ptr_offset = offsetof(ExpertPlan, ptr);
        size_t start_offset = offsetof(ExpertPlan, start);
        size_t groups_offset = offsetof(ExpertPlan, groups);
        size_t dst_offset = offsetof(ExpertPlan, dst);
        size_t tok_offset = offsetof(ExpertPlan, tok);
        if (n_tokens == 1) {
            std::copy_n(plan.ptr, n_gpu, single_plan.ptr);
            std::copy_n(plan.start, n_gpu + 1, single_plan.start);
            single_plan.groups[0] = n_gpu;
            std::copy_n(plan.dst, entries, single_plan.dst);
            std::copy_n(plan.tok, entries, single_plan.tok);
            host_plan = &single_plan;
            plan_bytes = sizeof(single_plan);
            ptr_offset = offsetof(ExpertPlanSingle, ptr);
            start_offset = offsetof(ExpertPlanSingle, start);
            groups_offset = offsetof(ExpertPlanSingle, groups);
            dst_offset = offsetof(ExpertPlanSingle, dst);
            tok_offset = offsetof(ExpertPlanSingle, tok);
        }
        if (cudaMemcpyAsync(expert_plan_, host_plan, plan_bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return fail_cuda("expert plan upload");
        const auto* device_plan = static_cast<const uint8_t*>(expert_plan_);
        kernels::native_expert_grouped_clamped(layout, geo_.swiglu_limit,
            reinterpret_cast<const unsigned long long*>(device_plan + ptr_offset),
            reinterpret_cast<const int32_t*>(device_plan + start_offset),
            reinterpret_cast<const int32_t*>(device_plan + groups_offset),
            reinterpret_cast<const int32_t*>(device_plan + dst_offset),
            reinterpret_cast<const int32_t*>(device_plan + tok_offset), n_gpu, entries,
            expert_input_q8_, expert_scratch_, d_expert_rows_, stream_);
    }

    if (cpu) tier->finish_cpu();
    if (reading) {
        if (!tier->await_reads(err_)) { err_ = "deepseek4: " + err_; return false; }
        tier->start_cpu_window(layer, h_expert_input_, lookup, use, h_expert_rows_, true);
        tier->finish_cpu();
    }
    // GPU expert kernels and the weighted sum share this stream; queue order provides their dependency.
    weighted_route_sum(d_expert_rows_, weights, n_tokens, geo_.n_embd, output, stream_);
    tier->promote(layer, lookup);
    tier->mark_gap(stream_);
    return true;
}

bool Session::fail_cuda(const char* operation) {
    err_ = std::string(operation) + ": " + cudaGetErrorString(cudaGetLastError());
    return false;
}

}  // namespace

std::unique_ptr<model::ModelSession> make_session_impl(DsModel& model, std::string& err) {
    auto session = std::make_unique<Session>(model);
    if (!session->init(err)) return nullptr;
    return session;
}

}  // namespace basalt::models::deepseek4
