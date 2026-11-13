// src/models/qwen35/adapter.cpp - the Qwen3.5-family dense hybrid (Qwen3.8-27B): registers the architecture, reads its
// pack into the shared dense stack and hands out its sessions.
#include "models/qwen35/adapter.hpp"

#include "basalt/core/load_progress.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/program/qwen_forms.hpp"
#include "models/qwen35/meta.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <map>
#include <optional>

namespace basalt::models::qwen35 {
namespace {

constexpr int kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBf16 = 30, kTypeNvfp4 = 40;

class Loaded final : public model::LoadedModel {
public:
    ~Loaded() override {
        model_.reset();
        for (void* p : dev_) cudaFree(p);
        if (embd_) cudaFreeHost(embd_);
        if (fd_ >= 0) platform::close_fd(fd_);
    }

    const model::ModelDesc& desc() const override { return desc_; }

    model::Result make_session(std::unique_ptr<model::ModelSession>& out) override {
        std::string err;
        out = dense::make_session(*model_, err);
        if (!out) return {model::Status::Error, err};
        return {};
    }

    bool load(const model::LoadRequest& req, std::string& err) {
        const std::string& path = req.paths.front();
        const GgufFile gg(path);
        desc_ = describe_meta(gg);
        dense::Geometry g = geometry(gg, err);
        if (!err.empty()) return false;
        for (const TensorInfo& t : gg.tensors()) tensors_[t.name] = t;
        fd_ = platform::open_fd_utf8(path);
        if (fd_ < 0) {
            err = "cannot open " + path;
            return false;
        }
        uint64_t total = 0;
        for (const TensorInfo& t : gg.tensors())
            if (!t.name.starts_with("mtp.")) total += tensor_payload_bytes(t);
        core::LoadProgress::announce_total(total);
        core::LoadProgress::plan(core::LoadProgress::cuda(0), total);
        core::LoadProgress progress(core::LoadProgress::cuda(0), total);
        progress_ = &progress;
        dense::Weights w;
        const int H = g.n_embd;
        // the token embedding stays in page-locked host memory; the kernels read the rows they need
        const TensorInfo* te = need(gg, "token_embd.weight", kTypeBf16, (int64_t) H * g.n_vocab, err);
        if (!te) return false;
        if (cudaHostAlloc(&embd_, tensor_payload_bytes(*te), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "out of page-locked memory for the token embedding";
            return false;
        }
        if (!platform::pread_all(fd_, embd_, tensor_payload_bytes(*te), gg.data_start() + te->offset)) {
            err = "short read of token_embd.weight";
            return false;
        }
        progress.add(tensor_payload_bytes(*te));
        void* dev_embd = nullptr;
        cudaHostGetDevicePointer(&dev_embd, embd_, 0);
        w.embd = (const uint16_t*) dev_embd;
        w.out_norm = floats(gg, "output_norm.weight", H, err);
        if (!w.out_norm || !linear(gg, "output", g.n_vocab, H, w.head, err)) return false;
        const int C = g.conv_channels(), VD = g.value_width(), HV = g.gdn_v_heads, AQ = g.n_head * g.head_dim;
        const int KV = g.n_kv * g.head_dim;
        for (int il = 0; il < g.n_layers; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            dense::Layer L;
            L.kind = g.kinds[(size_t) il];
            L.attn_norm = floats(gg, p + "attn_norm.weight", H, err);
            L.ffn_norm = floats(gg, p + "post_attention_norm.weight", H, err);
            if (!L.attn_norm || !L.ffn_norm) return false;
            if (L.kind == dense::Mixer::Gdn) {
                if (!linear(gg, p + "attn_qkv", C, H, L.qkv, err) || !linear(gg, p + "attn_gate", VD, H, L.z, err) ||
                    !linear(gg, p + "ssm_out", H, VD, L.out, err))
                    return false;
                L.w_ab = bf16_pair(gg, p + "ssm_alpha.weight", p + "ssm_beta.weight", (int64_t) HV * H, err);
                L.dt = floats(gg, p + "ssm_dt.bias", HV, err);
                L.ssm_a = floats(gg, p + "ssm_a", HV, err);
                L.conv = floats(gg, p + "ssm_conv1d.weight", (int64_t) C * g.gdn_conv, err);
                L.gamma = floats(gg, p + "ssm_norm.weight", g.gdn_state, err);
                if (!L.w_ab || !L.dt || !L.ssm_a || !L.conv || !L.gamma) return false;
            } else {
                if (!linear(gg, p + "attn_q", 2 * AQ, H, L.q, err) || !linear(gg, p + "attn_k", KV, H, L.k, err) ||
                    !linear(gg, p + "attn_v", KV, H, L.v, err) || !linear(gg, p + "attn_output", H, AQ, L.o, err))
                    return false;
                L.q_norm = floats(gg, p + "attn_q_norm.weight", g.head_dim, err);
                L.k_norm = floats(gg, p + "attn_k_norm.weight", g.head_dim, err);
                if (!L.q_norm || !L.k_norm) return false;
            }
            if (!linear(gg, p + "ffn_gate", g.n_ff, H, L.gate, err) || !linear(gg, p + "ffn_up", g.n_ff, H, L.up, err) ||
                !linear(gg, p + "ffn_down", H, g.n_ff, L.down, err))
                return false;
            w.layers.push_back(L);
        }
        if (req.draft && tensors_.count("mtp.fc.weight")) {   // the pack's drafter layer, for --spec
            dense::Mtp& d = w.mtp;
            dense::Layer& L = d.layer;
            L.kind = dense::Mixer::Attention;
            d.norm_e = floats(gg, "mtp.pre_fc_norm_embedding.weight", H, err);
            d.norm_h = floats(gg, "mtp.pre_fc_norm_hidden.weight", H, err);
            d.norm = floats(gg, "mtp.norm.weight", H, err);
            L.attn_norm = floats(gg, "mtp.attn_norm.weight", H, err);
            L.ffn_norm = floats(gg, "mtp.post_attention_norm.weight", H, err);
            L.q_norm = floats(gg, "mtp.attn_q_norm.weight", g.head_dim, err);
            L.k_norm = floats(gg, "mtp.attn_k_norm.weight", g.head_dim, err);
            if (!d.norm_e || !d.norm_h || !d.norm || !L.attn_norm || !L.ffn_norm || !L.q_norm || !L.k_norm ||
                !linear(gg, "mtp.fc", H, 2 * H, d.fc, err) || !linear(gg, "mtp.attn_q", 2 * AQ, H, L.q, err) ||
                !linear(gg, "mtp.attn_k", KV, H, L.k, err) || !linear(gg, "mtp.attn_v", KV, H, L.v, err) ||
                !linear(gg, "mtp.attn_output", H, AQ, L.o, err) || !linear(gg, "mtp.ffn_gate", g.n_ff, H, L.gate, err) ||
                !linear(gg, "mtp.ffn_up", g.n_ff, H, L.up, err) || !linear(gg, "mtp.ffn_down", H, g.n_ff, L.down, err))
                return false;
            d.present = true;
        }
        progress.finish();
        progress_ = nullptr;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        basalt::log::info("basalt qwen35", "%zu tensors, %.2f GiB on the GPU, %.2f GiB of embedding in page-locked RAM; "
                          "%.2f GiB free", dev_.size(), (double) dev_bytes_ / 1073741824.0, (double) tensor_payload_bytes(*te) / 1073741824.0,
                          (double) free_b / 1073741824.0);
        dense::Options o;
        o.max_context = req.max_context;
        o.sessions = req.sessions;
        o.spec = req.spec;
        o.draft = req.draft;
        o.vision = req.vision;
        o.arch = "qwen35";
        if (o.vision) {
            core::ModelFile file;
            core::ModelSources srcs;
            if (!file.open(path, err)) return false;
            core::model_sources(file, srcs);
            o.mmproj = srcs.vision;
        }
        if (g.context_length > 0 && o.max_context > g.context_length) {
            err = "qwen35: --max-context " + std::to_string((long long) o.max_context) + " is beyond the model's " +
                  std::to_string((long long) g.context_length);
            return false;
        }
        const uint64_t need_b = dense::shared_bytes(g, o.max_context) +
                                (uint64_t) o.sessions * dense::session_bytes(g, o.max_context, w.mtp.present && o.draft);
        const uint64_t reserve = (uint64_t) std::max(req.vram_reserve_mib, 0) << 20;
        if (need_b + reserve + (512ull << 20) > free_b) {
            err = "qwen35: " + std::to_string((long long) o.sessions) + " session(s) of " + std::to_string((long long) o.max_context) +
                  " cells need " + std::to_string((unsigned long long) (need_b >> 20)) + " MiB, " +
                  std::to_string((unsigned long long) (free_b >> 20)) + " MiB are free; lower --max-context or --parallel";
            return false;
        }
        model_ = dense::make_model(g, w, o, err);
        return model_ != nullptr;
    }

private:
    const TensorInfo* need(const GgufFile& gg, const std::string& name, int type, int64_t elems, std::string& err) {
        (void) gg;
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            err = "qwen35: the pack has no " + name;
            return nullptr;
        }
        const TensorInfo* t = &it->second;
        if ((int) t->type != type || (int64_t) t->elements() != elems) {
            err = "qwen35: " + name + " is not the expected type and size";
            return nullptr;
        }
        return t;
    }

    void* upload(const GgufFile& gg, const TensorInfo& t, std::string& err) {
        void* d = nullptr;
        if (cudaMalloc(&d, tensor_payload_bytes(t)) != cudaSuccess) {
            err = "qwen35: out of device memory loading " + t.name;
            return nullptr;
        }
        dev_.push_back(d);
        constexpr uint64_t kChunk = 64ull << 20;
        staging_.resize((size_t) std::min<uint64_t>(tensor_payload_bytes(t), kChunk));
        for (uint64_t off = 0; off < tensor_payload_bytes(t); off += kChunk) {
            const uint64_t n = std::min<uint64_t>(kChunk, tensor_payload_bytes(t) - off);
            if (!platform::pread_all(fd_, staging_.data(), n, gg.data_start() + t.offset + off) ||
                cudaMemcpy((char*) d + off, staging_.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "qwen35: reading " + t.name + " failed";
                return nullptr;
            }
            if (progress_) progress_->add(n);
        }
        dev_bytes_ += tensor_payload_bytes(t);
        return d;
    }

    const float* floats(const GgufFile& gg, const std::string& name, int64_t elems, std::string& err) {
        const TensorInfo* t = need(gg, name, kTypeF32, elems, err);
        return t ? (const float*) upload(gg, *t, err) : nullptr;
    }

    // two BF16 tensors in one buffer, the first then the second
    const uint16_t* bf16_pair(const GgufFile& gg, const std::string& a, const std::string& b, int64_t elems, std::string& err) {
        const TensorInfo* ta = need(gg, a, kTypeBf16, elems, err);
        const TensorInfo* tb = ta ? need(gg, b, kTypeBf16, elems, err) : nullptr;
        if (!tb) return nullptr;
        void* d = nullptr;
        if (cudaMalloc(&d, tensor_payload_bytes(*ta) + tensor_payload_bytes(*tb)) != cudaSuccess) {
            err = "qwen35: out of device memory loading " + a;
            return nullptr;
        }
        dev_.push_back(d);
        uint64_t at = 0;
        for (const TensorInfo* t : {ta, tb}) {
            staging_.resize((size_t) tensor_payload_bytes(*t));
            if (!platform::pread_all(fd_, staging_.data(), tensor_payload_bytes(*t), gg.data_start() + t->offset) ||
                cudaMemcpy((char*) d + at, staging_.data(), tensor_payload_bytes(*t), cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "qwen35: reading " + t->name + " failed";
                return nullptr;
            }
            at += tensor_payload_bytes(*t);
            if (progress_) progress_->add(tensor_payload_bytes(*t));
        }
        dev_bytes_ += at;
        return (const uint16_t*) d;
    }

    // `<base>.weight` (Q8_0 or NVFP4, rows x cols) and, for NVFP4, its tensor scale `<base>.scale`
    bool linear(const GgufFile& gg, const std::string& base, int64_t rows, int64_t cols, dense::Linear& l, std::string& err) {
        auto it = tensors_.find(base + ".weight");
        if (it == tensors_.end()) {
            err = "qwen35: the pack has no " + base + ".weight";
            return false;
        }
        const int type = (int) it->second.type;
        if (type != kTypeQ8_0 && type != kTypeNvfp4) {
            err = "qwen35: " + base + ".weight is " + it->second.type_name() + "; the dense kernels read Q8_0 and NVFP4";
            return false;
        }
        const TensorInfo* t = need(gg, base + ".weight", type, rows * cols, err);
        if (!t || (int64_t) t->shape[0] != cols) {
            if (err.empty()) err = "qwen35: " + base + ".weight has the wrong shape";
            return false;
        }
        l.type = type;
        l.rows = rows;
        l.cols = cols;
        l.w = upload(gg, *t, err);
        if (!l.w) return false;
        if (type == kTypeNvfp4) {
            l.scale = floats(gg, base + ".scale", 1, err);
            if (!l.scale) return false;
        }
        return true;
    }

    model::ModelDesc desc_;
    dense::ModelPtr model_;
    std::map<std::string, TensorInfo> tensors_;
    std::vector<void*> dev_;
    uint64_t dev_bytes_ = 0;
    void* embd_ = nullptr;
    int fd_ = -1;
    std::vector<uint8_t> staging_;
    core::LoadProgress* progress_ = nullptr;
};

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "qwen35"; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        if (req.paths.empty()) return {model::Status::Error, "qwen35: no model path"};
        auto m = std::make_unique<Loaded>();
        std::string err;
        if (!m->load(req, err)) return {model::Status::Error, err};
        out = std::move(m);
        return {};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "qwen"}; }
    PreTokenizer pretokenizer() const override { return {"qwen35", program::tokenizer::pretok_qwen35}; }
    model::TokenNames token_names() const override { return program::qwen_token_names(); }
    program::grammar::ToolForm tool_form() const override { return program::qwen_tool_form(); }
    int max_expert_cards() const override { return 0; }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        (void) code;
        if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
        else return false;
        return true;
    }

    // the pack's drafter runs in windows of 8 unless --spec says otherwise (0 or 1: plain decode)
    int planned_spec(const program::CommonOptions& o) const override {
        if (o.spec_given) return o.spec >= 2 ? o.spec : 0;
        return has_drafter(o.model) ? kDefaultSpec : 0;
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (o.spec_given && (o.spec < 0 || o.spec > 8)) {
            basalt::log::error("basalt qwen35", "--spec %d: 0 (off) to 8", o.spec);
            return 2;
        }
        if (!o.spec_given) o.spec = has_drafter(o.model) ? kDefaultSpec : 0;
        if (o.spec >= 2 && !has_drafter(o.model)) {
            basalt::log::warn("basalt qwen35", "--spec %d off: the pack has no drafter layer", o.spec);
            o.spec = 0;
        }
        return std::nullopt;
    }

private:
    static constexpr int kDefaultSpec = 8;
    static bool has_drafter(const std::string& path) {
        core::ModelFile file;
        std::string err;
        if (path.empty() || !file.open(path, err) || !file.load_meta(err)) return false;
        const core::Json* t = file.meta().get("tensors");
        return t != nullptr && t->get("mtp.fc.weight") != nullptr;
    }
    const char* model_name() const override { return "Qwen3.8-27B"; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::qwen35
