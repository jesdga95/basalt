// src/models/deepseek41/model.cpp - loads a DeepSeek-V4.1 pack onto the GPU.
#include "basalt/log.hpp"
#include "models/deepseek41/model.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/platform/file.hpp"
#include "models/deepseek41/context_policy.hpp"
#include "models/deepseek41/session.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

namespace basalt::models::deepseek41 {

DsModel::~DsModel() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (embd_fd_ >= 0) platform::close_fd(embd_fd_);
    if (experts_fd_ >= 0) platform::close_fd(experts_fd_);
}

bool DsModel::load(const std::string& path, std::string& err) {
    if (!open_pack(path, pack_, err)) return false;
    path_ = path;
    const Geometry& g = pack_.geo;
    desc_.arch = "deepseek41";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.n_expert_shared = g.n_shared;
    desc_.context_length = g.context_length;
    for (const ExpertLayer& e : pack_.experts) {
        desc_.expert_gate_up_type.push_back(e.gu_type);
        desc_.expert_down_type.push_back(e.d_type);
    }

    embd_fd_ = platform::open_fd_utf8(path);
    experts_fd_ = platform::open_fd_utf8(path);
    if (embd_fd_ < 0 || experts_fd_ < 0) {
        err = "cannot open " + path;
        return false;
    }

    // dense tensors to the device; the token embedding and the Engram table stay in the file
    std::vector<uint8_t> staging;
    uint64_t total = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") continue;
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, t.bytes) != cudaSuccess) {
            err = "out of device memory loading " + name;
            return false;
        }
        constexpr uint64_t kChunk = 64ull << 20;
        staging.resize((size_t) std::min<uint64_t>(t.bytes, kChunk));
        for (uint64_t off = 0; off < t.bytes; off += kChunk) {
            const uint64_t n = std::min<uint64_t>(kChunk, t.bytes - off);
            if (!platform::pread_all(embd_fd_, staging.data(), n, t.offset + off)) {
                err = "short read of " + name;
                cudaFree(d.ptr);
                return false;
            }
            if (cudaMemcpy((char*) d.ptr + off, staging.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "upload of " + name + " failed";
                cudaFree(d.ptr);
                return false;
            }
        }
        total += t.bytes;
        dev_.emplace(name, std::move(d));
    }
    basalt::log::info("basalt deepseek41", "%zu dense tensors, %.2f GiB on the GPU", dev_.size(),
                 (double) total / (1024.0 * 1024.0 * 1024.0));

    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        if (!engram_->open(path, e.file_offset, e.rows, 64, 1 << 16, err, e.row_bytes)) return false;
    }
    return true;
}

int64_t DsModel::ctxp_default() { return ctxp::kLegacyContext; }

bool DsModel::resolve_context(int64_t requested, bool set, std::string& err) {
    const Geometry& g = pack_.geo;
    std::vector<int> ratios;
    for (int il = 0; il < g.n_layers; ++il)
        if (ctxp::owns_cache(il, g.compress_ratio[(size_t) il])) ratios.push_back(g.compress_ratio[(size_t) il]);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        err = "deepseek41: cannot query the free device memory";
        return false;
    }
    const ctxp::Plan p = ctxp::plan_context(requested, set, g.context_length, free_b, ratios);
    if (!p.error.empty()) {
        err = "deepseek41: " + p.error;
        return false;
    }
    ctx_ = p.context;
    basalt::log::info("basalt deepseek41", "context %lld tokens%s (caches of a full context: %.2f GiB)%s%s", (long long) ctx_,
                 set ? "" : " (the model's maximum within the free memory)",
                 (double) ctxp::context_bytes(ctx_, ratios) / (1024.0 * 1024.0 * 1024.0), p.note.empty() ? "" : "; ",
                 p.note.c_str());
    return true;
}

const DevTensor* DsModel::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

bool DsModel::embedding_row(int32_t token, float* out, std::string& err) const {
    const TensorRef* t = pack_.find("token_embd.weight");
    if (!t || (t->type != 14 && t->type != 8) || t->shape.size() != 2) {
        err = "token_embd.weight is missing or not Q6_K or Q8_0";
        return false;
    }
    const uint64_t n_embd = t->shape[0];
    if (token < 0 || (uint64_t) token >= t->shape[1]) {
        err = "token id out of range";
        return false;
    }
    const bool q8 = t->type == 8;
    const uint64_t row_bytes = q8 ? n_embd / 32 * 34 : n_embd / 256 * 210;
    std::vector<uint8_t> row(row_bytes);
    if (!platform::pread_all(embd_fd_, row.data(), row_bytes, t->offset + (uint64_t) token * row_bytes)) {
        err = "cannot read the embedding row";
        return false;
    }
    if (q8) {
        for (uint64_t b = 0; b < n_embd / 32; ++b) dequantize_q8_0(row.data() + b * 34, out + b * 32);
    } else {
        for (uint64_t b = 0; b < n_embd / 256; ++b) dequantize_q6_K(row.data() + b * 210, out + b * 256);
    }
    return true;
}

bool DsModel::open_mmproj(vision::Mmproj& mp, std::string& err) const {
    if (!diag_mmproj_.empty()) return mp.open(diag_mmproj_, err, "deepseek41v");
    if (pack_.vision.empty() || pack_.vision.bytes == 0) {
        err = "this pack carries no vision projector (convert with --mmproj)";
        return false;
    }
    return mp.open_range(pack_.vision.path, pack_.vision.offset, pack_.vision.bytes, err, "deepseek41v");
}

model::Result DsModel::make_session(std::unique_ptr<model::ModelSession>& out) {
    std::string err;
    out = make_session_impl(*this, err);
    if (!out) return {model::Status::Error, err};
    return {};
}

}  // namespace basalt::models::deepseek41
