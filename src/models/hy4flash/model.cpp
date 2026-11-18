// src/models/hy4flash/model.cpp - loads a Hy4-flash pack: the dense tensors to the main card, the embedding to host memory,
// the routed experts to RAM.
#include "models/hy4flash/model.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <thread>

namespace basalt::models::hy4flash {
namespace {
constexpr uint64_t kGiB = 1ull << 30;
constexpr int kTypeQ8_0 = 8;
}  // namespace

Hy4Model::Hy4Model() = default;

Hy4Model::~Hy4Model() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (fd_ >= 0) platform::close_fd(fd_);
}

const DevTensor* Hy4Model::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

bool Hy4Model::embedding_row(int32_t token, float* out) const {
    const uint64_t n = (uint64_t) geo().n_embd;
    if (token < 0 || (int64_t) token >= geo().n_vocab) return false;
    const uint8_t* row = embd_.data() + (uint64_t) token * (n / 32) * 34;
    for (uint64_t b = 0; b < n / 32; ++b) dequantize_q8_0(row + b * 34, out + b * 32);
    return true;
}

ExpertBlob Hy4Model::expert(int layer, int e) const {
    const ExpertLayer& L = pack_.experts[(size_t) layer];
    return {ram_[(size_t) ram_index_[(size_t) layer]].data() + (uint64_t) e * L.blob_bytes, L.gu_type, L.d_type};
}

model::Result Hy4Model::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                             std::vector<std::unique_ptr<model::ModelSession>>&) {
    if (req.paths.empty()) return {model::Status::Error, "hy4flash: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "hy4flash: " + err};
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    desc_.arch = "hy4flash";
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
    ctx_ = req.max_context > 0 ? std::min<int64_t>(req.max_context, kMaxContext) : kDefaultContext;
    if (req.max_context > kMaxContext)
        basalt::log::warn("basalt hy4flash", "context %lld capped at %lld", (long long) req.max_context, (long long) kMaxContext);
    diag_dump_ = req.diag_dump;
    if (!diag_dump_.empty()) basalt::log::info("basalt hy4flash", "--diag-dump: each layer's streams to %s", diag_dump_.c_str());

    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "hy4flash: cannot open " + path};
    const int file = reads.file_fd(fd_);

    uint64_t dense = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight") {
            if (t.type != kTypeQ8_0 || t.shape.size() != 2 || t.shape[0] != (uint64_t) g.n_embd) {
                return {model::Status::Error, "hy4flash: token_embd.weight is not Q8_0 [n_embd][vocab]"};
            }
            embd_.resize(t.bytes);
            core::LoadItem it;
            it.file = file;
            it.off = t.offset;
            it.bytes = t.bytes;
            it.host = embd_.data();
            reads.add(it);
            continue;
        }
        if (name == "per_layer_token_embd.weight") continue;
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) return {model::Status::Error, "hy4flash: out of device memory loading " + name};
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        reads.add(it);
        dense += d.bytes;
        dev_.emplace(name, std::move(d));
    }
    basalt::log::info("basalt hy4flash", "%zu dense tensors, %.2f GiB on the GPU", dev_.size(), (double) dense / (double) kGiB);

    // every routed expert in RAM, one allocation per layer
    const core::ModelFile::Section* xs = pack_.file.find("experts");
    if (!xs) return {model::Status::Error, "hy4flash: the pack has no experts section"};
    ram_.resize(pack_.experts.size());
    ram_index_.assign(pack_.experts.size(), -1);
    uint64_t total = 0;
    for (size_t l = 0; l < pack_.experts.size(); ++l) {
        const ExpertLayer& L = pack_.experts[l];
        if (!L.present) continue;
        const uint64_t bytes = L.blob_bytes * (uint64_t) g.n_expert;
        if (!ram_[l].allocate(bytes, false)) return {model::Status::Error, "hy4flash: no RAM for the experts of layer " + std::to_string(l)};
        ram_index_[l] = (int) l;
        core::LoadItem it;
        it.file = file;
        it.off = xs->offset + L.offset;
        it.bytes = bytes;
        it.host = ram_[l].data();
        reads.add(it);
        total += bytes;
    }
    basalt::log::info("basalt hy4flash", "RAM tier: all %d experts of %d layers (%.1f GiB), run by the CPU", g.n_expert,
                      g.n_layers - g.n_dense, (double) total / (double) kGiB);

    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        if (!engram_->open(path, e.file_offset, e.rows, 64, 1 << 16, err, e.row_bytes))
            return {model::Status::Error, "hy4flash: " + err};
    }
    const int threads = req.pool_workers > 0 ? req.pool_workers : (int) std::max(2u, std::thread::hardware_concurrency() / 2);
    workers_ = std::make_unique<Workers>(threads);
    return {};
}

model::Result Hy4Model::attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    std::string err;
    for (int i = 0; i < std::max(req.sessions, 1); ++i) {
        std::unique_ptr<model::ModelSession> s = make_session(*this, err);
        if (!s) return {model::Status::Error, "hy4flash: " + err};
        sessions.push_back(std::move(s));
    }
    return {};
}

}  // namespace basalt::models::hy4flash
