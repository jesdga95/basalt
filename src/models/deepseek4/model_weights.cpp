#include "models/deepseek4/model_weights.hpp"

#include "basalt/platform/memory.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "models/deepseek4/expert_tier_spec.hpp"

#include <cuda_runtime.h>

#include <limits>

namespace basalt::models::deepseek4 {

ModelWeights::~ModelWeights() { reset(); }

void ModelWeights::reset() {
    int previous_device = -1;
    const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    const bool selected = device_ < 0 || cudaSetDevice(device_) == cudaSuccess;
    for (auto& [name, tensor] : tensors_) {
        (void) name;
        if (tensor.ptr) cudaFree(tensor.ptr);
    }
    if (have_previous && selected && previous_device != device_) cudaSetDevice(previous_device);
    tensors_.clear();
    if (fd_ >= 0) platform::close_fd(fd_);
    fd_ = -1;
    device_ = -1;
    path_.clear();
    pack_ = Pack{};
}

bool ModelWeights::plan(const std::string& path, int device, core::LoadPlan& reads, std::string& err) {
    if (path.empty() || device < 0 || !tensors_.empty() || fd_ >= 0) {
        err = "deepseek4 weights: invalid path, device, load plan, or already-planned owner";
        return false;
    }
    Pack candidate;
    if (!open_pack(path, candidate, err)) return false;

    const Geometry& g = candidate.geo;
    for (int layer = 0; layer < g.n_layers; ++layer) {
        const ExpertLayer& expert = candidate.experts[(size_t) layer];
        if (!kernels::native_expert_supported(expert.gate_up_type, expert.down_type, g.n_embd, g.n_ff_exp)) {
            err = "deepseek4 layer " + std::to_string(layer) + ": no grouped expert kernel for gate/up type " +
                  std::to_string(expert.gate_up_type) + " and down type " + std::to_string(expert.down_type);
            return false;
        }
    }

    const int fd = platform::open_fd_utf8(path);
    if (fd < 0) {
        err = "deepseek4 weights: cannot open " + path;
        return false;
    }

    std::map<std::string, DeviceTensor> planned;
    core::LoadPlan additions;
    const int file = additions.file_fd(fd);
    const auto release_planned = [&] {
        int previous_device = -1;
        const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        const bool selected = cudaSetDevice(device) == cudaSuccess;
        for (auto& [planned_name, item] : planned) {
            (void) planned_name;
            if (item.ptr) cudaFree(item.ptr);
        }
        if (have_previous && selected && previous_device != device) cudaSetDevice(previous_device);
    };
    constexpr char kHashSuffix[] = "ffn_gate_tid2eid.weight";
    for (const auto& [name, source] : candidate.tensors) {
        if (name == "token_embd.weight" ||
            (name.size() >= sizeof(kHashSuffix) - 1 &&
             name.compare(name.size() - (sizeof(kHashSuffix) - 1), sizeof(kHashSuffix) - 1, kHashSuffix) == 0))
            continue;

        DeviceTensor tensor;
        tensor.type = source.type;
        tensor.bytes = source.bytes;
        tensor.shape = source.shape;
        const bool widen_router = source.type == 30 && name.find("ffn_gate_inp.weight") != std::string::npos;
        if (widen_router) {
            if (source.bytes > std::numeric_limits<uint64_t>::max() / 2) {
                err = "deepseek4 weights: tensor size overflows while widening " + name;
                release_planned();
                platform::close_fd(fd);
                return false;
            }
            tensor.type = 0;
            tensor.bytes = source.bytes * 2;
        }

        int previous_device = 0;
        if (cudaGetDevice(&previous_device) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
            err = "deepseek4 weights: cannot select CUDA device " + std::to_string(device);
            release_planned();
            platform::close_fd(fd);
            return false;
        }
        const cudaError_t allocation = cudaMalloc(&tensor.ptr, (size_t) tensor.bytes);
        const cudaError_t restore = cudaSetDevice(previous_device);
        if (allocation != cudaSuccess || restore != cudaSuccess) {
            if (allocation == cudaSuccess) {
                cudaSetDevice(device);
                cudaFree(tensor.ptr);
                cudaSetDevice(previous_device);
            }
            err = "deepseek4 weights: CUDA allocation failed for " + name;
            release_planned();
            platform::close_fd(fd);
            return false;
        }

        core::LoadItem read;
        read.file = file;
        read.off = source.offset;
        read.bytes = source.bytes;
        read.dev = tensor.ptr;
        read.device = device;
        read.conv = widen_router ? core::Conv::Bf16ToF32 : core::Conv::Copy;
        additions.add(read);
        planned.emplace(name, std::move(tensor));
    }

    reads.append(additions);
    pack_ = std::move(candidate);
    tensors_ = std::move(planned);
    path_ = path;
    fd_ = fd;
    device_ = device;
    err.clear();
    basalt::log::info("basalt deepseek4", "%zu dense tensors queued from %s", tensors_.size(), path_.c_str());
    return true;
}

const DeviceTensor* ModelWeights::tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const DeviceTensor* ModelWeights::tensor(int layer, const char* suffix) const {
    return tensor("blk." + std::to_string(layer) + "." + suffix);
}

bool ModelWeights::attention_layer_weights(int layer, AttentionLayerWeights& weights, std::string& err) const {
    const Geometry& g = pack_.geo;
    if (layer < 0 || layer >= g.n_layers || device_ < 0) {
        err = "deepseek4 weights: attention layer requested before a valid pack was planned";
        return false;
    }
    const DeviceTensor* q_a = tensor(layer, "attn_q_a.weight");
    const DeviceTensor* q_norm = tensor(layer, "attn_q_a_norm.weight");
    const DeviceTensor* q_b = tensor(layer, "attn_q_b.weight");
    const DeviceTensor* kv = tensor(layer, "attn_kv.weight");
    const DeviceTensor* kv_norm = tensor(layer, "attn_kv_a_norm.weight");
    const DeviceTensor* sinks = tensor(layer, "attn_sinks.weight");
    const DeviceTensor* output_a = tensor(layer, "attn_output_a.weight");
    const DeviceTensor* output_b = tensor(layer, "attn_output_b.weight");
    const DeviceTensor* norm = tensor(layer, "attn_norm.weight");
    const DeviceTensor* hc_fn = tensor(layer, "hc_attn_fn.weight");
    const DeviceTensor* hc_scale = tensor(layer, "hc_attn_scale.weight");
    const DeviceTensor* hc_base = tensor(layer, "hc_attn_base.weight");
    const auto f32 = [](const DeviceTensor* t) { return t && t->ptr && t->type == 0; };
    const auto matrix = [](const DeviceTensor* t) {
        return t && t->ptr && kernels::native_mmvq_supported(t->type);
    };
    if (!matrix(q_a) || !f32(q_norm) || !matrix(q_b) || !matrix(kv) || !f32(kv_norm) || !f32(sinks) ||
        !matrix(output_a) || !matrix(output_b) || !f32(norm) || !f32(hc_fn) || !f32(hc_scale) || !f32(hc_base)) {
        err = "deepseek4 layer " + std::to_string(layer) + ": attention weights are missing or have unsupported types";
        return false;
    }

    AttentionLayerWeights bound;
    bound.hc = {g.n_embd, g.rms_eps, g.hc_eps, g.sinkhorn_iters};
    bound.hc_fn = static_cast<const float*>(hc_fn->ptr);
    bound.hc_scale = static_cast<const float*>(hc_scale->ptr);
    bound.hc_base = static_cast<const float*>(hc_base->ptr);
    bound.norm = static_cast<const float*>(norm->ptr);
    bound.attention.q = {q_a->ptr, q_a->type, static_cast<const float*>(q_norm->ptr), q_b->ptr, q_b->type,
                         g.rms_eps, g.n_embd, g.q_lora, g.rms_eps};
    bound.attention.kv_type = kv->type;
    bound.attention.kv = kv->ptr;
    bound.attention.kv_norm = static_cast<const float*>(kv_norm->ptr);
    bound.attention.sinks = static_cast<const float*>(sinks->ptr);
    bound.attention.output_a_type = output_a->type;
    bound.attention.output_a = output_a->ptr;
    bound.attention.output_b_type = output_b->type;
    bound.attention.output_b = output_b->ptr;
    weights = bound;
    err.clear();
    return true;
}

bool ModelWeights::ffn_layer_weights(int layer, FfnRouteWeights& route, SharedExpertWeights& shared,
                                     std::string& err) const {
    const Geometry& g = pack_.geo;
    if (layer < 0 || layer >= g.n_layers || device_ < 0) {
        err = "deepseek4 weights: FFN layer requested before a valid pack was planned";
        return false;
    }
    const DeviceTensor* norm = tensor(layer, "ffn_norm.weight");
    const DeviceTensor* gate = tensor(layer, "ffn_gate_inp.weight");
    const DeviceTensor* hc_fn = tensor(layer, "hc_ffn_fn.weight");
    const DeviceTensor* hc_scale = tensor(layer, "hc_ffn_scale.weight");
    const DeviceTensor* hc_base = tensor(layer, "hc_ffn_base.weight");
    const DeviceTensor* gate_bias = tensor(layer, "exp_probs_b.bias");
    const DeviceTensor* image_bias = tensor(layer, "exp_probs_b_vl.bias");
    const DeviceTensor* shared_gate = tensor(layer, "ffn_gate_shexp.weight");
    const DeviceTensor* shared_up = tensor(layer, "ffn_up_shexp.weight");
    const DeviceTensor* shared_down = tensor(layer, "ffn_down_shexp.weight");
    const auto f32 = [](const DeviceTensor* t) { return t && t->ptr && t->type == 0; };
    const auto matrix = [](const DeviceTensor* t) {
        return t && t->ptr && kernels::native_mmvq_supported(t->type);
    };
    if (!f32(norm) || !gate || !gate->ptr || gate->type != 0 || !f32(hc_fn) || !f32(hc_scale) || !f32(hc_base) ||
        (!gate_bias && layer >= g.hash_layers) || (gate_bias && !f32(gate_bias)) || !f32(image_bias) ||
        !matrix(shared_gate) || !matrix(shared_up) || !matrix(shared_down)) {
        err = "deepseek4 layer " + std::to_string(layer) + ": FFN weights are missing or have unsupported types";
        return false;
    }

    FfnRouteWeights bound_route;
    bound_route.hc = {g.n_embd, g.rms_eps, g.hc_eps, g.sinkhorn_iters};
    bound_route.hc_fn = static_cast<const float*>(hc_fn->ptr);
    bound_route.hc_scale = static_cast<const float*>(hc_scale->ptr);
    bound_route.hc_base = static_cast<const float*>(hc_base->ptr);
    bound_route.norm = static_cast<const float*>(norm->ptr);
    bound_route.gate = static_cast<const float*>(gate->ptr);
    bound_route.routing = router_mixed_args(g.n_expert, g.n_vocab, nullptr, nullptr, nullptr, nullptr,
                                             gate_bias ? static_cast<const float*>(gate_bias->ptr) : nullptr,
                                             static_cast<const float*>(image_bias->ptr));

    SharedExpertWeights bound_shared;
    bound_shared.gate_type = shared_gate->type;
    bound_shared.up_type = shared_up->type;
    bound_shared.down_type = shared_down->type;
    bound_shared.gate = shared_gate->ptr;
    bound_shared.up = shared_up->ptr;
    bound_shared.down = shared_down->ptr;
    bound_shared.n_embd = g.n_embd;
    bound_shared.n_ff = g.n_ff_exp;
    bound_shared.swiglu_limit = g.swiglu_limit;
    route = bound_route;
    shared = bound_shared;
    err.clear();
    return true;
}

bool ModelWeights::head_weights(HeadWeights& weights, std::string& err) const {
    const Geometry& g = pack_.geo;
    if (device_ < 0) {
        err = "deepseek4 weights: output head requested before a valid pack was planned";
        return false;
    }
    const DeviceTensor* hc_fn = tensor("hc_head_fn.weight");
    const DeviceTensor* hc_scale = tensor("hc_head_scale.weight");
    const DeviceTensor* hc_base = tensor("hc_head_base.weight");
    const DeviceTensor* norm = tensor("output_norm.weight");
    const DeviceTensor* output = tensor("output.weight");
    const auto f32 = [](const DeviceTensor* t) { return t && t->ptr && t->type == 0; };
    if (!f32(hc_fn) || !f32(hc_scale) || !f32(hc_base) || !f32(norm) ||
        !output || !output->ptr || output->type != 30) {
        err = "deepseek4 weights: output head tensors are missing or have unsupported types";
        return false;
    }
    HeadWeights bound;
    bound.hc_fn = static_cast<const float*>(hc_fn->ptr);
    bound.hc_scale = static_cast<const float*>(hc_scale->ptr);
    bound.hc_base = static_cast<const float*>(hc_base->ptr);
    bound.norm = static_cast<const float*>(norm->ptr);
    bound.output = static_cast<const uint16_t*>(output->ptr);
    bound.n_embd = g.n_embd;
    bound.n_vocab = g.n_vocab;
    bound.rms_eps = g.rms_eps;
    bound.hc_eps = g.hc_eps;
    weights = bound;
    err.clear();
    return true;
}

bool ModelWeights::embedding_row(int32_t token, std::vector<float>& row, std::string& err) const {
    return read_embedding_row(pack_, token, row, err);
}

void ModelWeights::warm_token_table() const {
    if (const TensorRef* t = pack_.find("token_embd.weight"))
        if (fd_ >= 0) platform::advise_willneed(fd_, t->offset, t->bytes);
}

bool ModelWeights::hash_route_table(int layer, std::vector<int32_t>& ids, std::string& err) const {
    return load_hash_route_table(pack_, layer, ids, err);
}

bool ModelWeights::read_expert_blob(int layer, int expert, void* dst, size_t bytes, std::string& err) const {
    if (fd_ < 0 || !dst || layer < 0 || layer >= (int) pack_.experts.size() ||
        expert < 0 || expert >= pack_.geo.n_expert) {
        err = "deepseek4 weights: invalid expert staging request";
        return false;
    }
    const ExpertLayer& el = pack_.experts[(size_t) layer];
    const core::ModelFile::Section* section = pack_.file.find("experts");
    if (!section || bytes != el.bytes ||
        (uint64_t) expert > (std::numeric_limits<uint64_t>::max() - el.offset) / el.bytes) {
        err = "deepseek4 weights: expert staging size or layout is invalid";
        return false;
    }
    const uint64_t rel = el.offset + (uint64_t) expert * el.bytes;
    if (rel > section->bytes || el.bytes > section->bytes - rel || section->offset > UINT64_MAX - rel ||
        !platform::pread_all(fd_, dst, bytes, section->offset + rel)) {
        err = "deepseek4 weights: cannot read expert blob from the pack";
        return false;
    }
    err.clear();
    return true;
}

bool ModelWeights::expert_tier_spec(const model::LoadRequest& request, core::ExpertTierSpec& spec,
                                    std::string& err) const {
    if (path_.empty() || fd_ < 0) {
        err = "deepseek4 weights: expert tier requested before the pack is planned";
        return false;
    }
    return make_expert_tier_spec(pack_, path_, fd_, request, spec, err);
}

}  // namespace basalt::models::deepseek4
