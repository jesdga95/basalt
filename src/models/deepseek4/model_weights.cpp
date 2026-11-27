#include "models/deepseek4/model_weights.hpp"

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

bool ModelWeights::embedding_row(int32_t token, std::vector<float>& row, std::string& err) const {
    return read_embedding_row(pack_, token, row, err);
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
