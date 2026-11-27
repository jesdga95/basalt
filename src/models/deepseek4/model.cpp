#include "models/deepseek4/model.hpp"

#include "basalt/core/load_progress.hpp"
#include "basalt/log.hpp"
#include "models/deepseek4/context_policy.hpp"
#include "models/deepseek4/dspark.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::deepseek4 {
namespace {
constexpr uint64_t kModuleRoom = 512ull << 20;

model::Result failure(const std::string& message) {
    return {model::Status::Error, "deepseek4: " + message};
}
}  // namespace

DsModel::~DsModel() {
    dspark_.reset();
    tier_.reset();
    if (h_expert_input_) cudaFreeHost(h_expert_input_);
    if (h_expert_rows_) cudaFreeHost(h_expert_rows_);
}

model::Result DsModel::ensure_tier() {
    if (tier_) return {};
    if (!planning_) return failure("the expert tier can only be planned during model start");
    const Geometry& g = weights_.geo();
    constexpr unsigned flags = cudaHostAllocPortable | cudaHostAllocMapped;
    if (cudaHostAlloc(reinterpret_cast<void**>(&h_expert_input_), (size_t) g.n_embd * sizeof(float), flags) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&h_expert_rows_), (size_t) g.n_used * g.n_embd * sizeof(float), flags) != cudaSuccess) {
        if (h_expert_input_) { cudaFreeHost(h_expert_input_); h_expert_input_ = nullptr; }
        if (h_expert_rows_) { cudaFreeHost(h_expert_rows_); h_expert_rows_ = nullptr; }
        return failure("cannot allocate mapped expert staging");
    }

    core::ExpertTierSpec spec;
    std::string err;
    if (!weights_.expert_tier_spec(request_, spec, err)) return failure(err);
    // The image encoder's cuBLAS handle takes host memory when created (the handle and its first products), so it opens
    // before the tier reads the free RAM: the RAM expert tier is sized from what is left after it.
    if (request_.vision) spec.host.open = [this](std::string& open_err) { return vision_blas_.open(open_err); };
    auto tier = std::make_shared<core::ExpertTier>(std::move(spec));
    core::ExpertTier::Reserve reserve;
    const ctxp::Footprint footprint = ctxp::cache_footprint(context_, g.compress_ratio);
    if (!footprint.error.empty()) return failure(footprint.error);
    uint64_t gpu_keep = footprint.total;
    if (!ctxp::add(gpu_keep, kModuleRoom, gpu_keep))
        return failure("cache reserve size overflows");

    // Auto sizing runs before the RAM spans are pinned. Leave their worst-case page-table mapping
    // free along with the full compressed-cache growth room; RAM load includes one spare per layer.
    if (request_.expert_cache <= 0) {
        uint64_t full_ram = 0, one_each = 0;
        for (const ExpertLayer& e : weights_.pack().experts) {
            uint64_t with_spare = 0;
            if (!ctxp::multiply(e.bytes, (uint64_t) g.n_expert + 1, with_spare) ||
                !ctxp::add(one_each, e.bytes, one_each) ||
                !ctxp::add(full_ram, with_spare, full_ram))
                return failure("pinned expert mapping reserve overflows");
        }
        if (request_.resident_budget > 0) {
            uint64_t budget_with_spare = 0;
            if (!ctxp::add(request_.resident_budget, one_each, budget_with_spare))
                budget_with_spare = std::numeric_limits<uint64_t>::max();
            full_ram = std::min(full_ram, budget_with_spare);
        }
        const uint64_t pin_map = core::pin_mapping_bytes(full_ram);
        if (!ctxp::add(gpu_keep, pin_map, gpu_keep))
            return failure("pinned expert mapping reserve overflows");
    }

    reserve.homes = gpu_keep;
    reserve.copies = gpu_keep;
    reserve.modules = kModuleRoom;
    if (!tier->open(reserve, h_expert_rows_, *planning_, err)) return failure(err);
    tier_ = std::move(tier);
    return {};
}

model::Result DsModel::plan(const model::LoadRequest& request, core::LoadPlan& reads,
                            std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (request.paths.size() != 1 || request.paths.front().empty()) return failure("one Basalt pack path is required");
    if (request.sessions != 1) return failure("this initial session path supports one conversation");
    if (!request.gpus.cards.empty() && request.gpus.cards.size() != 1)
        return failure("DeepSeek4 currently runs on one GPU");

    request_ = request;
    if (cudaGetDevice(&device_) != cudaSuccess) return failure("cannot query the current CUDA device");
    std::string err;
    if (!weights_.plan(request.paths.front(), device_, reads, err)) return failure(err);

    if (weights_.pack().dspark.present && request.spec >= 2) {
        const int file_handle = reads.file(request.paths.front());
        dspark_ = DsparkDrafter::plan_load(*this, file_handle, reads, std::max(request.sessions, 1), err);
        if (!dspark_) return failure(err);
    }

    const Geometry& g = weights_.geo();
    desc_.arch = "deepseek4";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.n_expert_shared = g.n_shared;
    desc_.context_length = g.context_length;
    desc_.expert_gate_up_type.reserve(g.n_layers);
    desc_.expert_down_type.reserve(g.n_layers);
    for (const ExpertLayer& layer : weights_.pack().experts) {
        desc_.expert_gate_up_type.push_back(layer.gate_up_type);
        desc_.expert_down_type.push_back(layer.down_type);
    }

    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        return failure("cannot query free CUDA memory after dense weights were planned");
    (void) total_bytes;
    const ctxp::Plan context_plan = ctxp::plan_context(request.max_context, g.context_length,
                                                        (uint64_t) free_bytes, g.compress_ratio);
    if (!context_plan.error.empty()) return failure(context_plan.error);
    context_ = context_plan.context;
    if (!context_plan.note.empty()) basalt::log::warn("basalt deepseek4", "%s", context_plan.note.c_str());
    basalt::log::info("basalt deepseek4", "context %lld tokens (cache reserve %.2f GiB)",
                      (long long) context_, (double) context_plan.footprint.total / (1ull << 30));

    planning_ = &reads;
    std::unique_ptr<model::ModelSession> session = make_session_impl(*this, err);
    if (!session) {
        planning_ = nullptr;
        return failure(err);
    }
    if (model::Result result = session->prepare(); !result.ok()) {
        planning_ = nullptr;
        return result;
    }
    planning_ = nullptr;
    sessions.push_back(std::move(session));
    return {};
}

model::Result DsModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    if (!tier_) return failure("expert tier was not planned");
    std::string err;
    if (!tier_->attach(err)) return failure(err);
    return {};
}

}  // namespace basalt::models::deepseek4
