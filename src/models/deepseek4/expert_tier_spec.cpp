#include "models/deepseek4/expert_tier_spec.hpp"

#include <algorithm>

namespace basalt::models::deepseek4 {

bool make_expert_tier_spec(const Pack& pack, const std::string& path, int experts_fd,
                           const model::LoadRequest& request, core::ExpertTierSpec& spec, std::string& err) {
    const Geometry& g = pack.geo;
    const core::FileRange range = pack.file.range("experts");
    if (path.empty() || experts_fd < 0 || range.empty() || g.n_layers <= 0 || g.n_expert <= 0 ||
        g.n_used <= 0 || g.n_embd <= 0 || g.n_ff_exp <= 0 || pack.experts.size() != (size_t) g.n_layers) {
        err = "deepseek4 expert tier: pack, descriptor, or geometry is incomplete";
        return false;
    }

    core::ExpertTierSpec out;
    out.area = "basalt deepseek4";
    out.name = "deepseek4";
    out.n_layers = g.n_layers;
    out.n_expert = g.n_expert;
    out.top_k = g.n_used;
    out.n_embd = g.n_embd;
    out.n_ff = g.n_ff_exp;
    out.layers.reserve(pack.experts.size());
    for (size_t layer = 0; layer < pack.experts.size(); ++layer) {
        const ExpertLayer& e = pack.experts[layer];
        if (e.gate_up_type < 0 || e.down_type < 0 || e.bytes == 0 || e.offset > range.bytes ||
            e.bytes > range.bytes - e.offset) {
            err = "deepseek4 expert tier: invalid expert blob for layer " + std::to_string(layer);
            return false;
        }
        out.layers.push_back({e.gate_up_type, e.down_type, e.offset, e.bytes, g.swiglu_limit});
        out.share_stride = std::max(out.share_stride, e.bytes);
    }

    out.path = path;
    out.fd = experts_fd;
    out.range = range;
    out.gpus = request.gpus;
    out.diag_tier_ram = request.diag_tier_ram;
    out.diag_tier_vram = request.diag_tier_vram;
    out.diag_tier_frozen = request.diag_tier_frozen;
    out.vram_reserve_mib = request.vram_reserve_mib;
    out.peer_reserve_mib = request.peer_reserve_mib;
    out.expert_cache = request.expert_cache;
    out.resident_budget = request.resident_budget;
    out.pool_workers = request.pool_workers;
    out.pool_affinity = request.pool_affinity;
    out.pool_affinity_given = request.pool_affinity_given;
    out.expert_profile = request.expert_profile;
    out.route_tail_skip = request.route.tail_skip;
    out.route_prior = request.route.prior;
    spec = std::move(out);
    err.clear();
    return true;
}

}  // namespace basalt::models::deepseek4
