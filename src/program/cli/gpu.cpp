#include "gpu.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

namespace basalt::program::cli {

std::optional<int> select_gpus(Options& o, std::vector<GpuCard>& chosen_cards, int max_expert_tiers) {
    namespace bp = basalt::program;
    bp::GpuSpec spec;
    std::string gerr;
    if (!bp::parse_gpu_spec(o.gpu, spec, gerr)) {
        if (o.gpu_plan) {
            bp::GpuChoice bad;
            bad.error = gerr;
            std::printf("%s\n", bp::gpu_plan_json({}, bad).c_str());
        } else {
            basalt::log::error("basalt", "%s", gerr.c_str());
        }
        return 2;
    }
    bool known = false;
    const std::vector<bp::GpuCard> visible = bp::visible_gpus(known);
    const bp::GpuChoice choice = bp::choose_gpus(spec, visible, known);
    if (o.gpu_plan) {
        std::printf("%s\n", bp::gpu_plan_json(visible, choice).c_str());
        return choice.error.empty() ? 0 : 2;
    }
    if (!choice.error.empty()) { basalt::log::error("basalt", "%s", choice.error.c_str()); return 2; }
    for (const std::string& s : choice.notes) basalt::log::info("basalt", "%s", s.c_str());
    if (!choice.warning.empty()) basalt::log::warn("basalt", "%s", choice.warning.c_str());
    bp::apply_gpus(visible, choice.cards);
    int seen = 0;
    if (cudaGetDeviceCount(&seen) != cudaSuccess || seen < (int) choice.cards.size()) {
        basalt::log::error("basalt", "gpu %s: CUDA sees %d of the %zu GPUs chosen", o.gpu.empty() ? "0" : o.gpu.c_str(),
                           seen, choice.cards.size());
        return 2;
    }
    bp::RoleInputs ri;
    for (int i : choice.cards) ri.cards.push_back((size_t) i < visible.size() ? visible[(size_t) i] : bp::GpuCard{});
    ri.peer_access = basalt::platform::gpu_peer_access();
    ri.max_expert_tiers = max_expert_tiers;   // the device-planned tail's peer windows
    const std::vector<bp::GpuRole> roles = bp::choose_roles(ri);
    std::string line;
    for (size_t d = 0; d < roles.size(); ++d) {
        const bp::GpuCard& c = ri.cards[d];
        line += std::string(d ? ", " : "") + (c.name.empty() ? "GPU" : c.name) +
                (c.smi >= 0 ? " (nvidia-smi " + std::to_string(c.smi) + ")" : "") + " " + bp::gpu_role_name(roles[d]);
        if (roles[d] == bp::GpuRole::ExpertTier) {
            o.peer_devices.push_back((int) d);
            o.peer_gbps.push_back(c.mem_gbps);
        }
    }
    basalt::log::info("basalt", "gpus: %s", line.c_str());
    if (std::count(roles.begin(), roles.end(), bp::GpuRole::Idle) > 0)
        basalt::log::warn("basalt", "%s", ri.peer_access
            ? ("expert tiers max " + std::to_string(ri.max_expert_tiers) + "; later cards idle").c_str()
            : "no GPU-to-GPU access; cards after the main idle");
    if (!o.peer_devices.empty()) o.peer_device = o.peer_devices[0];
    chosen_cards = ri.cards;
    return std::nullopt;
}

}  // namespace basalt::program::cli
