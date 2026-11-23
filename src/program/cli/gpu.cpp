#include "gpu.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <cuda_runtime.h>

namespace basalt::program::cli {

std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards, const model::ModelAdapter* adapter) {
    namespace bp = basalt::program;
    bp::GpuSpec spec;
    bp::GpuMode mode = bp::GpuMode::One;
    std::string gerr;
    if (!bp::parse_gpu_spec(o.gpu, spec, gerr) || !bp::parse_gpu_mode(o.gpu_mode, mode, gerr)) {
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
    bp::GpuChoice choice = bp::choose_gpus(spec, mode, o.gpu_share, visible, known);
    if (choice.error.empty() && adapter != nullptr) choice.error = bp::check_gpu_modes(choice, adapter->model_name(), adapter->gpu_modes());
    if (o.gpu_plan) {
        std::string lossy;
        for (const auto& f : o.lossy) lossy += (lossy.empty() ? "" : ",") + f.flag;
        std::printf("%s\n", bp::gpu_plan_json(visible, choice, o.arch, o.quality_flags, o.quality_window, o.plan_spec, o.max_context,
                                             o.model_name, lossy, o.community_kv).c_str());
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
    o.card_mode = choice.mode;
    bp::RoleInputs ri;
    for (int i : choice.cards) ri.cards.push_back((size_t) i < visible.size() ? visible[(size_t) i] : bp::GpuCard{});
    ri.mode = choice.mode;
    ri.prompt_split = adapter != nullptr && adapter->second_card_reads_prompts();
    const std::vector<bp::GpuRole> roles = bp::choose_roles(ri);
    std::string line;
    for (size_t d = 0; d < roles.size(); ++d) {
        const bp::GpuCard& c = ri.cards[d];
        line += std::string(d ? ", " : "") + (c.name.empty() ? "GPU" : c.name) +
                (c.smi >= 0 ? " (nvidia-smi " + std::to_string(c.smi) + ")" : "") + " " + bp::gpu_role_name(roles[d]);
        if (roles[d] != bp::GpuRole::Main) {
            o.peer_devices.push_back((int) d);
            o.peer_gbps.push_back(c.mem_gbps);
        }
    }
    int drv = 0, rt = 0;
    (void) cudaDriverGetVersion(&drv);
    (void) cudaRuntimeGetVersion(&rt);
    basalt::log::info("basalt", "system: %s, CUDA driver %d.%d / runtime %d.%d",
                      basalt::platform::host_os_name().c_str(),
                      drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
    basalt::log::info("basalt", "gpus: %s", line.c_str());
    if (!o.peer_devices.empty()) o.peer_device = o.peer_devices[0];
    if (!ri.cards.empty()) o.main_gbps = ri.cards[0].mem_gbps;
    chosen_cards = ri.cards;
    // The split hands a window's activations from one card to the next, and a tier card reads its experts d2d: both
    // want peer access, which every pair must opt into once per process.  A pair the driver refuses is left alone and
    // the copy goes the long way.
    if (chosen_cards.size() > 1) {
        int pairs = 0, tried = 0, prev = 0;
        cudaGetDevice(&prev);
        for (int a = 0; a < (int) chosen_cards.size(); ++a)
            for (int b = 0; b < (int) chosen_cards.size(); ++b) {
                int can = 0;
                if (a == b || cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
                    (void) cudaGetLastError();
                    continue;
                }
                ++tried;
                if (cudaSetDevice(a) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) ++pairs;
                (void) cudaGetLastError();
            }
        (void) cudaSetDevice(prev);
        if (tried > 0) basalt::log::info("basalt", "peer access: %d of %d ordered pairs enabled", pairs, tried);
    }
    return std::nullopt;
}

}  // namespace basalt::program::cli
