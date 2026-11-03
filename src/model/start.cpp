// src/model/start.cpp - ModelAdapter::start, the start every session model runs (adapter.hpp).  Its own translation unit:
// only a program that starts a model (and links core, which places the bytes) pulls it in.
#include "basalt/core/load_progress.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace basalt::model {

Result ModelAdapter::start(const LoadRequest& req, Started& out) const {
    std::unique_ptr<LoadedModel> m = make_model();
    if (!m) return {Status::NotImplemented, std::string(arch()) + ": this model runs its own start"};
    std::string err;
    core::Loader loader;
    // its copy streams on the main card (the current device) before the plan sizes anything there; a tier card's come with
    // its experts, after its slots are sized
    if (!loader.prepare({-1}, err)) return {Status::Error, std::string(arch()) + ": " + err};
    core::LoadPlan reads;
    std::vector<std::unique_ptr<ModelSession>> sessions;
    // destroyed before the model and the sessions: a start that left GPU work unconfirmed ends here, before their CUDA
    // teardown waits for that work
    struct GpuLeftRunningExit {
        ~GpuLeftRunningExit() {
            if (!core::gpu_left_running().load()) return;
            basalt::log::error("basalt start", "the GPU did not confirm the start's work: ending without the CUDA teardown");
            std::fflush(stdout);
            std::_Exit(1);
        }
    } left_running;
    core::LoadProgress::planning();
    if (Result r = m->plan(req, reads, sessions); !r.ok()) return r;
    if (!core::place_start(loader, reads, err)) return {Status::Error, std::string(arch()) + ": reading the model failed: " + err};
    if (Result r = m->attach(req, sessions); !r.ok()) return r;
    if (sessions.size() != (size_t) std::max(req.sessions, 1))
        return {Status::Error, std::string(arch()) + ": the start made " + std::to_string(sessions.size()) + " sessions"};
    core::LoadProgress::complete();
    out.model = std::move(m);
    out.sessions = std::move(sessions);
    return {};
}

}  // namespace basalt::model
