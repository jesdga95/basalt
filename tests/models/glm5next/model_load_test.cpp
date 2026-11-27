// tests/models/glm5next/model_load_test.cpp - GlmModel's plan/attach against a tiny generated pack: the geometry reaches
// ModelDesc, the dense tensors land on the device (non-null pointers) with their reads queued, and per-layer tensor
// lookups resolve by layer kind. The fixture is the same tiny pack the pack test uses: GLM5NEXT_TINY_PACK names one, or
// the test generates it, or it SKIPs - and, needing a device, it skips without an sm_120 card.
#include "models/glm5next/model.hpp"

#include "basalt/core/loader.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

std::string fixture() {
    if (const char* e = std::getenv("GLM5NEXT_TINY_PACK"); e && *e && file_exists(e)) return e;
#ifdef GLM_SRC_DIR
    const std::string out = "/tmp/glm5next_tiny_test.basalt";
    const std::string cmd = "python3 '" GLM_SRC_DIR "/tools/convert/glm5next_tiny_pack.py' '" + out + "' >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0 && file_exists(out)) return out;
#endif
    return "";
}

bool have_sm120() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) return true;
    }
    return false;
}

}  // namespace

int main() {
    const std::string path = fixture();
    if (path.empty()) {
        std::printf("SKIP: no tiny pack (set GLM5NEXT_TINY_PACK or provide python + the converter)\n");
        return 0;
    }
    if (!have_sm120()) {
        std::printf("SKIP: no sm_120 device\n");
        return 77;
    }

    basalt::models::glm5next::GlmModel model;
    basalt::core::LoadPlan reads;
    std::vector<std::unique_ptr<basalt::model::ModelSession>> sessions;
    basalt::model::LoadRequest req;
    req.paths = {path};

    const auto r = model.plan(req, reads, sessions);
    check(r.ok(), "plan succeeds");
    if (!r.ok()) {
        std::printf("plan error: %s\n", r.message.c_str());
        return 1;
    }
    check(model.attach(req, sessions).ok(), "attach succeeds");

    const auto& d = model.desc();
    check(d.arch == "glm5next", "desc arch");
    check(d.n_layers == 3, "desc model layers exclude the drafter");
    check(d.n_embd == 256, "desc n_embd");   // the fixture's geometry, as pack_test pins it
    check(d.n_vocab == 3, "desc n_vocab");
    check(d.n_expert == 32, "desc n_expert");
    check(d.context_length > 0, "desc context_length");
    check(!d.expert_gate_up_type.empty(), "desc expert gate/up types");

    check(model.device() >= 0, "device set");
    check(!reads.empty(), "plan queues reads");
    check(reads.counted() > 0, "reads count bytes");

    const auto* mla = model.tensor("blk.2.attn_q_a.weight");
    check(mla != nullptr, "MLA layer 2 tensor present");
    check(mla && mla->ptr != nullptr, "MLA tensor has a device pointer");
    check(mla && mla->shape.size() == 2, "MLA tensor shape preserved");
    check(model.tensor(2, "attn_q_a.weight") == mla, "layer/suffix lookup matches name lookup");
    check(model.tensor("blk.1.attn_q_a.weight") == nullptr, "KDA layer 1 has no MLA query tensor");
    const auto* ssm = model.tensor("blk.0.ssm_a");
    check(ssm != nullptr && ssm->ptr != nullptr, "KDA layer 0 tensor on device");

    if (g_failures != 0) {
        std::printf("MODEL LOAD: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("MODEL LOAD: ok (%zu queued reads)\n", reads.items().size());
    return 0;
}
