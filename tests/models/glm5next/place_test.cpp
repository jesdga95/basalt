// tests/models/glm5next/place_test.cpp - the load pipeline end to end: GlmModel's plan, placed by the shared loader,
// lands every dense tensor's bytes exactly where the plan says. Reads a loaded weight back off the device and byte-
// compares it to the pack file at that tensor's offset - a wrong section, offset or conversion shows up here, not in
// the shape-only load test. GPU-gated and fixture-gated like the load test.
#include "models/glm5next/model.hpp"
#include "models/glm5next/pack.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        std::printf("SKIP: no tiny pack\n");
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

    check(model.plan(req, reads, sessions).ok(), "plan succeeds");
    std::string err;
    basalt::core::Loader loader;
    check(loader.run(reads, err), "loader places the plan");
    if (!err.empty()) std::printf("loader: %s\n", err.c_str());

    // A weight that lives in the gguf section (Conv::Copy): what is on the device must equal the file bytes at its offset.
    const char* name = "blk.2.attn_q_a.weight";
    const auto* t = model.tensor(name);
    check(t != nullptr && t->ptr != nullptr && t->bytes > 0, "weight loaded to device");
    if (t && t->ptr && t->bytes > 0) {
        const uint64_t bytes = t->bytes;
        std::vector<uint8_t> dev(bytes), file(bytes);
        check(cudaMemcpy(dev.data(), t->ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess, "read device bytes");

        const int fd = basalt::platform::open_fd_utf8(path);
        check(fd >= 0, "open pack for readback");
        if (fd >= 0) {
            const uint64_t off = model.pack().find(name)->offset;
            const bool got = basalt::platform::pread_all(fd, file.data(), bytes, off);
            basalt::platform::close_fd(fd);
            check(got, "read file bytes at offset");
            check(std::memcmp(dev.data(), file.data(), bytes) == 0, "device bytes == file bytes");
        }
    }

    // The routed experts are not dense tensors. The tier reads them out of the pack's experts section, addressed off
    // the section's own base rather than the absolute offset the tensor table carries - the same class of mistake this
    // test exists to catch - and keeps each one in a pinned RAM place for the CPU pool and, for the ones it homes, a
    // VRAM slot for the card. So read a place back and byte-compare it with the file at that address.
    {
        check(model.attach(req, sessions).ok(), "attach opens the tier's places");
        check(model.tier() != nullptr, "a pack with routed experts opens the tier");
        check(model.has_experts(1) && !model.has_experts(0), "the dense lead routes to none, the next layer routes");
        using Pack = basalt::models::glm5next::Pack;
        using Layout = basalt::kernels::NativeExpertLayout;
        using Source = basalt::core::ExpertTier::Source;
        const Pack& p = model.pack();
        const Layout& L = model.expert_layout(1);
        check(L.gu_type == p.experts[1].gu_type && L.d_type == p.experts[1].d_type && L.bytes == p.experts[1].blob_bytes,
              "the layout the card computes from is the pack's types and blob size");
        check(model.expert_layout(0).bytes == 0, "the dense lead layer has no expert layout");

        const int fd = basalt::platform::open_fd_utf8(path);
        check(fd >= 0, "open pack for readback");
        if (model.tier() && fd >= 0) {
            // first expert of the first routed layer, last expert of the next: a missing section base, layer offset or
            // expert stride each shows up in one of the two.
            const int probes[2][2] = {{1, 0}, {2, model.geo().n_expert - 1}};
            for (const auto& pr : probes) {
                const int layer = pr[0], expert = pr[1];
                const std::string ctx = "layer " + std::to_string(layer) + " expert " + std::to_string(expert);
                const Source src = model.tier()->locate(layer, expert);
                // The tier decides what it can afford to pin; on a machine with no room for the RAM tier it keeps
                // every expert in the file and there is no place to read back. Say so rather than fail on the
                // machine's memory: the byte compare below is what a normal start gets.
                if (!src.pinned && !src.dev) {
                    std::printf("      SKIP: the tier keeps %s in the file (no RAM to pin it)\n", ctx.c_str());
                    continue;
                }
                std::vector<uint8_t> got(p.experts[layer].blob_bytes), file(p.experts[layer].blob_bytes);
                bool read = false;
                if (src.pinned) {
                    std::memcpy(got.data(), src.pinned, got.size());   // the pool's own bytes
                    read = true;
                } else if (src.dev) {
                    read = cudaMemcpy(got.data(), src.dev, got.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
                }
                check(read, ("read " + ctx + " back from the tier").c_str());
                const uint64_t off = p.experts_range.offset + p.experts[layer].offset +
                                     (uint64_t) expert * p.experts[layer].blob_bytes;
                const bool n = basalt::platform::pread_all(fd, file.data(), file.size(), off);
                check(n, ("read " + ctx + " at its experts-section offset").c_str());
                check(read && n && std::memcmp(got.data(), file.data(), file.size()) == 0,
                      (ctx + " bytes == the experts section bytes").c_str());
            }
        }
        if (fd >= 0) basalt::platform::close_fd(fd);
    }

    if (g_failures != 0) {
        std::printf("PLACE TEST: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("PLACE TEST: ok\n");
    return 0;
}
