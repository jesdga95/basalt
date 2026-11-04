// src/models/deepseek41/fixture_ops.cpp - DEV ONLY, not part of the shipped path.
//
// Replays the attention, routed-expert and shared-expert outputs recorded from the reference implementation, so the
// rest of the layer loop (embedding, Engram, hyper-connections, norms, router, head) is checked against the same
// recording without those kernels. `--diag-fixture <prefix>` reads `<prefix>.bin` and `<prefix>.idx`; the index
// has a first line `n_prompt N` and then one `step name layer offset nbytes` line per recorded tensor.
#include "models/deepseek41/layer_ops.hpp"
#include "models/deepseek41/model.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

namespace basalt::models::deepseek41 {
namespace {

class FixtureOps final : public LayerOps {
public:
    bool open(const std::string& prefix, int n_embd, std::string& err) {
        n_embd_ = n_embd;
        std::ifstream idx(prefix + ".idx");
        if (!idx) {
            err = "cannot read " + prefix + ".idx";
            return false;
        }
        std::string line;
        while (std::getline(idx, line)) {
            std::istringstream ls(line);
            std::string a;
            ls >> a;
            if (a == "n_prompt") {
                ls >> n_prompt_;
                continue;
            }
            int step = std::atoi(a.c_str());
            std::string name;
            int layer;
            unsigned long long off, nb;
            if (ls >> name >> layer >> off >> nb) tab_[{step, name, layer}] = {off, nb};
        }
        bin_.open(prefix + ".bin", std::ios::binary);
        if (!bin_ || n_prompt_ <= 0) {
            err = "bad fixture " + prefix;
            return false;
        }
        return true;
    }

    void begin_token(int64_t pos) override { pos_ = pos; }

    bool attention(int l, int64_t pos, int32_t, const float*, float* out, void*, std::string& err) override {
        return serve("attn_out", l, pos, out, err);
    }
    bool routed_experts(int l, const float*, const int32_t*, const float*, float* out, void*, std::string& err) override {
        return serve("ffn_moe_out", l, pos_, out, err);
    }
    bool shared_expert(int l, const float*, float* out, void*, std::string& err) override {
        return serve("ffn_shexp", l, pos_, out, err);
    }

private:
    bool serve(const char* name, int layer, int64_t pos, float* out, std::string& err) {
        pos_ = pos;
        const int64_t step = pos < n_prompt_ ? 0 : pos - n_prompt_ + 1;
        const int64_t col = pos < n_prompt_ ? pos : 0;
        auto it = tab_.find({(int) step, name, layer});
        if (it == tab_.end()) {
            err = std::string("fixture has no ") + name + " for layer " + std::to_string(layer) + " step " + std::to_string(step);
            return false;
        }
        const size_t per = (size_t) n_embd_ * sizeof(float);
        std::vector<char> h(per);
        bin_.clear();
        bin_.seekg((std::streamoff) (it->second.first + (uint64_t) col * per));
        bin_.read(h.data(), (std::streamsize) per);
        if (!bin_) {
            err = "short read of the fixture";
            return false;
        }
        cudaMemcpy(out, h.data(), per, cudaMemcpyHostToDevice);
        return true;
    }

    int n_embd_ = 0, n_prompt_ = 0;
    int64_t pos_ = 0;
    std::map<std::tuple<int, std::string, int>, std::pair<uint64_t, uint64_t>> tab_;
    std::ifstream bin_;
};

}  // namespace

std::unique_ptr<LayerOps> make_fixture_ops(const DsModel& model, const std::string& fixture, std::string& err) {
    auto o = std::make_unique<FixtureOps>();
    if (!o->open(fixture, model.geo().n_embd, err)) return nullptr;
    return o;
}

}  // namespace basalt::models::deepseek41
