#include "basalt/log.hpp"
#include "models/deepseek4/pack.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::puts("skip: pass a DeepSeek-V4 Basalt pack to validate its C++ runtime manifest");
        return 77;
    }
    basalt::models::deepseek4::Pack pack;
    std::string err;
    if (!basalt::models::deepseek4::open_pack(argv[1], pack, err)) {
        basalt::log::error("deepseek4 pack test", "%s", err.c_str());
        return 1;
    }
    const auto& g = pack.geo;
    if (g.n_layers != 43 || g.n_embd != 4096 || g.n_vocab != 129280 || g.n_expert != 256 ||
        g.n_used != 6 || g.hash_layers != 3 || pack.vision.empty()) {
        basalt::log::error("deepseek4 pack test", "the opened pack does not have the verified V4 Vision geometry");
        return 1;
    }
    std::printf("deepseek4 pack ok: %d layers, %d vocab, %zu tensor records, vision section %llu bytes\n",
                g.n_layers, g.n_vocab, pack.tensors.size(), (unsigned long long) pack.vision.bytes);
    return 0;
}
