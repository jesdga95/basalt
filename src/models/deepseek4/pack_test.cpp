#include "basalt/log.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "models/deepseek4/pack.hpp"
#include "models/deepseek4/meta.hpp"
#include "basalt/vision/mmproj.hpp"

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
    if (!pack.file.register_gguf_view(err)) {
        basalt::log::error("deepseek4 pack test", "GGUF metadata view: %s", err.c_str());
        return 1;
    }
    basalt::GgufFile gguf(argv[1]);
    if (!(err = basalt::models::deepseek4::validate_meta(gguf)).empty()) {
        basalt::log::error("deepseek4 pack test", "adapter metadata validation: %s", err.c_str());
        return 1;
    }
    const auto desc = basalt::models::deepseek4::describe_meta(gguf);
    if (desc.arch != "deepseek4" || desc.n_layers != pack.geo.n_layers || desc.n_embd != pack.geo.n_embd ||
        desc.n_vocab != pack.geo.n_vocab || desc.context_length != pack.geo.context_length) {
        basalt::log::error("deepseek4 pack test", "adapter description differs from the validated pack geometry");
        return 1;
    }
    basalt::vision::Mmproj mmproj;
    if (!mmproj.open(pack.vision, err, "deepseek4v")) {
        basalt::log::error("deepseek4 pack test", "embedded projector: %s", err.c_str());
        return 1;
    }
    const auto& g = pack.geo;
    const auto& v = mmproj.hp();
    const char* image_rows[] = {"v.token_embd.img_start", "v.token_embd.img_end", "v.token_embd.img_pad", "v.image_newline"};
    for (const char* name : image_rows) {
        const auto* tensor = mmproj.find(name);
        if (!tensor || tensor->type != 0 || tensor->elems() != g.n_embd) {
            basalt::log::error("deepseek4 pack test", "projector is missing the expected F32 image row %s", name);
            return 1;
        }
    }
    if (g.n_layers != 43 || g.n_embd != 4096 || g.n_vocab != 129280 || g.n_expert != 256 ||
        g.n_used != 6 || g.hash_layers != 3 || v.layers != 32 || v.embd != 1024 || v.heads != 16 ||
        v.ffn != 2816 || v.patch != 14 || v.merge != 3 || v.proj != 4096 || v.min_pixels != 147456 ||
        pack.vision.empty()) {
        basalt::log::error("deepseek4 pack test", "the opened pack does not have the verified V4 Vision geometry");
        return 1;
    }
    std::printf("deepseek4 pack ok: %d layers, %d vocab, %zu tensor records; vision %d layers / %d width / %d heads / %d FFN / %d patch / %d merge; section %llu bytes\n",
                g.n_layers, g.n_vocab, pack.tensors.size(), v.layers, v.embd, v.heads, v.ffn, v.patch, v.merge,
                (unsigned long long) pack.vision.bytes);
    return 0;
}
