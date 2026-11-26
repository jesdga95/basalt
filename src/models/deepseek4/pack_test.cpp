#include "basalt/log.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "models/deepseek4/pack.hpp"
#include "models/deepseek4/meta.hpp"
#include "models/deepseek4/context_policy.hpp"
#include "models/deepseek4/expert_tier_spec.hpp"
#include "basalt/vision/mmproj.hpp"

#include <algorithm>
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
    const auto cache_plan = basalt::models::deepseek4::ctxp::plan_context(
        g.context_length, g.context_length, UINT64_MAX, g.compress_ratio);
    if (!cache_plan.error.empty() || cache_plan.context != g.context_length) {
        basalt::log::error("deepseek4 pack test", "native cache plan: %s",
                           cache_plan.error.empty() ? "native context was changed" : cache_plan.error.c_str());
        return 1;
    }
    basalt::model::LoadRequest tier_request;
    tier_request.diag_tier_ram = 7;
    tier_request.diag_tier_vram = 11;
    tier_request.vram_reserve_mib = 512;
    tier_request.expert_cache = 64;
    tier_request.resident_budget = 3ull << 30;
    tier_request.route.tail_skip = 2;
    tier_request.route.prior = 0.125f;
    basalt::core::ExpertTierSpec tier;
    if (!basalt::models::deepseek4::make_expert_tier_spec(pack, argv[1], 0, tier_request, tier, err)) {
        basalt::log::error("deepseek4 pack test", "expert tier spec: %s", err.c_str());
        return 1;
    }
    uint64_t expected_stride = 0;
    for (const auto& expert : pack.experts) expected_stride = std::max(expected_stride, expert.bytes);
    if (tier.name != "deepseek4" || tier.n_layers != g.n_layers || tier.n_expert != g.n_expert ||
        tier.top_k != g.n_used || tier.layers.size() != pack.experts.size() || tier.range.bytes == 0 ||
        tier.share_stride != expected_stride || tier.diag_tier_ram != 7 || tier.diag_tier_vram != 11 ||
        tier.vram_reserve_mib != 512 || tier.expert_cache != 64 || tier.resident_budget != (3ull << 30) ||
        tier.route_tail_skip != 2 || tier.route_prior != 0.125f) {
        basalt::log::error("deepseek4 pack test", "expert tier spec differs from the pack or requested tier policy");
        return 1;
    }
    for (size_t layer = 0; layer < pack.experts.size(); ++layer) {
        const auto& expected = pack.experts[layer];
        const auto& actual = tier.layers[layer];
        if (actual.gu_type != expected.gate_up_type || actual.d_type != expected.down_type ||
            actual.offset != expected.offset || actual.blob_bytes != expected.bytes || actual.clamp != 0.f) {
            basalt::log::error("deepseek4 pack test", "expert tier layer %zu differs from experts.layout", layer);
            return 1;
        }
    }
    std::vector<int32_t> route_ids;
    for (int layer = 0; layer < g.hash_layers; ++layer) {
        if (!basalt::models::deepseek4::load_hash_route_table(pack, layer, route_ids, err) ||
            route_ids.size() != (size_t) g.n_vocab * 6) {
            basalt::log::error("deepseek4 pack test", "hash route table %d: %s", layer,
                               err.empty() ? "unexpected table size" : err.c_str());
            return 1;
        }
        for (const int token : {0, 1, g.n_vocab - 1}) {
            for (int slot = 0; slot < 6; ++slot) {
                const int32_t expert = route_ids[(size_t) token * 6 + slot];
                if (expert < 0 || expert >= g.n_expert) {
                    basalt::log::error("deepseek4 pack test", "hash route table %d has invalid sampled expert id", layer);
                    return 1;
                }
                for (int earlier = 0; earlier < slot; ++earlier) {
                    if (expert == route_ids[(size_t) token * 6 + earlier]) {
                        basalt::log::error("deepseek4 pack test", "hash route table %d duplicates an expert at token %d",
                                           layer, token);
                        return 1;
                    }
                }
            }
        }
    }
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
    std::printf("deepseek4 pack ok: %d layers, %d vocab, %zu tensor records; native cache capacity %.2f GiB; vision %d layers / %d width / %d heads / %d FFN / %d patch / %d merge; section %llu bytes\n",
                g.n_layers, g.n_vocab, pack.tensors.size(), (double) cache_plan.footprint.total / (1024.0 * 1024.0 * 1024.0),
                v.layers, v.embd, v.heads, v.ffn, v.patch, v.merge,
                (unsigned long long) pack.vision.bytes);
    return 0;
}
