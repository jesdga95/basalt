// src/models/deepseek41/expert_support_test.cpp - the grouped expert kernels' support for the DeepSeek packs (the
// published ones and UD-Q2_K_XL) and their quantization pairs at the expert geometry: every gate/up x down pair those
// packs carry is taken, MXFP4 is refused as gate/up. The check is a host function; no GPU.
#include "basalt/kernels/iq_kernels.hpp"

#include "basalt/ggml/ggml.h"

#include <cstdio>

namespace {

// the DeepSeek packs' expert geometry
constexpr int64_t kNEmbd = 5120, kNFf = 2304;

}  // namespace

int main() {
    using basalt::kernels::native_expert_supported;
    const int gu[] = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS};
    const int down[] = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
                        GGML_TYPE_IQ3_S, GGML_TYPE_MXFP4};
    int bad = 0, pairs = 0;
    for (int g : gu)
        for (int d : down) {
            bad += !native_expert_supported(g, d, kNEmbd, kNFf);
            ++pairs;
        }
    bad += !native_expert_supported(GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, kNEmbd, kNFf);
    ++pairs;
    std::printf("pack gate/up x down pairs at %lld x %lld: %d pairs, %s\n", (long long) kNEmbd, (long long) kNFf, pairs,
                bad ? "FAIL" : "all taken");
    // MXFP4 has down kernels only: refused as gate/up with every down type, its own included
    int bad2 = 0;
    for (int d : down) bad2 += native_expert_supported(GGML_TYPE_MXFP4, d, kNEmbd, kNFf);
    std::printf("MXFP4 as gate/up: %s\n", bad2 ? "FAIL" : "refused");
    return bad || bad2 ? 1 : 0;
}
