#pragma once

#include "basalt/kernels/sparse_mqa.hpp"
#include "basalt/models/deepseek4/cache_rows.hpp"

#include <cstdint>

namespace basalt::models::deepseek4 {

constexpr int kAttnHeads = kernels::kSparseMqaHeads;
constexpr int kAttnHeadDim = kernels::kSparseMqaDim;
constexpr int kAttnNopeDim = kAttnHeadDim - kernels::kSparseMqaRopeDim;
constexpr int kAttnWindow = kernels::kSparseMqaWindow;
constexpr int kAttnChunk = 64;

struct PrefillMqaArgs {
    int n_tokens = 1;
    const float* q = nullptr;
    const void* win = nullptr;
    CacheFormat win_type = CacheFormat::Fp8;
    int win_cap = kAttnWindow;
    const int32_t* win_start = nullptr;
    const int32_t* win_n = nullptr;
    const void* comp = nullptr;
    CacheFormat comp_type = CacheFormat::Fp8;
    const int32_t* sel = nullptr;
    int n_sel = 0;
    const int32_t* comp_limit = nullptr;
    const float* sinks = nullptr;
    float scale = 0.044194174f;
    const int32_t* pos = nullptr;
    kernels::MqaRopeTable rope{};
    bool round_output_bf16 = true;
    float* out = nullptr;
};

void sparse_mqa_tc(const PrefillMqaArgs& a, float* workspace, void* stream);

}  // namespace basalt::models::deepseek4
