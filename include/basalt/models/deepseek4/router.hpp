#pragma once

#include "basalt/kernels/router.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::deepseek4 {

struct RouteRows {
    std::vector<int32_t> fixed_rows;
    std::vector<uint8_t> fixed_mask;
    std::vector<uint8_t> alternate_mask;
};

/// V4's scored and hash-selected rows share the generic router's mixed selector. Keep the checkpoint's
/// top-6, SqrtSoftplus, and 1.5 weight scale in one model-owned constructor so sessions cannot drift
/// from the pack contract. All pointers are device pointers; row buffers are produced from build_route_rows.
inline kernels::RouterMixedArgs router_mixed_args(int n_expert, int token_vocab, const int32_t* fixed_table,
                                                  const int32_t* fixed_rows, const uint8_t* fixed_mask,
                                                  const uint8_t* alternate_mask, const float* bias,
                                                  const float* alternate_bias) {
    kernels::RouterMixedArgs args;
    args.n_expert = n_expert;
    args.k = 6;
    args.weight_scale = 1.5f;
    args.gate = kernels::RouterGate::SqrtSoftplus;
    args.fixed_table = fixed_table;
    args.fixed_rows = fixed_table ? fixed_rows : nullptr;
    args.fixed_table_rows = fixed_table ? token_vocab : 0;
    args.fixed_mask = fixed_table ? fixed_mask : nullptr;
    args.alternate_mask = alternate_mask;
    args.bias = bias;
    args.alternate_bias = alternate_bias;
    return args;
}

/// Scores normalized token rows with the V4 F32 gate matrix, then selects the mixed fixed/hash or biased routes.
/// `logits` is caller-owned [n_tokens][n_expert] workspace; `ids` and `weights` are [n_tokens][6].
inline void router_step(const float* x, const float* gate, int n_tokens, int n_embd,
                        const kernels::RouterMixedArgs& selection, float* logits,
                        int32_t* ids, float* weights, void* stream) {
    if (!x || !gate || !logits || !ids || !weights || n_tokens <= 0 || n_embd <= 0 ||
        selection.n_expert <= 0 || selection.k != 6)
        throw std::invalid_argument("deepseek4 router step: invalid input or routing geometry");
    kernels::RouterArgs scoring;
    scoring.n_expert = selection.n_expert;
    scoring.n_embd = n_embd;
    scoring.k = selection.k;
    scoring.weight_scale = selection.weight_scale;
    scoring.gate = selection.gate;
    kernels::router_logits(x, gate, n_tokens, scoring, logits, stream);
    kernels::router_select_mixed(logits, n_tokens, selection, ids, weights, stream);
}

/// Builds the host-side selector inputs from token IDs. Text rows in hash-routed layers index `tid2eid`; image IDs use
/// score selection with `bias_vl` in every layer.
inline bool build_route_rows(const int32_t* tokens, int n_tokens, int token_vocab, int layer, int n_layers,
                             int hash_layers, RouteRows& out, std::string& err) {
    if (!tokens || n_tokens <= 0 || token_vocab <= 0 || n_layers <= 0 || layer < 0 || layer >= n_layers ||
        hash_layers < 0 || hash_layers > n_layers) {
        err = "deepseek4 route rows: invalid token or layer geometry";
        return false;
    }

    RouteRows rows;
    rows.fixed_rows.resize((size_t) n_tokens);
    rows.fixed_mask.resize((size_t) n_tokens);
    rows.alternate_mask.resize((size_t) n_tokens);
    const bool hash_layer = layer < hash_layers;
    for (int t = 0; t < n_tokens; ++t) {
        const int32_t token = tokens[t];
        if (token < 0) {
            err = "deepseek4 route rows: negative token id at row " + std::to_string(t);
            return false;
        }
        const bool image = token >= token_vocab;
        rows.fixed_rows[(size_t) t] = !image && hash_layer ? token : 0;
        rows.fixed_mask[(size_t) t] = (uint8_t) (!image && hash_layer);
        rows.alternate_mask[(size_t) t] = (uint8_t) image;
    }
    out = std::move(rows);
    err.clear();
    return true;
}

}  // namespace basalt::models::deepseek4
