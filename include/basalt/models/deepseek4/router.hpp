#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::deepseek4 {

struct RouteRows {
    std::vector<int32_t> fixed_rows;
    std::vector<uint8_t> fixed_mask;
    std::vector<uint8_t> alternate_mask;
};

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
