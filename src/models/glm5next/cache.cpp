// src/models/glm5next/cache.cpp - the cache sizing formulas.
#include "models/glm5next/cache.hpp"

namespace basalt::models::glm5next {

CacheDims cache_dims(const Geometry& g) {
    CacheDims d;
    const int model_layers = g.n_model_layers();
    for (int i = 0; i < model_layers && i < (int) g.kinds.size(); ++i)
        if (g.kinds[i] != 0) ++d.n_mla;
        else ++d.n_kda;
    d.n_idx = d.n_mla;   // the DSA indexer runs on the sparse layers
    d.n_head = g.n_head;
    d.head_dim = g.kda_head_dim;
    d.conv = g.conv;
    d.kv_lora = g.kv_lora;
    d.idx_dim = g.idx_dim;

    // one KDA layer: the delta-rule state n_head*head_dim*head_dim, plus 3 conv streams of (conv-1)*n_head*head_dim.
    const uint64_t heads = g.n_head, hd = g.kda_head_dim, cw = (uint64_t)(g.conv > 0 ? g.conv - 1 : 0);
    const uint64_t state_per_layer = heads * hd * hd;
    const uint64_t conv_per_layer = 3 * cw * heads * hd;
    d.kda_state_elems = (uint64_t) d.n_kda * state_per_layer;
    d.kda_conv_elems = (uint64_t) d.n_kda * conv_per_layer;

    d.mla_latent_elems = (uint64_t) d.n_mla * g.kv_lora;
    if (g.nextn > 0) {
        d.n_mla_draft = g.nextn;
        d.draft_latent_elems = (uint64_t) g.nextn * g.kv_lora;
    }
    d.kpool = g.kpool;
    d.idx_pooled_elems = (uint64_t) d.n_idx * g.idx_dim;
    // the pool being filled is the only one with un-pooled members, and it holds at most kpool-1 of them
    d.idx_pending_elems = (uint64_t) d.n_idx * g.idx_dim * kIndexerPendingRows * (g.kpool > 1 ? g.kpool - 1 : 0);
    return d;
}

uint64_t cache_bytes(const CacheDims& d, uint64_t context, int n_seq) {
    if (n_seq <= 0) return 0;
    const uint64_t per_seq = d.kda_state_elems + d.kda_conv_elems + d.idx_pending_elems +
                             context * (d.mla_latent_elems + d.draft_latent_elems) +
                             d.n_pools(context) * d.idx_pooled_elems;
    return (uint64_t) n_seq * per_seq * sizeof(float);
}

}  // namespace basalt::models::glm5next
