#pragma once
#include <cstdint>

#include "basalt/kernels/resident_tabs.hpp"

namespace basalt::kernels {

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// Residency-biased routing (opt-in, changes the output): n_tok rows as native_router_top10_multi, except that an
/// expert a GPU holds (any table of `held` has a slot for it) is picked as if its probability were exp(lambda) times
/// larger. Only the selection is biased: the weights are the router's own probabilities of the picked experts,
/// renormalised over the ten. lambda 0 gives native_router_top10_multi's output bit for bit.
/// counts (device, optional): [3] entries, [4] entries picked only because of the bias, [5] entries no GPU holds (added to).
void route_prior_top10(const float* logits, const ResidentTabs& held, float lambda, int32_t* ids, float* weights, int n_tok,
                       unsigned long long* counts, void* stream);
}
