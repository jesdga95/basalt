#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/kernels.hpp"
#include <algorithm>
#include <future>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::inputs() {
    // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
    // previous stage handed on
    if (owner.hand_in_ != nullptr) {
        if (cudaMemcpyAsync(m.R, owner.hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice, m.cs) !=
            cudaSuccess) {
            err = "prefill: the layer split's hand-off upload failed";
            return false;
        }
    }
    // the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
    // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
    bool batched = first_stage && m.tok_dev != nullptr;
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (batched && nemb == nullptr &&
        (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
         (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        batched = false;
    for (int64_t t = 0; batched && t < T; ++t) {
        const int64_t tok = tokens[c0 + t];
        if ((owner.embd_rows && owner.embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
        else m.tok_host[(size_t) t] = (int32_t) tok;
    }
    if (batched) {
        if (cudaMemcpyAsync(m.tok_dev, m.tok_host.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                            m.cs) != cudaSuccess) {
            err = "prefill: the token id upload failed";
            return false;
        }
        if (nemb) {
            nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            basalt::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                  wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
        }
    }
    for (int64_t t = 0; first_stage && !batched && t < T; ++t) {
        const float* row = owner.embd_rows ? owner.embd_rows[p0 + t] : nullptr;
        if (row) {
            if (cudaMemcpyAsync(m.emb + t * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                err = "prefill: the image embedding upload failed";
                return false;
            }
        } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
            return false;
        }
    }
    if (first_stage) gr_broadcast(m.emb, m.R, T, m.cs);
    // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
    // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
    // layer 1 on, and gathering them here first would leave the GPU idle for the whole read
    if (ple_on && !ple_next.valid())
        ple_next = std::async(std::launch::async, [&run = run, c0 = c0, b = ple_buf] {
            return run.ple_gather(c0, b, run.ple_next_err);
        });
    for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
    // ---- the QSA step records of every position in the chunk
    for (int64_t t = 0; t < T; ++t) basalt::kernels::qsa_step_fill(m.steps_host.data() + t * basalt::kernels::kStepCount, p0 + t, s);
    cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * basalt::kernels::kStepCount * 4,
                    cudaMemcpyHostToDevice, m.cs);
    for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
    return true;
}

}  // namespace basalt::prefill
