// include/basalt/models/inkling2/prompt_kernels.hpp - the kernels of Inkling 2's prompt path (chunks of many rows): the
// prompt attention with each query's relative bias and sink, the experts' plan by expert, and the sliding-window keys a
// chunk staged moved into the window rings.
//
// Activations are fp32 rows, every pointer is device memory, `stream` may be null. A chunk's rows sit at consecutive
// positions pos0, pos0 + 1, ...
#pragma once

#include "basalt/models/inkling2/kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::inkling2 {

/// What one prompt attention call reads.
struct AttnPromptArgs {
    const float* q = nullptr;          ///< [rows][kHeads][kHeadDim], as attn_prepare writes them
    const float* r = nullptr;          ///< [rows][kHeads][kRel]
    const float* sink = nullptr;       ///< [rows][kHeads]
    const float* rel_table = nullptr;  ///< [kRel][kExtent]: bias(d) = r . table[:, d] for 0 <= d < kExtent
    int64_t pos0 = 0;                  ///< row i at position pos0 + i
    const void* k_cache = nullptr;     ///< keys before pos0: [cap][kKvHeads][kHeadDim], position p in cell p % cap
    const void* v_cache = nullptr;
    int64_t cap = 0;
    const void* k_chunk = nullptr;     ///< keys from pos0 on: [chunk_cap][kKvHeads][kHeadDim], position p in cell p % chunk_cap
    const void* v_chunk = nullptr;     ///< (a full layer passes its cache again; a sliding-window layer the chunk's stage)
    int64_t chunk_cap = 0;
    bool fp8 = false;                  ///< unit-scale E4M3 codes instead of fp16
    bool int8 = false;                 ///< symmetric INT8 codes with two per-row FP16 scales
    int window = 0;                    ///< > 0: keys at distance < window only
    float scale = 0.08838834764831845f;
};

/// out[i][h] = sum_j softmax_j(scale q.k_j + bias(p_i - p_j)) v_j over the keys at positions p_j <= p_i (and within the
/// window), exp(sink[i][h]) beside them in the denominator: attention_decode's result for each row, on the tensor cores
/// (q, the relative states and the bias table rounded to fp16 for the products, fp32 accumulation and softmax).
void attention_prompt(const AttnPromptArgs& a, int rows, float* out, void* stream);

/// ids[i] = map[ids[i]] for `n` entries: a spilled layer's experts by their slot (spill.hpp).
void remap_ids(int32_t* ids, int n, const int32_t* map, void* stream);

/// pos[i] = pos0 + i for a chunk's `rows` rows, and bounds = {0, rows} (the one-matrix products' row bounds).
void prompt_rows(int32_t* pos, int64_t pos0, int rows, int32_t* bounds, void* stream);

/// The last positions of a chunk's staged sliding-window keys and values (position p in cell p % stage_cap) into the ring
/// (cell p % ring_cap): positions max(pos0, end - keep) .. end - 1, end = pos0 + rows. A cell is [kKvHeads][kHeadDim] of
/// `row_bytes` bytes per KV head row (fp16, E4M3 or scaled INT8).
void stage_to_ring(const void* k_stage, const void* v_stage, int64_t stage_cap, void* k_ring, void* v_ring, int64_t ring_cap,
                   int64_t pos0, int rows, int keep, int row_bytes, void* stream);

/// Fused prompt loop controller down projection: RMS-normalizes dest and msg independently and computes
/// hid[r][o] = w_down[o][0..kWidth) . rms(dest[r]) + w_down[o][kWidth..2*kWidth) . rms(msg[r]).
void ctrl_down_prompt(const float* dest, const float* msg, const uint16_t* w_down, int rows, int rank, float* hid, void* stream);

/// Fused prompt loop controller up projection + blend (out may alias dest):
/// out[r][c] = (1 - tanh(w_forget[c] . hid[r])) * dest[r][c] + tanh(w_gain[c] . hid[r]) * msg[r][c].
void ctrl_up_blend_prompt(const float* dest, const float* msg, const float* hid, const uint16_t* w_gf, int rows, int rank,
                          float* out, void* stream);

/// Fused RMSNorm + Q8 planes quantization of [rows][kWidth] directly into prefill::q8's codes + coalesced scales.
void rms_norm_q8_planes_prompt(const float* x, const float* w, int rows, float eps, void* act, void* stream);

/// Fused attention sigmoid gate + Q8 planes quantization of [rows][kHeads * kHeadDim] directly into prefill::q8's layout.
void attn_gate_q8_planes_prompt(const float* attn, const float* yq, int64_t yq_ld, int rows, void* act, void* stream);

/// Deterministic vectorized weighted sum of k fp16 expert outputs per row into fp32 y[rows][kWidth].
void moe_combine_f16(const uint16_t* out, const float* w, int rows, int k, float* y, void* stream, const int32_t* ids = nullptr);

/// gpu-mode peer: the rows of ids [rows][k] (raw expert choices) with a choice on the second card (slot_map[expert] < 0), in
/// row order: row_ix[0 .. *count), *count (device or mapped host memory).
void peer_rows_plan(const int32_t* ids, int rows, int k, const int32_t* slot_map, int32_t* row_ix, int32_t* count, void* stream);
/// Row row_ix[c] of x [.][kWidth], ids and w [.][k] as row c of x_c, ids_c and w_c, for c < count.
void peer_rows_gather(const float* x, const int32_t* ids, const float* w, int k, const int32_t* row_ix, int count, float* x_c,
                      int32_t* ids_c, float* w_c, void* stream);
/// y[row_ix[c]] += part[c] for c < count (rows of kWidth floats).
void peer_rows_add(const float* part, const int32_t* row_ix, int count, float* y, void* stream);

/// y[r] += x[r] for `rows` rows of kWidth floats: another card's partial sum of the experts joins the combine's.
void add_rows(const float* x, float* y, int rows, void* stream);

}  // namespace basalt::models::inkling2
