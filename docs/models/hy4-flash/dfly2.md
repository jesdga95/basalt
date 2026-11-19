# DFly2, Hy4-flash's drafter

The forward the engine runs (`src/models/hy4flash/dfly*`), where it comes from, and every assumption left. Nothing here has run on
the model yet: the tests are synthetic (a double-precision CPU reference of the equations below on small random weights, kernel
parity, the exactness of verification against a fake target). Acceptance only changes speed, never output.

## Source

An inference implementation exists, in review: **vLLM PR 68296**, "[Model] Add HY4 Flash and multimodal support"
(https://github.com/vllm-project/vllm/pull/68296, branch `jeejeelee/hy4-flash-support`, one consolidated commit, 2026-11-18). The
drafter is `vllm/models/hy_v4/nvidia/dfly.py` (`HYV4DFlyModel`, `HYV4DFlyFusion`, `HYV4DFlyHiddenCorrection`),
`dfly_speculator.py` (the block: `HYV4DFlyQueryModel.forward`), `dfly_inputs.py` (anchor, positions, the index snapshot),
`dfly_target_reuse.py` (KV and index sharing) and `dfly_config.py` / `hy_v4_flash/dfly_config.py` (the derived config), with
`docs/features/speculative_decoding/dfly.md`. The same model package is mirrored in flagos-ai/vllm-plugin-FL PR 774 (without the
drafter files). A separate vLLM PR 50246 adds DFly for Qwen3, unrelated to Hy4. Tencent's AngelSpec (https://github.com/Tencent/AngelSpec,
`angelspec/models/draft/dfly.py`) holds the training-side `HiddenStatesCorrection` for the Qwen-style DFly: same formula, no Hy4
drafter. SGLang's `--speculative-algorithm DFLY` and llama.cpp's dflash2 (#32016, generic DFlash) were not needed. The `hy4-flash`
vLLM image was not pulled. The model card's diagram and text agree with the code.

## Weights

`model.dfly_layers.*`, 87 BF16 tensors, 1.69 GB (the last shard, `model-00148-of-00148.safetensors`), stored whole in the pack's
`dfly` section by `convert.py --dfly FILE --into PACK --undo-dir DIR` (meta.dfly2 records every tensor's offset, shape and the
config's `dfly_*` settings). Per layer i < 5: `input_layernorm`, `self_attn.{q_a_proj 1024x3072, q_a_layernorm, q_b_proj 8192x1024,
kv_a_proj_with_mqa 576x3072, kv_a_layernorm 512, kv_b_proj 14336x512, linear_gate 8192x3072, o_proj 3072x8192, post_proj_norm,
learnable_sink_param 32}`, `post_attention_layernorm`, `mlp.{gate,up}_proj 9216x3072, down_proj`. Global: `final_norm`,
`mtp_fusion.{eh_proj 3072x6144, enorm, hnorm, gate 3072}`, `hidden_correction.{gate,up}_proj 3072x6144, down_proj 3072x3072,
hidden_norm, embed_norm}`, `confidence_head.proj {weight 1x3072, bias}`. The embedding and the output head are the target's.
Settings (config.json): 5 layers, `dfly_mask_token_id` 120009, `dfly_target_kv_layers` = `dfly_target_index_layer` = [25, 29, 33, 37,
41], `dfly_target_index_topk` 2048, no local window, slot0 fusion, hidden correction on. The block pass's projections (every
matrix but `kv_b`, which the per-head kernels read as BF16) are rounded to Q8_0 as they load (`core::Conv::Bf16ToQ8_0`, ggml's
reference rounding), so the drafter takes 0.87 GiB of the main card instead of 1.58 and the expert tier, sized after it, gets 4
more experts per layer (131 to 135 of 512 with the NVFP4 pack on a 5090); their products quantize the rows to q8_1 and run MMVQ.
The norms, the sink, the fusion gate and the confidence head are widened to FP32 at load. The drafter only proposes: the target
verifies every draft, so the rounding changes which drafts are accepted, never the output.

## Forward (as vLLM's, with the Basalt mapping)

A block has the anchor `a` (the last pick: its position `pos` is the next the target runs) and `n` (1..5) mask tokens, rows at
positions `pos .. pos + n`. `s = pos - 1` is the **source row**: the target row whose output chose `a`.

1. **Gated Fusion**, row 0 only: `x0 = e(a) + gate * eh_proj(enorm(e(a)) :: hnorm(h_s))`, `h_s` the target's last-layer hidden state at
   the source row after the output norm (the vector the head multiplies). Rows 1..n: `e(mask)` unchanged.
2. **Five dense layers**, plain single-stream residual (no iHC), pre-norm: `n = input_ln(x)`; attention (below); `x += post_proj_norm(o_proj(out))`;
   `x += mlp(post_attention_ln(x))` with a plain SwiGLU of 9216 (no clamp). `z = final_norm(x)`.
3. **Attention** of layer i = the target's absorbed MLA, with the drafter's own weights: q path `q_b(q_a_ln(q_a(n)))`, 32 heads of 192 nope + 64 rope;
   latent `kv_a_ln(kv_a(n)[:512])`, rope part `kv_a(n)[512:]`; interleaved-pair rope at base 1e7 (the target's sparse layers), position `pos + r`;
   `kv_b_proj` rows per head: 192 key rows then 256 value rows, used absorbed (`q_abs = W_k^T q_nope`, value `W_v o_lat`); scale 1/16; learnable sink in the
   denominator; sigmoid gate `linear_gate(n)` over the 8192 outputs; `o_proj`. **KV Share**: the keys and values are the cache rows of target layer
   `kv_layers[i]` (the rows of 576 the target wrote; with `--kv fp8` its FP8 rows, decoded exactly). **Index Share**: the rows are the cells the target's indexer selected in layer `idx_layers[i]` for
   the source row `s` (every position before `pos` where the indexer keeps all of them), at most 2,048, the same for every block row; each block row also
   attends the block's own rows 0..r (causal), whose K/V the drafter computes from its own `kv_a` and holds as BF16 in scratch (vLLM writes them into the
   target's cache slots at `pos..`, where verification overwrites them; Basalt writes nothing into the target's cache).
4. **Hidden-Correction**, sequential, draft `j = 0..n-1` from row `j + 1` and the token before it (`a`, then draft `j - 1`):
   `c = hidden_norm(z_{j+1}) :: embed_norm(e(prev))`, `h' = z_{j+1} + down(silu(gate(c)) * up(c))`.
5. **Head**: `logits = output_head(h')`, greedy argmax (vLLM can also sample at the request's temperature; greedy drafts keep verification exact at any
   temperature, only acceptance of sampled drafts suffers). Row 0's output is not used (its K/V feed later rows).

## Assumptions

- `h_s` is the target's final hidden state after the output norm, as vLLM passes `last_hidden_states` to `propose` (no auxiliary layers). Not measured.
- The confidence head is `sigmoid(confidence_head(h'_j))`: vLLM ignores the head (`_ignored_prefixes`), so its input is a guess. It only feeds the
  draft policy's window choice, which re-learns the acceptance per stated chance; a bad guess costs speed.
- Positions are plain text positions. With images the target's MRoPE delta would shift the drafter's rope positions (`mrope_delta` in vLLM); not done until vision lands.
- The sink value is the raw logit added as `exp(sink)` to the denominator, as the target's `attn_sinks`.
- Rows not selected, SWA layers and the target's indexer are untouched; the draft layers need sparse (indexed) target layers, which the loader checks.
- No local window of the context (`dfly_context_local_window` null); a pack that sets one is refused.

## Engine

- Pack: `dfly` section and meta.dfly2 (`tools/convert/dfly_attach.py`); weights placed with the dense ones, before the context is sized.
- Session hooks (marked `DFLY HOOK`): the tap copies each decode-window row's hidden state and, per sparse layer a draft layer reads, that row's selection
  (`icells_`, or "all"); `draft_begin` enqueues the block pass on the session stream, `draft_end` runs the heads and returns the drafts with their chances.
- Policy: the generic one. The drafter returns up to 5 drafts with stated chances; `DraftPolicy::window` picks the window by expected tokens per ms from
  measured round costs. A window is verified through the existing `decode_window` row path, so the stream equals plain decoding's. When the tap holds no
  row for the position (a prompt's last row run without logits) the round drafts nothing.
- Flags, as DeepSeek's: `--spec N` (default 6 when the pack has a `dfly` section, else 0; 0 or 1 = plain), `--spec-oracle FILE`, `--diag-teacher`, `--draft-log FILE`.

## Owed (model level)

Acceptance per position on code and prose; greedy identity with and without `--spec` (tiers pinned) and sampled identity with a fixed seed; tok/s against plain
decode; the Q8_0 weights; the confidence head's calibration; MRoPE positions with images; the head's cost (5 sequential 742 MB reads) against a restricted draft vocabulary.
