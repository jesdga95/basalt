# Hy4-flash pack and converter

The first pack is a test pack from the AngelSlim Q4_K_M GGUF (`general.architecture` `hy_v4_flash`, llama.cpp PR 32517 names), so
the engine can run before the NVFP4 pack exists. The NVFP4 pack from the BF16 GGUF is [below](#the-nvfp4-pack-from-the-bf16-gguf). Architecture id: `hy4flash`
(`src/models/hy4flash`, `tools/convert/arch/hy4flash.py`).

## Source facts (gguf-py over the two shards, 1,416 tensors)

- Shard 1 (105.0 GB) holds the metadata and every tensor but one; shard 2 (63.4 GB) holds only `per_layer_token_embd.weight`.
- 43 blocks: 0 to 41 are the trunk, block 42 is llama.cpp's MTP (`nextn`) layer, 11.3 GB of BF16 (experts 7.2 GB). **Dropped**:
  Hy4 drafts with DFly2, not with this layer.
- The metadata the engine needs is in the GGUF, except MRoPE: `rope.dimension_sections` is absent, so the converter writes
  `meta.rope = {"mrope_section": [12, 10, 10], "interleaved": true}` from the model config (text-only code ignores it).
- Heads per layer (`attention.head_count`: 32 sparse, 64 SWA), `sliding_window_pattern`, `indexer.types`, `indexer.kpool` 4,
  `kpool_select_tail`, `expert_latent_length` 1536, `ple.*` and `engram.*` are copied to `meta.geometry` and `meta.engram`.

## What is kept as is, what moves

Nothing is requantized. Quantized tensors keep their source bytes (`gguf` section, 64 B aligned) under the GGUF names, so the
engine's `pack.cpp` maps names the way deepseek41's does.

| Tensors (per layer unless noted) | Source type | Pack |
|---|---|---|
| routed experts `ffn_{gate,up,down}_exps` (layers 1 to 41) | gate, up Q4_K; down Q6_K on 19 layers (1-4, 7, 10, ..., 37, 38-41), Q4_K on 22 | `experts`: per layer 512 blobs `[gate \| up \| down]`, bytes unchanged |
| `ffn_routed_down`, `ffn_routed_up` (latent projections), shared expert, layer-0 dense FFN, `attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_gate`, `attn_output`, `attn_k_b`, `attn_v_b`, `engram_kv`, indexer `attn_k`, `attn_q_b`, `compressor_gate`, `compressor_value`, `token_embd` | Q8_0 | `gguf`, native |
| `output.weight` (3072 x 120,832) | BF16 | `dense`, as stored (742 MB) |
| norms, `attn_sinks`, `hc_*` (fn, base, scale; `output_hc_*`), `ffn_gate_inp`, `exp_probs_b`, `engram_{gate,hc_scale,kv_b,lookup_norm,pre_gamma}`, indexer `proj`, `k_norm`, `compressor_{ape,norm}` | F32 | `dense`, as stored |
| `per_layer_token_embd.weight` (the Engram table, 96 x 660,006,416) | F8_E4M3 (ggml type 43, no scale) | own section `engram`, 2 MiB aligned, rows of 96 bytes |
| `blk.42.*` (MTP) | BF16, F32 | dropped |

Expert blob: every matrix of an expert is 1536 x 1536 (the latent is 1536 wide, and so is the expert), K = 1536 = 6 blocks of
256. A Q4_K matrix is 1,327,104 B, a Q6_K matrix 1,935,360 B. The blob (Q4_K, Q4_K, Q4_K) is 3,981,312 B, (Q4_K, Q4_K, Q6_K)
4,589,568 B; a layer's routed bytes 2.04 or 2.35 GB; all 41 layers 89.49 GB. `experts.layout` is v5 (`gu_type` 12, `d_type` 12
or 14), one row per MoE layer, layers 1 to 41 (layer 0 has no row; the engine reads the `layer` column).

Pack size: experts 89.49 GB + Engram 63.36 GB + Q8_0 6.96 GB (embedding included) + BF16 head 0.74 GB + F32 0.04 GB + tokenizer
= about 160.6 GB (149.6 GiB). The engine reads Engram rows by `pread` (O_DIRECT, row-aligned), so the section takes no RAM.

Reinterpretations that move no byte (the dense index has two dimensions per row, `ne1` = the product of the rest; `meta.tensors`
keeps the GGUF shape): `attn_k_b` (192, 512, H) is H x 512 rows of 192 (row = h x 512 + latent index; the absorbed query product
is one batched matrix-vector per head); `attn_v_b` (512, 256, H) is H x 256 rows of 512 (row = h x 256 + d); the 3-D expert
tensors are sliced by expert as deepseek41's are.

## Converter changes

1. `tools/convert/arch/hy4flash.py`: `Hy4Flash(Arch)` registered for `hy_v4_flash`. `side_sections = False`, `own_sections =
   {per_layer_token_embd.weight: "engram"}`, `form_table` empty (floats as stored), `sampling_kv` temp 0.9, top_p 1, top_k 0.
   `license()` is the model repository's LICENSE (Tencent Hy License 1.0, Apache-derived with a MaaS clause; see the decisions
   in the design doc). `meta_extra`, as deepseek41's: `arch`, `geometry` (the keys above), `engram` {section, tensor, type
   `F8_E4M3_RAW`, type_id 43, rows, row_values 96, row_bytes 96, bytes, the `ple.*` and `engram.*` hash constants,
   `token_map: gguf_kv:hy_v4_flash.engram.token_map`}, `rope`. `check()` refuses, with the reason, a table that is not type 43
   of shape [96, rows], rows different from the sum of `ple.head_vocab_sizes` (4 modules x 16 heads), or Engram layers other
   than [1, 11, 21, 31].
2. `Arch.drops(name, kv)` (default: nothing). Hy4's drops `blk.N.*` with N >= `block_count - nextn_predict_layers` (block 42).
   Applied in `source_tensors`, `Dense` and `Experts`: dropped tensors are in no section and not in `meta.tensors`.
3. `Experts`: the layers are those that have all three `_exps` tensors (1 to 41), not `0 .. max`; the router-width check reads
   the first of them (`blk.1.ffn_gate_inp`), not `blk.0`.
4. `Dense`: a native (quantized) tensor of 3 dimensions is allowed, its index row `(ne0, prod(rest))`. Float tensors keep the
   two-dimension limit (all of Hy4's are 1-D or 2-D).
5. `tools/convert/test_convert_hy4flash.py`, modelled on `test_convert_deepseek41.py`: a tiny synthetic `hy_v4_flash` GGUF (3
   trunk layers with the first dense, 4 experts, a nextn block, an F8_E4M3 table). It checks the sections, layout rows for MoE
   layers only, the dropped block, the 3-D native tensors, and every tensor's bytes against the source.
6. `--mmproj`, `--dspark` and `--mtp-*` die with the normal side-section message. Vision and DFly2 attach later (`--mmproj`,
   `--dfly`) as sections `vision/...` and `dfly/...` without copying weights, as `--dspark` does.

## Checks before the engine reads it

- `convert.py --dry-run` prints the section sizes; the sum is within 1 MiB of the figures above.
- Per tensor, the SHA-256 of the pack bytes equals the source tensor's; per expert, the blob equals the three source slices; the
  Engram section's SHA-256 equals the source tensor's.
- The loader's pack parse lists 42 trunk layers, 41 expert rows, 4 Engram layers and no `blk.42`.

## What the engine must do with these types (milestone 1)

- Q8_0 dense: `native_mmvq` (supported). Q4_K gate/up and Q6_K down on the GPU: `iq_kernels.cu` has Q4_K (`Fmt<12>`) and Q6_K
  (`Split<14>`), but the grouped expert path has been exercised with Q4_K gate/up and Q5_1 or Q8_0 down only, so the
  (Q4_K, Q4_K, Q6_K) combination gets a parity test first. Prefill's `moe_mmq.cu` has both types.
- CPU experts: `kq_avx2.cpp` has Q4_K x Q8_K and no Q6_K. A `q6k_dot` (ggml's AVX2 `ggml_vec_dot_q6_K_q8_K`, bit-exact) is added
  to it for the 19 down matrices, with its parity test, before RAM-tier decode runs.
- The F8_E4M3 Engram rows are decoded by a 256-entry table (host first, a GPU gather later); the lookup norm and everything
  after it are FP32.

## The NVFP4 pack from the BF16 GGUF

`convert.py --gguf Hy4-flash-BF16.gguf --nvfp4 --hf-dir REPO --mmproj MMPROJ.gguf --dfly hy4-dfly.safetensors --out P.basalt`
(`tools/convert/hy4flash_bf16.py`). The BF16 GGUF is llama.cpp's converter run on the checkpoint: 42 blocks (no next-n block), the
Engram tables as one BF16 tensor of 660,006,416 rows, the head as F32 holding BF16 numbers. Every rounding is deterministic and
runs once, in pieces by worker processes (no tensor is held whole, the source's page cache is dropped behind each read).

| Tensors | Pack |
|---|---|
| routed experts (layers 1 to 41) | NVFP4 (type 40), blobs `[gate \| up \| down]` + FP32 tail `{s_gate, s_up, s_down, 0}`: 3,981,328 B each, 83.58 GB; per matrix the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least squared error among half to four times amax / 6 (the 27B pack's rounding) |
| BF16 projections (attention, latent, shared expert, dense FFN, `engram_kv`, indexer, embedding) | Q8_0, ggml reference rounding: byte-identical to AngelSlim's Q8_0 tensors. `--dense-q6k`: Q6_K for `Q6K_TARGETS` with whole 256-value rows (attention q, gate, output, kv_a, shared expert, latent projections, dense FFN), the rest Q8_0 |
| head | BF16 (the F32 values are BF16 numbers; checked) |
| norms, hc, router, biases | F32 as stored |
| Engram table | E4M3 codes (type 43), to nearest, ties to even, no scale; the exact fraction and RMS error go to `meta.quantized_from.engram` and the log |
| tokenizer, template | the GGUF's, refused unless ids, merges, special ids and chat template equal `--hf-dir`'s `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` |
| `vision/mmproj`, `dfly` | the side files, whole (`meta.dfly2` as `--dfly --into` writes it) |

`general.architecture` is `hy4flash`; the other keys keep the GGUF's `hy_v4_flash.` prefix, which the engine reads. Size: 158.7 GB
(experts 83.58, Engram 63.36, dense 7.45 + 0.97, mmproj 1.04, DFly 1.69).

`--verify P.basalt --gguf Hy4-flash-BF16.gguf` checks every section's SHA-256 and repeats each rounding against the BF16 GGUF,
comparing the bytes (it takes as long as the build). Tests: `tools/convert/test_convert_hy4flash_bf16.py`.
