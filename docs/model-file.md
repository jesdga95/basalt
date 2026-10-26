# The .basalt model file

A model is one file. It holds everything the engine, the server and the vision sidecar load: the weights, the
tokenizer and chat template, the MTP drafter, the expert profile, the 8-bit hyper-connection copies, the vision
projector, and the default sampling. It carries no engine flags: those come from `--config` or the command line.

The layout diagram is in the [README](../README.md#models).

- **Layout:** a 64 KiB header (magic `BASALTMF`, version, a table of up to 511 named sections, each with its offset,
  size and SHA-256), then the sections, each aligned to 2 MiB so it can be memory-mapped or read with `O_DIRECT`.
- **Sections:** `meta` (JSON: geometry, the source GGUF metadata, every tensor's type, shape and offset,
  sampling, provenance and license), `tokenizer/*`, `dense`, `experts` and `experts.layout` (the routed
  experts, laid out per layer), `gguf` (every non-expert source tensor), `mtp/*` (the drafter), `profile`, `hc8` and
  `vision/*`. The authoritative description is [include/basalt/core/model_format.hpp](../include/basalt/core/model_format.hpp).
- **Weights as published:** a pack built from a GGUF copies every tensor byte for byte, except with `--compat-bf16`,
  for GGUFs that store the small projections the engine reads as BF16 in another type (it converts them and records
  that in `meta`). The NVIDIA NVFP4 pack keeps NVIDIA's NVFP4 experts bit for bit and quantizes its dense tensors once
  from BF16 ([below](#nvfp4-pack)).

Build one from the source GGUF shards (give shard 1; the others are found beside it):

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--profile FILE] [--hc8 FILE] [--compat-bf16] [--name NAME]
```

`python3 tools/convert/convert.py --help` describes every input.

## Packs

A pack you convert yourself from a GGUF, without the official files' expert profile, hc8 copies and tuned drafter, will be significantly slower than the official `.basalt` files. How precision is handled and how quality is measured: [precision](precision/index.md).

## NVFP4 pack

The official NVFP4 file is NVIDIA's NVFP4 checkpoint
([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), revision `fc694b54`)
with the dense projections stored as 6-bit Q6_K instead of 8-bit Q8_0, which takes the tensor-core dense path in
decode. The per-tensor breakdown and its quality are on the Hugging Face card.

Basalt's converter reads NVIDIA's safetensors directly (`convert.py --checkpoint`). It moves each expert's NVFP4 codes
and scales into Basalt's blocks without changing a bit, quantizes each dense projection once from BF16, folds the "+1"
of the norm weights into F32, puts the GDN value heads in the order the engine reads them, and writes the metadata and
tokenizer from NVIDIA's `config.json`, tokenizer files and model card. `meta.checkpoint` records each step.

BF16 PLE shards stay unquantized; FP8 PLE shards still become Q8_0. This is independent of `--dense-q8-0`.
Rebuild from BF16 sources rather than an existing Q8_0 table. Disk/RAM costs and precision:
[BF16 PLE tables](precision/features.md#bf16-ple-tables).

Build either variant yourself from NVIDIA's checkpoint. The base pack supplies only the drafter, vision projector and
expert profile (and its `hc8` copies if it has any), copied byte for byte. The official file takes them from
UD-Q4_K_XL, which has no `hc8`, so the engine makes the hyper-connection copies from NVIDIA's weights; use a base
without `hc8` for the same result. The base's vocabulary must be NVIDIA's:

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE` replaces the dense part of the drafter copied from the base with a tuned layer. The file is the raw
bytes of the `mtp/dense` section of an official IQ3_S, IQ3_XXS or Q2_0 pack (for example
`ModelFile(path).read("mtp/dense")` from `tools/convert/model_file.py`); its size must match the base's section, and the
converter records its SHA-256 in `meta.mtp.dense` with the tuning `self_distilled`. Add it to either command:

```sh
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --mtp-dense tuned-mtp-dense.bin \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

The output with BF16 PLE is about 167 GiB (the 8-bit dense variant 168 GiB). How NVFP4 is computed on each tier:
[precision](precision/index.md).

## CPU PLE quantization in RAM

`--ple-ram-quant none|q8_0` defaults to `none`, leaving the stored PLE format unchanged.
`q8_0` requires a BF16 source and `--ple-io ram`; other modes or source formats are errors.

```sh
basalt serve --model MODEL-BF16-PLE.basalt --ple-io ram --ple-ram-quant q8_0
```

In a server config, the engine entries are:

```json
{"engine": {"ple-io": "ram", "ple-ram-quant": "q8_0"}}
```

Startup allocates and locks the full Q8_0 destination, then returns while background workers stream the BF16 table
through two bounded 80 MiB input buffers. A read of an unfinished 128-row region packs it on demand through a separate
small read buffer. Atomic region ownership prevents duplicate writers; readers wait for publication before decoding.
Both paths widen to fp32 and call the same reference Q8_0 encoder. Warm rows need only a readiness check, with no
allocation or conversion. There is no full BF16/fp32 host copy or GPU conversion, and the source file is never written.
Worker count follows `--pool-workers`, or the automatic topology, `--pool-affinity` and CPU-quota policy.

For 320,001,536 rows of 160 values, BF16 uses 320 bytes per row and 95.368 GiB on disk; resident Q8_0 uses 170 bytes
per row and 50.664 GiB. The output allocation rounds up to 2 MiB. Additional memory covers the input buffers,
80 KiB scratch per worker plus the demand reader,
region states and normal engine reserves. Large pages are inherently resident; ordinary pages must be completely
OS-locked. Linux reserved huge pages are admitted separately from ordinary-RAM reserves. Allocation or partial locking
failure is an error, not a pageable fallback. Full allocation and locking stay before expert-tier planning, so background
packing cannot consume RAM already promised to experts. Conversion workers, source handles and temporary buffers end
when packing completes; close cancels and joins unfinished work before releasing the table.

Q8_0 is int8 codes with an FP16 scale per 32 values, not FP8. It is lossy relative to BF16 and uses the same rounding
as CPU model-file quantization, not a new quantizer. The BF16-versus-Q8_0 PPL/KLD quality comparison is not measured.

The independently converted, 54,400,261,120-byte offline Q8_0 reference has SHA-256
`91743e33b6de2272b50a6269a6fe692f56244946fdecf19ca1eba0ba640b303f`.
Background and on-demand packing match the CPU encoder over 1,031 fixture rows, with zero encoded or decoded
mismatches. Against the synchronous loader, 300 greedy tokens each for conv, struct and prose match exactly with
all 24,576 experts fixed in VRAM, spec 8, context 262,144 and Q8_0 RAM PLE.
CPU tests cover reference-byte parity, concurrent demand/background reads, boundaries, duplicates, decoded gathers,
validation failures, early close, reopen and source removal after completion.

Development A/B on Windows, RTX PRO 6000, GPU power limit **396 W**, 15 workers: two alternating before/after pairs
using the BF16-PLE UD-Q4_K_XL pack, one adapter, GPU-0 vision, FP16 KV, context 262,144 and all 24,576 experts in VRAM.
These are local development measurements, not headline benchmarks.

| Startup measure, 396 W | Before | Background PLE packing |
|---|---:|---:|
| PLE open on the startup thread | 29.9–32.0 s | 11.1–15.3 s |

Background PLE packing completed before server readiness in both runs. On-demand behavior is covered by concurrent
reader parity tests, not by these timings.
