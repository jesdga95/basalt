<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png">
    <img src="assets/basalt-lockup-light.png" alt="Basalt" width="480">
  </picture>
</p>

> **Native Windows x64 support** lives in this tree: the same engine, kernels and precision rules, plus a platform
> layer for MSVC and CUDA 13. Build steps, the test recipe and the platform caveats are in
> **[docs/WINDOWS.md](docs/WINDOWS.md)**. Every Windows change sits behind `#if defined(_WIN32)` /
> `if(WIN32)` guards: the Linux code path is unchanged.

Basalt is a local inference engine for the **Qwen3.8-Flash-Next** mixture-of-experts model. It targets one thing: an
NVIDIA Blackwell GPU (sm_120) under Linux or Windows x64. The experts that do not fit in VRAM live in system RAM (and optionally on
a second GPU used as a peer expert tier), so a large MoE model runs on a single workstation.
Ready-to-run `.basalt` packs: see [Models](#models).

It is a fork of [Strata](https://github.com/Niko1221/Strata) (MIT), stripped down to that one target. See
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

<p align="center">
  <img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720">
</p>

<p align="center"><sub>The web UI at <code>/</code> while a reply streams: live decode speed, draft accept rate, context, GPU load and the request log.</sub></p>

## What is in the repository

- `src/`, `include/`: the C++/CUDA engine (one binary, `basalt`).
- `serve/`: a Python server with an OpenAI- and Anthropic-compatible API, and a stats page.
- `tools/convert/`: the model-file converter (`convert.py`), its section builders and the drafter fetcher (`mtp_fetch.py`).
- `tools/`: the .basalt reader and writer (`model_file.py`), the PPL/KLD quality gate (`ppl_gate.py`), the greedy-identity
  check and the tokenizer.
- `tests/`: parity and unit tests.

## Requirements

- Linux x86-64 with AVX2 (AVX-512 is used when present).
- Windows x64 builds natively too (MSVC 2022 + CUDA 13, same sm_120 requirement); the working toolchain, the build
  recipe and the test caveats are in [docs/WINDOWS.md](docs/WINDOWS.md).
- An NVIDIA Blackwell GPU (compute capability 12.0) and CUDA 13.
- CMake 3.24 or newer and a C++20 compiler.
- Enough system RAM to hold the experts that do not fit in VRAM.
- Python 3.10 or newer for the server: `pip install -r requirements.txt`.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j        # bin/basalt, bin/basalt-bench, bin/basalt-vision
```

### Tests

The tests are off by default. Configure with `-DBUILD_TESTING=ON`, build, then run `ctest`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build
```

The kernel parity tests need the GPU; the file-format, cache and memory unit tests do not. `expert_parity` and `pool_test`
need local expert data (`experts.bin` in the source directory, not shipped), and `platform_memory_test` needs a locked-memory
limit raised first (`ulimit -l`); without that, those three fail.

`basalt-bench` reads its pre-tokenized prompts from `data/bench` (token ids for the Qwen3.8-Flash-Next tokenizer: the
three decode prompts and a prefill filler); run it from the repository root or pass `--fixtures DIR`. `basalt-bench
--help` says how the files are made.

The build needs nothing outside this repository and fetches nothing. The parts of ggml and llama.cpp that Basalt uses
(quantization formats, CPU dot products, the prompt path's MMQ kernels) are extracted into `src/` and `include/basalt/ggml/`,
with their MIT licence in `THIRD_PARTY_LICENSES`.

## Run

```sh
bin/basalt serve --model /path/to/model.basalt --host 127.0.0.1 --port 8080 [--alias NAME]
```

The server listens on 127.0.0.1 by default; do not expose it beyond that without `--api-key`.

### Settings: `--config`

The engine's built-in defaults are untuned (4096 context, fp16 KV, no speculation, no expert cache), and the model
file carries no engine flags. Put every setting in one `basalt_config.json` and pass `--config`:

```json
{
  "server": {
    "model": "model.basalt",
    "alias": "Qwen 3.8 FN",
    "gpu": "0,1",
    "host": "127.0.0.1",
    "port": 8080
  },
  "engine": {
    "expert-cache": "auto",
    "prefill": "auto",
    "spec": 7,
    "spec-min-p": 0.5,
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "vram-reserve-mib": 700
  }
}
```

```sh
bin/basalt serve --config basalt_config.json                     # everything from the file
bin/basalt serve --config basalt_config.json --max-context 32768  # the command line wins
```

### Concurrent requests: `--parallel N`

`--parallel N` (1..8, also `"parallel"` in the config's `server` section) serves N requests at once in the
engine's batch slots; the default 1 is the single-stream path. Further requests wait in arrival order (a later one
may go first only to resume a slot that holds its conversation, a few times at most). A slot is freed when its
request finishes, is cancelled or its client leaves. `/metrics` reports `live.slots` (`total`, `active`,
`queue_depth`). As in llama.cpp, `--max-context` is shared: each slot holds `--max-context / N` tokens (`--parallel 4
--max-context 262144` gives four slots of 65536; ask for `--max-context 1048576` for four of 262144). The solo
path, used while one request runs alone, has the same length, so a request that fits a slot never outgrows it, and
one that does not is refused with a 400 naming that length. `/slots`, `/props` and `/v1/models` report the context
each slot has. `--slot-full-context` (also `"slot-full-context"` in the config's `server` section) gives every slot
the whole `--max-context` instead, at N times the KV cache; the startup line says so. The slots' KV cache, and one
solo session of the same length, come out of the expert cache.

On one GPU (not with `--kv-resident` or `--vram-elastic`) the KV cache is elastic: each session backs its first 32768
cells at startup (the K/V pools, the indexer's block keys, the RoPE tables and the draft layer's K/V alike), and the
rest of that VRAM holds the expert cache's last slots until a request needs it. A request backs its prompt plus 16384 cells before the prompt is read (and 16384 more whenever decode runs past that) by taking
those slots back; their experts become CPU misses read from their RAM copies, which the RAM copy keeps for exactly this.
The next request that needs less gives the cells back and the cache refills the slots from RAM. A short request so
decodes with the cache a 32k context would have, whatever `--max-context` is. The startup line `elastic KV: ...` says
how much of the cache is lent this way, and each move is logged (`elastic KV: N cells backed ...`).

While other slots are decoding, a new prompt is read beside them: the prompt path runs on its own thread, the slots'
decode windows keep running (their kernels take the higher stream priority), and the few steps that touch the draft
layer or the output run between two windows. A long prompt therefore does not stop the other requests. A prompt that
arrives while no other slot is decoding is read as in the single-stream path.

At startup the server prints one `[server] kv cache:` line per device with all the session state placed there, for
example `12.5 GiB on cuda2 (solo 262144 tokens + 2 slots x 262144 tokens + draft K/V 0.5 GiB)`. With elastic KV (the
default on one GPU) it reads `0.6 / 4.4 GiB on cuda0 (elastic)`: the memory backed at startup, and what the cache can
grow to at `--max-context`. Because that memory is
taken from the expert cache, the engine warns (`basalt warning: the N batch slots' sessions ... take X GiB of CUDA0
...`) when it leaves the cache short of the expert set, or at under half of what it would hold without the slots, and
names the remedies: a smaller `--max-context`, no `--slot-full-context`, fewer `--parallel` slots. If a slot's session
does not fit at all, the engine says which slot (`WARNING: --batch N: slot K's session does not fit`) and starts with
the slots that did fit; if fewer than two fit, `--batch is off` and the server runs one request at a time.

- `server` holds the server's options (`python -m serve.server --help`), named as the flags without the dashes:
  strings or numbers, `true`/`false` for a switch (`lazy`, ...), a list for the repeatable ones
  (`allowed-host`, `cors-origin`). `model` and `tokenizer` paths are relative to the file.
- `engine` holds engine flags (`bin/basalt --help`), the same way: a string or number, `true` for a switch, `false`
  to leave it off. Any flag the engine accepts works, on the command line as well (`--kv int8`, `--resident-experts`).
  A name the engine does not accept stops the server at startup, as does one in the wrong section.
- Precedence: command line > config file > built-in defaults. `spec`, `spec-min-p`, `mtp-max-t`, `peer-exclusive` and
  `max-context` are engine flags; they are also server flags on the command line.
- The API key is command line only (`--api-key`); a config file that sets it is refused.
- `"relaxed-drafting": 0.3` (or `true` for the default 0.5; `--relaxed-drafting [P]` on the command line) trades
  exactness for speed under sampling: a draft the model gives probability P or more is kept even when the model's own
  draw would have been another token. Lower P is faster and further from the model's true sampling distribution; the
  measured speed and cost per value are in [docs/precision.md](docs/precision.md#relaxed-drafting). Off by default:
  without it, speculative decoding returns exactly what plain sampling would. Greedy (temperature 0) requests,
  `logit_bias`, JSON `response_format`, control tokens and everything inside a tool call stay exact. A request can set
  its own value with `"relaxed_drafting": P` (or `false` for exact). Single-stream decoding only; `--parallel` slots
  stay exact.
- The startup line `[server] engine: ...` lists the settings that came from the file or the command line; without
  `--config` it says the built-in defaults apply.

### Prompt chunks: `--prefill`

A prompt is read in chunks. `--prefill auto` (the usual setting) takes the largest chunk whose buffers the expert cache
can lend: the buffers are carved from the cache's last slots for the length of the prompt and the experts in those slots
are copied back afterwards, so a short prompt lends only what it needs. A bare `auto` reads a prompt in the fewest equal chunks of at most 16384 tokens, because
a chunk's cost per token stops falling past a few thousand tokens; it is still limited to what the lend can fund, which with
`--resident-experts` is the lent slots whose experts the RAM copy holds plus a fixed amount of file reading (7 GiB), and without
`--resident-experts` is 8192. `--prefill auto:N` (N = 8192, 16384 or 32768) sets the ceiling yourself; a number sets a
fixed chunk. A prompt longer than one chunk is split into equal chunks (a short tail chunk costs nearly a whole pass over
the streamed experts). The startup line `prompt chunk auto: N tokens` says what was
chosen.

### Memory floors

Basalt refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below 256 is refused at startup: with less VRAM free once everything is loaded, a later
  allocation can make the driver page GPU memory and requests stall.
- When some experts are read from the SSD (they fit in neither the GPUs nor the RAM copy), the RAM copy leaves a
  floor free for the OS file cache: the RAM margin (2 GiB or 5% of the RAM, the larger) plus 3.5 GiB plus two layers'
  experts, which is what one prompt chunk reads from the file. The startup line `RAM floor ...` gives the figures.
  When every expert has a place, only the margin is kept, as before.
- A server's RAM copy also leaves room for what serving takes after startup: the conversation checkpoints (each the
  model's whole running state, `--prompt-cache` of them per slot plus one), `--conversation-cache-mib`, one image's
  buffers and a request's own buffers. The startup line `RAM ... free, guard ...` gives the figures.
- An explicit `--resident-budget-gib` that would leave less than that floor free is refused at startup; the error
  states the RAM free and the largest safe value ('max').
- While serving, a request that finds MemAvailable below the margin plus 3.5 GiB (the margin alone when no expert is
  read from the SSD) stops with an error (`RAM low (... free < ... guard): request stopped`, also returned to the
  client); the server keeps running and takes the next request.

### Vision: `--vision`

Image input is one server option, `--vision off | on | peer | dynamic | cpu` (config: `"vision": "peer"` in the
`server` section). It is `off` by default. The encoder is Basalt's own and shares its kernels in every mode, so all
five give bitwise identical embeddings.

| mode | where the encoder runs | VRAM cost | fresh-image latency (IQ3_S, 400 W) |
| --- | --- | --- | --- |
| `off` | no image input | none | |
| `on` | sidecar `bin/basalt-vision` on the main GPU, always loaded | about 2.2 GiB of the main card's expert cache | TTFT 0.77 s at 1k image tokens, 1.4 s at 3k, 1.8 s at 4k |
| `peer` | the same sidecar on the second `--gpu` card | about 1.8 GiB of the peer card's cache, none on the main card | 0.86 s, 2.1 s, 2.9 s |
| `dynamic` | inside the engine, on the main GPU | none at rest (0.87 GiB of pinned RAM) | 0.77 s, 1.45 s, 1.8 s |
| `cpu` | sidecar on the CPU | none | about 6 s at 1k, 40 s at 3k, 70 s at 4k |

`dynamic` borrows up to 1.55 GiB of expert-cache slots per image (the prompt path's lend), uploads the weights from the
pinned copy (about 30 ms), encodes on a side stream, and refills the same slots from RAM (about 60 ms), so the cache is
identical afterwards. Other `--parallel` slots pause for the encode. An image already seen is served from the
encoder cache in tens of milliseconds in every mode. `peer` on a single GPU warns and loads the encoder on that GPU;
with two cards, `on` says so at startup and names `peer` as the way to put it on the second card.

```json
{ "server": { "gpu": "0,1", "vision": "peer" } }
```

`on` and `peer` use the sidecar, which the main build also produces (`tools/vision` also builds on its own); the
server looks for it next to `bin/basalt`. `dynamic` needs no sidecar.

The image encoder is Basalt's own, written for this model's vision tower (no llama.cpp or ggml): bf16 tensor-core
products and flash attention with fp32 accumulation on the GPU, AVX-512/AVX2 on the CPU. It replaces llama.cpp's
mtmd encoder as a drop-in, is faster than it at every image size, and stays closer to an fp32 reference: mtmd's fp16
attention drifts on large images, ours does not.

## The `.basalt` model file

A model is one file. It holds everything the engine, the server and the vision sidecar load: the weights, the
tokenizer and chat template, the MTP drafter, the expert profile, the 8-bit hyper-connection copies, the vision
projector, and the default sampling. It carries no engine flags: those come from `--config` or the command line.

- **Layout:** a 64 KiB header (magic `BASALTMF`, version, a table of up to 511 named sections, each with its offset,
  size and SHA-256), then the sections, each aligned to 2 MiB so it can be memory-mapped or read with `O_DIRECT`.
- **Sections:** `meta` (JSON: geometry, the source GGUF metadata, every tensor's type, shape and offset,
  sampling, provenance and license), `tokenizer/*`, `dense`, `experts` and `experts.layout` (the routed
  experts, laid out per layer), `gguf` (every non-expert source tensor), `mtp/*` (the drafter), `profile`, `hc8` and
  `vision/*`. The authoritative description is [include/basalt/core/model_format.hpp](include/basalt/core/model_format.hpp).
- **No requantization:** tensors are copied byte for byte from the source GGUF. The only exception is
  `--compat-bf16`, for GGUFs that store the small projections the engine reads as BF16 in another type; it converts
  them and records that in `meta`.

Build one from the source GGUF shards (give shard 1; the others are found beside it):

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--profile FILE] [--hc8 FILE] [--compat-bf16] [--name NAME]
```

`python3 tools/convert/convert.py --help` describes every input.

## Models

Ready-to-run `.basalt` packs are on [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt). A pack
you convert yourself will be significantly slower than the official `.basalt` files.

Quality measured with `basalt-bench --quality`: 20 chunks of wikitext-2 (512 tokens each), scored through the decode
path against the Q8_0 reference. Lower KLD and higher top-1 agreement are closer to Q8_0. How precision is handled and how quality is measured:
[docs/precision.md](docs/precision.md).

| model | source | file | experts | PPL | KLD mean | same top-1 |
|---|---|---:|---:|---:|---:|---:|
| GSQ-RCO-Coder IQ1_M (256 experts) | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 57.4 GiB | 23.4 GiB | | 1.03 | 69% |
| GSQ-RCO Q2_0 | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 64.9 GiB | 31.6 GiB | 2.971 | 0.486 | 80.6% |
| GSQ-RCO IQ3_XXS | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 74.4 GiB | 40.0 GiB | 2.535 | 0.277 | 86.8% |
| UD-Q2_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 76.5 GiB | 42.9 GiB | 2.755 | 0.323 | 84.3% |
| UD-IQ3_XXS | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 79.4 GiB | 45.3 GiB | 2.564 | 0.244 | 87.0% |
| GSQ-RCO IQ3_S | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 81.0 GiB | 46.8 GiB | 2.387 | 0.177 | 89.4% |
| UD-Q3_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 86.8 GiB | 52.0 GiB | 2.457 | 0.160 | 89.3% |
| UD-Q4_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 106.7 GiB | 71.7 GiB | 2.321 | 0.068 | 93.0% |
| Q8_0 | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 178.3 GiB | 119.5 GiB | 2.282 | reference | |

The IQ3_S, IQ3_XXS and Q2_0 packs carry an MTP layer fine-tuned on the model's own output
(self-distillation), which raises draft acceptance. Output is unchanged, because verify decides every token. Q4_K_XL
and Q8_0 keep the original MTP layer. `--mtp-dense FILE` embeds that tuned layer (the `mtp/dense` section's raw bytes, checked against the
source layout) and records it as self-distilled, with its SHA-256, in `meta.mtp`. The tuned layer ships with the
packs, inside the model file; the converter input is only needed to build a pack yourself.

The model files are derived from Qwen3.8-Flash-Next and are distributed under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), not Basalt's MIT
license. The GSQ-RCO quants come from IST Austria's Deep Algorithms and Systems Lab; please cite the
[GSQ](https://arxiv.org/abs/2604.18556) and [RCO](https://arxiv.org/abs/2605.00649) papers if you use them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
