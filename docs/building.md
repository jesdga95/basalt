# Building and testing

Clone, requirements and the build commands are in the [README](../README.md). This page has the rest.

## What is in the repository

- `src/`, `include/`: the C++/CUDA engine (one binary, `basalt`).
- `serve/`: a Python server with an OpenAI- and Anthropic-compatible API, and a stats page.
- `tools/convert/`: the model-file converter (`convert.py`), its section builders and the drafter fetcher (`mtp_fetch.py`).
- `tools/`: the .basalt reader and writer (`model_file.py`), the PPL/KLD quality gate (`ppl_gate.py`), the greedy-identity
  check and the tokenizer.
- `tests/`: parity and unit tests.

## Tests

The tests are off by default. Configure the README's build with the extra option `-DBUILD_TESTING=ON`, build, then run `ctest --test-dir build`.

The kernel parity tests need the GPU; the file-format, cache and memory unit tests do not. `expert_parity` and `pool_test`
run on a synthetic expert fixture; pass `--file experts.bin` (the pack's experts section, not shipped) to check real blobs.
`platform_memory_test` locks 256 MiB when the locked-memory limit (`ulimit -l`) allows it and a smaller region otherwise,
and says so; it skips the lock case only when the limit is too low to lock anything.

The server tests run with `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`
(the server's requirements). Two tokenizer tests read the pack's
vocabulary and skip without a pack; the same checks run on a synthetic vocabulary in the same file. The test that runs
the built engine with every flag needs `bin/basalt`.

The engine tokenizes request text itself (`GENT`). `tokenizer_test` checks the C++ tokenizer on a synthetic vocabulary;
`tokenizer_parity` compares its ids with `tools/tokenizer.py` over a corpus, and is registered when cmake gets
`-DBASALT_TOKENIZER_DIR=<a pack's extracted tokenizer/ directory>` (with python3 and the `regex` package; `-DBASALT_TOKENIZER_BIG`
adds `--big FILE...` to the corpus). `tokenizer_bench DIR FILE...` times a cold and a warm encode.

`basalt-bench` reads its pre-tokenized prompts from `data/bench` (token ids for the Qwen3.8-Flash-Next tokenizer: the
three decode prompts and a prefill filler); run it from the repository root or pass `--fixtures DIR`. `basalt-bench
--help` says how the files are made.

The build needs nothing outside this repository and fetches nothing. The parts of ggml and llama.cpp that Basalt uses
(quantization formats, CPU dot products, the prompt path's MMQ kernels) are extracted into `src/` and `include/basalt/ggml/`,
with their MIT licence in `THIRD_PARTY_LICENSES`.

`basalt-bench --community` runs the fixed preset `community-1` for the public table in [community runs](community.md):
the three decode fixtures, a 16384-token prefill and a 65536-token prefill, each once, with int8 KV cache and the
engine's own defaults otherwise (drafting included). The decode fixtures and each prefill run in their own fresh engine
process, with no warm-up pass and no repetition, so every number is the cold first run. It
prints the usual table, adds its row to `docs/community/benchmarks.json` of the checkout the build came from (`--community-file PATH` picks another file) and prints it. Review the row, commit it and open a pull request:

```sh
bin/basalt-bench --community --model Qwen3.8-FN-NVFP4.basalt --submitter "Your Name"
```

`prefill_16k_tps` and `prefill_64k_tps` are the engine tokens per second of those two prompts; `decode_tps` is the generated tokens of conv, struct and prose over their summed decode time (token-weighted). The row
records the engine commit, the GPUs the engine used (main first) with VRAM and enforced power limit from NVML, system
RAM, the OS (`Linux`, `Windows` or `WSL2`), the pack's quant label. Every expert must sit in VRAM, the peer GPU or RAM: if the engine would read any from the SSD, the bench stops before measuring, says how many GiB would spill and roughly how much more free RAM is needed, exits non-zero and prints no row. NVML is required;
without it, or without a git build, the run stops with a message. Only `--model`, `--submitter`, `--fixtures` (the
repository's `data/bench`), `--logs` and `--print-command` are accepted: any other bench option or engine flag changes
the workload and is refused. A second visible GPU is used as the peer tier, as in the plain bench; hide it with
`CUDA_VISIBLE_DEVICES=0` for a single-GPU row. The preset's version changes whenever the workload does.
