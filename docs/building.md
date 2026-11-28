# Building and testing

The build commands are in the [README](../README.md). Windows: [platforms](platforms.md).

## What is in the repository

- `src/`, `include/`: the engine (`basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision`).
- `serve/`: the Python server with the OpenAI and Anthropic API and the stats page.
- `tools/`: the converter (`tools/convert/convert.py`), the `.basalt` reader (`model_file.py`), the quality suite (`quality_suite/`), the wikitext KLD gate (`ppl_gate.py`), the
  greedy-identity check and the tokenizer.
- `tests/`: parity and unit tests. `data/`: expert profiles (`data/profiles`) and benchmark fixtures (`data/bench`).

The build fetches nothing. The ggml and llama.cpp parts Basalt uses (quantization formats, CPU dot products, MMQ) are in `src/` and
`include/basalt/ggml/`, under the MIT licence in `THIRD_PARTY_LICENSES`.

`-DWERROR=ON` makes project warnings errors (CI sets it on Linux).

## Building only some models

`-DBASALT_MODELS=<ids>` is a semicolon list of architecture ids, or `all`, which builds every model, including any added later. The option is required: a configure that leaves it out stops and says so. A model left out is not compiled at
all: its C++ and CUDA sources, its kernels and its tests are skipped, which is most of the build time for DeepSeek.

| Model | Id |
| --- | --- |
| Qwen3.8-Flash-Next | `qwen4exp` |
| DeepSeek-V4.1-Flash | `deepseek41` |
| Qwen3.8-27B | `qwen35` |
| Hy4-flash | `hy4flash` |
| Inkling 2 Mini | `inkling2` |
| ERNIE-5.5-Flash | `ernie55` |

```sh
cmake -S . -B build -DBASALT_MODELS=qwen4exp                  # Flash-Next only
cmake -S . -B build -DBASALT_MODELS=deepseek41                # DeepSeek only
cmake -S . -B build -DBASALT_MODELS=qwen35                    # the 27B only
cmake -S . -B build -DBASALT_MODELS=hy4flash                  # Hy4-flash only
cmake -S . -B build -DBASALT_MODELS=inkling2                  # Inkling 2 Mini only
cmake -S . -B build -DBASALT_MODELS=ernie55                   # ERNIE-5.5-Flash only
cmake -S . -B build "-DBASALT_MODELS=qwen4exp;deepseek41"     # a list; quote the semicolon
```

An unknown id, an empty value or no value stops the configure and names the valid ids. A pack of a model the build left out fails at load with
`architecture is 'deepseek41', a Basalt model this build left out (built with BASALT_MODELS=qwen4exp); rebuild with cmake
-DBASALT_MODELS="qwen4exp;deepseek41;qwen35" to load it`. The value is kept in the build folder's cache: to change it, pass the option again (`-DBASALT_MODELS=all` goes back to every
model).

## Tests

Configure with `-DBUILD_TESTING=ON`, build, and run `ctest --test-dir build`.

- Kernel parity tests need the GPU. `expert_parity` and `pool_test` use a synthetic fixture; `--file experts.bin` checks real blobs.
- Server tests: `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`.
- Tokenizer parity: register `tokenizer_parity` with `-DBASALT_TOKENIZER_DIR=<a pack's tokenizer/ directory>`.

`basalt-bench` reads its prompts from `data/bench` (run from the repository root, or pass `--fixtures DIR`).
