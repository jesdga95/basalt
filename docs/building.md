# Building and testing

The build commands are in the [README](../README.md). Windows: [platforms](platforms.md).

## What is in the repository

- `src/`, `include/`: the engine (`basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision`).
- `serve/`: the Python server with the OpenAI and Anthropic API and the stats page.
- `tools/`: the converter (`tools/convert/convert.py`), the `.basalt` reader (`model_file.py`), the quality gate (`ppl_gate.py`), the
  greedy-identity check and the tokenizer.
- `tests/`: parity and unit tests. `data/`: expert profiles (`data/profiles`) and benchmark fixtures (`data/bench`).

The build fetches nothing. The ggml and llama.cpp parts Basalt uses (quantization formats, CPU dot products, MMQ) are in `src/` and
`include/basalt/ggml/`, under the MIT licence in `THIRD_PARTY_LICENSES`.

`-DWERROR=ON` makes project warnings errors (CI sets it on Linux).

## Tests

Configure with `-DBUILD_TESTING=ON`, build, and run `ctest --test-dir build`.

- Kernel parity tests need the GPU. `expert_parity` and `pool_test` use a synthetic fixture; `--file experts.bin` checks real blobs.
- Server tests: `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`.
- Tokenizer parity: register `tokenizer_parity` with `-DBASALT_TOKENIZER_DIR=<a pack's tokenizer/ directory>`.

`basalt-bench` reads its prompts from `data/bench` (run from the repository root, or pass `--fixtures DIR`).
