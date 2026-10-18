<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

> **Windows x64 is community-supported**: the same engine, kernels and precision rules, with the
> Windows-specific code kept in a platform layer (MSVC + CUDA 13). Linux is the supported target and
> the regression oracle; the Windows tier promises a build on one GPU, not performance. Build steps,
> the test recipe and the platform caveats are in **[docs/WINDOWS.md](docs/WINDOWS.md)**.

Basalt is a local inference engine for the **Qwen3.8-Flash-Next** mixture-of-experts model.
It runs on Linux with an NVIDIA Blackwell GPU (sm_120); Windows x64 is community-supported.

Experts that do not fit in VRAM live in system RAM, and optionally on a second GPU used as a peer expert tier.
A large MoE model runs on one workstation, behind an OpenAI- and Anthropic-compatible API.

<p align="center"><img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720"></p>

## Requirements

- Linux x86-64 with AVX2 (AVX-512 is used when present), and an NVIDIA Blackwell GPU (compute capability 12.0) with CUDA 13.
- Windows x64 builds natively as a community-supported target (MSVC 2022 + CUDA 13, same sm_120 requirement, one GPU);
  the working toolchain, the build recipe and the test caveats are in [docs/WINDOWS.md](docs/WINDOWS.md).
- CMake 3.24 or newer, a C++20 compiler, and Python 3.10 or newer.
- Enough system RAM to hold the experts that do not fit in VRAM.

## Quick start

```sh
git clone https://github.com/jesdga95/basalt && cd basalt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-vision
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4.basalt --local-dir models
```

Put your settings in `basalt_config.json`, with `model` pointing at the downloaded file:

```json
{
  "server": { "model": "models/Qwen3.8-FN-NVFP4.basalt", "gpu": "0,1", "port": 8080 },
  "engine": { "expert-cache": "auto", "prefill": "auto", "kv": "int8", "resident-experts": true }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1 by default.

## Models

<p align="center"><img src="docs/assets/basalt-format.png" alt="Inside a .basalt file" width="800"></p>

Ready-to-run packs, with the quality of each measured against Q8_0, are on
[Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt):
`Qwen3.8-FN-NVFP4`, `-UD-Q4_K_XL`, `-UD-Q5_K_XL`, `-UD-Q6_K_XL`, `-Q8_0`, `-GSQ-RCO-IQ3_S`, `-GSQ-RCO-IQ3_XXS` and `-GSQ-RCO-Q2_0` (each `.basalt`).
A model is one file: weights, tokenizer, MTP drafter, expert profile and vision projector.

## Docs

- [Configuration](docs/configuration.md): every setting.
- [Running](docs/running.md): parallel requests, prompt chunks, memory floors, image input.
- [The .basalt model file](docs/model-file.md): the layout, and building a pack yourself (including NVFP4).
- [Windows x64](docs/WINDOWS.md): the community-supported build, its recipe and its caveats.
- [Building and testing](docs/building.md), [Precision and quality](docs/precision.md), [Contributing](CONTRIBUTING.md).

## Credits

- [Strata](https://github.com/Niko1221/Strata): the project Basalt forks (MIT).
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model. The packs are under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), not Basalt's MIT license.
- [unsloth](https://huggingface.co/unsloth): the UD and Q8_0 GGUFs the packs are built from.
- IST Austria DAS Lab: the GSQ-RCO quants; please cite [GSQ](https://arxiv.org/abs/2604.18556) and [RCO](https://arxiv.org/abs/2605.00649).
- [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): the NVFP4 experts.
- [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp): quantization formats and kernels extracted into `src/` (MIT, see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)).

## License

MIT, see [LICENSE](LICENSE).
