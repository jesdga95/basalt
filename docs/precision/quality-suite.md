# Quality suite

The quality suite is how Basalt measures the quality of a model's weights, kernels and options: mean KLD, top-1 agreement and
the PPL ratio against a stored BF16 reference, on chat conversations, through the same decode path that serves requests. Its scripts
are in `tools/quality_suite/`. The KLD numbers on some model pages come from [wikitext KLD](quality.md#wikitext-kld).

## What it is

- **Corpus:** 417 conversations of 416,134 tokens from public datasets with permissive licences. The prompts are fixed (sources, ids,
  seeds and token caps are in `gen.py`), so every model is scored on the same prompts. Only the model that writes the assistant turns
  changes.
- **Assistant turns:** the BF16 model writes them itself, greedy, through vLLM. A conversation is text the model would produce, which
  is the regime the scores are meant to describe.
- **Reference:** a vLLM `prompt_logprobs` pass over the finished conversations stores, for every scored position, the likeliest
  tokens with their fp32 log-probabilities, and the leftover probability of all the others.
- **Test:** the model under test is run over the same token streams (teacher-forced) and its log-probabilities at the stored ids are
  compared with the reference's.

| Kind | Conversations | Source |
| --- | --- | --- |
| chat | 150 | `HuggingFaceH4/ultrachat_200k` (MIT), half with a second user turn |
| code | 80 | `ise-uiuc/Magicoder-OSS-Instruct-75K` (MIT) |
| reasoning | 60 | `openai/gsm8k` (MIT), "Think step by step." |
| multilingual | 70 | `OpenAssistant/oasst1` (Apache-2.0), non-English root prompts in 20 languages |
| tools | 50 | `glaiveai/glaive-function-calling-v2` (Apache-2.0), rendered with the checkpoint's tool format |
| long | 7 | `abisee/cnn_dailymail` 3.0.0 (Apache-2.0), 8K to 31K tokens: many articles and one question |

Token ids are exact. Prompt ids come from the chat template (thinking off, so the empty thought header is part of the prompt) and the
generated ids are appended as vLLM produced them; nothing is re-tokenized. Assistant turns are capped at 500 to 900 new tokens, so
some end without the end-of-turn token. A tool conversation keeps the model's tool call and appends the dataset's canned tool
response before the model continues.

## Design

- **Assistant rows only.** Every token has a role: assistant (generated, including the end-of-turn token and the tool-call stop
  token), other (system, user, template markup, injected tool responses) or BOS. The headline numbers use the assistant rows. Other
  rows behave like wikitext (text the model did not write); they are stored but flagged, and not for gating.
- **Honest error bars.** Rows inside a conversation are correlated, so a per-token standard error is far too small. The scorer
  bootstraps whole conversations; its error bar is several times the naive one.
- **Paired A/B mode.** Two packs scored on the same rows are compared row by row, and the paired error bar of the difference is
  tighter than the two unpaired ones.
- **Truncation bias.** The file stores each row's tokens in descending order until the cumulative probability reaches 1 - 1e-6 (at
  most 256 tokens), and the rest as one leftover bucket on both sides of the KLD. The cut drops a little probability mass, which
  biases the KLD low. Flat text has high-entropy rows that hit the 256 cap, so the bias there is large; assistant text is
  peaked, and the bias is a small fraction of the error bar. That is why the suite scores assistant rows and keeps a
  [bias check](#bias-check).

## What it reports

`score.py` prints, over the chosen rows, each with a conversation-bootstrap error bar:

- mean KLD of reference to test, and its median, p90, p95, p99, p99.9 and maximum;
- top-1 agreement ("same top"), and the mean and RMS change in probability of the reference's top token;
- PPL of the test, PPL of the reference, ln of their ratio and the ratio, over the rows whose next token is among the stored ids.

The test model's input is its log-probability at the stored ids, its leftover (1 minus the sum of probabilities at those ids) and its
argmax; that is all the KLD with the leftover bucket needs.

## The reference file

A safetensors file: `tokens` (int32), `roles` (uint8), `conv_off` (conversation offsets), `row_off` (row offsets), `topk_ids` (int32),
`topk_lp` (fp32, descending per row) and `leftover` (fp32). Row i is the distribution over token i+1 given tokens 0 to i, and exists
when token i+1 has a role above 0. The metadata holds the vLLM version, the generation and extraction settings (dtype, CPU offload,
context length, logprobs mode, cut), the model path, the SHA-256 of the checkpoint index, and the conversation ids and sources.

`slim_ref.py --roles 1` keeps the assistant rows only (about a fifth of the full file); a cut of 1e-5 halves that again with no
measurable bias on this corpus. The scorers take either.

## Build a reference

One NVIDIA GPU and the BF16 checkpoint. vLLM is the only reference engine, and it is not a Basalt dependency: install it only to build
a reference. A model that does not fit the card runs with CPU offload; the settings are recorded in the file's metadata.

The scripts download nothing and the repository ships no data. The sources are public datasets under MIT or Apache-2.0
(the table above); fetch them with the hf CLI into one directory, in the layout `gen.py` reads:

```sh
D=data
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --include "data/test_sft*" --local-dir $D/ultrachat
hf download ise-uiuc/Magicoder-OSS-Instruct-75K --repo-type dataset --include "data-oss_instruct-decontaminated.jsonl" --local-dir $D/magicoder
hf download openai/gsm8k --repo-type dataset --include "main/test*" --local-dir $D/gsm8k
hf download OpenAssistant/oasst1 --repo-type dataset --include "data/train*" --local-dir $D/oasst1
hf download glaiveai/glaive-function-calling-v2 --repo-type dataset --include "glaive-function-calling-v2.json" --local-dir $D/glaive
hf download abisee/cnn_dailymail --repo-type dataset --include "3.0.0/validation*" --local-dir $D/cnn
```

```sh
pip install vllm transformers pandas safetensors numpy      # vLLM 0.34.0 built the published references
python3 tools/quality_suite/gen.py --model CHECKPOINT_DIR --data $D --out WORK [--offload-gb N] [--max-model-len 34000] [--max-seqs 320]
python3 tools/quality_suite/extract.py WORK/corpus.jsonl WORK/ref.safetensors --model CHECKPOINT_DIR [--offload-gb N]
python3 tools/quality_suite/slim_ref.py WORK/ref.safetensors WORK/ref_assistant.safetensors --roles 1
```

`--offload-gb` is vLLM's `cpu_offload_gb`, `--max-seqs` its `max_num_seqs`. A mixture-of-experts model needs a vLLM MoE backend that
avoids the fused-MoE JIT (it can start many compiler processes and fill RAM): `--moe-backend triton`. A dense model needs none.
Generation is limited by the KV cache and, with offload, by PCIe; extraction is prefill only and runs faster per token.
`extract.py --model-name` and `--model-sha` set the name and checkpoint hash recorded in the metadata.

The corpus builder and the role masks read Gemma 4's chat markup (turn, thought-channel and tool-response tokens, and its tool-call
format). Another model family needs its own markup in `common.py` and `gen.py` before it can have a reference.

## Score a Basalt pack

Run the quality suite directly through `basalt-bench`:

```sh
basalt-bench --model PACK.basalt --quality ref.safetensors [--smoke] [--quality-top 256] [--quality-role assistant] [--quality-max-context 5120] [--quality-boot 2000]
```

Alone, `--quality` runs only the quality suite; with `-p` or `-t` it runs after them. `--quality-max-context` (default 5,120) sets
`--max-context` for the engine (unless overridden after `--`) and skips conversations longer than that; raise it to cover the long
ones. Add `-o json` for machine-readable output under the `"quality"` key.

Under the hood, the engine scores each conversation itself: `--ppl-out FILE` names the rows file, and `GEN 1 score=N score_top=K IDS`
on `--serve` teacher-forces the whole stream (ids 0 to N are the prompt and every later row is written with its K likeliest ids, K in
0..256; 0 writes the whole row, binary). For offline scoring or paired A/B comparisons, `basalt_score.py` and `score.py` drive the
same protocol and save per-conversation row files:

```sh
python3 tools/quality_suite/basalt_score.py --pack PACK.basalt --ref ref.safetensors --out DIR [--smoke]
python3 tools/quality_suite/score.py ref.safetensors --basalt DIR/conv_*.txt [--smoke]
```

Which models: any model whose session returns head logits (`copy_logits`), which every model in the tree does. The request goes
through the shared serving loop and the same teacher-forced path as the wikitext gate (`ppl=K`), so the model plugs in nothing of
its own. The stream is forced token by token, with no drafting. A stream with a token outside the vocabulary, a scored position
outside the stream, a `score_top` above 256 or an image in the request is refused with an `ERR` line.

## Score a GGUF

`tools/quality_suite/llama_project.cpp` links against a llama.cpp build and projects a GGUF onto the reference's token streams. It is
not part of Basalt's build. Build llama.cpp as shared libraries with CUDA, then, from a checkout `$LLAMA` and its build directory
`$LLAMA/build`:

```sh
g++ -O2 -std=c++17 tools/quality_suite/llama_project.cpp -o llama_project -I$LLAMA/include -I$LLAMA/ggml/include \
    -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin
```

Use the llama.cpp commit that supports the model's architecture; the projection is only as good as that port.

```sh
python3 tools/quality_suite/make_req.py ref.safetensors req.bin [--smoke]
llama_project project MODEL.gguf req.bin test.bin 99 2048
python3 tools/quality_suite/score.py ref.safetensors test.bin [--role assistant|other|all] [--boot 2000] [--json out.json]
```

`project` writes, per requested row, the log-probability of every stored id, the leftover and the argmax over the full vocabulary.

## Paired A/B

```sh
python3 tools/quality_suite/score.py ref.safetensors a.bin --test2 b.bin
```

Reports A, B and A minus B on the same positions, with the paired and the unpaired error bar. A difference smaller than the paired
error bar is noise.

## Bias check

`llama_project check A.gguf B.gguf req.bin out.bin` runs two models in one engine over the same rows and computes their KLD over the
full vocabulary and with the stored cut (caps 64, 256, 1024 and 4096). The truncation bias is the gap; it must be a small fraction of
the conversation-bootstrap error bar before a reference is released. `analysis.py` repeats the check from the stored file and
measures the error bars at smaller subsamples.

## Smoke slice

`--smoke` (in `basalt-bench --quality`, `score.py`, `make_req.py` and `basalt_score.py`) selects every 20th conversation shorter than 5,000 tokens: 21
conversations, 14,226 tokens, 11,864 assistant rows, a mix of chat, code, reasoning, multilingual and tool conversations. The test inputs
must cover exactly those conversations. It is a quick check of a change, not the gate. `gen.py --smoke` writes a corpus of two
conversations per kind (no long ones) to try the settings and the speed.

## Gate

A change to numerics is judged by the bar in [contributing](../../CONTRIBUTING.md): the full-set assistant-row KLD and top-1 are the
same or better than the previous build's, within the error bar (use the paired mode). Report the KLD, top-1 and PPL ratio with their
error bars, the reference file's model and checksum, and the command.
