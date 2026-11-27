# Quality suite

The quality suite scores a model on chat conversations against a stored BF16 reference. It is for instruction-tuned models, where the
wikitext gate of [How quality is measured](quality.md) does not describe use, and it reports the same statistics (KLD, top-1, PPL).
Its scripts are in `tools/quality_suite/`.

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

## Why it replaces the wikitext gate for chat models

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

One NVIDIA GPU and the BF16 checkpoint. vLLM is the only reference engine. A model that does not fit the card runs with
`cpu_offload_gb`; the settings are recorded in the file's metadata.

```sh
pip install vllm transformers pandas safetensors numpy      # vLLM 0.34.0 built the published references
# download the six datasets above with the hf CLI into the data directory named at the top of gen.py
python3 tools/quality_suite/gen.py                           # corpus.jsonl: assistant turns by vLLM, greedy
python3 tools/quality_suite/extract.py corpus.jsonl ref.safetensors
python3 tools/quality_suite/slim_ref.py ref.safetensors ref_assistant.safetensors --roles 1
```

`QSV2_MODEL` selects the model in `common.py`; `QSV2_OFFLOAD_GB`, `QSV2_MAXLEN` and `QSV2_MAX_SEQS` size the offload, the context and
the batch. A mixture-of-experts model needs a vLLM MoE backend that avoids the fused-MoE JIT (it can start many compiler processes
and fill RAM); a dense model needs none. Generation is limited by the KV cache and, with offload, by PCIe; extraction is prefill only
and runs faster per token.

## Score a Basalt pack

The engine scores a token stream itself. `--ppl-out FILE` names the rows file, and `GEN 1 score=N score_top=K IDS` on `--serve`
teacher-forces the whole stream: ids 0 to N are the prompt and every later row (the distribution over the next token) is written with
its K likeliest ids (K is 0 to 256; 0 writes the whole row, binary). `basalt_score.py` drives it, one request per conversation:

```sh
python3 tools/quality_suite/basalt_score.py --pack PACK.basalt --ref ref.safetensors --out DIR [--smoke]
python3 tools/quality_suite/score.py ref.safetensors --basalt DIR/conv_*.txt [--smoke]
```

The script's `--max-context` (default 5,120) skips the longer conversations; raise it to cover the long ones.

## Score a GGUF

`tools/quality_suite/llama_project.cpp` links against a llama.cpp build and projects a GGUF onto the reference's token streams:

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

`--smoke` (in `score.py`, `make_req.py` and `basalt_score.py`) selects every 20th conversation shorter than 5,000 tokens: 21
conversations, 14,226 tokens, 11,864 assistant rows, a mix of chat, code, reasoning, multilingual and tool conversations. The test inputs
must cover exactly those conversations. It is a quick check of a change, not the gate. `gen.py --smoke` writes a corpus of two
conversations per kind (no long ones) to try the settings and the speed.

## Gate

A change to numerics is judged by the bar in [contributing](../../CONTRIBUTING.md): the full-set assistant-row KLD and top-1 are the
same or better than the previous build's, within the error bar (use the paired mode). Report the KLD, top-1 and PPL ratio with their
error bars, the reference file's model and checksum, and the command.
