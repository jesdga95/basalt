# How quality is measured

Basalt scores a model against a full-precision reference on the same text, through the decode path that generates tokens. Both
models use this method. Their numbers: [Qwen3.8-Flash-Next](qwen38-flash-next.md) and [DeepSeek-V4.1-Flash](deepseek41.md).

## The gate

`basalt-bench --quality` runs the gate. `tools/ppl_gate.py` runs it offline against a built engine ([commands](#reproduce-the-reference)).

- **Text:** wikitext-2 in 512-token chunks. The first 256 tokens of each chunk are prefilled.
- **Scored rows:** 256 to 510, through the decode path in verify windows of `--quality-t` tokens (default 3). The window tokens are the
  reference text, so the drafter's output is not used.
- **Reference:** a llama.cpp KL-divergence base file (`--kl-divergence-base`).
- **KLD** is KL(base || test) over the full vocabulary, from the fp32 log-softmax, summed in double. Base terms at log-probability -16 or
  below are skipped, a tiny underestimate.
- **Reported:** PPL, mean KLD (the tables also give median and p99) and top-1 agreement.
- **Length:** 20 chunks for a quick check. A change to numerics needs 100.
- **Coverage:** the 256-token prefill halves run the MMQ prompt path. The gate does not reach the FP4 prompt path (used from 1,024-token
  chunks) or the split prompt across cards; those are measured separately ([Flash-Next prefill](qwen38-flash-next.md#prefill)).
- **Determinism:** two runs of one build match row for row, with the tiers pinned (see [greedy identity](#greedy-identity)).

## The BF16 reference

Every score is against the original model in BF16:

| Model | Base | Base PPL |
| --- | --- | --- |
| Qwen3.8-Flash-Next | unsloth's BF16 GGUF through `llama-perplexity` (llama.cpp commit 436f6f89e), 100 chunks of 512, `-b 8192 -ub 8192` | 3.2749 (3.2733 on the scored rows) |
| DeepSeek-V4.1-Flash | a BF16 GGUF by @giveen, made with `convert_hf_to_gguf --outtype bf16` (llama.cpp 330f0189), wikitext-2, 100 chunks of 512 | 2.7394 (2.7314 from the stored rows) |

The official Flash-Next packs are scored against the first base on their
[Hugging Face card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality). That gate is 100 chunks, T 3, int8 KV, resident
experts and tiers pinned, on an RTX 5090 at 400 W with an RTX 5060 Ti at 180 W.

Both bases are published as the [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) dataset on Hugging Face,
with checksums: pass one to `basalt-bench --quality --kld-base FILE` to score a build or a self-converted pack.

A quantized base is not a reference. llama.cpp's Q8_0 GGUF adds its own rounding, so every pack reads 0.003 to 0.010 KLD higher against a
Q8_0 base than against BF16 (see [Flash-Next](qwen38-flash-next.md#summary)).

Numerics changes: report PPL, mean KLD and top-1 next to the previous build. Bitwise-neutral changes: show greedy identity.

## Greedy identity

Pin the placement first, or experts moving between the CPU pool and the GPUs will round differently: `--adapt-swaps 0 --peer-adapt-swaps 0`,
a fixed `--expert-cache N`, and a fixed peer tier.

Forced-window identity (the same window width must give the same rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base flashnext-bf16-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --kv int8 --resident-experts --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP on and off, token by token: one engine session per configuration over `--serve`:

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7} --expert-cache N --prefill auto --kv int8 \
    --vram-reserve-mib 700 --resident-experts --peer-exclusive --gpu 0,1 --max-context 8192 \
    --adapt-swaps 0 --peer-adapt-swaps 0 --exp concur9_logit_hash=1
```

After `READY`, `GEN 256 temperature=0 ID,ID,...` on stdin returns `T ID` lines and ends with `DONE`; `QUIT` ends the session. `LH` hash lines on stderr. Compare per position, matching rows by position.

## Reproduce the reference

Flash-Next base (the model does not fit in RAM and VRAM, so the experts stream from disk):

```sh
llama-perplexity -m Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-bf16-base.kld -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99 --n-cpu-moe 99 -t 16
```

Score a pack against it with the same command and `-m PACK.gguf --kl-divergence` added.
