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
- **Coverage:** the 256-token prefill halves run each model's prompt path for that length: MMQ for Flash-Next, whose FP4 prompt path
  (used from 1,024-token chunks) and split prompt across cards the gate does not reach; those are measured separately
  ([Flash-Next prefill](qwen38-flash-next.md#prefill)). Inkling 2 Mini's halves run its FP4 experts.
- **Determinism:** two runs of one build match row for row, with the tiers pinned (see [greedy identity](#greedy-identity)).

## The BF16 reference

Every score is against the original model in BF16:

| Model | Base | Base PPL |
| --- | --- | --- |
| Qwen3.8-Flash-Next | unsloth's BF16 GGUF through `llama-perplexity` (llama.cpp commit 436f6f89e), 100 chunks of 512, `-b 8192 -ub 8192` | 3.2749 (3.2733 on the scored rows) |
| DeepSeek-V4.1-Flash | a BF16 GGUF by @giveen, made with `convert_hf_to_gguf --outtype bf16` (llama.cpp 330f0189), wikitext-2, 100 chunks of 512 | 2.7394 (2.7314 from the stored rows) |
| Qwen3.8-27B | unsloth's BF16 GGUF through `llama-perplexity` (llama.cpp 5676dfbe), wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192` | 6.7437 (6.7411 on the scored rows) |
| Hy4-flash | Tencent's BF16 checkpoint converted to a BF16 GGUF (llama.cpp 0a054d0b0), through `llama-perplexity` with the llama.cpp port's Engram gate norm epsilon set to FP32's (#556) and `add_bos_token=true`, wikitext-2, 100 chunks of 512; `hy4flash-bf16-base.kld` | 18.416 (over the 25,500 scored rows) |
| Inkling 2 Mini | the official release with its experts dequantized to BF16 (no BF16 checkpoint exists; the release ships NVFP4 and MXFP8 experts): a BF16 GGUF from `convert_hf_to_gguf --outtype bf16` (llama.cpp 29b52d398), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`. Not comparable with the other models' true-BF16 bases | 7.0714 |

The official Flash-Next packs are scored against the first base on their
[Hugging Face card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality). That gate is 100 chunks, T 3, int8 KV, resident
experts and tiers pinned, on an RTX 5090 at 400 W with an RTX 5060 Ti at 180 W.

The bases are published as the [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) dataset on Hugging Face,
with checksums: pass one to `basalt-bench --quality --kld-base FILE` to score a build or a self-converted pack.

A quantized base is not a reference. llama.cpp's Q8_0 GGUF adds its own rounding, so every pack reads 0.003 to 0.010 KLD higher against a
Q8_0 base than against BF16 (see [Flash-Next](qwen38-flash-next.md#summary)).

Numerics changes: report PPL, mean KLD and top-1 next to the previous build. Bitwise-neutral changes: show greedy identity.

## Lossy routing flags

Two opt-in flags on every mixture-of-experts model trade accuracy for fewer experts that no GPU holds. Each one given is named in
the start's one lossy warning, `--serve` reports it (`INFO lossy=`), `basalt-bench` marks the run LOSSY and `--community`
refuses it.

- `--route-tail-skip [R]`: in a decode window, an expert no GPU holds is neither copied nor computed when every entry of the window
  that names it ranks R or lower by router weight (0-based of the model's k routed experts). Its entries get weight 0; the other
  weights stay as the router gave them (no renormalisation). R is 1 to k-1, lower drops more; the flag alone is k-1.
- `--route-prior [LAMBDA]`: a decode window picks its top k as if every expert a GPU holds had LAMBDA added to the score the
  router ranks by: the log probability for a softmax router (Flash-Next), the biased score `s + bias` for a sqrt-softplus or
  sigmoid one (DeepSeek, Hy4). The weights stay the router's own scores of the picked experts, normalised over the picks as the
  model normalises them. LAMBDA is in (0, 16]; the flag alone is each model's own default (0.5 on Flash-Next, 0.1 on
  DeepSeek, 0.01 on Hy4), since the scores' scales differ.

"Held" means a main-card slot or a second card's. Both flags are off for prompts and for the drafters' own routing, and off when
every expert is held by a GPU (nothing to skip). DeepSeek and Hy4 read which experts are held at the start of each decode window,
so a copy that lands during a window counts from the next one. Routing then follows placement, and placement follows history, so
the same prompt can give different text on different runs. Their final `TIER` lines count what `--route-tail-skip` dropped.
Each model's page has the measured cost and speed: [Flash-Next](qwen38-flash-next.md#lossy-routing-flags),
[DeepSeek](deepseek41.md#lossy-routing-flags), [Hy4-flash](hy4-flash.md#lossy-routing-flags).

## Greedy identity

Pin the placement first, or experts moving between the CPU pool and the GPUs will round differently: `--adapt-swaps 0 --peer-adapt-swaps 0`,
a fixed `--expert-cache N`, and a fixed peer tier.

Forced-window identity (the same window width must give the same rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --architecture qwen4exp --engine-bin bin/basalt --base flashnext-bf16-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --kv int8 --resident-experts --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP on and off, token by token: one engine session per configuration over `--serve`:

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7} --expert-cache N --prefill auto --kv int8 \
    --vram-reserve-mib 700 --resident-experts --peer-exclusive --gpu 0,1 --gpu-mode peer --max-context 8192 \
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
