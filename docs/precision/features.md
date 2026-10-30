# Precision of the features

The decode-path precision tables are in [index.md](index.md); the quality method is in [quality.md](quality.md). This page covers the NVFP4 dense projections, the two-stage verify head, logit bias, speculative decoding and relaxed drafting.

## 6-bit dense projections (NVIDIA NVFP4 pack)

The official NVIDIA NVFP4 pack stores its dense projections as Q6_K where the Q8_0 variant stores them as Q8_0. This is
a weight-format change: `convert.py --checkpoint` quantizes each projection once from NVIDIA's BF16.
`--requant-dense-q6k PACK --checkpoint DIR` gives the same bytes. Without `--checkpoint` it quantizes from the pack's Q8_0
values, which rounds twice; the official file is not made that way.

- **What moves:** attention q/k/v/o, GDN qkv, z and out, and the shared experts' gate and up: 252 tensors, 2,948 MiB of
  dense weights down by 654 MiB. The shared experts' down (640-value rows are not whole 256-value blocks), the head and
  the token embedding stay Q8_0; the router, hyper-connection matrices, indexer, ssm_alpha/beta and PLE key/value stay
  BF16; the experts, the PLE table, the drafter and vision are untouched.
- **Source and rounding:** each tensor is quantized once from NVIDIA's BF16 weight, never from a Q8_0 copy (which
  would round twice). The GGUF lists the 48 GDN value heads in a different order from the checkpoint; the converter
  reorders them. `--requant-dense-q6k --checkpoint` compares each BF16 tensor with the pack's Q8_0 copy first (the
  largest distance is 0.62 Q8_0 steps; it stops above 1.05, which catches a wrong tensor or head order); `--checkpoint`
  with `--sections-from` reads the BF16 tensors by name and gives the same Q6_K bytes.
- **Method:** per 256-value block, least-squares scales per 16 values from a signed-max starting grid, 8-bit scales
  against one fp16 super-scale, values rounded against the scales as stored (ggml's `block_q6_K` layout). The error is
  about 1.9% of sigma on Gaussian groups, and the converter tests bound it by the 6-bit step.
- **Why it is faster:** 6-bit dense weights take the tensor-core dense path in decode, while Q8_0 dense weights fall back to
  the mmvq kernel, and they are about 22% fewer bytes (654 of 2,948 MiB).

Speed, RTX 5090 at 400 W with a 5060 Ti peer card, the two packs alternated in one session (ABBA, 6 fresh processes per
arm), development measurement; mean over 6:

| test | 8-bit dense tok/s | 6-bit dense tok/s | change | ms per window |
|---|---:|---:|---:|---|
| prose | 213.4 | 229.9 | +7.8% | 10.00 to 9.24 |
| conversation | 251.0 | 265.1 | +5.6% | 10.34 to 9.35 |
| structured | 420.4 | 463.5 | +10.2% | 15.88 to 14.39 |
| prefill, 8K and 32K | 3,806 and 7,791 | 3,845 and 7,856 | +1% | |

Draft acceptance is unchanged within the run-to-run spread, so the gain is window time.

Quality, against the Q8_0 reference through the decode path, 100 chunks, 400 W (three runs of the Q8_0-dense pack on three builds, KLD spread
0.0009; two of them are in the quality table in [quality.md](quality.md)):

| dense format | PPL | KLD mean | KLD p99 | same top-1 |
|---|---:|---:|---:|---:|
| Q8_0 | 3.287-3.294 | 0.0612-0.0621 | 0.692-0.699 | 91.80-91.87% |
| Q6_K | 3.3081 | 0.0680 | 0.775 | 91.46% |

The paired per-chunk KLD difference is +0.0063 (95% CI +0.0048 to +0.0077), so the loss is outside the run-to-run noise:
about 10% of the KLD and 0.4 points of top-1. On long prompts (12 chunks of 4,096 tokens, no Q8_0 reference, so PPL only)
PPL goes from 2.9547 to 2.9816; paired NLL +0.0091 (95% CI +0.0049 to +0.0132). Q6_K dense runs through dense_tc and
dense_tc_pair; the same arithmetic applies as for any Q6_K dense tensor (q8_1 activations, int8 tensor-core dot).

## Two-stage verify head

A Q6_K output head (the IQ3_S pack's) runs in two stages in the verify window (`q6k_vhead`, q6k_head.cu). Every one of the
248,320 rows is scored from a 2-bit copy (the draft head's fit, head_proxy.cuh) on int8 tensor cores against the same
q8_1 codes the exact kernel reads. Each block of rows adds its per-column maximum to a histogram; the lower edge of the bin
holding the 64th largest block maximum is at or below the column's 64th largest approximate logit, and every row within 4
of it is rescored exactly with the plain Q6_K kernel's arithmetic (the same bits). The other rows keep their approximate
logits, so log-probabilities stay finite. The block partition does not depend on the window width, so a token's logits
are the same bits at every T.

What it changes: the argmax and the sampler's top-64 list come from exact rows unless the copy's error exceeds the margin;
the softmax tail outside the rescored rows carries the copy's error (rms about 0.7 logit). A request whose sampled list
is longer than 64 (top_k 0 or above 64) reads every row, so its windows run the plain Q6_K head again over the same
columns (`Verifier::exact_head_rows`): every logit exact.

Offline, on 3,614 real verify-head inputs dumped from the engine (conv, struct, prose fixtures), all 248,320 rows, exact =
Q6_K x q8_1 in float:

| Fixture | Inputs | Top-64 recall, margin 4 | Margin 6 | Margin 8 | Rows rescored per column, margin 4 (mean / p99) | Greedy pick unchanged | KLD exact -> two-stage (mean) |
|---|---|---|---|---|---|---|---|
| conv | 1,232 | 99.971% | 100% | 100% | 1,333 / 6,286 | 100% | 4.7e-4 |
| prose | 1,124 | 99.979% | 100% | 100% | 1,098 / 2,988 | 100% | 5.1e-4 |
| struct | 1,258 | 99.968% | 100% | 100% | 1,981 / 12,790 | 100% | 2.5e-5 |

Margins 6 and 8 rescore 5,569-18,649 and 27,962-78,806 rows per column. Neither a least-squares scale for the copy nor a
per-column correction of the tail's bias lowers the KLD (the tail's mass is unbiased; the cost is its noise).

Quality gate (IQ3_S, 100 chunks, against the Q8_0 base, tiers pinned (expert cache 9000 slots, peer 7562), 400 W;
the gate leaves the PCIe share dynamic, so two runs of one build differ):

| Head | Run | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| exact Q6_K | 1 | 3.3778 | 0.15355 | 0.03400 | 1.7316 | 87.36% |
| exact Q6_K | 2 | 3.3728 | 0.15288 | 0.03397 | 1.7082 | 87.18% |
| two-stage | 1 | 3.3852 | 0.15507 | 0.03484 | 1.7681 | 87.24% |
| two-stage | 2 | 3.3827 | 0.15472 | 0.03497 | 1.7381 | 87.40% |

The two-stage head raises KLD by about 0.0017 (1.1%) and PPL by 0.2%, more than the 0.0007 between two runs of one build:
the gate scores the whole softmax, including the approximate tail. Same top-1 is unchanged. Greedy decoding with the tiers
pinned (`tools/greedy_identity.py`, the three bench prompts, 900 tokens) is token-identical to the exact head.

## Request logit bias

`logit_bias` (Chat Completions) is applied by the verifier's own sampling, on every row of every window, so a bias is
part of the distribution speculation must reproduce: the host-side sampling that already follows each window (the same
one a temperature or a penalty uses) runs with the bias, and a draft is accepted only if it equals that pick. The draft
head ignores the bias; it only proposes tokens, so a banned token it proposes is rejected, and a biased request
accepts fewer drafts. A request without a bias takes the unchanged captured greedy sample.

Checked on IQ3_S with the tiers pinned as in the section below: requests with a bias (greedy and seeded, a ban, a
boost, a mixed map, with and without a presence penalty) return the same text with `--spec 7` as with
`--spec 0`, and a request with an empty map matches one without the field. A -100 bias never yields the token and
a +100 bias yields it at every step.

## Request grammar

A request with a `grammar` key (engine started with `--grammar on`) is checked on the host after each verify window.
The tokens are fed to the grammar in order; at the first one it refuses, that row is drawn again with the grammar's
mask added to the request's logit bias (same filters, penalties and position draw), and the window ends there, so the
drafts after it are dropped like a rejection. A window the grammar accepts is untouched: no extra launch or sync.
Under `--batch` each slot holds its own guard and the same rule runs per slot: its kept picks are scanned, the first
refused row is drawn again with that slot's sampling and position under the mask, and the slot's segment ends at that
row. A request that moves from the solo run into a slot replays its generated tokens through the grammar first. Slots
and requests without a grammar run the unchanged path.
Tool-call grammars start at the call opener, JSON grammars at the first token after the thinking block. Relaxed
drafting is off for a request with a grammar. The prompt is never constrained. Requests without the key, and the engine
with `--grammar off`, run the unchanged path. The schema grammar is llama.cpp's (integer bounds, string length, patterns,
array sizes and formats are enforced); what llama.cpp does not express (`uniqueItems`, number bounds, a pattern with no
grammar form) is not enforced.

## Speculative decoding does not change output

The standalone generator starts verification at the last prompt token, including position zero for a one-token
prompt. The initial window has one row; subsequent windows use the same verifier and draft policy as longer prompts.

Verify guarantees that a token's logits are bit for bit those of plain decode, whatever the window width
(verify.hpp:3-8). This section tests that end to end.

The default window is 8 tokens (`--spec 8`; an explicit `--spec N` is used as given). Greedy decoding with the drafter
stays exact at any window. A sampled request is still drawn from the model's own distribution, but the window length
sets which random draw each position uses, so a seeded sampled request can return different text than under another
window. The tables below were measured with `--spec 7`; the exactness argument does not depend on the width.

**Method.** 275 prompts: 272 chat-templated prompts with thinking off (152 prose, 60 chat, 60 code) plus the 3
`data/bench` decode prompts (conv, struct, prose). 256 new tokens each, temperature 0, one engine session per
configuration, same prompt order. A logit-hash diagnostic prints an FNV-1a hash of the full-vocabulary fp32 head logits
of every accepted row. Token ids and the hash at each generated position are compared between `--spec 0` and
`--spec 7`.

Expert tiers are pinned (`--adapt-swaps 0 --peer-adapt-swaps 0`, with the expert cache and the peer tier
fixed to the sizes auto-sizing picks), so the same experts sit on the same device in both runs.

| Pack | Prompts | Identical tokens | Logit rows compared | Bit-identical rows | First divergence | Accepted drafts | Decode tok/s, spec 0 → spec 7 |
|---|---|---|---|---|---|---|---|
| IQ3_XXS | 275 | 275 (100%) | 62,271 | 62,271 | none | 37,892 / 54,206 (69.9%) | 183.9 → 319.8 |
| IQ3_S | 275 | 275 (100%) | 60,878 | 60,878 | none | 36,650 / 52,312 (70.1%) | 145.1 → 239.5 |

Throughput is decode-only over these 275 requests, at 400 W. Verify windows in the spec 7 run have widths 1 to 7, so the
compared rows include logits computed at every window width T, all equal to the T=1 logits of the spec 0 run. On an
earlier build, a repeated IQ3_XXS spec 0 run was bit-identical to the first (62,491 of 62,491 rows), so the setup is
deterministic run to run.

### Forced-window KLD

The quality gate is teacher-forced: the window tokens are the reference text, forced to width `--t`, and the drafter
output is not used (`src/program/serve/request_decode.cpp`, the `ppl != nullptr` branches). Forcing the window widths the drafter
produces exercises the same verify kernels as real drafting.

IQ3_XXS, 20 chunks of wikitext-2, against the llama.cpp Q8_0 base (base PPL 2.2815), int8 KV, tiers pinned, 400 W:

| Forced window T | PPL | KLD mean | KLD median | KLD p99 | Same top-1 | rows.tsv md5 |
|---|---|---|---|---|---|---|
| 1 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |
| 3 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |
| 7 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |

All 5,100 per-row lines (NLL and KLD) are byte identical across T=1, 3 and 7 (`ppl_gate.py --diff`: max |dKLD|
0.000000). Window width does not change the logits, so KLD with MTP on equals KLD with MTP off; the 0.278 is the quant
against the base.

### Scope

- Identity is bitwise with the expert tiers pinned. With adaptation on, experts move between the CPU pool and the GPU,
  which round differently: a rounding difference, not a quality difference. The same holds for the slots the elastic KV
  takes from the expert cache and gives back. An idle conversation moved to RAM and back keeps its bytes exactly: with
  the tiers pinned, its next turns write the same tokens and accept the same drafts as when it stays in VRAM.
- The GPU tiers (VRAM and peer card) share one arithmetic; the CPU tier quantizes with ggml's vec_dot_type (Q8_K per 256
  for i-quant gate/up, against q8_1 per 32 on the GPU; native_expert.cpp:48-49, 76-86). In the quality gate about 0.8% of
  routed experts were computed on the CPU (IQ3_XXS, default placement). The split is fixed per placement, so it is the
  same for spec 0 and spec 7.
- Prompts are 19 to 395 tokens with 256 new tokens: this tests the decode path, not long context.

### Overlapped windows on a layer split

A two-card layer split can start window K+1 on the first card while the second card still verifies window K (see
[GPUs](../configuration/gpus.md)). The first card commits K as if every row were kept, from a snapshot of the state that
commit changes (recurrent state, conv history, the indexer's pooled rows and block position, the PLE history), and runs
K+1 from the guess that K is accepted whole and that its last pick is the drafter's next token. When the guess fails the
snapshot is put back and K+1 is run again as without the overlap; when it holds, the second card runs K+1 behind K's
commit. Every row is computed by the same kernels on the same inputs, so greedy output is that of the plain split: the
two builds write the same 400 tokens for conv, struct and prose and for 3-5K-token prompts when their expert caches are
pinned (an expert on the CPU tier and the same expert on a GPU differ at 8e-8, which flips a near-tie token between
unpinned runs). Scored (`ppl=`) requests, requests with a repetition penalty or a grammar, relaxed drafting and batch
slots never overlap windows.

## Relaxed drafting

Lossy flags are listed together in one start warning; the routing ones are in [routing.md](routing.md).

`--relaxed-drafting P` (opt-in; off by default) gives up the guarantee above for sampled requests. In exact mode a draft
is kept only when it equals the target's own draw at that row (`src/program/serve/request_decode.cpp`, the acceptance loop after
the verify window). In relaxed mode a draft is also kept when the target gives it probability P or more under the request's
final distribution (top_k, top_p, min_p, temperature and penalties applied). The sampler reports that probability next to
its draw (`sample_tokens_probe`, src/kernels/cuda/sampler.cu:1375); the logits and the draws themselves do not change.

**What changes.** The drafts are the draft head's argmax, fixed by the context. At a row where the draft d has
p(d) ≥ P, the output is d every time instead of a draw from p: that row is greedy. Its total-variation distance from the
model's distribution is exactly 1 − p(d), and 0 at every other row. Rows after a kept draft are conditioned on it,
as plain decoding would be. What stays exact: greedy requests (nothing is drawn), requests with `logit_bias`, JSON
`response_format` (the server sends `relax=0`), the quality gate, `--parallel` batch slots, any drafted added token
(ids from `<|endoftext|>` on: control tokens, think and tool tags, end of turn), and every token from `<tool_call>` to
`</tool_call>`, so a tool name and its arguments are sampled exactly.

**Exact mode is unchanged.** With the flag off and the tiers pinned, IQ3_XXS, the three bench prompts greedy plus conv
and prose at temperature 0.6 and 1.0 with two seeds each (600 tokens): all 11 token streams are identical to the build
before the flag. With the flag on, the three greedy streams are identical too.

**Distribution cost per token (offline replay).** Real verify windows were logged (per row: the draft, the target's
draw, the target's final probabilities) for the bench prose and conv prompts (three seeds) and five natural chat
prompts, at temperature 0.6 and 1.0 (top_p 0.95, top_k 20), on IQ3_S and IQ3_XXS, about 3,500 tokens per cell. The
replay keeps a draft when the rule would, and averages 1 − p(d) over the relaxed rows per generated token
(TV/token; 0 for exact). Token gain is the replayed extra tokens per window at the logged window widths; the live
gain below is larger because the window controller lengthens windows when more drafts are kept.

| Model | Prompts | T | Exact accept | P = 0.5 gain / TV | P = 0.3 gain / TV | P = 0.2 gain / TV |
|---|---|---|---|---|---|---|
| IQ3_XXS | prose | 0.6 | 64.1% | +4.3% / 0.026 | +6.9% / 0.042 | +9.2% / 0.052 |
| IQ3_XXS | natural | 0.6 | 64.5% | +5.3% / 0.030 | +8.4% / 0.050 | +10.7% / 0.063 |
| IQ3_XXS | prose | 1.0 | 61.6% | +5.6% / 0.035 | +11.1% / 0.064 | +13.9% / 0.080 |
| IQ3_XXS | natural | 1.0 | 62.9% | +6.6% / 0.043 | +12.1% / 0.073 | +15.2% / 0.089 |
| IQ3_S | prose | 0.6 | 63.7% | +3.0% / 0.022 | +5.9% / 0.037 | +7.2% / 0.046 |
| IQ3_S | natural | 0.6 | 64.5% | +4.9% / 0.030 | +8.0% / 0.049 | +9.8% / 0.060 |
| IQ3_S | prose | 1.0 | 61.1% | +5.1% / 0.035 | +9.3% / 0.061 | +12.4% / 0.079 |
| IQ3_S | natural | 1.0 | 62.3% | +7.1% / 0.042 | +11.9% / 0.072 | +14.7% / 0.089 |

Every relaxed keep moves 1 − p(d) of probability mass, so tokens gained per unit of TV is nearly the same at every
threshold: about +1.5% per 0.01 TV/token. Medusa-style typical acceptance (a threshold that falls with the entropy of p)
was replayed too and lands on the same line; it is not a better trade, so the flag is the plain probability threshold.

**Speed (400 W).** `basalt-bench` (conv and prose at temperature 0.6, struct greedy), with `--spec 7
--kv int8 --vram-reserve-mib 500 --resident-experts --peer-exclusive`, each value in three rounds
interleaved with the others, the cold repetition dropped:

| Model | Fixture | Exact | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|---|
| IQ3_XXS | prose | 348.7 | 366.8 (+5.2%) | 373.8 (+7.2%) | 382.4 (+9.7%) |
| IQ3_XXS | conv | 383.8 | 421.5 (+9.8%) | 431.4 (+12.4%) | 423.3 (+10.3%) |
| IQ3_XXS | struct (greedy) | 660.2 | 658.5 | 659.4 | 658.0 |
| IQ3_S | prose | 323.8 | – | 347.2 (+7.2%) | – |
| IQ3_S | conv | 354.2 | – | 392.9 (+10.9%) | – |

In one engine session per model, the prose prompt and the five natural prompts, two seeds, the four settings
interleaved per prompt (decode tok/s, 400 W):

| Model | T | Exact | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|---|
| IQ3_XXS | 1.0 | 332.5 | 366.9 (+10.3%) | 385.5 (+15.9%) | 399.1 (+20.0%) |
| IQ3_XXS | 0.6 | 351.2 | 370.1 (+5.4%) | 383.1 (+9.1%) | 380.9 (+8.5%) |
| IQ3_S | 1.0 | 336.3 | 356.8 (+6.1%) | 369.7 (+9.9%) | 392.3 (+16.7%) |
| IQ3_S | 0.6 | 347.0 | 359.2 (+3.5%) | 370.9 (+6.9%) | 382.4 (+10.2%) |

**Quality.** The teacher-forced KLD gate cannot see this change: it scores logits, and relaxed drafting changes which
token is emitted, not the logits. So the evidence is the replayed TV above plus sampled tasks. IQ3_S through the server
with the same flags, temperature 1.0, top_p 0.95, top_k 20, thinking off, one seed per item so the settings are
paired, the four settings in one server session (`relaxed_drafting` per request), 400 W:

| Task | Exact | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|
| gsm8k, 300 items (flexible extract) | 0.947 ± 0.025 | 0.973 (+0.027 ± 0.020) | 0.980 (+0.033 ± 0.022) | 0.977 (+0.030 ± 0.023) |
| MMLU-Pro, 20 per subject (280) | 0.682 ± 0.055 | 0.714 (+0.032 ± 0.040) | 0.725 (+0.043 ± 0.045) | 0.721 (+0.039 ± 0.046) |
| Drafts kept by the rule, per generated token | 0 | 3.0% | 5.0% | 5.6% (with the tool requests) |

The ± are 95% intervals (paired difference against exact in brackets). An earlier session in which every arm ran exact
(the per-request key had not reached the engine) scored gsm8k 0.960 to 0.970 and MMLU-Pro 0.696 to 0.714 across four
exact runs, with 6 to 9 and 24 to 35 items flipping between runs (the adaptive expert tiers round differently run to
run): this session's exact arm is at the low end of that spread. No loss is measurable at this size; the relaxed arms
sit at or above the exact runs, as expected of a change that makes confident rows greedy (a lower effective
temperature). The cost is diversity, which these tasks do not measure; TV/token above is the direct measure.

Tool calls, IQ3_S, the same sampling, 50 requests over 8 tools (similar names such as `replace_in_file` and
`read_file`), thinking on for half: exact 53 calls and P = 0.2 52 calls, every name a declared tool, no leaked call
text, all arguments valid JSON. The engine log shows the rule kept almost no drafts in these requests (the calls
themselves are exact).

**Choosing P.** `--relaxed-drafting` alone means 0.5: the smallest cost (TV/token 0.03 to 0.04), roughly +5% at
temperature 0.6 and +6 to 10% at 1.0. 0.3 doubles both the gain and the cost; 0.2 is the aggressive end. Values above
0.5 keep few drafts (0.7: about +2.5% at TV 0.015).

## Prefill CPU share

The CPU pool and the GPU round a streamed prompt expert differently (see the CPU-share paragraph in [index.md](index.md)), so
`--prefill-cpu-share` can move numbers. NVFP4 pack, quality gate 20 chunks against the Q8_0 reference, T 3, int8 KV,
`--spec 0`, 400 W, share 0 (all GPU) against 1 (all CPU): PPL 2.3046 / 2.3142 (standard error 0.06 each), KLD 0.07766 /
0.07717, top-1 92.86% / 92.82%. The two ends differ by 0.0095 PPL, 0.0005 KLD and 0.04 points of top-1, inside the
run-to-run size of this gate, so any share between them is bounded by that. `basalt-probe` may print a different share
as a suggestion ([probe](../probe.md)); it never changes it on its own.
