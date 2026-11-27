# GLM-5.3-Flash

Architecture id `glm5next`, file architecture `glm5-next`. One `.basalt` file: 45 model blocks and the pack's
next-n drafter as the last one, KDA (linear attention) blocks with sparse-MLA blocks between them, 288 routed
experts a layer with one shared expert, and mHC hyper-connections over four streams. What it supports next to
the others: [models](index.md).

## Pack

No published pack. Packs are converted from Unsloth's GGUFs with

```sh
python3 tools/convert/convert.py --gguf GLM-5.3-Flash-UD-IQ1_S-00001-of-0000N.gguf --out GLM5.3-Flash-UD-IQ1_S.basalt
```

(the shards after the first are found by name, the architecture is read from the GGUF, and the next-n block is
kept as the last block). The quantisations this page's numbers come from are UD-IQ1_S, 93,332,244,714 bytes for
the whole file, and UD-IQ2_XXS. A tiny runnable pack for the tests - 28 MiB, four blocks of hidden 256, its
routed experts IQ2_XXS over IQ3_XXS down:

```sh
python3 tools/convert/glm5next_tiny_pack.py
```

A pack keeps the source GGUF's `general.architecture`, `glm5-next`, and that is the name `basalt-bench` looks a
model's own fixtures up by: they live in `data/bench/glm5-next`, not under the build id.

Geometry: hidden 4,096; vocab 154,880; 288 experts a layer, 8 routed plus one shared 2,048 wide; indexer 32
cells of 128 over pools of 4, top-2,048; mHC 4 streams, 20 Sinkhorn steps; SwiGLU clamp 10; the first three
blocks dense. The dense part is 1,282 tensors and 5.88 GiB (the load line's 6,021 MiB), and stays on the
card; the routed experts are 78.0 GiB (about 6 MiB each) and live in RAM, with the card holding copies of
26 of the 288 a layer by default (7.04 GiB).

## Drafting

`--spec` is not raised for this model: it stays at 0, so the pack's next-n block (`blk.45`) drafts only when
asked, as `--spec 5`. It reads the model's own indexer cells and its
own latents, and its experts get a tier of their own; a card that cannot compute them warns at load and runs
every token as a full forward rather than failing the start.

## Memory and the expert tier

The tier is the model's, not a variant: experts are homed in RAM when RAM takes them all, and the main card
fills the rest of its free memory with copies (`expert homes of 288 per layer: main card 0, ... RAM 288`,
`VRAM expert tier 7.04 GiB, 26 of 288 experts per layer`). A window's routed experts are then split between
the card's grouped kernels and the CPU pool, and the line `TIER final: N lookups: A% VRAM, B% bus copy, C%
CPU` says how it went. That line is the only account the model gives: `ModelSession::expert_counts()` is not
implemented for this arch, so the per-request counters a batch or serve run prints when a request finishes
(`decode routed experts 0: VRAM hits 0.0%, peer GPU/PCIe 0.0%, CPU 0.0%`) read zero whatever the tier did.

**The two paths are not the same numbers, and the tier's choice between them is not a function of the input.**
The card multiplies a `q8_1` activation quantized on the device; the pool multiplies the activation
`kernels::cpu::native_quant_act` makes on the host. Each sits a few `1e-3` from the float reference, in its own
direction ([`tests/kernels/native_expert_parity.cpp`](../../tests/kernels/native_expert_parity.cpp)), and the
model's own pass test measures the consequence at the logits: **0.4% at the first token, 2.5% two positions
in**, against a card capped at `--diag-tier-vram n_used` so that the pool computes the rest
([`tests/models/glm5next/forward_test.cpp`](../../tests/models/glm5next/forward_test.cpp)). Which path an
expert takes follows which expert holds one of the card's slots, and the tier exchanges slots as it goes, so a
greedy run is not reproducible: two runs of one 128-token command parted at pick 6, with `TIER final` reading
26.9%/14.7%/58.4% in one and 33.6%/14.7%/51.7% in the other.

Pin it when the number has to be read across runs:

- `--diag-tier-vram 8` (the pack's `n_used`, which is also the floor the tier enforces: fewer is refused)
  leaves exactly a token's worth of slots, so nothing is left for the clock to decide. Measured deterministic,
  128 picks of 128 the same, and 71.8% of the lookups still go to the CPU: the cap removes the slack, not the
  pool.
- `--diag-tier-frozen` exchanges nothing after the start (`9.3% VRAM, 0.0% bus copy, 90.7% CPU; promoted 0`).
  Measured deterministic too.

The same holds for every tiered model; GLM only shows it more plainly because its experts are quantized so hard.

Within one process a pin fixes only the *starting* placement - the tier keeps adapting as it goes - so it is not
enough for comparing two requests inside a run. Measured with the cap on: two prefills of one 512-token chunk in
the same process differ at 162 of the 255 scored rows by more than 0.01 nats (worst 1.23), while their means
agree to 0.01 nats; the aggregate is stable and the individual row is not. Two separate processes with either
pin agree pick for pick, which is why every number in this page is quoted with one of them.

## Two cards

`--gpu 1,0 --gpu-mode peer` deals a second card a block of layers' expert homes, in the shared tier the other
mixture-of-experts models use ([several GPUs](../configuration/gpus.md)): the start adds
`peer GPU 1: 2205 experts` - the peer's CUDA ordinal, which is 1 inside the narrowed view however `nvidia-smi`
numbers it - and reads `expert homes of 288 per layer: main card A, second card B, RAM C, file D`, and `TIER final`
gains a `B% second card` term between VRAM and bus copy. That line is still the only account.

The peer computes its experts with the card's own grouped kernels, over a `q8_1` copy of the same block activations
the card quantizes on the device, so its rows are card rows rather than pool rows: the difference between the two
paths above is card-versus-pool, not one card versus another. Its share of a block goes out before the lookup, so the
peer works while the main card's launches and the pool's rows are in flight, and its rows land in the same pinned
block the pool writes into: `cudaDeviceCanAccessPeer` is 0 between consumer cards (checked on a GeForce RTX 4070 Ti
SUPER and a 5080, both ways), so every row a peer computes comes home over the host. This model has one expert path,
so a peer-held expert is computed in a prompt's blocks as it is in a decode window, and the peer gets no prompt
staging buffers: the model reads no prompt stages.

An sm_89 (Ada) card is accepted as the peer of a `-DPEER_ADA=ON` build ([an older peer
card](../configuration/gpus.md#an-older-peer-card)); the tier's kernels this path uses are among those it compiles for
sm_89. The drafter's tier stays on the main card whatever the cards: a draft is one row, and its block would compete
with 45 model layers for the peer's slots.

Peer speed, measured against the RTX 5080 alone with a GeForce RTX 4070 Ti SUPER as its peer - IQ1_S pack,
`--max-context 4096 --max-new 128 --diag-tier-frozen`, one 6-token prompt: 128 tokens in **8,288 ms** against
**7,369 ms** with the peer, 15.4 against **17.4 tok/s (+12.5%)**, and those five prompt rows in 323 against 295 ms.
The peer took `peer GPU 1: 2205 experts`: 49 of the 288 a layer across layers 0-44, 12.9 GiB of its 16 GiB at the
default `--peer-reserve-mib 600`. `TIER final` read `10.4% VRAM, 0.0% second card, 89.6% CPU` on one card and
`10.6% VRAM, 15.5% second card, 73.8% CPU` with it - the 6,977 lookups the peer took are the ones the pool gave up,
and the pool is the bottleneck path, so the move pays almost as much as it costs: two thirds of the ideal gain, the
rest being the host wait `peer->finish` puts between a block's launches and its sum. DeepSeek 4.1 hides that wait
with a word the sum waits for on the card (`wait_word`, raised after `finish`); this pass has not.

Pin the tier when a number has to read across runs; the arms never agree pick for pick, because a peer takes homes
the main card or RAM would otherwise have had:

```
bin/basalt --model GLM5.3-Flash-UD-IQ1_S.basalt --gpu 1              --diag-tier-frozen ...
bin/basalt --model GLM5.3-Flash-UD-IQ1_S.basalt --gpu 1,0 --gpu-mode peer --diag-tier-frozen ...
```

`--diag-tier-vram` is the other pin, but it caps the main card's slots only: it leaves the deal of the peer's homes
alone, so use `--diag-tier-frozen` across a peer comparison.

## Parity

- Against the original checkpoint, tensor by tensor: every F32 tensor of all 46 blocks (norms, conv taps,
  biases, `A_log`, `dt_bias`, router and indexer weights, mHC bases and scales) at 0.0000% relative rms, none
  unmapped - the checkpoint's tensors read over HTTP, each against the pack's stored row.
  `tools/convert/verify_pack.py` reports
  1412 of 1412 tensors equal to the source GGUF for IQ2_XXS. Outside the routed experts the pack is
  byte-identical to the GGUF it came from.
- The routed experts carry the loss: against the checkpoint's FP8 weights times `weight_scale_inv`, the gate/up
  rows of the IQ1_S pack differ by 48.8% relative rms (correlation 0.891) and the IQ2_XXS pack by 34.5%
  (0.943); the shared expert, IQ3_XXS down, 18.8% (0.983).
- Every kernel against a double-precision CPU reference on the model's own geometry, in
  [`tests/models/glm5next/kernels/`](../../tests/models/glm5next/kernels): the KDA recurrence and its front and
  output norm, the key pooling, the indexer's scores and selection, the sparse-MLA read, the cell expansion. The
  engine against the checkpoint's own forward pass over the tiny pack, teacher-forced on the ids the reference
  itself fed: the prefill row's logits at cosine 0.9998 with the same top-5, so KDA, the indexer, the MoE, mHC and
  the head are right across a window
  ([`tests/models/glm5next/forward_test.cpp`](../../tests/models/glm5next/forward_test.cpp)).
- Engine against **llama.cpp over the same IQ1_S bytes** - two engines, one set of weights, so this measures
  Basalt's decode path and not the quantisation. `tools/ppl_gate.py`, 20 chunks of wikitext-2, 255 scored rows
  each, the tier pinned (`--diag-tier-vram 8`), the drafter off the scored rows:

  | | PPL | KLD mean (median, p99) | top-1 vs the base |
  | --- | --- | --- | --- |
  | llama.cpp's own rows (the base file) | 5.5144 ± 0.1813 | - | - |
  | Basalt, IQ1_S pack | **5.5217 ± 0.1821** | 0.0447 (0.0179, 0.378) | 91.84% |

  A perplexity 0.13% off llama.cpp's, and per chunk between ×0.94 and ×1.04 with no trend: the decode path
  reproduces the other engine's distribution over the same weights. The 91.84% is agreement between two
  engines, not accuracy - llama.cpp's own top-1 against the corpus token is 60.06% on these rows. It says
  nothing about how far IQ1_S sits from BF16 either: that is the tensor audit above, and this pack is not in
  the [BF16 gate](../precision/quality.md), which is scored against a BF16 base this model does not have.
  Reproduce it with
  `python3 tools/ppl_gate.py --model GLM5.3-Flash-UD-IQ1_S.basalt --base <llama.cpp's .kld> --out <dir> --architecture other --chunks 20`
  and the engine's own flags after it, which the gate hands to the engine: `--gpu 1 --vram-reserve-mib 512
  --diag-tier-vram 8` for the run above. Name the card even when one is enough: the gate puts the model on
  `--gpu 0,1 --gpu-mode peer` by itself when it sees two cards. The peer arm of this table is not measured yet.

## Not measured

`gpu-mode split`; idle sleep and wake; `--kv` alternatives; pictures; prompt skip. `--logprobs` with
`K` is refused: the session reads a window's rows for the perplexity gate (`copy_logits`) but does not
implement `row_logprobs`, so a request asking for `K` gets "this model keeps no logits".
