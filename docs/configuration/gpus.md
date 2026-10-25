# GPUs

One setting picks the cards: `gpu`. It means the same in the server (`"gpu"` in the config's `server` section, or
`--gpu`), in `basalt` itself, in `basalt-bench` and in `basalt-probe`.

| Value | Cards used | Main card |
| --- | --- | --- |
| not set | GPU 0 alone; one warning at start names the other visible cards | GPU 0 |
| `"0,1"`, `"1,0"`, ... | exactly these, in this order | the first one listed |
| `"auto"` | every visible Blackwell (sm_120) card; any other card is skipped with one line naming it | the one with the most memory bandwidth, then the most VRAM |

Cards are numbered as `nvidia-smi` numbers them. When `CUDA_VISIBLE_DEVICES` is set, the numbers count within its list
(`CUDA_VISIBLE_DEVICES=1,0` makes `gpu 0` the card `nvidia-smi` calls 1), and its numbers are read as `nvidia-smi`'s
too. A number past the visible cards, a card listed twice or a card that is not Blackwell stops the start with an error.

The main card holds the dense weights, the KV cache, the drafter and its own expert cache. The engine decides what the
other cards do and prints one line at start, for example `gpus: RTX 5090 (nvidia-smi 0) main, RTX 5060 Ti (nvidia-smi 1)
expert tier`:

- **Expert tier.** Each card after the main one holds experts the main card does not, computes their rows in every
  decode window and follows the conversation like the main card's cache does. The experts are dealt out hottest first,
  each to the tier with the least work so far for its memory bandwidth, until every tier is full. Every tier computes
  its experts with the main card's kernels, so where an expert sits does not change its rows. The second card also
  computes its experts' share of each prompt, and when its memory is faster than the main card's it takes a share of
  each GDN mixer and of the head (one start line each; see [the GDN split](../precision/gdn-split.md)).
- **Idle.** Where a process gets no GPU-to-GPU access (Windows), the cards after the main one stay idle and a warning
  says so. The engine runs up to three expert tiers: a fifth card and later ones stay idle.

Three or more cards are community-tested: the path is checked here on two cards, with two expert tiers sharing the
second one. Reports from machines with three or four cards are welcome (see [community](../community.md)).

## What several cards do together

With two or more cards the engine prices three mechanisms at start, as milliseconds per decode window, from the cards'
memory bandwidth, their VRAM and the model's size, and prints one line, for example
`multi-gpu: 2 cards unlike; ms per window: expert tier 7.9, layer split 14.0, tensor parallel 9.0 (not built); picked
expert tier`. The line appears for every multi-card setting, alike cards or not. With one card (unset, or one number)
there is no choice: nothing is priced and no other card is used or even visible to the process.

- **Expert tier**: the main card runs every layer and the other cards hold and compute experts (above).
- **Layer split**: each card runs a block of layers, one after another. The cut is searched at start. It plans every
  layer's experts on the host and hands the window to the next card, which costs a fixed few milliseconds per window,
  so it only wins when that is smaller than what a second window in flight would hide. Windows are not overlapped.
- **Tensor parallel**: every layer is split across the cards. It is priced and not built: the dense part is a third of
  a window's bytes, so splitting it saves about 2 ms between two cards and the exchanges between the cards (host-staged,
  about 5.4 us per 10 KiB) cost about as much; the expert tier already shares the experts.

Only cards that match (memory bandwidth and VRAM within 10%) may pick a mechanism other than the expert tier, and only
when it prices at least 5% below it. Unlike cards always use the expert tier. The prices are estimates fitted to
measured windows (a 5090 with a 5060 Ti: tier 8.0 ms, layer split 13.8 ms at the best cut; two 5080s: 147 tok/s
reported against 141 priced), good for the choice and not for forecasting a rate.
