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

The layer split (each card a block of layers) is not a choice: it measured slower than the expert tier and than the main
card alone on the same machine, so the engine never picks it.
