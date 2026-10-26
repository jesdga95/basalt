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
  computes its experts' share of each prompt chunk: of any chunk when it is as fast as the main card, of chunks up to
  2,559 rows when it is slower (a slower card costs a longer chunk more than the experts it saves the main card's link).
  When its memory is faster than the main card's it takes a share of each GDN mixer and of the head (one start line
  each; see [the GDN split](../precision/gdn-split.md)).
- **Idle.** Where a process gets no GPU-to-GPU access (Windows), the cards after the main one stay idle and a warning
  says so. The engine runs up to three expert tiers: a fifth card and later ones stay idle.

Three or more cards are community-tested: the path is checked here on two cards, with two expert tiers sharing the
second one. Reports from machines with three or four cards are welcome (see [community](../community.md)).

## What several cards do together

With two or more cards the engine prices three mechanisms at start, as milliseconds per decode window, from the cards'
memory bandwidth, their VRAM and the model's size, and prints one line, for example
`multi-gpu: 2 cards unlike; ms per window: expert tier 7.9, layer split 13.5, tensor parallel 9.0
(not built); picked expert tier`. The line appears for every multi-card setting, alike cards or not. With one card (unset, or one number)
there is no choice: nothing is priced and no other card is used or even visible to the process.

- **Expert tier**: the main card runs every layer and the other cards hold and compute experts (above).
- **Layer split**: each card runs a block of layers, one after another. The cut is searched at start, before any weight
  is placed, and every card holds only the dense weights of its own layers, so its VRAM goes to experts as in the tier.
  Every card plans its layers' experts on its own device, as a single card does, and the window is handed to the next
  card once, which costs a fixed part of a window. The drafter stays on the first card and reads the last card's rows
  of each window; the last card holds the head. The experts no card holds are read from one pinned RAM copy that all
  stages share (the resident RAM mode, as in the tier), and each card's adaptive swaps stay within its own layers.
- **Layer split of alike cards, overlapped windows**: with two cards, a server and the drafter, one conversation at a
  time, the split also overlaps consecutive windows. Once the first card has served window K and the drafter's chain
  holds a draft past it, the first card commits K as if every row were kept and starts window K+1 from the guess that K
  is accepted whole, while the second card still verifies K. A guess that holds sends K+1 on to the second card at once;
  one that fails puts the first card's state back and runs the window again, so the text is the same either way. The
  price of the split does not count the overlap. The engine log prints `pipelined windows: N launched, M held` per
  request.
- **Tensor parallel**: every layer is split across the cards. It is priced and not built: the dense part is a third of
  a window's bytes, so splitting it saves about 2 ms between two cards and the exchanges between the cards (host-staged,
  about 5.4 us per 10 KiB) cost about as much; the expert tier already shares the experts.

Only cards that match (memory bandwidth and VRAM within 10%) may pick a mechanism other than the expert tier, and only
when it prices at least 5% below it. Unlike cards always use the expert tier. The prices are estimates fitted to
measured windows (a 5090 with a 5060 Ti: tier 7.9 ms, plain layer split 13.4-13.7 ms measured at the cuts
tried; two 5080s: 147 tok/s reported against 141 priced), good for the choice and not for forecasting a rate.

## Long prompts across the cards

In the expert tier, a long fresh part of a prompt can be read as a pipeline over the main card and the tier cards while
no other conversation decodes (with `--parallel`, while every other slot is idle). The main card reads the first layers
of each chunk and hands the chunk on; each tier card reads the next block of layers of it while the cards before it read
the next chunks, the last one ending at the last layer. The main card reads the last chunk's later layers itself once
the tier cards have given their layers' state back and got their experts back. A tier card's weights, prompt state and
buffers go into its tier's last slots for the read (its weights from a pinned RAM copy made at start), so outside a
prompt the tiers hold what they held before and decode does not change. The engine prices each part on the main card
alone and over every number of stages and every set of cuts, from each card's memory bandwidth, the probed link rates
(each card's alone and all the cards' at once, which a shared switch or the RAM can hold below their sum) and the
experts each card holds, and reads it across the cards only when that is at least 5% cheaper. Tier cards join in
order of memory bandwidth, the fastest reading the last layers. At start each card gets the layers the stage counts
worth it for a prompt of the context's length (up to 131,072 tokens) give it, plus a few. The start and request lines
begin with `prompt split:` and print the prices. The chunks are the main card's own, so a part of one chunk (up to
32,768 tokens) stays on the main card. On three or more cards it runs when they all match (memory bandwidth and VRAM
within 10%); on three or more unlike cards it is off until measured on such a set, and the start line prints what it
would cost. Off with `--peer-exclusive`, without the RAM copy of the experts, with KV streaming, a
control vector, experts other than NVFP4 (the price is fitted to those), or when RAM or a tier's slots are short. A
prompt that arrives while other slots decode is read on the main card beside them, as before. A much shorter request
that arrives during a read across the cards gets its turn at the end of the main card's current chunk: the main card
reads that chunk's later layers itself and the rest of the prompt is read later, as with a main-card read.

## Two 16 GB cards

`gpu "0,1"` with two cards of 16 GB (two RTX 5080s, say) runs the expert tier: each card holds experts in what its VRAM has
left, and the second card also reads its experts' share of each prompt. The engine sizes the second card's tier to leave
room for the buffers of that share (about 200 MiB, the size is computed, not a setting) and `peer-reserve-mib` more
(default 600 MiB; at least 128) for everything else the card needs after its experts are in. If the start prints
`prefill peer: the peer's buffers do not fit`, raise `peer-reserve-mib` (the card shows a display or another program using
its VRAM) and the second card's experts shrink by the same amount. The start line `peer GPU 1: N experts` says how many
it holds.

## The engine log

The server writes the engine's output, the `basalt ...` lines the engine prints at start and while it runs, to
`~/.cache/basalt/<id>/engine.log`, one folder per model file (`<id>` is 16 hex digits taken from the model file).
`[server] engine log: PATH` names it when the engine stops. The `[server]` lines are the server's own and go to its terminal.
