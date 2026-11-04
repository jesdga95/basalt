# The elastic KV cache

With `--max-context` above 32768 the KV cache is elastic (not with `--kv-resident`, `--vram-elastic` or a layer split; a peer card is
fine): each session backs its first 32768 cells at startup (the K/V pools, the indexer's block keys, the RoPE tables
and the draft layer's K/V alike), and the rest of that VRAM holds the expert cache's last slots until a request needs
it. A request backs its prompt plus 16384 cells before the prompt is read (and 16384 more whenever decode runs past
that) by taking those slots back; their experts become CPU misses (a long prompt's go to a peer card). The next request
that needs less gives the cells back and the cache refills the slots from RAM, so a short request decodes with the cache
a 32k context would have. With `--parallel` a slot whose share is above 4096 tokens backs 4096 cells at startup and
grows by 4096. An idle conversation's KV stays until requests beside it have run 15 s, then moves to RAM (outside
`--conversation-cache-mib`, above the RAM floors; else only to the disk tier holding it) when a request needs VRAM or
the engine is idle; its next turn reads it back. The startup line and each move say `elastic KV: ...` (a move out:
`slot N's idle conversation ...`). The cache never goes below the slots a prompt chunk needs: a request whose KV would
take it there even after the other sessions and idle conversations give back what they can is refused with that error,
and a reply that grows past it ends with it (a `--parallel` slot ends as at its length limit); the engine keeps running.

The RAM plan keeps the lent slots' RAM copies, the KV's whole reach included, as far as RAM holds them after the
experts no slot holds. When RAM is short at startup the lowest lent slots go without a copy; before a KV growth or a
prompt's loan takes such a slot, its expert is copied into a RAM home (`VRAM: N experts moved from the cache to RAM`),
as many as leave the RAM floor and the serving budget free, the slots the KV takes first. The others are read from the
model file while their slots are lent (`elastic KV: RAM homes for ...: the others are read from the file`). A home is
freed when the cache grows back and its expert is in a slot again (`elastic KV: N RAM homes released`).

## Host K/V: `--exp kv_host`

A request that starts decoding at 99328 tokens or deeper keeps its K/V in pinned RAM instead (the host K/V): attention
reads it through a VRAM pool of 32768 cells, so the cache keeps its tail slots. A conversation switches at a turn that
reads at most 32768 new prompt tokens (a continuation); a turn that reads a longer prompt stays in VRAM, since a long
streamed read costs more than its decode wins back, and the conversation's next turn switches. The pin covers the
prompt, up to 16384 new tokens and 4096 more, on a 16384-cell grid; a reply that outgrows it gets a pool for its next
16384 tokens between two decode windows, or goes back to the VRAM tail when the RAM is short. The next shallower
request goes back to the VRAM tail too, and so does a streamed session when a request arrives with the RAM below the
guard's floor (the pin is freed). A pin, and each larger one, is taken only when it leaves the RAM guard's floor free
plus what serving can still take (its budget less what it holds now: the checkpoints and the freed blocks kept for
the next ones, and the parked conversations) and the RAM copies
the prompt path's loan then needs; otherwise the request keeps its K/V in VRAM (`host K/V: kept in VRAM ...`). It is
on by default with one GPU, the experts' RAM copy, a prompt path that lends cache slots and a KV other than `k8v4`; not
with `--parallel`, a layer split, `--kv-resident` or `--vram-elastic`, where the K/V stays in VRAM.
`"exp": "kv_host=0"` in the config's `engine` section turns it off. A streamed request decodes as from the VRAM tail
at the same expert placement; its prompt read lends a few more cache slots (the staging pool), which can move the last
bits of a short read's K/V, and the freed VRAM holds more experts, so in serving the text can differ
([precision](precision/index.md)). Until a request streams, the VRAM, the expert cache and the
prompt chunks are what they are with it off.
