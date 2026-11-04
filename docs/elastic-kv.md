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
