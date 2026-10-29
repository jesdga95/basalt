# The elastic KV cache

With `--max-context` above 32768 the KV cache is elastic (not with `--kv-resident` or `--vram-elastic`; a peer card is
fine): each session backs its first 32768 cells at startup (the K/V pools, the indexer's block keys, the RoPE tables
and the draft layer's K/V alike), and the rest of that VRAM holds the expert cache's last slots until a request needs
it. A request backs its prompt plus 16384 cells before the prompt is read (and 16384 more whenever decode runs past
that) by taking those slots back; their experts become CPU misses read from their RAM copies. The RAM copy covers the
prompt path's loan at the full cache only: once the KV holds part of the tail the loan sits lower, and at a request's
admission the experts it then reaches without a RAM copy are copied from their slots into RAM homes (`VRAM: N experts
moved from the cache to RAM`), within the RAM guard's reserve, and given back when the cache grows again. The same
happens when the RAM at startup kept fewer of the loan's copies than planned (`N of the prompt path's M lendable slots
keep ...`). With `--parallel` the RAM copies cover the loan's whole reach instead, at the RAM the tail's experts take.
The next request that needs less gives the cells back and the cache refills the slots from RAM. A short request so
decodes with the cache a 32k context would have, whatever `--max-context` is. With `--parallel` every slot is elastic
once its share of the context is above 4096 tokens: it backs 4096 cells at startup and grows by 4096, so idle and short
slots leave their VRAM to the expert cache. The startup line `elastic KV: ...` says how much of the cache is lent this
way, and each move is logged (`elastic KV: N cells backed ...`). The cache never goes
below the slots a prompt chunk needs: a request whose KV would take it there, even after the other sessions give back
the cells they do not hold, is refused with that error, and a reply that grows past it ends with it (a `--parallel`
slot ends as at its length limit); the engine keeps running.
