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
