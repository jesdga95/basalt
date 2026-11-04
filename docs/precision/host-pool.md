# Host K/V pool growth

A registered host pool reserves virtual address space for the session's maximum context, with array bases carved at
that capacity. The carve is widened to a host-page cell quantum and each layer's stride is page aligned. Only used
array prefixes are committed. Growth commits the additional complete host pages, unregisters the old array prefix,
and registers the longer prefix at the same base, keeping one registration per array without moving bytes.
The visible extent remains rounded to K/V cells, independently of host-page padding.

This is a byte-preserving storage change: FP16, INT8 codes and scales, and Q4_0 bytes keep their formats and values.
Growth publishes each streamed layer's `host_cells` bound without resetting the residency map or recapturing graphs.
There is no change to a kernel, reduction order, activation type or accumulation type. Admission charges the additional
page-rounded bytes; the keep rule, RAM guard and mid-request floor still apply. Placement can differ when growth avoids
a VRAM fallback, so identity comparisons pin expert tiers as for other host K/V checks.

Every registered range must map to the identical host and device address. Without that capability, or if initial
registration fails, the pool uses a mapped host allocation and growth copies into a replacement. A refused longer
registration rolls every changed array back to its previous registered prefix before returning, then decommits the
new tails. A failed unregistration or restoration is fatal: the engine must never continue with missing device
mappings or unmap memory still owned by the device.

`platform_vm_test` and `kv_host_pool_stub_test` cover range commit, rollback, alias mismatch, fallback and ownership
on the CPU; `kv_host_pool_restore_failure_test` covers a fatal restoration refusal. `auto_plan_test` covers the
growth's admission. `kv_host_grow_test` compares every layer and array with a copied pool, whole-prefix transfers in
both directions and replays a graph captured before re-registration. `kv_host_registration_test` isolates each copy
direction, direct kernel access and pre-existing graph replay in fresh processes over adjacent, single and
re-registered ranges. `kv_mode_switch_test` checks streamed attention and snapshot identity after in-place growth,
and that the residency map and its counters survive growth.

The runtime check is greedy identity against the preceding build with pinned expert tiers across a pin crossing.
`kv_host_grow_test --bench POWER_W` compares streamed decode windows from registered ordinary pages and mapped host
allocations; the argument records the GPU power limit. Transparent huge pages are requested for committed ranges;
disable them for the benchmark process when measuring ordinary pages. Host-page backing and captured pointer validity
after re-registration are driver-sensitive. Re-pinning the full prefix may cost as much as pinning a new pool; net RAM
growth equals the newly committed tails.

## The pin and the RAM homes

A take reads the free RAM again right before it pins and pins only when that reading still admits the pool beside the
keep and the RAM guard. The keep counts what serving can still take: its budget less the checkpoints, the blocks
their pool keeps for the next ones and the parked conversations it holds now.

A session nearing the crossover (`kv_host_lead`) prefaults its first pool: the registered layout is reserved and its
first pin's pages are committed and written with zeros on a background thread that makes no device call, under the
same keep as the take. The take waits for the writer and registers that layout in place. The bytes a pool holds when
it is first read are unchanged (new pages are zero either way); only when the OS backs them moves. Admission and the
take count the prefaulted bytes as free, since the pin reuses them; the RAM guard's shed, a shallower request, a deep
request refused for RAM and RAM homes that would not fit release them. `kv_host_pool_stub_test` covers the layout
reuse, a refused pin's fallback and a release while writing; `kv_host_grow_test` runs its checks over a prefaulted
pool too, and `--pin-bench POWER_W` times the first pin of a 131072-cell pool from fresh and prefaulted pages.

The RAM homes a prompt's loan takes (experts without a RAM copy in the slots it lends) follow the admitted request's
own loan: its read's chunk laid out from the cache's end, never below the full chunk's loan (`homes_from`). When the
RAM cannot hold them all, the highest slots get homes first and the rest are read from the file. None of this changes
a weight, K/V, activation, accumulation or reduction format; it moves placement (the CPU/GPU split), so validation
pins expert tiers.
