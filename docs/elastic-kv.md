# The elastic KV cache

With `--max-context` above 32768, the server backs KV cells as requests need them and lends unused VRAM to the
expert cache. With `--parallel`, each slot whose share is above 4096 cells starts at 4096 and grows on that grid.
The feature is off with `--kv-resident` or `--vram-elastic`.

The RAM budget first holds the nonresident experts, then as many copies of lent cache slots as fit. Before a KV
shrink or prompt loan overwrites a resident expert without a copy, the server tries to place it in a RAM home. If
the home cannot be allocated, the expert is read from its file. Homes can be released as cache slots return.

Idle parallel conversations can move to RAM or the disk tier when another request needs their VRAM. If the requested
KV would take the expert cache below its prompt floor after other sessions give back their available cells, the
request is refused or its reply ends; the engine stays running. The `elastic KV:` lines report changes.
