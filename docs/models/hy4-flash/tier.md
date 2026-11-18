# Shared expert tier (M3)

How DeepSeek's expert tier becomes `core::ExpertTier` and what Hy4-flash uses of it. Companion to
[Hy4-flash](../hy4-flash.md#expert-tier-and-what-becomes-shared); where they differ, this page is the interface.

## Steps and what each turned out to need

1. `include/basalt/core/tier_policy.hpp` (namespace `basalt::core::policy`): moved as is; `count_use(uses, threshold)` takes the
   admission threshold (default `kPromoteUses`), so Hy4's k = 1, 2, 3 is a parameter, not a copy.
2. `core::ExpertTier` (`include/basalt/core/expert_tier.hpp`, `src/core/expert_tier.cpp`): the class of `deepseek41/tier.cpp`.
3. Engram table reader: **nothing to move.** `ngram::PleReader` already is it (worker thread, direct reads, `row_bytes` up to a
   page, hot-row cache of `cache_rows`); Hy4's 96-byte FP8 rows open it as `open(path, offset, rows, inflight, cache_rows, err, 96)`.
   The row decode (E4M3) and the hash stay the model's.
4. `core::ElasticArray` (`include/basalt/core/elastic_array.hpp`): `ElasticBuf` renamed and moved, nothing else changes.
5. `file_reads` (blob reads around the file cache) moves with step 2 to `core::FileReads`.

## The latent hook is a number

DeepSeek's tier runs experts over the layer input (`n_embd` x `n_ff`); Hy4's run over the 1536 latent that its own
`routed_down` makes. The tier only sees the expert's input width and inner width, so there is no `latent` hook: Hy4 passes
`n_embd = 1536`, `n_ff = 1536` and runs `routed_down` and `routed_up` itself, around `lookup` and `start_cpu`. The clamp is the
`clamp` of each layer's descriptor (`native_fmt` takes it).

## Descriptor (what the tier is given)

```cpp
struct ExpertTierSpec {
    const char* area;                      // log area, "basalt deepseek41" / "basalt hy4flash"
    int n_layers, n_expert, top_k;         // the tier's layer 0 is the first MoE layer: the model passes MoE layers only
    int n_embd, n_ff;                      // input width and inner width of an expert
    struct Layer { int gu_type, d_type; uint64_t offset, blob_bytes; float clamp; };
    std::vector<Layer> layers;             // offset: inside the experts section
    std::string path; int fd;              // the pack and a descriptor of it
    core::FileRange experts_range;
    model::LoadRequest flags;              // --expert-cache, --resident-budget-gib, --vram-reserve-mib, --pool-workers, ...
    std::vector<int> peer_devices;         // second cards, in block order (2 to 4 cards = 1 to 3 peers)
    int diag_ram = -1, diag_vram = -1; bool diag_frozen = false;
    int admit_uses;                        // CPU uses that earn a slot (policy::count_use threshold); 0: DeepSeek's rule
    HostHooks host;                        // below
};
```

`kMaxK` becomes 128 (a window of 8 rows at top-16); DeepSeek's kernel argument structs keep their own 48.
`Layer::layer` numbering is the tier's own: Hy4 maps model layer `il` to `il - 1` (layer 0 is dense) before calling.

Admission rule. DeepSeek: with every expert in RAM a repeat miss within `kPromoteWindow` tokens earns a slot; with experts in the
file, `kPromoteUses` CPU uses with a half-life. Hy4: `admit_uses = k` selects the counting rule with threshold k and the same
half-life, for every placement. DeepSeek's two rules stay as they are (`admit_uses == 0`), so its output does not change.

## Host hooks (what DeepSeek's prompt path owns)

The tier sizes the RAM tier after the prompt path's host buffers exist, and asks the model for a few things at fixed points. They
are `std::function`s in `HostHooks`, each optional:

| Hook | When | DeepSeek | Hy4 |
|---|---|---|---|
| `open(err)` | before the RAM tier is sized | vision BLAS, `PromptHost::open` | its own staging buffers |
| `bytes()` | the RAM ceiling | `PromptHost::bytes()` | its buffers |
| `set_ring(pieces, err)` | experts stay in the file | `PromptHost::set_ring` | same ring |
| `staged_stride(layer)` | peer prompt share sizing | `staged_stride` | same |
| `ram_only()` | no expert in the file | `warm_token_table` | none |

`PromptHost` itself stays DeepSeek's (image buffers, Engram reads); the tier no longer owns it: `ExpertTier::host()` goes away
and DeepSeek holds it in `DsModel::SharedTier`.

## What Hy4 calls

Per MoE layer of a token or window, as DeepSeek's decode does:

```
tier.begin_token(n);
tier.lookup(layer, ids, 16, lk, err);        // slots, peers, RAM, file; stamps; lands the layer's queued copies
tier.fetch(layer, lk);                       // bus copies of misses beyond the CPU's half (not with the file read tier)
tier.start_cpu(layer, latent, lk, rows);     // or start_cpu_window with the per-expert row lists
  ... launch the GPU share on lk.dev / lk.peer ...
tier.finish_cpu();  tier.await_fetch(main);
tier.promote(layer, lk);                     // admission by admit_uses
tier.mark_gap(main);  tier.pump(layer, pieces);
```

Prompts use `locate(layer, e)`, `share_for(layer)`, `lend_all`, `reclaim`, `settle`. Placement: `open(reserve, h_rows, plan, err)`
queues the start's reads, `attach(err)` pins the RAM tier and seeds the slots. 2 to 4 cards: `peer_devices` of 1 to 3 cards; each
owns a contiguous block of layers (`split_by_bytes`), and layers are the tier's 41, not the model's 42.

## Verification

Every step: DeepSeek greedy identity against the previous build with tiers pinned (`tools/greedy_identity.py`), and ABBA decode
speed, numbers in the commit. Steps 1 and 4 touch no hot path; steps 2 and 5 are the ones that need the DeepSeek pack.
