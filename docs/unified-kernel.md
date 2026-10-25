# Unified kernel

Two pieces of the unified-kernel work are in the engine; the full-layer kernel it started from was retired.

## What runs

| Piece | Source | What it does |
|---|---|---|
| Mid chain (`k_mid`) | `mid_chain.cu`, `attn_chain_kernel.cuh`, `attn_chain_bound.cuh` | A GDN layer's middle in one persistent launch: the qkv + z projection pair, the recurrence and the output projection. It runs between the layer's two hyper-connection reads, which stay `hc_fused_read`. |
| Commit finisher | `verify_kernels.cu` | The GDN commit claims its items from a counter on its own stream; the next window starts with a finisher that takes what is left. With the device-planned tail and a whole GDN mixer, the commit stream takes only the first GDN layer: the later ones are claimed inside the next window, each before its own GDN kernels and ahead while the window waits for the CPU tier. The same launches read the next layer's projection weights into L2 while a combine waits for a tier (the CPU pool, or the peer's rows when the layer has no CPU entries). |

- The mid chain runs for windows of one to four rows, on every pack whose GDN projection triple (qkv, z, out) `mid_chain_supported` accepts. Types accepted for each of the three: Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS. `(14, 14, 14)` and `(23, 21, 12)` have specialised instances; any other combination of those types uses the generic one. A pack with Q8_0 or another type in a GDN projection keeps the three separate kernels.
- The pair ring always has an even slot count, because the two tile groups take alternate tiles: an odd ring would give a slot consecutive tiles from different groups, and one group could pass its wait before the data arrived. The parity test checks the ring size for every type triple.
- The grid is one CTA per SM, at most 170. Below 170 SMs the pair tiles are dealt round-robin, the recurrence heads go to the last min(48, SMs) CTAs (several per CTA when there are fewer) and a CTA takes several output tiles; a tile's bits do not depend on which CTA computes it. `hc_fused_read` claims its work items from a queue, so it runs on any SM count too. A device with fewer than 160 SMs has no one-tile output projection in the unfused path: there the mid chain's output projection uses the one-tile reduction order, which differs from that device's unfused kernels (and matches the larger device's).
- QSA layers, windows of five or more rows, batched windows and profiled runs keep the separate kernels.
- Arithmetic and reduction order are those of the kernels it replaces: bitwise identical output and state, no numerics change.
- The kernel needs every CTA resident, so it is a programmatic dependent of the first read and the second read is its dependent. The attention-chain header keeps the phase code (pair, recurrence, output) and the bounded waits (`attn_chain_bound.cuh` traps instead of hanging).
- `dense_tc.cu`, `dec_gdn_rec.cu` and `hc_fused.cu` carry `FZ_DEV_ONLY` guards so `mid_chain.cu` includes their device code without their host side. The translation unit is built without fast math.

## Tests

- `mid_chain_parity`: the kernel against `dense_tc_pair`, `dec_gdn_rec` and `dense_tc` in sequence, bitwise, every type triple, windows of one to four rows, then smaller grids against the device's own; it prints the phase ledger and has a microbench.
- `dec_commit_parity` and `dec_commit_finisher` (`--finisher`): the finisher run before, during and after the commit, with a microbench.
- `dec_commit_lazy` (`--lazy`): the in-window commit against the commit kernel and a CPU reference, and its read-ahead, with a microbench.

## What was retired

The branch first fused a whole layer into one kernel: the attention chain, the experts half (plan, gate/up, SwiGLU, down, shared expert, tier wait, combine) and a QSA attention half. Outputs were bitwise identical to the separate kernels, but it decoded slower than the separate kernels, so only the GDN middle and the finisher stayed.

Same session, IQ3_S pack, `--peer-exclusive`, prose fixture, speculation 7, GPU at 400 W (agent development A/Bs):

| build | window (ms) | tokens/s |
|---|---|---|
| separate kernels | 6.675 | 324.6 |
| full-layer kernel | 7.541 | 288.2 |

Per GDN layer at T = 3 the attention half was level with the separate kernels; the experts half lost about 11 us (plan + gate/up +21 us, the rest of the half -11 us). The QSA half lost about 35 us per layer (projections +13, output projection +11, hc2 + router +6). Per window that is about +0.4 ms from each.

Why it lost:

- The experts half is bound by cross-CTA dependency latency (gate/up round 2, then the h rows, then down), not by DRAM: about 58 us at T = 3 against a byte floor of about 30 us.
- The QSA half serialises projections that the separate kernels overlap with side kernels and adds grid barriers.
- Two CTAs per SM do not fit (99 KB arena, 512 threads at 128 registers).
- The layer kernel was about 1.5 MB of SASS; gate/up warps stalled on instruction fetch (20% of PC samples). Compiling per window width cut it to 1.13 MB and about 1% per window, with no change in the gate/up phase's end.
- Windows of one and five to seven rows gained nothing, and the fused layer needed the tag protocol off, which cost about 1% on windows of five to seven rows.
- The standalone harness predicted gains the engine did not show (ratio 0.1 to 0.3): it started every CTA with the experts known, while in the engine they arrive about 7 us after the attention output.

Persistent QSA attention and the in-kernel phase stamps (`BASALT_PROFILE`) went with it.

## Lessons

- Check register spills on the real engine binary (`cuobjdump --dump-resource-usage`). `nvcc --split-compile` on these units made the layer kernel spill about four times more and cost about 9% tokens/s. Never use it here.
- A kernel that does not fit its dynamic shared memory silently falls back; confirm the fused path runs before timing it.
- Time with the GPU lock held, no builds running and no per-launch memsets in hot loops; compare stamp numbers only inside one session, ABAB.
- Convert per-layer gains to window gains with the window-width mix (about 0.73 x 36 per layer-microsecond).
- A harness must reproduce the engine's start state (late ids, tier wait, shared-expert tail) before its deltas predict anything.
- Profile stall reasons with CUPTI continuous PC sampling, mapped to source through `nvdisasm -gi` on a `-lineinfo` build.
