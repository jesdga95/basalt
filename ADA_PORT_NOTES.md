# Ada (sm_89) port of Basalt — technical notes

Upstream Basalt is Blackwell-only (sm_120 / sm_120a). This port builds and runs the same engine on
Ada Lovelace (compute capability 8.9, RTX 40-series) by giving every sm_90+ instruction a
compile-out trap stub plus a runtime-dispatched fallback kernel — without changing what a Blackwell
build sees: the sm_120 SASS of every file this port touches is upstream's, apart from two kernels in
`verify_kernels.cu` whose constant-bank address load reads offset 8 instead of 0 (a module-scope
sink added ahead of them in the address table; the instructions are the same). The first port
changed 14 files; the sections below record what each newer upstream line needed on 8.9.

## What sm_89 does not have (so what any compile error will be about)

Thread-block clusters and distributed shared memory, TMA / `cp.async.bulk` and mbarrier
`expect_tx`/`try_wait`, `griddepcontrol` (PDL), `fence.proxy.async`, `mul.bf16x2` (all sm_90+), and
the block-scaled MMAs, `cvt e2m1` and the direct `cvt.rn.bf16x2.e4m3x2` (sm_120a / sm_100+).

It does have int8, bf16, f16 and plain e4m3 `mma.sync` (whose e4m3 accumulation keeps too few bits
for a W8A8 product — see the dense stack), `cvt.rn.f16x2.e4m3x2`, `cp.async`, and 99 KB opt-in
shared memory per block. nvcc builds whole-program (no rdc), so unreachable `__device__` helpers
are dropped: only code reachable from a compiled kernel must compile for sm_89.

## The pattern, and how to diagnose a failure

- Wrap sm_90+ instructions in `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` (files use local
  names for it: `HC_SM90_ONLY`, `DM_SM90_ONLY`, `REC_NO_CLUSTER`, `INK_NO_CLUSTER`); the kernel
  traps below that, and a **runtime** predicate picks the fallback so the trap is never launched.
- Parity tests that freeze kernel bits per architecture go through `frozen_for_device`
  (`kernels/test_util.hpp`): this port's values below compute 9.0, upstream's from 9.0 up.
- An illegal-instruction or launch failure in a new build means a `__trap()` stub was launched:
  rerun with `CUDA_LAUNCH_BLOCKING=1` (and `compute-sanitizer`) to find the kernel, then the
  runtime predicate that should have excluded it.

## Build and architecture rules

- Windows needs Visual Studio 2022 C++ tools, CUDA 13.x, CMake 3.24+, Ninja (see
  `docs/platforms.md`); a shell that does not load `vcvars64.bat` cannot find the toolchain.
- Single-architecture Ada: `-DCMAKE_CUDA_ARCHITECTURES=89`. Mixed Blackwell + Ada:
  `-DCMAKE_CUDA_ARCHITECTURES="120;89"`.
- Upstream reads the target capability through `cmake/target_arch.cmake` and `core/cuda_cc.hpp`
  (one 12.0 or 12.1 per build). This port lets `89` be built beside a Blackwell arch or alone; the
  reference capability then stays 12.0, so message texts and the gpu_select rigs are unchanged.
- `include/basalt/build_archs.hpp` (checked in `cli/gpu.cpp` and `core/device.cu`) refuses at
  startup a card the build has no device code for — otherwise, say, a Blackwell card would run the
  8.9 PTX with its sm_90 paths trapping. Upstream has no such gate.
- `PEER_ADA=ON` (upstream's #343) means "add 89 to `CMAKE_CUDA_ARCHITECTURES`" here, since every
  kernel has an 8.9 form.
- Libraries upstream builds for `120a-real` only (`window_gemm`, `prefill_fp8`, the dense FP4
  product `prefill_fp4_dense`, and the `narrow_bf16_parity` / `silu_parity` tests) build for
  `BASALT_CUDA_ARCHS_A` instead: `120a-real` in a 120 build (bit-for-bit upstream's), `89-real` in
  a pure 89 build. The MoE experts' FP4 kernels (`src/prefill/moe_fp4.cu`) stay `120a-real` and are
  never launched on 8.9.
- On a mixed set, FP4 prompt kernels run only when every engine card is 12.x
  (`fp4_tensor_cores()`, `layout.cpp`; `fp4_cores()` itself answers per device, since the layer
  split may pair a Blackwell and an Ada card), so all cards read NVFP4 prompts through MMQ; the
  buffers, chunk plan, peer share and row rounding are card-independent, and `fp4_pack()` /
  `fp4_every_layer()` decided once on the main card hold everywhere. Decode never uses FP4 tensor
  cores, so only prompt speed pays the difference.

## How the fallbacks are chosen

- **GDN recurrence**: `dec_gdnf_mode()` (`gdn_window.cu`) returns 0 below sm_90, so `record_gdn.cpp`
  takes the unfused sequence. `k_mid_ada` (`mid_chain_ada.cu`) is one persistent kernel bitwise
  equal to the separate kernels after a join on the L2 read-ahead; `--exp mid_ada=0` runs the
  separate kernels (gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, z on native_mmvq),
  `--exp mid_ada=2` the kernel with z on dense_tc. Dev A/B: about +2 % decode over the split path.
- **Dense projections**: the persistent `dense_tc` kernel (cp.async, int8 `mma.sync`) runs on 8.9;
  its bulk-copy variants (tile, one-tile, pair) stay off through `tc_bulk_ok()`, and so does
  `qsa_qproj_half` (`qsa_mid.cu`).
- **Lazy GDN commit**: off below sm_90 (`lifecycle.cpp`) — its read-ahead blocks wait beside the
  MoE grid and deadlock without PDL. The read-ahead itself (next layer's projections and FFN read
  matrices into L2) runs alone instead, launched beside the combine (`gdn_l2_ahead` in
  `record_moe.cpp`; `--exp l2_ahead=0` turns it off). Dev A/B: +0.7 % and +1.5 % decode, greedy
  output unchanged.
- **HC read**: one persistent kernel, `k_hc_ada` (`hc_read_ada.cu`), bitwise equal to hc8_gr's three
  kernels when no side branch is open, after a join on the L2 read-ahead; `--exp hc_ada=0` runs the
  three kernels. The router, the q8_1 rows and the draft graphs stay outside it. Dev A/B: about
  +3 % decode over the split path.
- **Draft reads**: the draft graphs' hyper-connection reads, their router and q8_1 rows run as one
  kernel, `hc_fused_ada` (`hc_fused_ada.cu`), where `hc_fused` does not run (upstream's
  `hc_fused_supported` already requires device major 12); verify stays on `k_hc_ada`;
  `--exp hc_fused_ada=0` keeps the separate kernels. A one-row read takes fused_gr_read_multi's
  bits instead of gr_read's, which only the drafts see — greedy, sampled with swaps off, and gate
  rows at equal slots are identical. `hc_fused_ada_parity`: 0 differing cases at T 1..8, 0 of 208
  stress reads, 0 stuck with 100 SMs held. Microbench, T = 1, hot/cold: read 24.4/32.7 ->
  12.5/26.3 us; with the router 38.1/44.2 -> 17.0/33.4 us.
- **Route**: without a peer, staged share or shared gate, `dec_moe_route` takes `dm_plan_ada_kernel`
  (block 0 plans, the others go straight to the q8_1 rows) and `dm_ring_ada_kernel` (one block
  publishes the header and x in 16-byte stores and rings; blocks for the warm-up's rows do not hold
  the doorbell back, so those rows may land after it — which changes no output).
  `--exp dm_route_ada=0` runs `dm_route_kernel`. `dm_route_ada_parity`: 0 differing bytes in 2560
  cases; greedy identical. The pooled gain is small and mostly on prose (route busy time
  7.6 -> 4.3 % of the cycle there); expert hit rates are unchanged.
- **HC read + route together**: about +6 % decode end to end over the build without them (community
  bench suite, rotated pairs); the gate's rows (100 chunks, T 3) stay byte-identical, greedy and
  sampled with `--adapt-swaps 0`.
- **PDL launches** (`launch_pdl`, `launch_dependent`) check `pdl_kernel_ok` and launch plain below
  9.0: a programmatic edge without `griddepcontrol` only adds latency — 25 to 42 us once a
  fork/join has moved the chain to another of the graph's internal streams.
- **NVFP4 prefill**: `fp4_tensor_cores()` is true only for major >= 12, and `fp4_pack()`,
  `fp4_every_layer()` and `fp4_l` (`experts.hpp`) depend on it, so NVFP4 experts take the MMQ path
  (`src/prefill/mmq`, `mmq_nvfp4`) and the MMQ buffers are sized for every chunk length. A buffer
  overflow or wrong-size error in prefill points first at `moe_bufs` / `group_rows_cap` in
  `layout.cpp`.
- **VRAM TMA expert kernel** (`dec_moe_vram` kind 1) is never selected by default (`dm_->vram` is
  always -1 on 8.9).
- **GDN split by key head** (`gs_split`): its Blackwell half used `dense_tc_pair` and the cluster
  `dec_gdn_rec`, both unavailable on 8.9; it runs qkv and z as two `dense_proj` plus
  `dec_gdn_rec_split` + `dec_gdn_rec_norm` (the same bits, `gdn_window.hpp`).
- **Sampler / qsa_select cluster kernels** check the device major.
- **DeepSeek-V4.1**: its kernels use no bulk copies, griddepcontrol or FP4 tensor cores, and DSpark
  decodes MXFP4 in software. One kernel uses a cluster — the index key (`index_key_kernel`,
  `attn_index.cu`, eight blocks a token); below 9.0 it compiles without its cluster and traps, and
  `index_keys()` runs `index_key_block_kernel` instead (a block a token, the same row sums
  `index_key_row` and finish `index_key_finish`, so the keys keep the cluster form's bits —
  `block_sum`'s extra warps add zeros). The `ds41_*` parity tests pass on 8.9 (the peer test on one
  card).
- **Inkling 2** (`src/models/inkling2`) checks no compute capability. Its fused window attention (a
  cluster of eight blocks) compiles out and traps below 9.0 (`INK_NO_CLUSTER`); `attention_window`
  takes the split and the combine — upstream's own path for longer windows, the same bits. Its
  prompt NVFP4 experts take MMQ below 12.x (`fp4_cores()`). Bulk L2 prefetches in its glue and ops
  (and Hy4-flash's ops) run a line at a time below 9.0. Its tensor-core decode attention (#626,
  #606) is f16 mma and ldmatrix sized by the card's opt-in limits — nothing to port — and the
  `!cluster_device()` gate on the fused window stays beside upstream's `mma_decode()` route since
  both send the window to the split kernels.
- **Inkling's drafter pick (a real Ada codegen difference)**: `head_final_kernel` read the pick back
  from the restrict-qualified `*prev` thread 0 stored; the sm_89 compiler hoists that load above the
  barrier (sm_120's does not), so warps copied the previous draft's w1 row. Below 9.0 the pick comes
  from shared memory (`inkling2_dspark_parity` checks the row). The batched `head_final_batch_kernel`
  added later (0.5.7+79) needed the same fix: without it, batching drifted from solo in 5 of 64 ids
  (worst confidence delta 0.13); with it the path is bitwise and the parity test passes unrelaxed.

## The dense stack (Qwen3-27B class) on 8.9

- The 27B pack takes 16.0 GiB on the card with its drafter — it fits a 16 GB Ada card. Every kernel
  the stack calls has an 8.9 path, checked by upstream's kernel tests.
- Decode windows: `window_gemm` (cp.async, bf16 mma) runs as on 12.0. Its NVFP4 codes decode by a
  byte permute from the eight magnitudes' patterns, and its scale multiply is `fma.rn.bf16x2` with
  a -0 addend (`narrow_bf16.cuh`) — the same bits as 12.0a's conversions and `mul.rn.bf16x2`
  (`narrow_bf16_parity`: every input against the host values). `dec_gdn_rec` runs as
  `dec_gdn_rec_split` + `dec_gdn_rec_norm` (the same bits, `state_out` included; the raw output in
  `a.o`, or else `a.y`); batch windows use `dec_gdn_rec_seg_split` on every card;
  `dec_gdn_rec_seg` refuses below 9.0.
- `gdn_gating_proj` has a flat form: a block of 8 warps per 16 tokens, warp q taking the cluster
  form's rank q (a copy of its loop, so the cluster kernel stays upstream's source), partials added
  in rank order.
- **FP8 W8A8 (`fp8::gemm`)**: 8.9's e4m3 mma aligns its products to the largest in a 14-bit window
  that also holds the accumulator (f16 mma: 25 bits) — measured error 1.2e-3 of sum |w a| at K 6144
  (1.5e-4 with each k32 step's sum moved into fp32) against the test's 1e-5. So the codes decode to
  f16 (exact) and the product runs on f16 mma with fp32 accumulation: 1.6e-6, 134-161 TFLOP/s. The
  FP8 tensor cores would reach 220-268 TFLOP/s (1.65-1.85x) at 1.5e-4 — an open item, since the
  precision target forbids it. Only packs with F8_E4M3 projections call this.
- **FP4 W4A4 (`fp4_dense`)**: on 8.9 `kernel_ada` runs one e4m3 mma m16n8k16 per 16-value scale
  group from a zero accumulator, then adds the sum times the token's and the row's group scales in
  one fp32 fma. Why it is exact: every E2M1 code is an E4M3 value; a group's code products are
  multiples of 1/4 of at most 36, so all their bits lie within 8 bits of the largest product, inside
  the 14-bit window, and the sum (at most 576) is exact in fp32. Codes become E4M3 bytes by a byte
  permute; the epilogue is the TMA kernel's arithmetic; the quantizer rounds to E2M1 in software
  (`nvfp4_planes.cu`'s rounding). Measured error of sum |w x| in `fp4_dense_tma_test`: plain 6.6e-8,
  add 6.7e-8 (the f16 form it replaced: 6.4e-8, 8.4e-8); at 27B shapes gate (K 5120) 5.4e-8 and down
  (K 17408) 1.7e-8, against 3.0e-7 and 8.3e-8 before. Speed: 1.60-1.68x the f16 form (e.g. gate+up
  SwiGLU 213-215 against 128 TFLOP/s at 2048 tokens; down 172-175 against 107). Rows are bitwise at
  any chunk length; the sums run in another order than 12.0's block-scaled mma, so an Ada card's
  prompt rows are not a Blackwell card's bits.
- Inkling's two prompt GEMMs added in 0.5.6 were built for sm_120a alone (Ada prompt runs stopped at
  "no kernel image"); the Q8_0 projections take `gemm_ada_kernel` below 9.0 — the TMA kernel's tiles,
  fragments and int32 block dots loaded by cp.async, which `q8_gemm_test` replays bit for bit. When
  upstream later replaced the TMA kernels with two new ones whose tiles are kernel-local, the
  file-scope set the Ada kernel reads was restored beside them and both new kernels took the
  standard `< 900` trap (the dispatcher already keeps 8.9 on `gemm_ada_kernel`).
- MXFP8 experts (e.g. Inkling's MX8 pack) run each K=32 block as two f16 `mma.sync` on exactly
  decoded E4M3 codes with the UE8M0 scales on the block's sum — not the block-scaled mma's bits
  (`moe_mxfp8_test` holds).

## Expert-cache and L2 behaviour on Ada

- Below 9.0 a decode window reads no experts into L2 beside the attention (upstream 6d7dc337): the
  guessed experts' prefetch held the SMs at the wrong L1/shared-memory split and took the
  attention's bandwidth. Plain decoding +12 %; conv/prose +2.9 / +4.7 %, same text, both packs alike.
- Drafts accepted 56 / 97 / 40 % where a 5090 with the same build accepted 68 / 97.5 / 51 %: deep
  speculative drafts pay off less on Ada; keep draft depth modest.
- int8 KV roughly doubles what fits: a token's KV cache is 288 against 512 MiB at 16K, so int8 fits
  about 1.68x the context of fp16 with the same expert residency; a short-context window costs the
  same either way, so conv/struct/prose track the other texts. Experts stay on the card up to the
  largest context that fits and move to RAM above it; #542's dead zone just past the limit (a few
  tens of MiB that refused to start) was fixed upstream by #643.
- Staging a share of each layer's CPU expert misses over PCIe during solo windows
  (`--exp solo_stage=1`, `stage_frac`/`stage_min`; see `docs/configuration/diagnostics.md`) does not
  pay: no share beats the plain path beyond noise, and the stage kernel, launch and join cost
  0.2-0.5 ms a pass. Staging changes the sampled text, so retest over seeds.

## Multi-card on Ada

- GeForce cards have no GPU-to-GPU access; cards exchange data through pinned host buffers
  (`StagedCopy`) and `cudaMemcpyPeer`, as upstream does. Every kernel choice (`dec_gdnf_mode`,
  `tc_bulk_ok`, `hc_fused_supported`, `pdl_kernel_ok`, `fp4_cores`) is made per card, so in an
  all-Ada setup each card takes the same 8.9 paths one card would.
- 0.5.7's `gpu` + `gpu-mode` (`peer`/`split`/`tensor`) + `gpu-share` over the shared `model::GpuCards`
  layer needed no kernel fallback — the whole delta adds no TMA, `tcgen05` or cluster instruction to
  any file an 8.9 build compiles, only `ldmatrix` and `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`,
  both native to sm_89. What Ada does need is the host rule: upstream's `gpu_main_supported()` is
  sm_120 only, so `src/program/gpu_select.cpp` allows "12.0 or 8.9" for the main and the second
  card, and `gpu_peer_supported()` is the same predicate because this build gives every kernel an
  8.9 form.
- For a single-card box the multi-card machinery is config-neutral: `gpu "0"` is "unchanged, and
  silent", and `gpu-mode` is an error with one card.
- `gdn_head_split_parity` runs on 8.9 (split recurrence only, both ranks on one card) and is bitwise
  except one tolerance: the combined out projection's max error against the CPU must stay within 2x
  the whole layer's and is 2.3-3.0x in 11 of 60 cases (RMS within 1.2x) — the rounding of a
  two-part sum, the same arithmetic sm_120 uses; the threshold is left as it is. `dec_gdn_parity`
  (modes 2 and 3 and the batch-window segment kernels) passes bitwise; `peer_share_parity` and
  `ds41_peer_parity` pass with the share on one card. The two-card data paths (expert tier, layer
  split with overlapped windows, prompt split, GDN/head split, DeepSeek's second tier) have not
  been run on two physical Ada cards.

## Test conventions, frozen values, known fails

- sm_120-only tests skip below their compute capability: `hc_fused_parity`, `prefill_fp4_test`,
  `mid_chain_hold*`, `qsa_mid_parity`.
- Values frozen on Blackwell that take Ada entries (via `frozen_for_device`): the INT8 KV graph's PDL
  launches, `gqa_window_parity` (the one-tile kernel's own 8.9 bits), Inkling's full-attention
  hashes, and `gqa_attn_parity` (checked identical before and after upstream's next-chunk fetch by
  undoing that change in a test build). The Ada window-ring hash moved with the 16-row window wave,
  0x37a69c7be6281e30 -> 0xc31b1c9779f823ff (measured twice, identical; the reference check holds);
  Blackwell's side keeps upstream's re-frozen 0x72538aec1379b868. The DS4.1 set on 8.9: router
  f4ec7c76f483174c, gemv 258e734e874cae57, compressor d2ec8ec03dfea810, DSpark MoE
  c7b1acee87ad0e77, and (since the dp4a rows of 0.3.2+80, which changed the bits by design)
  9c13389dbf259e5f — the last one pins the kernel's own bits rather than a replaced kernel's.
- Measuring on a dev box: plain basalt-bench enables the per-window log unless `--no-windows` (about
  2 % slower — compare its tok/s only with its own), and it keeps the last of a repeated engine flag,
  so two experiment gates go in one `--exp a=0,b=0`. Gate rows follow the GPU slot count, and a
  bigger binary leaves less VRAM (a 24-slot drop changed nearly every row), so fix `--expert-cache`
  when comparing builds — with it equal, rows are byte-identical. conv/prose texts move results more
  than most changes do: compare over several seeds, and keep a look at `utilization.decoder` — a
  background app once cost a conv run 21.5 % across every window width.
- Known Windows fails unrelated to Ada: `file_expert_repin_test` assumes a registered copy's device
  alias is its host address (true under Linux's unified addressing); `iq_avx512_test`'s rows differ
  from its frozen hashes under MSVC (max rel error against double 4e-6); `ds41_vision_reserve_test`
  trips because WDDM lets an allocation past free VRAM succeed in system memory.
  `dense_ops_parity`'s old baseline (`bf16_rows_dot` + `gdn_gates` at T = 2048, 3.17e-4 against
  3e-4) is its tf32 mma accumulating below fp32 — the engine no longer calls that path.
  `prefill_mmq_mxfp4_test` fails IQ2_XS/IQ2_S gate/up (2.96e-4 against 1e-4) the same at every SM
  count and on every architecture — not an 8.9 issue.
- Fixed on the way, both not Ada-specific: the vision sidecar kept the image open until after its OK
  line, so the server's delete failed on Windows (WinError 32) and every sidecar image request was
  dropped; and the model-pack ctest entries were never registered because python3 was looked up
  after them. The server does not parse `gpu_select.cpp`'s error strings.

## Current state

- The suite on one Ada card passes: CPU 109/109, GPU 163/163 (the skips are missing packs
  and two-card rigs). Greedy and sampled output with swaps off are
  identical to the pre-port baseline with `--kv int8`; with the default fp16 KV they differ from the
  first token on by upstream's own design (its fp16 prompt attention rounds q and p to fp16 and runs
  one fp16 mma instead of hi + lo parts, c8f54503).
- The NVFP4 Flash-Next pack runs end to end (prompt/stop/grammar tests pass, vision sidecar
  included); the other Flash-Next formats (IQ1_S to Q8_0, the K-quants, Q2_0) pass their kernel
  parity tests (`iq_*`, `dec_moe_parity_kq`, `prefill_mmq_*`, `native_expert_parity_*`). 27B-class
  and Hy4 packs are kernel-verified on 8.9; end-to-end runs depend on having the pack.
