// basalt/models/qwen4exp/options.hpp - Flash-Next's flags on top of the common ones.
#pragma once

#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/program/options.hpp"

namespace basalt::models::qwen4exp {

struct Options : CommonOptions {
    /// THE CONTEXT EXTENSION (rope scaling, rope_scaling.hpp).  These knobs resolve to ONE process config,
    /// set once before `session_init` builds the rope table and the graphs capture the kernels.  There is
    /// deliberately no per-request form: K sits in the cache POST-RoPE, so one cache must never mix two
    /// scalings.  Precedence: an EXPLICIT flag over the model file's rope keys over the struct defaults -
    /// `none` and `1` are explicit values (the opt-outs), the absent flag is not.
    std::string rope_scaling;           ///< --rope-scaling none|linear|yarn (llama.cpp's names); empty = the flag is absent
    double rope_scale = 0;              ///< --rope-scale F: the extension factor; 0 = the flag is absent (the model file's, else 1 = off)
    double rope_freq_base = 0;          ///< --rope-freq-base N: 0 = the model's (1e7)
    double rope_freq_scale = 0;         ///< --rope-freq-scale F: the raw ggml knob; 0 = 1/--rope-scale
    double yarn_orig_ctx = 0;           ///< --yarn-orig-ctx N: 0 = the model's, else 262144
    double yarn_ext_factor = -1.0;      ///< --yarn-ext-factor F: <0 = auto (1 for yarn, 0 otherwise)
    double yarn_attn_factor = 1.0;      ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;       ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;        ///< --yarn-beta-slow F
    /// How the n-gram table is read.  Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 256;   // the prompt path reads a chunk's rows at once, so the SSD stays busy
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the large GEMV does not.
    std::string dump_mixed;
    /// The routing trace: one record per layer per position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is
    /// what a hit-rate curve for a candidate VRAM expert cache is computed from, and it needs no new kernels - the
    /// doorbell already publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< pre + post + head, the true per-token GPU floor
    bool mmap_experts = false;    ///< opt OUT of the resident arena, back to a file mapping
    std::string shared_expert_arena; ///< optional file backing for the resident arena shared by processes
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// `--resident-experts` (the low-RAM resident mode): `--resident-cpu-experts` with the copy
    /// page-locked when the driver allows (else locked in the working set), the RAM reserve as headroom, and plain mmap
    /// (with a warning) when even the experts no slot holds do not fit.
    bool resident_pin = false;
    uint64_t resident_headroom = 8ull << 30;
    bool resident_soft = false;
    bool resident_cpu_explicit = false;   // --resident-cpu-experts given by itself (not only implied)
    int64_t peer_slots = 0;
    int peer_adapt_swaps = -1;
    /// The peer's experts are held only on the peer, not also in the RAM complement (needs --resident-experts).
    bool peer_exclusive = false;
    /// --no-peer-prefill: the prompt path stays on the primary, the peer computes none of its rows
    bool no_peer_prefill = false;
    /// --serve (opt-in): where to save what the adaptive tier learned, as a profile (the resident experts first,
    /// then the routing counted since the start; the format of data/profiles); on QUIT and every
    /// `expert_profile_save_min` minutes between requests.  Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// Capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    /// Opt-in: the expert cache in physical segments (CUDA virtual memory management), so the serve loop's
    /// `VRAM <reserve_mib>` command can give part of it back to other programs and take it again.  Off: one cudaMalloc.
    bool vram_elastic = false;
    int64_t vram_segment_mib = 512;
    /// Batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  Every expert a chunk
    /// routes to is streamed once per chunk, so a bigger chunk streams fewer bytes per token (the "ubatch" effect).
    bool prefill_auto = false;
    bool prefill_given = false;   // --prefill on the command line
    /// Opt-in: the largest chunk `--prefill auto` may take - 8192 by default; `--prefill auto:16384` or
    /// `auto:32768` lets it go further, never past the context
    int64_t prefill_auto_max = 8192;
    bool prefill_auto_fit = false;   // a bare `auto`: the ceiling is what the lend can fund (see auto_ceiling)
    /// Validation: batch only positions [0, P) and run the rest of the prompt through the token path (teacher-forced),
    /// so the logits of positions >= P - which depend on the batched state - can be scored against the oracle at many
    /// positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    /// After every processed position, append the residual after the last layer (hc x n_embd floats, the MTP draft
    /// head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// --relaxed-drafting [P]: under sampling, a draft the target gives probability >= P is kept even when the target's
    /// own draw differs (0 = exact; the flag alone = kRelaxedDraftingDefault).  This changes the output distribution:
    /// docs/precision/qwen38-flash-next.md#relaxed-drafting.  Solo decode only; a request's relax=P key overrides it.
    float relaxed_drafting = 0.0f;
    /// --route-tail-skip [R]: in a decode window an expert no GPU holds is neither copied nor computed when every
    /// entry of the window that names it ranks R or lower by router weight (0 = off; the flag alone =
    /// kRouteTailSkipDefault).  The others keep their router weights.  This changes the output:
    /// docs/precision/qwen38-flash-next.md#lossy-routing-flags.
    int route_tail_skip = 0;
    /// --route-prior [LAMBDA]: a decode window picks its top-10 as if every expert a GPU holds had LAMBDA added to its
    /// router logit; the weights stay the router's own (0 = off; the flag alone = kRoutePriorDefault). This changes
    /// the output:
    /// docs/precision/qwen38-flash-next.md#lossy-routing-flags.
    float route_prior = 0.0f;
    /// Never relaxed: the added tokens (control and tool tags, ids from relax_special_from on) and every token
    /// between the tool-call tags, so tool calls are sampled exactly.
    int64_t relax_special_from = 248044;
    int64_t relax_call_open = 248058, relax_call_close = 248059;
    /// The --batch slots in this many groups pipelined through the stages of a layer split (stage k
    /// runs one group while stage k+1 runs another).  1 = every slot in one window, stage after stage.
    int batch_groups = 1;
    int spec_corrupt = 0;         // --spec-corrupt N > 0 replaces every Nth draft with a wrong token
    /// Names the model's draft layer when it has one (the file); empty means no drafter.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// The share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the pinned arena while the
    /// CPU computes the rest (host-planned verify windows).  Set from the startup link probe.
    double pcie_frac = -1.0;
    /// Every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into the VRAM tier in
    /// place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    bool adapt_every_given = false;   // --adapt-every on the command line
    bool peer_exclusive_given = false;   // --peer-exclusive on the command line
    float adapt_decay = 0.7f;   ///< the usage counts are multiplied by this after each adaptation (--adapt-decay)
    bool adapt_decay_set = false;
    double prefill_cpu_share = 0.6;   ///< the share of the streamed prompt experts the CPU computes (--prefill-cpu-share)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// The most experts one adaptive round swaps into the main card's tier; a saturated pipeline (see `kSatBoost`) may
    /// swap `kSatBoost` times as many.
    int adapt_swaps = 96;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (basalt/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// A control vector on the residual stream (basalt/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
};

}  // namespace basalt::models::qwen4exp
