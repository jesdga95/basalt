// basalt/program/options.hpp - the `basalt generate` flag set, resolved into one struct.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "basalt/kernels/cpu/pool.hpp"

namespace basalt::program {

struct Options {
    std::string model;                ///< --model: the .basalt file every loader reads from
    std::string dump_hc8;             ///< --dump-hc8: write the 8-bit hyper-connection copies here, then exit
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
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
    /// The files the source tensors are read from: the model file, which serves them as a GGUF.
    std::vector<std::string> native_shards;
    /// How the n-gram table is read.  Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 256;   // the prompt path reads a chunk's rows at once, so the SSD stays busy
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    std::string kv = "fp16";   // KV storage, fp16 (default) or int8 (half the VRAM)
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
    int pool_workers = 0;         ///< >0 = the user's worker count; otherwise the start-up pick
    /// The pool's core layout; the start-up pick unless given
    basalt::kernels::cpu::PoolAffinity pool_affinity = basalt::kernels::cpu::PoolAffinity::Auto;
    bool mmap_experts = false;    ///< opt OUT of the resident arena, back to a file mapping
    std::string shared_expert_arena; ///< optional file backing for the resident arena shared by processes
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// `--resident-experts` (the low-RAM resident mode): `--resident-cpu-experts` with the copy
    /// page-locked when the driver allows (else locked in the working set), the RAM margin as headroom, and plain mmap
    /// (with a warning) when even the experts no slot holds do not fit.
    bool resident_pin = false;
    uint64_t resident_headroom = 8ull << 30;
    bool resident_soft = false;
    bool resident_cpu_explicit = false;   // --resident-cpu-experts given by itself (not only implied)
    /// CS-T `--resident-budget-gib N`: the resident mode with a RAM budget - the N GiB of experts the GPU cache does
    /// not hold that the expert profile ranks hottest are copied into RAM, the rest are read from the files in place
    /// (the GGUF shards when the model file has no experts section).  0 = the whole complement (--resident-experts).
    uint64_t resident_budget = 0;
    /// Slots of VRAM-resident experts.  **0 = off, and off is the default.**  Switching the cache on moves work off the
    /// CPU pool (`src/kernels/cuda/s2_expert_grouped.cu`, wired via `expert_hit_run`).
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::array<bool, 3> expert_cache_remote_auto{};
    bool remote_expert_opt = false; ///< optional optimization of the existing remote-expert decode path
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    /// Multi-GPU: a second expert tier on CUDA device `peer_device` (-1 = off), `peer_reserve_mib` left free
    /// there, `peer_slots` caps its size (0 = as many as fit), `peer_adapt_swaps` per adaptive round
    /// (-1 = adapt_swaps).
    int peer_device = -1;
    /// every expert tier's device, the first one peer_device, and each card's memory bandwidth (GB/s, 0 unknown)
    std::vector<int> peer_devices;
    std::vector<double> peer_gbps;
    int peer_reserve_mib = 600;
    int64_t peer_slots = 0;
    int peer_adapt_swaps = -1;
    /// The peer's experts are held only on the peer, not also in the RAM complement (needs --resident-experts).
    bool peer_exclusive = false;
    /// the prompt path's rows per layer the peer computes: -1 = half of chunk x top-k, 0 = the prompt path stays on
    /// the primary
    int64_t peer_prefill_rows = -1;
    /// Names the model's expert profile when it carries one (the file and `#profile`); empty means none.  A profile
    /// decides residency by routing frequency instead of by first use.
    std::string expert_profile;
    /// --serve (opt-in): where to save what the adaptive tier learned, as a profile (the resident experts first,
    /// then the routing counted since the start; tools/convert/convert.py reads it as `--profile`); on QUIT and every
    /// `expert_profile_save_min` minutes between requests.  Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// Capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    /// `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session and the KV state, minus
    /// this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   // --vram-reserve-mib on the command line (no smaller automatic reserve)
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
    /// Speculative decoding with a verify window of this many tokens (the last accepted token and spec-1 drafts);
    /// 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for the exactness test);
    /// `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
    int spec = 0;
    bool spec_given = false;      // --spec on the command line
    /// --relaxed-drafting [P]: under sampling, a draft the target gives probability >= P is kept even when the target's
    /// own draw differs (0 = exact; the flag alone = kRelaxedDraftingDefault).  This changes the output distribution:
    /// docs/precision/features.md.  Solo decode only; a request's relax=P key overrides it.
    float relaxed_drafting = 0.0f;
    /// --grammar on|off: a request's "grammar" key is enforced on the sampled tokens (the vocabulary is read from the
    /// pack at start).  Off: the key is ignored and the decode loop pays one pointer test per window.
    bool grammar = false;
    /// Never relaxed: the added tokens (control and tool tags, ids from relax_special_from on) and every token
    /// between the tool-call tags, so tool calls are sampled exactly.
    int64_t relax_special_from = 248044;
    int64_t relax_call_open = 248058, relax_call_close = 248059;
    /// --serve runs up to this many sequences at once, one token each per batch window (0 = off).  A request
    /// sent as `BGEN <slot> <max_new> ...` reads its prompt and its first token through the usual path, then
    /// continues in slot <slot> of the batch windows (`BT <slot> <id> [<id> ...]` lines, then `BDONE <slot> ...`).
    int batch = 0;
    /// The --batch slots in this many groups pipelined through the stages of a layer split (stage k
    /// runs one group while stage k+1 runs another).  1 = every slot in one window, stage after stage.
    int batch_groups = 1;
    std::string spec_oracle;
    int spec_corrupt = 0;
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
    bool resident_budget_given = false;  // --resident-budget-gib on the command line
    bool pool_affinity_given = false;    // --pool-affinity on the command line
    bool pool_workers_given = false;     // --pool-workers on the command line
    float adapt_decay = 0.7f;   ///< the usage counts are multiplied by this after each adaptation (--adapt-decay)
    bool adapt_decay_set = false;
    double prefill_cpu_share = 0.6;   ///< the share of the streamed prompt experts the CPU computes (--prefill-cpu-share)
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// --gpu: the cards this run uses (gpu_select.hpp); empty = GPU 0 alone.  --gpu-plan prints the choice and exits.
    std::string gpu;
    bool gpu_plan = false;
    /// Stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    /// --vision-mmproj PATH: the image encoder runs in this process on the model's GPU (implies --vision).  Its weights
    /// stay in pinned host memory; each ENC command borrows expert-cache slots, encodes in them and refills them.
    std::string vision_mmproj;
    /// The most experts one adaptive round swaps into the main card's tier; a saturated pipeline (see `kSatBoost`) may
    /// swap `kSatBoost` times as many.
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole prompt
    /// again).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and the PLE history (~118 MB of host
    /// RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    int64_t conversation_cache_mib = 0; // opt-in host RAM for independent conversations
    int conversation_cache_slots = 4;
    int64_t conversation_cache_min_free_mib = 2560;
    /// --serve: the folder of the disk tier (the server names one inside the model cache folder): a conversation
    /// that leaves the session or the RAM cache, and every one the process holds when it ends, is written there and
    /// read back when a prompt starts with it, after a restart too (conversation_store.hpp)
    std::string conversation_cache_dir;
    int64_t conversation_cache_disk_mib = 0;   ///< byte budget of the disk tier, least recently used first; above 0 turns it on
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve (opt-in): the role token of a short turn the server puts right before the new assistant turn (the
    /// "system" of a trailing reasoning-effort turn).  When the turn before the last <|im_start|> opens with it,
    /// the checkpoint goes in front of that turn instead, so it holds the conversation and not the effort text.
    /// -1 = off (the checkpoint at the last <|im_start|>).
    int64_t tail_role_token = -1;
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
};

}  // namespace basalt::program
