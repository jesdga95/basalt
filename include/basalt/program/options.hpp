// basalt/program/options.hpp - the `basalt generate` flag set, resolved into one struct.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/platform/memory.hpp"

namespace basalt::program {

/// The context of every model without --max-context.
inline constexpr int64_t kDefaultContext = 8192;

/// The flags every model takes.
struct CommonOptions {
    virtual ~CommonOptions() = default;
    std::string model;                ///< --model: the .basalt file every loader reads from
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = kDefaultContext;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    /// The files the source tensors are read from: the model file, which serves them as a GGUF.
    std::vector<std::string> native_shards;
    int pool_workers = 0;         ///< >0 = the user's worker count; otherwise the start-up pick
    /// The pool's core layout; the start-up pick unless given
    basalt::kernels::cpu::PoolAffinity pool_affinity = basalt::kernels::cpu::PoolAffinity::Auto;
    /// CS-T `--resident-budget-gib N`: the resident mode with a RAM budget - the N GiB of experts the GPU cache does
    /// not hold that the expert profile ranks hottest are copied into RAM, the rest are read from the files in place
    /// (the GGUF shards when the model file has no experts section).  0 = the whole complement (--resident-experts).
    uint64_t resident_budget = 0;
    /// Slots of VRAM-resident experts.  **0 = off, and off is the default.**  Switching the cache on moves work off the
    /// CPU pool (the model's hit kernel).
    int expert_cache = 0;
    /// Multi-GPU: a second expert tier on CUDA device `peer_device` (-1 = off), `peer_reserve_mib` left free
    /// there, `peer_slots` caps its size (0 = as many as fit), `peer_adapt_swaps` per adaptive round
    /// (-1 = adapt_swaps).
    int peer_device = -1;
    /// every expert tier's device, the first one peer_device, and each card's memory bandwidth (GB/s, 0 unknown)
    std::vector<int> peer_devices;
    std::vector<double> peer_gbps;
    int peer_reserve_mib = 600;
    bool no_prompt_skip = false;  ///< --no-prompt-skip: every prompt row through every layer
    std::string kv;               ///< --kv: the KV storage, as each model names it (empty: the model's default)
    /// The engine's expert profile for the model (--expert-profile, else data/profiles/<key>.prof, cli::profile_key),
    /// or the --expert-profile-save file a start places from; empty means none.  A profile decides residency by routing
    /// frequency instead of by first use.
    std::string expert_profile;
    /// `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session and the KV state, minus
    /// this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   // --vram-reserve-mib on the command line (no smaller automatic reserve)
    /// --grammar on|off: a request's "grammar" key is enforced on the sampled tokens (the vocabulary is read from the
    /// pack at start).  Off: the key is ignored and the decode loop pays one pointer test per window.
    bool grammar = false;
    /// --serve runs up to this many sequences at once, one token each per batch window (0 = off).  A request
    /// sent as `BGEN <slot> <max_new> ...` reads its prompt and its first token through the usual path, then
    /// continues in slot <slot> of the batch windows (`BT <slot> <id> [<id> ...]` lines, then `BDONE <slot> ...`).
    int batch = 0;
    bool resident_budget_given = false;  // --resident-budget-gib on the command line
    bool pool_affinity_given = false;    // --pool-affinity on the command line
    bool pool_workers_given = false;     // --pool-workers on the command line
    /// Stop when the model emits an end-of-turn token (the pack's own, or --eos-ids).
    bool stop_eos = false;
    /// --ram-reserve-mib: the RAM left free for the rest of the system (platform::ram_reserve); 0 turns the guard off.
    int64_t ram_reserve_mib = (int64_t) (basalt::platform::kDefaultRamReserveBytes >> 20);
    std::vector<int64_t> eos_ids;
    /// --gpu: the cards this run uses (gpu_select.hpp); empty = GPU 0 alone.  --gpu-plan prints the choice and exits.
    std::string gpu;
    bool gpu_plan = false;
    std::string arch;                  ///< the --model file's architecture
    std::string ppl_base, ppl_out;     ///< --ppl-base, --ppl-out
    std::string model_name;            ///< the adapter's community-row name, for --gpu-plan
    std::string quality_flags;         ///< the flags the quality gate starts this model with, for --gpu-plan
    int quality_window = 1;            ///< the widest window the gate scores on this model (0: as the request asks)
    int plan_spec = 0;                 ///< the verify window this start would run, for --gpu-plan (0: plain decode)
    /// Speculative decoding with a verify window of this many tokens (the last accepted token and spec-1 drafts);
    /// 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for the exactness test).
    int spec = 0;
    bool spec_given = false;           ///< --spec on the command line
    std::string spec_oracle;
    int draft_experts = 32;            ///< --draft-experts N: the drafter's routed experts each stage keeps (0: all)
    int drafter_device = -1;           ///< --diag-drafter-gpu N: the CUDA device it runs on (-1: the main card)
    bool diag_teacher = false;         ///< --diag-teacher: --spec-oracle's tokens go on whatever the picks
    bool diag_tier_frozen = false;     ///< --diag-tier-frozen: the VRAM tier keeps what it holds
    std::string diag_dump;                 ///< --diag-dump DIR: a model's per-layer tensors, off by default
    int diag_tier_ram = -1, diag_tier_vram = -1;   ///< --diag-tier-ram / --diag-tier-vram: fixed expert-tier sizes
    bool diag_serial_prompt = false;       ///< --diag-serial-prompt: an adapter model reads its prompt one decode step at a time
    std::string diag_logits;               ///< --diag-logits FILE: an adapter model writes the logits of each decoded row
    std::vector<std::string> images;       ///< --image FILE ... with --image-at N: the picture's block goes before token N
    std::vector<int64_t> image_at;
    bool image_pad_given = false;
    int64_t image_pad_id = -1;             ///< --image-pad-id: the prompt token that stands for a picture's rows (--serve)
    /// Stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    bool generates = true;             ///< false: this run produces no tokens (a model flag says so)
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests (--vision and
    /// --vision-external).
    bool vision = false;
    /// --vision (not --vision-external): the image encoder runs in this process on the model's GPU.  It reads the
    /// model file's `vision/mmproj` section in place; its weights stay in pinned host memory; each ENC command borrows
    /// expert-cache slots, encodes in them and refills them.
    bool vision_in_engine = false;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole prompt
    /// again).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and the PLE history (~118 MB of host
    /// RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    int64_t conversation_cache_mib = 0; // opt-in host RAM for independent conversations
    int conversation_cache_slots = 4;
    int64_t conversation_cache_min_free_mib = 2560;
    int conversation_cache_min_tokens = 0;   ///< --serve: park only conversations of at least N tokens (0 = park every one)
    /// --serve: the folder of the disk tier (the server names one inside the model cache folder): a conversation
    /// that leaves the session or the RAM cache, and every one the process holds when it ends, is written there and
    /// read back when a prompt starts with it, after a restart too (conversation_store.hpp)
    std::string conversation_cache_dir;
    int64_t conversation_cache_disk_mib = 0;   ///< byte budget of the disk tier, least recently used first; above 0 turns it on
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (the pack's own, or --turn-token).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = -1;
    bool turn_token_given = false;
    /// --serve (opt-in): the role token of a short turn the server puts right before the new assistant turn (the
    /// "system" of a trailing reasoning-effort turn).  When the turn before the last <|im_start|> opens with it,
    /// the checkpoint goes in front of that turn instead, so it holds the conversation and not the effort text.
    /// -1 = off (the checkpoint at the last <|im_start|>).
    int64_t tail_role_token = -1;
    /// A flag on in this start that changes the model's output: the flag as typed and the warning line that says how.
    struct LossyFlag {
        std::string flag, effect;
    };
    /// Filled by the model's finish_flags; the start warns about each and --serve reports them (INFO lossy=).
    std::vector<LossyFlag> lossy;
};

}  // namespace basalt::program
