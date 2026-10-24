// include/basalt/program/gpu_select.hpp - which GPUs a run uses and what each one does.
//
// The `gpu` setting means the same in the server, the engine, basalt-bench and basalt-probe:
//   unset       the first visible card alone (one warning names the others when there are any);
//   A,B,...     exactly these cards, in this order: the first is the main card;
//   auto        every visible Blackwell card, the fastest (memory bandwidth, then VRAM) as the main card.
// Cards are numbered as nvidia-smi numbers them; under CUDA_VISIBLE_DEVICES the numbers count within that list. The
// choice is made before any CUDA call and narrows CUDA_VISIBLE_DEVICES to the chosen cards, main first, so the engine
// always runs on device 0 with the rest after it. The decisions are pure functions over a card table.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::program {

/// One card this process can see.
struct GpuCard {
    std::string token;              ///< its CUDA_VISIBLE_DEVICES spelling: an nvidia-smi number or a UUID
    int smi = -1;                   ///< its nvidia-smi number; -1 when unknown
    std::string name;               ///< "RTX 5090"; empty when unknown
    int cc_major = 0, cc_minor = 0; ///< compute capability; 0 when unknown
    double mem_gbps = 0;            ///< peak memory bandwidth; 0 when unknown
    uint64_t vram = 0;              ///< bytes; 0 when unknown
};

/// The `gpu` setting as written.
struct GpuSpec {
    enum class Kind { Unset, List, Auto };
    Kind kind = Kind::Unset;
    std::vector<int> list;          ///< visible numbers, main first (List)
};

/// Parses "", "auto" or "A,B,..."; false with `err` naming what is wrong.
bool parse_gpu_spec(const std::string& text, GpuSpec& out, std::string& err);

/// Whether Basalt runs on the card: compute capability 12.0, or unknown (nothing to check against).
bool gpu_supported(const GpuCard& c);

/// "GPU 1 (RTX 5060 Ti)": the visible number the user types, and the name when known.
std::string gpu_label(const std::vector<GpuCard>& visible, int index);

/// The cards a run uses, as indices into the visible table, main first, and the lines to print once.
struct GpuChoice {
    std::vector<int> cards;
    std::vector<std::string> notes; ///< info lines: the cards `auto` skipped
    std::string warning;            ///< unset with more cards visible
    std::string error;              ///< non-empty: no run
};

/// The choice for `spec` over the visible cards. `known` is false when the visible cards could not be listed (no
/// management library and no CUDA_VISIBLE_DEVICES): numbers are then taken as given and `auto` is an error.
GpuChoice choose_gpus(const GpuSpec& spec, const std::vector<GpuCard>& visible, bool known = true);

/// What a chosen card does.
enum class GpuRole { Main, ExpertTier, Idle };
const char* gpu_role_name(GpuRole r);

/// The inputs of the role decision: the chosen cards, main first, and what this platform and engine can run.
struct RoleInputs {
    std::vector<GpuCard> cards;
    bool peer_access = true;        ///< the platform gives a process GPU-to-GPU access (the expert tier needs it)
    int max_expert_tiers = 1;       ///< how many cards the engine can run as expert tiers
};

/// The role of each card, in order: the cards after the main one are expert tiers, as many as the engine runs, where
/// the platform allows it, and idle otherwise. The layer split (each card a block of layers) is never picked: it
/// measured slower than the expert tier and than the main card alone (docs/configuration/gpus.md).
std::vector<GpuRole> choose_roles(const RoleInputs& in);

/// One expert tier's room for experts (bytes) and its card's memory bandwidth (0: unknown, counted as equal).
struct TierRoom {
    uint64_t bytes = 0;
    double gbps = 0;
};

/// Deals ranked experts (their slot bytes, hottest first) to the expert tiers: each goes to the tier that has taken the
/// fewest so far for its bandwidth among those it still fits, so the hot experts split by bandwidth and every tier
/// fills. A tier closes at the first expert that no longer fits it. Returns each expert's tier, -1 when none takes it.
std::vector<int> spread_experts(const std::vector<uint64_t>& bytes, const std::vector<TierRoom>& tiers);

/// The visible cards: CUDA_VISIBLE_DEVICES when set (integers read as nvidia-smi numbers), else every card the NVIDIA
/// management library lists, with what that library says about each. `known` false: neither was available.
std::vector<GpuCard> visible_gpus(bool& known);

/// Narrows this process's CUDA view to `cards` of `visible`, main first; children inherit it. Call before any CUDA call.
void apply_gpus(const std::vector<GpuCard>& visible, const std::vector<int>& cards);

/// The `gpu` value that means "every visible card, in order": what a launcher passes the engine once it narrowed the view.
std::string gpu_identity(size_t n);

/// The choice as one JSON line (the engine's --gpu-plan answer, read by the server).
std::string gpu_plan_json(const std::vector<GpuCard>& visible, const GpuChoice& c);

/// For a launcher (basalt-bench, basalt-probe): resolves `text` ("" = unset), prints its lines under `label`, narrows
/// the environment the engine inherits, and returns the engine's `gpu` value in `engine_value` and the chosen cards in
/// `used`. False with `err` set when the setting cannot run.
bool select_gpus(const std::string& text, const char* label, std::string& engine_value, std::vector<GpuCard>& used,
                 std::string& err);

}  // namespace basalt::program
