// include/basalt/program/gpu_select.hpp - which GPUs a run uses and what each one does.
//
// Three settings mean the same in the server, the engine, basalt-bench and basalt-probe:
//   gpu         unset: the first visible card alone (one warning names the others when there are any);
//               A,B,...: exactly these cards, in this order, the first is the main card; all: every visible card, in order;
//   gpu-mode    peer, split or tensor: what the cards do together. Required with two or more cards, refused with one;
//   gpu-share   one positive number per card, in the order of gpu: their relative shares. Refused with one card.
// Nothing is picked for the user: a setting the run cannot honour is refused with the reason (architecture/multi-gpu.md).
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

/// What the cards do together: One is a single card (no gpu-mode).
enum class GpuMode { One, Peer, Split, Tensor };
/// "" for One, else "peer", "split" or "tensor".
const char* gpu_mode_name(GpuMode m);

/// The `gpu` setting as written.
struct GpuSpec {
    enum class Kind { Unset, List, All };
    Kind kind = Kind::Unset;
    std::vector<int> list;          ///< visible numbers, main first (List)
    std::string text;               ///< the setting as the errors name it: "", "all" or "A,B,..."
};

/// Parses "", "all" or "A,B,..."; false with `err` naming what is wrong.
bool parse_gpu_spec(const std::string& text, GpuSpec& out, std::string& err);

/// Parses "" (One) or "peer", "split" or "tensor"; false with `err` for anything else.
bool parse_gpu_mode(const std::string& text, GpuMode& out, std::string& err);

/// Whether the card can be the main card: compute capability 12.0, or unknown (nothing to check against).
bool gpu_main_supported(const GpuCard& c);
/// Whether the card can be a second card: as the main card, or 8.9 in a build with -DPEER_ADA=ON.
bool gpu_peer_supported(const GpuCard& c);

/// "GPU 1 (RTX 5060 Ti)": the visible number the user types, and the name when known.
std::string gpu_label(const std::vector<GpuCard>& visible, int index);

/// The cards a run uses, as indices into the visible table, main first, and the lines to print once.
struct GpuChoice {
    std::vector<int> cards;
    GpuMode mode = GpuMode::One;
    std::vector<double> shares;     ///< gpu-share normalized to sum 1, one per card; empty when none was given
    std::vector<std::string> notes; ///< info lines
    std::string warning;            ///< unset with more cards visible
    std::string error;              ///< non-empty: no run
};

/// The choice for the three settings over the visible cards (`mode` One and `share_text` "" when unset). `known` is false
/// when the visible cards could not be listed (no management library and no CUDA_VISIBLE_DEVICES): numbers are then taken
/// as given and `all` is an error.
GpuChoice choose_gpus(const GpuSpec& spec, GpuMode mode, const std::string& share_text, const std::vector<GpuCard>& visible,
                      bool known = true);

/// What a model runs: the most cards each mode takes, 0 = the model does not run it.
struct GpuModes {
    int peer = 0, split = 0, tensor = 0;
    bool peer_ada = false;          ///< its peers' kernels are among those a -DPEER_ADA=ON build compiles for sm_89
};

/// Empty when `model` runs the choice's mode on that many cards of `visible`; else the refusal naming the model and the
/// modes it runs, or the peer card its peer kernels are not built for.
std::string check_gpu_modes(const GpuChoice& choice, const std::string& model, const GpuModes& modes,
                            const std::vector<GpuCard>& visible = {});

/// The visible cards: CUDA_VISIBLE_DEVICES when set (integers read as nvidia-smi numbers), else every card the NVIDIA
/// management library lists, with what that library says about each. `known` false: neither was available.
std::vector<GpuCard> visible_gpus(bool& known);

/// Narrows this process's CUDA view to `cards` of `visible`, main first; children inherit it. Call before any CUDA call.
void apply_gpus(const std::vector<GpuCard>& visible, const std::vector<int>& cards);

/// The `gpu` value that means "every visible card, in order": what a launcher passes the engine once it narrowed the view.
std::string gpu_identity(size_t n);

/// The choice as one JSON line (the engine's --gpu-plan answer, read by the server), with its "mode" and "shares".
/// `arch`: the model's architecture when one was given; `spec` the verify window and `max_context` the context the start
/// would run (0: no model given); `lossy` the flags given that change the model's output, comma separated.
std::string gpu_plan_json(const std::vector<GpuCard>& visible, const GpuChoice& c, const std::string& arch = {},
                          const std::string& quality_flags = {}, int quality_window = 1, int spec = 0, int64_t max_context = 0,
                          const std::string& model_name = {}, const std::string& lossy = {},
                          const std::string& community_kv = {});

/// For a launcher (basalt-bench, basalt-probe): resolves the three settings ("" = unset), prints its lines under `label`,
/// narrows the environment the engine inherits, and returns the engine's `gpu` value in `engine_value` and the chosen cards
/// in `used`. False with `err` set when the settings cannot run. The model's own limits are the engine's to check.
bool select_gpus(const std::string& text, const std::string& mode_text, const std::string& share_text, const char* label,
                 std::string& engine_value, std::vector<GpuCard>& used, std::string& err);

}  // namespace basalt::program
