// include/basalt/model/multi_gpu.hpp - the cards a run uses and what shared code decides for them: the shares, a split's
// cut, a peer's expert deal and prompt stages, the refusals and the two start lines.
//
// The user picks the cards (gpu), what they do together (gpu-mode) and their proportions (gpu-share); nothing here prices
// or picks a mode. A model states what it can divide (bytes per layer, experts by heat) and gets its part back, or the
// refusal with the numbers. Pure functions over byte counts and the card table: no GPU calls (architecture/multi-gpu.md).
#pragma once

#include "basalt/program/gpu_select.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::model {

/// The cards a load receives, main first.
struct GpuCards {
    struct Card {
        int device = 0;                 ///< the CUDA ordinal; the view is narrowed to the chosen cards, so the main card is 0
        program::GpuCard info;          ///< name, compute capability, memory bandwidth
        int index = 0;                  ///< the number the user types for it in gpu
        std::string label;              ///< "GPU 1 (RTX 5060 Ti)", that number and the name
        uint64_t free = 0;              ///< bytes free at start, before any load
    };
    program::GpuMode mode = program::GpuMode::One;
    std::vector<Card> cards;
    std::vector<double> given;          ///< gpu-share normalized to sum 1, one per card; empty when not given
    int diag_cut = 0;                   ///< --diag-gpu-cut: a two-card split's cut taken as given (0: from the shares)
    bool diag_same = false;             ///< --diag-gpu-same: a split's every part on the main card (the split's bitwise reference)
    /// The devices after the main card.
    std::vector<int> peers() const;
};

/// What a share divides; each has its own default when gpu-share is not given.
enum class ShareUse {
    Split,      ///< a split's layer bytes: free VRAM at start
    Experts,    ///< a peer's experts outside the main card: free VRAM at start
    Stages,     ///< a peer's prompt stages, the layers of a chunk: memory bandwidth
};

struct Shares {
    std::vector<double> of;             ///< one per card, summing to 1
    std::string from;                   ///< "gpu-share", "free VRAM", "bandwidth" or "equal" (nothing known)
};

/// The shares of `use` for the cards.
Shares shares_for(const GpuCards& c, ShareUse use);

/// The shares of the cards after the main one, summing to 1 among them.
std::vector<double> peer_shares(const GpuCards& c, ShareUse use);

// ---- split: the layers divided between the cards in order, each layer's KV and state on its card

struct SplitSpec {
    std::vector<uint64_t> layer;        ///< per layer: its weights plus its KV and state at the context and slots
    std::vector<uint64_t> fixed;        ///< per card: what it holds whatever its layers (the main card's head, drafter, scratch)
    std::vector<uint64_t> room;         ///< per card: the bytes it has for the model, past its reserves
};

struct SplitCut {
    std::vector<int> first;             ///< per card: its first layer; card i runs [first[i], first[i + 1]), the last up to n
    std::vector<uint64_t> need;         ///< per card: fixed plus its layers
    std::string error;                  ///< non-empty: no run (a card would hold nothing, or its part does not fit)
};

/// The cut nearest the Split shares (or --diag-gpu-cut): card i ends where the bytes of cards 0..i come nearest their shares
/// of every card's bytes. A card whose share is less than a layer would hold nothing and is refused; so is a card whose
/// part does not fit its room, with its numbers.
SplitCut split_cut(const GpuCards& c, const SplitSpec& s);

/// "layers 0-29" of card `i` of a cut over `n_layers`.
std::string cut_layers(const SplitCut& cut, size_t i, int n_layers);

// ---- peer: the main card runs every layer, the others hold experts and read prompt stages

/// Deals experts (their bytes, hottest first) to the peers: each to the open peer with the fewest taken for its share, a
/// peer closing at the first expert that no longer fits its room. Returns each expert's peer, -1 when none takes it.
std::vector<int> deal_experts(const std::vector<uint64_t>& bytes, const std::vector<uint64_t>& room,
                              const std::vector<double>& shares);

/// The refusal of a peer that would hold nothing: `card` indexes the cards (1 = the first peer), `why` says why ("the main
/// card holds every expert").
std::string peer_holds_nothing(const GpuCards& c, size_t card, const std::string& why);

/// A long prompt's chunk over the cards by the Stages shares: the main card reads layers [0, first[1]), card i layers
/// [first[i], first[i + 1]), the last up to `n_layers`. A stage reads within [lo, n_layers), the layers its weights may
/// hold, and every card reads one layer at least.
std::vector<int> stage_cuts(const GpuCards& c, int n_layers, int lo);

// ---- the start lines

/// Before the load: "gpus: peer; RTX 5090 (nvidia-smi 0) main, share 0.80; RTX 5060 Ti (nvidia-smi 1) peer, share 0.20
/// (bandwidth)"; one card: "gpus: RTX 5090 (nvidia-smi 0)". Peer shows the Stages shares, split the Split shares.
std::string gpu_start_line(const GpuCards& c);

/// After the load, from each card's part as its model reports it: "gpus: RTX 5090 dense, KV, ...; RTX 5060 Ti ...".
/// Empty when no card reports a part.
std::string gpu_parts_line(const GpuCards& c, const std::vector<std::string>& parts);

/// "15.2 GiB".
std::string gib(uint64_t bytes);

}  // namespace basalt::model
