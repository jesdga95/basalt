// src/models/inkling2/walk.hpp - which row buffers each visit of one Inkling 2 forward reads and writes.
//
// A forward keeps one buffer of rows per role below. Each visit reads its input, the finished blocks before it and, in the
// loop's later passes, the message bank; it writes its output where the next visit needs it kept: the layer before a block
// boundary writes straight into the buffer that becomes a finished block (or the bank), every other layer alternates two
// buffers. No step's output overwrites a buffer read later, so a forward copies nothing. The decode forward and the prompt
// forward both follow it.
#pragma once

#include "basalt/models/inkling2/kernels.hpp"
#include "models/inkling2/pack.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

/// The row buffers of one forward, by role (the layer numbers are Inkling 2 Mini's).
enum class Buf : uint8_t {
    X0,       ///< the normed embedding: layer 0's input, the first finished block
    Pre,      ///< the prelude's output: every pass's input of the loop's first layer, a finished block
    I12a,     ///< the input of the loop's second block (layer 12) in even passes, read by the next pass as a message
    I12b,     ///< the same in odd passes
    BankOut,  ///< a pass's output, read by the next pass as a message
    I18,      ///< the coda's input (the last pass's output), a finished block
    H0,       ///< every other layer's output, alternately
    H1,
};
constexpr int kBufs = 8;

struct Step {
    int visit = 0;               ///< index into Geometry::visits
    Buf hid = Buf::X0;           ///< the layer's input
    Buf out = Buf::H0;           ///< its output
    int n_blocks = 0;            ///< the finished blocks before it, oldest first
    Buf blocks[kMaxMix] = {};
    bool msgs = false;           ///< a later pass of the loop: its inputs take messages mixed from the bank
    Buf bank[2] = {};            ///< the previous pass's input of the loop's second block, and its output
};

struct Walk {
    std::vector<Step> steps;     ///< one per visit, in order
    Buf final_hid = Buf::H0;     ///< the last layer's output: the head mixes it with the finished blocks
    int n_blocks = 0;
    Buf blocks[kMaxMix] = {};
};

/// A run of steps on one card of a layer split, and the row buffers it pulls from the card before it first.
struct Segment {
    int card = 0;
    size_t s0 = 0, s1 = 0;       ///< walk steps [s0, s1)
    bool head = false;           ///< the head runs after them (always on card 0, the last segment)
    std::vector<Buf> pull;       ///< buffers its steps (or the head) read whose last write was on the other card
    std::vector<char> mix_msgs;  ///< [step - s0]: a later pass's first step on this card mixes the pass's messages here
};

/// The segments of `w` when layer l runs on card `card_of[l]` (0 or 1); the embedding and the head are card 0's. One card:
/// one segment, the whole walk and the head, the messages mixed at the loop's first layer as before.
std::vector<Segment> split_walk(const Geometry& g, const Walk& w, const std::vector<int>& card_of);

/// The hand-offs between the cards (segments - 1) of a window whose second card runs layers [cut, n_layers).
int split_handoffs(const Geometry& g, int cut);

/// The walk of one forward over `g`'s visits. False with `err` for a layout it does not cover: it takes a prelude of one
/// block, a loop of two blocks and a coda of one block (Inkling 2 Mini's 6 + 12 x passes + 6 layers).
bool walk(const Geometry& g, Walk& w, std::string& err);

}  // namespace basalt::models::inkling2
