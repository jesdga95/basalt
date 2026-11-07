// src/models/deepseek41/prompt_host.hpp - the host memory prompts and images take at their peak, allocated once at start.
#pragma once

#include "basalt/vision/preprocess.hpp"
#include "models/deepseek41/pack.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

/// Bytes between experts in a staging buffer: the layer's blob rounded up to whole blocks of its two formats.
uint64_t staged_stride(const ExpertLayer& e);

/// What the prompt path and the image path take in host RAM at their peak. It is allocated before the RAM tier reads the
/// free memory, so the tier sizes beside it, and nothing large is allocated per request after it. Prompts are read one at
/// a time, so the prompt buffers are one set; each slot has its own image buffers.
class PromptHost {
public:
    PromptHost() = default;
    ~PromptHost();
    PromptHost(const PromptHost&) = delete;
    PromptHost& operator=(const PromptHost&) = delete;

    /// `ring_pieces`: experts the pinned ring of file and second-card reads holds (0: none); `slots`: sessions;
    /// `vision`: each slot gets the buffers of the largest image.
    bool open(const DsModel& m, int ring_pieces, int slots, bool vision, std::string& err);
    /// The ring at `pieces` experts (its contents are not kept).
    bool set_ring(int pieces, std::string& err);
    /// Page-locked bytes, which the driver keeps tracking memory for.
    uint64_t pinned_bytes() const { return pinned_; }
    uint64_t bytes() const { return bytes_; }

    size_t slot = 0;                    ///< bytes of one staged expert (staged_stride of the largest layer)
    uint8_t* ring = nullptr;            ///< [ring_pieces][slot], pinned
    int ring_pieces = 0;
    uint8_t* stage[2] = {};             ///< [kStageExperts][slot] each, pinned: a short read's two launches
    float* hx = nullptr;                ///< [share_rows][n_embd] a short read's activations, pinned
    float* hrows = nullptr;             ///< [kShareRows][n_embd] rows the CPU returns, pinned and mapped
    uint8_t* act = nullptr;             ///< [share_rows][kNativeActBytes] the CPU's quantized activations
    int share_rows = 0;                 ///< most tokens of a short read
    std::vector<float> embed;           ///< [kEmbedRows][4 streams][n_embd]
    std::vector<float> engram_rows;     ///< [kEngramRows * n_cols][row_values] dequantized
    std::vector<uint8_t> engram_raw[2]; ///< a chunk's Engram reads and the next one's: [layer][kMaxChunk * n_cols][row_bytes]
    size_t engram_layer = 0;            ///< bytes of one layer in engram_raw

    struct Image {
        vision::Image picture;          ///< the decoded image
        std::vector<float> patches;     ///< its patch rows
    };
    /// The next slot's image buffers (each session takes one), nullptr when there are none left.
    Image* take_image();

private:
    std::vector<Image> images_;
    int images_taken_ = 0;
    uint64_t pinned_ = 0, bytes_ = 0;
};

}  // namespace basalt::models::deepseek41
