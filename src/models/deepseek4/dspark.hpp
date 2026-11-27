#pragma once

#include "basalt/core/loader.hpp"
#include "models/deepseek4/pack.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek4 {

class DsModel;

constexpr int kDsparkBlock = 5;     ///< block rows: [anchor, noise x4]
constexpr int kDsparkWindow = 128;  ///< SWA context window before pos0
constexpr int kDsparkWinCap = 256;  ///< ring capacity (> kDsparkWindow + verify window)

struct DsparkDraft {
    int n = 0;
    int32_t token[kDsparkBlock] = {};
    float conf[kDsparkBlock] = {};
};

/// The experts each of `stages` stages keeps: its first `keep` in the ranked (stage, expert) pairs, topped up in index
/// order when the ranking names fewer, in ascending order.
std::vector<std::vector<int>> dspark_kept_experts(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                                  int stages, int experts, int keep);

class DsparkDrafter {
public:
    ~DsparkDrafter();

    /// Loads the pack's dspark section: the stage weights and each stage's `keep` most routed MXFP4 experts on the card
    /// (the ranking's order; 0 or every expert keeps all), the Markov head's first matrix on the card too. `ranking` is
    /// a profile (data/profiles): a pair per stage and expert, most routed first.
    static std::unique_ptr<DsparkDrafter> plan_load(const DsModel& main, int file_handle, core::LoadPlan& reads,
                                                    int max_windows, const std::string& ranking, int keep_asked,
                                                    std::string& err);

    int open_window();
    int feature_slot(int layer) const;
    int feature_floats() const;
    int block_size() const;
    int n_stages() const;
    int kept_per_stage() const;
    uint64_t device_bytes() const;
    uint64_t host_bytes() const;
    const std::string& kept() const;

    void forget_from(int w, int64_t pos);
    void reset(int w);
    bool inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream, std::string& err);
    /// Latents of a verify window's rows (at most 8, positions pos0..) made beside the window at low priority from
    /// their target features (device, on this card) once `ready` (an event of the writer's stream) has passed; `commit`
    /// then moves the accepted rows into window `w`'s rings.
    bool stage(int w, int64_t pos0, int n_rows, const float* d_features, void* ready, std::string& err);
    /// The first `n_rows` staged rows of window `w` from pos0 into its rings, as inject_device of their features would
    /// leave them. False (nothing done) when those rows were not staged.
    bool commit(int w, int64_t pos0, int n_rows, std::string& err);
    /// Starts the block over window `w` for the anchor token `anchor` at position `pos` on the drafter's stream; the
    /// first `n_out` rows become drafts. The caller's work queued on `caller_stream` runs before it.
    bool draft_begin(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream, std::string& err);
    /// Waits for the block draft_begin started and reads its picks and confidences.
    bool draft_end(DsparkDraft& out, std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    explicit DsparkDrafter(std::unique_ptr<Impl> impl);
};

// Internal CUDA helpers (dspark_kernels.cu)
void dspark_markov_argmax(const float* base_logits, const float* markov_bias, int n_vocab,
                          float* block_max, int32_t* block_idx, int32_t* out_tok, void* stream);
void dspark_confidence(const float* conf_w, const float* h_out, const float* me,
                       int n_embd, int rank, float* out_conf, void* stream);
void dspark_add_rows(const float* a, const float* b, float* out, int64_t count, void* stream);
void dspark_moe_gate_up(const void* gate_w, const void* up_w, int n_in, int n_ff, const void* x_q8_1,
                        const int32_t* ids, int n_tokens, int k, float clamp, const int32_t* live_rows, float* h,
                        void* stream);
void dspark_moe_down(const void* down_w, int n_ff, int n_out, const void* h_q8_1, const int32_t* ids, int n_tokens,
                     int k, const int32_t* live_rows, float* d, void* stream);
void dspark_markov_gather(const uint16_t* w1, const int32_t* tok, float* out, void* stream);
void dspark_markov_bias(const uint16_t* w2, const float* w1p, int n_vocab, float* col, void* stream);
void dspark_commit_rows(const float* staged, int stage_slots, float* ring0, float* ring1, float* ring2,
                        int ring_slots, int64_t pos0, int n, void* stream);

}  // namespace basalt::models::deepseek4
