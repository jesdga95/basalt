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

class DsparkDrafter {
public:
    ~DsparkDrafter();

    static std::unique_ptr<DsparkDrafter> plan_load(const DsModel& main, int file_handle, core::LoadPlan& reads,
                                                    int max_windows, std::string& err);

    int open_window();
    int feature_slot(int layer) const;
    int feature_floats() const;
    int block_size() const;
    int n_stages() const;
    uint64_t device_bytes() const;
    uint64_t host_bytes() const;

    void forget_from(int w, int64_t pos);
    bool inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream, std::string& err);
    bool draft(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream, DsparkDraft& out, std::string& err);

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

}  // namespace basalt::models::deepseek4
