// src/program/quality_suite.hpp - safetensors BF16 reference reader, Basalt score_top row parser,
// per-row KLD / top-1 / dp / PPL computation, and conversation-level bootstrap for basalt-bench --quality.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::program::quality {

enum class RoleFilter { Assistant, Other, All };

bool parse_role_filter(const std::string& s, RoleFilter& out);
const char* role_filter_name(RoleFilter r);
bool role_matches(uint8_t role, RoleFilter filter);

/// Stored BF16 reference file (.safetensors) produced by tools/quality_suite/extract.py (or slim_ref.py).
struct Reference {
    std::vector<int64_t> conv_off;
    std::vector<int64_t> row_off;
    std::vector<int32_t> tokens;
    std::vector<uint8_t> roles;
    std::vector<int32_t> topk_ids;
    std::vector<float> topk_lp;
    std::vector<float> leftover;
    std::string model_name;   // __metadata__["model"], if present

    int64_t num_convs() const { return conv_off.size() >= 2 ? static_cast<int64_t>(conv_off.size()) - 1 : 0; }
};

/// Loads a quality-suite .safetensors reference from `path` (supports I64/I32/U8/F32/F16 dtypes).
bool load_reference(const std::string& path, Reference& ref, std::string& err);

/// Selects conversation indices to run, matching tools/quality_suite/common.py:smoke_convs and basalt_score.py.
/// `skipped` receives the count of conversations excluded because their length exceeds `max_context`.
std::vector<int64_t> select_conversations(const Reference& ref, bool smoke, int64_t max_context, int64_t& skipped);

/// First relative position (i - conv_off[conv_idx]) with a stored reference row matching `role`, or -1 if none.
int64_t first_scored_pos(const Reference& ref, int64_t conv_idx, RoleFilter role);

/// One parsed line from Basalt `--ppl-out` (`pos target lp_target lse id:lp ...`).
struct ParsedRow {
    int64_t pos = 0;
    int32_t target = 0;
    int32_t top1_id = 0;
    float min_lp = 0.0f;
    std::vector<std::pair<int32_t, float>> entries;   // sorted by token id for binary search
};

/// Parses text rows from one conversation chunk written by `GEN 1 score=0 score_top=K`.
bool parse_basalt_rows(const char* data, size_t len, std::vector<ParsedRow>& out, std::string& err);

/// Per-row metrics for one reference row aligned with a Basalt output row.
struct ScoredRow {
    int64_t conv = 0;
    uint8_t role = 0;
    double kld = 0.0;
    double same = 0.0;
    double dp = 0.0;
    bool has_target = false;
    double nllb = 0.0;
    double nllq = 0.0;
};

/// Incremental scorer: feed one conversation at a time as the engine finishes it.
class Scorer {
public:
    explicit Scorer(const Reference& ref, RoleFilter role = RoleFilter::All) : ref_(ref), role_(role) {}

    /// Aligns `rows` for conversation `conv_idx` against the reference and appends per-row metrics.
    bool add_conversation(int64_t conv_idx, const std::vector<ParsedRow>& rows, std::string& err);

    int64_t missing_ref_ids() const { return missing_ref_ids_; }
    const std::vector<ScoredRow>& scored_rows() const { return scored_; }

private:
    const Reference& ref_;
    RoleFilter role_ = RoleFilter::All;
    int shift_ = -1;   // -1 = undetected, 0 = pos is target index, 1 = pos is source index
    int64_t missing_ref_ids_ = 0;
    std::vector<ScoredRow> scored_;
};

/// Aggregated quality-suite report with conversation-level bootstrap standard errors.
struct Summary {
    bool ran = false;
    std::string err;
    std::string ref_path;
    std::string ref_model;
    RoleFilter role = RoleFilter::Assistant;
    bool smoke = false;
    int top_k = 256;
    int max_context = 5120;
    int boot = 2000;

    int64_t n_rows = 0;
    int64_t n_convs = 0;
    int64_t n_convs_skipped = 0;
    int64_t missing_ref_ids = 0;
    int64_t rows_target_outside_topk = 0;

    double kld_mean = 0.0, kld_se = 0.0, kld_naive_se = 0.0;
    double kld_median = 0.0, kld_p90 = 0.0, kld_p95 = 0.0, kld_p99 = 0.0, kld_p99_9 = 0.0, kld_max = 0.0;

    double top1 = 0.0, top1_se = 0.0;          // fraction in [0, 1]
    double dp_mean = 0.0, dp_mean_se = 0.0;    // fraction in [-1, 1]
    double dp_rms = 0.0, dp_rms_se = 0.0;      // fraction in [0, 1]

    double ppl_q = 0.0, ppl_q_se = 0.0;
    double ppl_base = 0.0, ppl_base_se = 0.0;
    double ln_ppl_ratio = 0.0, ln_ppl_ratio_se = 0.0;
    double ppl_ratio = 0.0, ppl_ratio_se = 0.0;

    double load_s = 0.0;
    double run_s = 0.0;
    std::string engine_args;
};

/// Summarizes `scored` rows matching `role`, computing percentiles and `boot` conversation-level
/// bootstrap resamples with fixed `seed` (default 0).
bool summarize(const std::vector<ScoredRow>& scored, RoleFilter role, int boot, uint64_t seed,
               Summary& out, std::string& err);

}  // namespace basalt::program::quality
