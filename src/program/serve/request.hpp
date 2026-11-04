// src/program/serve/request.hpp - one request's state and execution phases.
#pragma once

#include "request_base.hpp"

namespace basalt::program::serve {

class QfnRequest final : public Request {
public:
    explicit QfnRequest(QfnRuntime& runtime) : Request(runtime), runtime_(runtime) {}

private:
    QfnRuntime& runtime_;
    using PfPart = QfnRuntime::PfPart;
    basalt::core::SessionState* rview{};
    int64_t kv_need{};
    struct KeepRouted {   // only this admission's fits, while slots decode
                bool& on;
                ~KeepRouted() { on = false; }
            };
    std::optional<KeepRouted> keep_routed;
    std::optional<KeepRouted> defer_peer;
    bool kv_ok{};
    // `copied`: the copies already ran (and were waited for) on the overlapped read's thread, so only the table is left
    bool ov_copied{};
    // --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one
    // run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots
    // decode for a share of the time the chunk took (half), so a long prompt does not stop the others.  The
    // slots' windows then see the cache without the slots this prompt borrowed (marked missing: those experts
    // run on the CPU), never a prompt buffer.
    static constexpr double decode_share = 0.5;
    bool batch_fatal{};
    int64_t il_parts{};
    double il_ms{};
    int64_t yielded_at{};
    int yielded_slot{};
    // ---- the prompt-time layer split: a long part of the prompt is read across the cards when its price there
    // (split_price) beats the main card's alone; the chunk, its head chunk, the stage count and the cuts, or none
    struct SplitPlan { int64_t chunk = 0, head = 0; int stages = 0; std::vector<int> cuts; };
    // per-request sampling for the verify window's head (greedy when temperature is absent)
    basalt::kernels::SamplerParams req_sp{};
    bool relax_on{};
    bool in_call{};
    int64_t relaxed_kept{};
    int hist_n{};
    bool cancelled{};
    int64_t turn_at{};
    int64_t root_at{};
    int64_t at{};
    int64_t fold_p0{};
    double prompt_ms{};
    int64_t p{};
    int32_t x{};
    std::vector<int32_t> drafts{};
    std::vector<int32_t> window{};
    std::vector<int32_t> outv{};
    std::vector<float> dprob{};
    basalt::core::DraftCumAdapt cum_adapt{};   // this request's draft threshold
    std::vector<int32_t> sbuf{};
    bool first_window{};
    int64_t produced_n{};
    int64_t sfx_windows{};
    int64_t sfx_drafts{};
    int64_t sfx_ok{};
    int64_t draft_offered{};
    int64_t draft_accepted{};
    // what the session holds once this request is done: the prompt read so far, then every committed token
    std::vector<int32_t> consumed{};
    const char* finish{};
    std::string kv_stop{};   // the elastic KV could not back the next window: the request ends on this error
    basalt::core::Verifier* cv{};
    basalt::core::Verifier* ov{};
    struct HeldWindow { bool on = false; int T = 0; std::vector<int32_t> win, out; };
    HeldWindow held{};
    std::vector<int32_t> win_b{};
    int n_chain{};
    double full_ema{};
    bool pair_hint{};
    bool pair_req{};
    int64_t pair_runs{};
    int64_t pair_holds{};
    Clock::time_point d0{};
    bool dec_timing{};
    struct DecSnap {
        double wait, pool, host, plan, actq, jobs, run, peer_wait;
        int64_t misses, entries, hits, pcie, peer;
    };
    DecSnap ds0{};
    double dt_run{};
    double dt_commit{};
    double dt_draft{};
    int64_t dec_windows{};
    int64_t dec_T{};
    int64_t decode_hits0{};
    int64_t swapped0{};
    int64_t noplace0{};
    int64_t ram0{};
    int64_t files0{};
    uint64_t file_bytes0{};
    int64_t decode_look0{};
    int64_t offload0{};
    int adapt_age{};
    bool applied_early{};
    JobRunner apply_thr{};
    JobRunner adapt_thr{};
    bool copy_live{};
    bool copy_done{};
    struct TkStep { int n = 0; int32_t id[4] = {}; float p[4] = {}; };
    TkStep tk_prev[16]{};
    bool tk_valid{};
    double decode_ms{};
    int64_t req_hits{};
    int64_t req_look{};
    int64_t req_offload{};
    int64_t fresh{};
    int64_t read_n{};
    bool ram_out{};
    // "12288 of 98179" when cancelled mid-read, the rate from what was read
    char read_txt[64]{};
    bool windows_ok(int64_t a, int64_t b);
    bool read_windows(int64_t a, int64_t b, std::string& e);
    bool refill_issue(PfPart& p, std::string& e, bool assign = true);
    bool refill_wait(PfPart& p, std::string& e);
    bool refill_one(PfPart& p, std::string& e);
    bool refill(std::string& e);
    bool lend(int64_t tokens, std::string& e, int64_t chunk = 0, int64_t head = 0);
    bool can_yield(int64_t q, int64_t b0, int ys, bool beside = false);
    bool park_read(int64_t q, int ys, std::string& ye, int64_t chunk = 0);
    SplitPlan split_plan(int64_t a0, int64_t b0, const char* no_split);
    bool read_split(int64_t a0, int64_t b0, const SplitPlan& plan, bool& ran, std::string& e);
    bool read_part(int64_t a0, int64_t b0, std::string& e);
    DecSnap dec_snap();
    Flow parse() override;
    Flow prepare_image() override;
    Flow read_prompt() override;
    Flow decode() override;
    Flow complete() override;
    bool cvec_wanted() const override;
    void set_cvec() override;
    void select_view() override;
    bool slot_draft_in(int slot) override;
    void drop_stale_reuse() override;
    bool release_room() override;
    bool checkpoint_mount(const ConvCheckpoint& c) override;
    bool restore_begin() override;
    void slot_draft_out() override;
    Flow slot_bind() override;
    void validate_incoming() override;
    Flow make_room() override;
    Flow restore_image() override;
    void adopt_checkpoints() override;
    void retain_image() override;
    bool session_reset() override;
    void draft_begin() override;
    void prompt_parts_reset() override;
};

}  // namespace basalt::program::serve
