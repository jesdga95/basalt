// src/models/qwen4exp/serve/qfn_request.hpp - a Flash-Next request: its state and its phases.
#pragma once

#include "basalt/models/qwen4exp/shared_names.hpp"
#include "../../../program/serve/request.hpp"
#include "qfn_state.hpp"

namespace basalt::models::qwen4exp::serve {

class QfnRequest final : public Request {
public:
    explicit QfnRequest(QfnRuntime& runtime) : Request(runtime), runtime_(runtime) {}

    static constexpr bool kSpeculates = true;
    // ---- the window loop's calls; defined in the one file that instantiates it
    Flow decode_begin();
    void seed_tiers();
    void decode_state_begin();
    void decode_stats_begin();
    int max_window() const { return runtime_.S; }
    int draft_window();
    void lookup_window(int& T, bool& from_sfx, int& sfx_match);
    void fill_drafts(int T, bool from_sfx);
    int before_window(int T, int64_t pos);
    int32_t* history_stage() { return runtime_.hist_stage.data(); }
    void upload_history(int T) {
        const basalt::core::OnDevice on_h(runtime_.hist_dev);
        cudaMemcpy(runtime_.d_hist, runtime_.hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
    }
    void mark_before_run() {
        basalt::core::decode_gaps().mark(19);
        basalt::core::decode_events().stamp(3);
    }
    void spawn_adapt(bool early);
    void tier_before(int T, int e0, bool from_sfx);
    void tier_after_commit();
    void tier_finish();
    /// The adaptive step of one window.
    class TierStep {
    public:
        explicit TierStep(QfnRequest& request) : r_(request) {}
        ~TierStep() { r_.tier_finish(); }
        void before_window(int T, int e0, bool from_sfx) { r_.tier_before(T, e0, from_sfx); }
        void after_commit() { r_.tier_after_commit(); }
        void finish() { r_.tier_finish(); }
        bool ok() const { return r_.adapt_ok; }
    private:
        QfnRequest& r_;
    };
    bool run_window(int T, int e0, int& TB);
    bool take_held(int T);
    bool window_failed() const { return runtime_.drive.d.failed; }
    const char* window_error() const {
        return runtime_.drive.d.failed && runtime_.drive.d.fail ? runtime_.drive.d.fail : runtime_.err.c_str();
    }
    void relax_note_forced(int a);
    int accept_drafts(int T);
    bool resample_row(int row, int64_t pos, const std::vector<float>& bias, int32_t& tok, std::string& e) {
        return cv->resample_row(row, pos, bias, tok, e);
    }
    bool copy_logits(int row, float* out) { return cv->copy_logits(row, out); }
    bool row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) {
        return cv->row_logprobs(row0, n, picks, top, out);
    }
    bool after_accept(int T, int& a, int e0, bool from_sfx, bool held_run, bool paired, bool& hold);
    void note_draft_probs(bool from_sfx);
    void stamp_before_commit() { basalt::core::decode_events().stamp(9); }
    bool commit(int keep, std::string& e, int phase) { return cv->commit(keep, e, phase); }
    void on_token(int32_t tok) { if (runtime_.o.suffix_draft > 0) runtime_.sfx->append(tok); }
    void count_round() { ++runtime_.rounds; }
    template <class After>
    bool draft_next(int T, int a, int e0, int TB, bool from_sfx, bool hold, bool will_draft, After& after_launch);
    bool draft_fatal() const { return draft_failed; }
    void window_timing(int T, int a, bool from_sfx, Clock::time_point round0, Clock::time_point tw0,
                       Clock::time_point tw1, Clock::time_point tw2);
    void mark_after_window() {
        basalt::core::decode_events().stamp(1);
        basalt::core::decode_gaps().mark(16);
    }
    void observe_policy(bool from_sfx, int T, int a, int sfx_match, Clock::time_point round0) {
        runtime_.policy->observe(from_sfx, T, a, sfx_match,
                                 std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
    }
    void mark_loop_end() { basalt::core::decode_gaps().mark(17); }
    void shed_low() { runtime_.shed_if_low(); }
    void stamp_loop_end() { basalt::core::decode_events().stamp(2); }
    void decode_end();

private:
    QfnRuntime& runtime_;
    bool try_pair{};
    bool do_adapt{};
    bool adapt_ok{true};
    bool adapt_early{};
    std::atomic<int> adapt_go{0};
    void* user_c{};
    void* user_o{};
    FILE* draft_log{};
    FILE* tk_log{};
    float wprob[16]{};
    int wprob_n{};
    bool draft_failed{};
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
    // ---- the prompt-time layer split: a long part of the prompt is read across the cards when its price there
    // (split_price) beats the main card's alone; the chunk, its head chunk, the stage count and the cuts, or none
    struct SplitPlan { int64_t chunk = 0, head = 0; int stages = 0; std::vector<int> cuts; };
    bool relax_on{};
    bool in_call{};
    int64_t relaxed_kept{};
    std::vector<int32_t> drafts{};
    std::vector<float> dprob{};
    basalt::core::DraftCumAdapt cum_adapt{};   // this request's draft threshold
    std::vector<int32_t> sbuf{};
    int64_t sfx_windows{};
    int64_t sfx_drafts{};
    int64_t sfx_ok{};
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
    int64_t req_hits{};
    int64_t req_look{};
    int64_t req_offload{};
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
    bool parse_begin() override;
    float default_relax() const override;
    Flow prepare_image() override;
    bool prompt_begin() override;
    Flow set_sampling() override;
    void limit_parts() override;
    Flow read_to(int64_t to) override;
    Flow prompt_end() override;
    Flow decode() override;
    Flow complete_begin() override;
    DoneCounts done_counts() override;
    bool tiers_settle() override;
    bool admit_draft() override;
    void slot_state_back(int slot) override;
    bool slot_draft_keep(int slot) override;
    void slot_set_sampling(int slot) override;
    void slot_init(BSlot& sl) override;
    bool slot_leave() override;
    void report_request(const DoneCounts& dc) override;
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

}  // namespace basalt::models::qwen4exp::serve
