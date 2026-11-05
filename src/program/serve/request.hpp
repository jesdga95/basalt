// src/program/serve/request_base.hpp - one request's state and the phases every model shares.
#pragma once

#include "state.hpp"

namespace basalt::program::serve {

class Request {
public:
    explicit Request(Runtime& runtime) : runtime_(runtime) {}
    virtual ~Request() = default;
    Flow execute();

protected:
    Runtime& runtime_;
    using BSlot = Runtime::BSlot;
    using SlotHeld = Runtime::SlotHeld;
    bool geni{};
    char* endp{};
    long long max_new{};
    float req_temperature{};
    float req_top_p{};
    int req_top_k{};
    unsigned long long req_seed{};
    bool req_seed_given{};
    float req_min_p{};
    float req_penalty_repeat{};
    float req_penalty_freq{};
    float req_penalty_present{};
    int req_penalty_last_n{};
    std::string req_bias_text{};   // logit_bias=id:value,...: added to the verifier's logits (see ver.set_logit_bias)
    bool req_bias_present{};
    int64_t req_gprompt{};
    std::string req_grammar_text{};   // grammar=<compact ASCII JSON, spaces as \u0020>: the request's grammar (--grammar on)
    int req_cvec{};
    float req_relax{};
    int req_ppl{};
    int req_ppl_t{};
    int req_clip{};
    std::vector<float> req_bias{};
    std::string emb_path{};
    std::vector<int64_t> ids{};
    std::string pe{};
    bool ids_ok{};
    bool from_text{};
    int64_t n{};
    std::unique_ptr<basalt::program::grammar::Guard> gguard{};
    std::vector<float> grammar_bias{};
    int64_t grammar_resamples{};
    basalt::program::PplGate* ppl{};
    bool bad{};
    std::vector<int32_t> ids32{};
    Clock::time_point r0{};
    bool want_cvec{};
    int slot_b{};
    bool in_slot{};
    bool solo_req{};
    using SoloBook = Runtime::SoloBook;
    SoloBook solo_book{};
    bool solo_moved{};
    bool own_partial{};
    bool own_partial_from0{};
    int64_t own_partial_chunk{};
    int64_t resume{};
    bool from_live{};
    int slot_source{};
    int64_t slot_tokens{};
    bool resumed_from0{};
    int64_t resumed_chunk{};
    const ConvCheckpoint* slot_ck{};
    int own_slot{};
    basalt::core::ConversationCache::Match parked{};
    std::optional<basalt::core::SavedConversation> incoming{};
    int64_t in_tokens{};
    bool in_live{};
    basalt::core::ConversationStore::Match on_disk{};
    bool from_disk{};
    Clock::time_point restore_t0{};
    int64_t read_from{};

    // per-request sampling (greedy when temperature is absent)
    basalt::kernels::SamplerParams req_sp{};
    bool cancelled{};
    std::vector<int32_t> window{};
    std::vector<int32_t> outv{};
    bool first_window{};
    int hist_n{};
    int64_t p{};
    int32_t x{};
    // what the session holds once this request is done: the prompt read so far, then every committed token
    std::vector<int32_t> consumed{};
    const char* finish{};
    std::string kv_stop{};   // the next window could not be backed: the request ends on this error
    int64_t produced_n{};
    int64_t draft_offered{};
    int64_t draft_accepted{};
    double decode_ms{};
    Clock::time_point d0{};
    int64_t fresh{};
    int64_t read_n{};
    bool ram_out{};
    // "12288 of 98179" when cancelled mid-read, the rate from what was read
    char read_txt[64]{};
    /// DONE's tier fields.
    struct DoneCounts { int64_t hits = 0, lookups = 0, ram_blobs = 0, file_blobs = 0; double file_mb = 0; int64_t offloaded = 0; };
    int64_t yielded_at{};
    int yielded_slot{};
    int64_t turn_at{};
    int64_t root_at{};
    int64_t at{};
    int64_t fold_p0{};
    double prompt_ms{};
    bool starts_with(const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs);
    Flow restore();
    Flow parse();
    Flow prepare_prompt();
    virtual Flow prepare_image() = 0;
    /// `ppl=K`: opens chunk K of the gate's base, with at most `most_rows` scored per window.
    Flow begin_quality_gate(const std::string& base, const std::string& out, int most_rows);
    Flow read_prompt();
    virtual Flow decode() = 0;
    Flow complete();
    template <class B> Flow decode_loop(B& b);
    virtual bool parse_begin() { return true; }
    virtual float default_relax() const { return 0.0f; }
    // ---- the end of a request
    virtual Flow complete_begin() { return Flow::proceed; }
    virtual DoneCounts done_counts() { return {}; }
    virtual bool tiers_settle() { return true; }
    virtual bool admit_draft() { return true; }
    virtual void slot_state_back(int) {}
    virtual bool slot_draft_keep(int) { return true; }
    virtual void slot_set_sampling(int) {}
    virtual void slot_init(BSlot&) {}
    virtual bool slot_leave() { return true; }
    virtual void report_request(const DoneCounts&) {}
    /// The decode's expert cache line; the server's Experts panel shows the same counts from DONE.
    void report_expert_counts(const DoneCounts& dc) const;
    // ---- reading the prompt, per part.  read_to: Flow::quit when a stop ended the read.
    virtual bool prompt_begin() { return true; }
    virtual Flow set_sampling() = 0;
    virtual void limit_parts() {}
    virtual void prompt_parts_begin() {}
    virtual Flow read_to(int64_t to) = 0;
    virtual Flow prompt_end() { return Flow::proceed; }
    /// A template whose turn token opens the user's turn: the turn's checkpoint sits before the prompt's last token.
    virtual bool turn_checkpoint_at_last_token() const { return false; }
    // ---- restoring a conversation: what it does to the model's sessions.  Per request.
    virtual bool cvec_wanted() const { return true; }
    virtual void set_cvec() {}
    virtual void select_view() {}
    virtual bool slot_draft_in(int) { return true; }
    virtual void drop_stale_reuse() {}
    virtual bool release_room() = 0;
    virtual bool checkpoint_mount(const ConvCheckpoint& c) = 0;
    virtual bool restore_begin() { return true; }
    virtual void slot_draft_out() {}
    virtual Flow slot_bind() { return Flow::proceed; }
    virtual void validate_incoming() {}
    virtual Flow make_room() { return Flow::proceed; }
    virtual Flow restore_image() { return Flow::proceed; }
    virtual void adopt_checkpoints() {}
    virtual void retain_image() {}
    virtual bool session_reset() { return true; }
    virtual void draft_begin() {}
    virtual void prompt_parts_reset() {}
};

}  // namespace basalt::program::serve
