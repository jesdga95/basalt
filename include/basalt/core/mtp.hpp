// include/basalt/core/mtp.hpp - the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer.  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (the converter quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "basalt/core/layer.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/core/session.hpp"
#include "basalt/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::core {

class NativeHead;

/// The drafter's files: the `mtp/...` sections of a .basalt.
struct MtpFiles {
    FileRange dense, index, experts, draft_vocab;   ///< dense, dense.index, experts, draft_vocab (optional)
    static MtpFiles from(const ModelSources& s) {
        MtpFiles f;
        f.dense = s.mtp_dense;
        f.index = s.mtp_dense_index;
        f.experts = s.mtp_experts;
        f.draft_vocab = s.mtp_draft_vocab;
        return f;
    }
};

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads the draft layer's files and allocates its K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized.
    bool load(const MtpFiles& files, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768);
    /// The bytes `load` places on the device (the dense weights and the experts; the K/V and buffers are working
    /// memory), from the section sizes and the dense index.
    static bool plan_bytes(const MtpFiles& files, const ModelGeometry& g, uint64_t& out, std::string& err);
    /// The draft layer's hyper-connection matrices that have 8-bit copies, as ("mtp." + tensor name, key) for
    /// hc8_dump.  After `load`.
    std::vector<std::pair<std::string, const void*>> hc8_keys() const;
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// A new request: the draft head's extra rows are cleared, and the tokens of the prompt that the
    /// draft subset lacks queue for them.
    void begin_request(const int64_t* prompt, int64_t n);
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    /// Captures every conditional draft graph (each draft-head tier, each window width) now rather than at its first
    /// use. After `bind` and `set_max_drafts`.
    bool prewarm(std::string& err);
    /// The cumulative draft rule's threshold for the next draft() chain (default: the rule's own).
    void set_cum_th(float th) { cum_th_ = th; }
    uint64_t vram_bytes() const { return vram_; }
    /// The draft layer's K/V state (read-only: the serve state check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// KV streaming: refill the ring of the drafter's window from its host copy for a sequence that continues at
    /// `upto` (a conversation-cache resume). No-op unless the drafter's K/V is a ring.
    void kv_restore(int64_t upto);
    /// The VRAM bind() will allocate for a native head of `head_row_bytes` per vocabulary row: the draft logits and
    /// the draft head over rt/draft_vocab.bin's subset.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const;
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err);

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, int* n_drafts = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, int* n_drafts = nullptr);

    void set_ple_session(SessionState* s) { ple_ss_ = s; }

    /// BATCH WINDOWS: one draft K/V per sequence.  `n` stores for the slots (set before `load`); the working state
    /// `st_` the graphs address holds one sequence at a time, `use_store` swaps another in (the old one's written
    /// cells go to its store).  Store `n` is the main sequence (the prompt path and solo requests), so `use_main`
    /// must precede anything that reads or writes it.  Needs the K/V fully resident.
    /// `cells[i]` (optional) is slot i's context: its store holds that many cells, the main store the full state.
    void set_stores(int n, std::vector<int64_t> cells = {}) { n_stores_ = n; store_cells_ = std::move(cells); }
    int main_store() const { return n_stores_; }
    bool use_store(int id, std::string& err);
    bool use_main(std::string& err) { return use_store(n_stores_, err); }
    /// Cells [0, cells) of sequence `from` into sequence `to`.
    bool copy_store(int from, int to, int64_t cells, std::string& err);
    /// `draft` for sequence `id` from the window rows `R_rows` (device, row 0 of the sequence's segment).
    /// Copies a verify window's rows (all slots) out of the buffer the draft graphs also use, so each slot's
    /// `draft_seq` / `defer_rows` reads its own rows intact; null on failure.
    const float* keep_rows(const float* R_rows, int rows, std::string& err);
    bool draft_seq(int id, const float* R_rows, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
                   std::string& err, float* probs, int* n_drafts);
    /// The K/V cells [cell0, cell0 + n) of sequence `id` from window rows, no draft (`prefill` on that sequence).
    bool catch_up(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
    /// The same, later: the rows are kept and their cells are computed by the next `draft_seq` (or `flush`) of the
    /// sequence, in one pass.  The copy runs on the drafter's stream: `idle` before the rows' buffer is reused.
    bool defer_rows(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
    bool flush(int id, std::string& err);
    /// Diagnostic (gate spec_log): the top-k tokens and softmax probabilities of the last draft's step j, from the
    /// draft head's logits kept per step.  Returns the count written (0 when the diagnostic is off).
    int diag_top(int j, int k, int32_t* ids, float* probs) const;
    /// Called once by the next draft() right after its graphs are launched (the host's own work in their shadow).
    void set_launch_hook(std::function<void()> fn) { launch_hook_ = std::move(fn); }

    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

    /// the prompt path computes this layer's prompt K/V in batches (Prefill::draft_kv): its tensors, its K/V
    /// state, the first cell a round can still read, its device, and a wait for its own stream.
    const float* tensor_f32(const char* name) const { return f32(name); }
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    QsaState& kv_state_rw();
    int64_t first_needed() const { return (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0; }
    int device() const { return device_; }
    bool idle(std::string& err) {
        if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
        return true;
    }

private:
    /// reuse_row: the full layer's row 0 from the catch-up's row *row_ (full, T == 1)
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err, bool reuse_row = false);
    void record_hidden(const float* R, int T, uint8_t* xq, cudaStream_t cs);
    void end_branches();
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< without the mapped staging (inputs copied on device)
    /// all_rows: copy every step / position record (the chain's too) and the threshold bits, for a graph without per-step
    /// copies
    bool record_round(int T, bool all_rows, std::string& err);
    bool record_step(int j, bool copy_records, std::string& err);
    bool capture_round(int T, std::string& err);
    bool capture_step(int j, std::string& err);
    cudaGraphExec_t step_exec_[2][9] = {};   ///< [tier_]
    /// The round and `steps - 1` chain steps in one graph, step j an IF node on draft j-1's
    /// probability passing the cumulative rule; false (with the per-step path from then on) where it cannot be built
    bool capture_cond(int T, int steps);
    cudaGraphExec_t cond_exec_[2][9] = {};   ///< [tier_]
    int cond_steps_[2][9] = {};
    bool cond_off_ = false;
    int chain_j_ = 0;   ///< record_forward: the chain step being recorded (the diagnostic logits' slot)
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    SessionState* ple_ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    std::function<void()> launch_hook_;
    void run_launch_hook() { if (launch_hook_) { auto f = std::move(launch_hook_); launch_hook_ = nullptr; f(); } }
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int max_drafts_ = 1 << 30;
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    // the branch stream of the round and chain graphs, its fork / join events, its Q8_1 input;
    // par_ while those graphs are recorded, pre_next_ / have_pre_ for the next step's hidden input computed beside the
    // head, fork_open_ while that branch is not joined yet
    cudaStream_t cs2_ = nullptr;
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr;
    uint8_t* xq2_ = nullptr;
    bool par_ = false, pre_next_ = false, have_pre_ = false, fork_open_ = false;
    struct CondTail { unsigned long long h[8]; int n; };
    const CondTail* ctail_ = nullptr;   ///< the conditional handles the step being recorded ends with
    bool ctail_done_ = false;           ///< that step's end already set them
    bool step_end(int j);
    cudaStream_t cs3_ = nullptr;          ///< the weight prefetch stream of the round and chain graphs
    cudaEvent_t ev_pf_ = nullptr, ev_pfj_ = nullptr;
    bool pf_open_ = false;
    void pf_fork(cudaStream_t from);
    void prefetch_end(cudaStream_t cs);
    void fork(cudaStream_t cs) { cudaEventRecord(ev_fork_, cs); cudaStreamWaitEvent(cs2_, ev_fork_, 0); }
    void join(cudaStream_t cs) { cudaEventRecord(ev_join_, cs2_); cudaStreamWaitEvent(cs, ev_join_, 0); }
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t prefill_dev_exec_[9] = {};
    int32_t* pf_dev_ = nullptr;   ///< a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaGraphExec_t round_exec_[2][9] = {};   ///< [tier_]

    struct Tensor {
        std::string name, kind;
        int64_t rows = 0, cols = 0;
        uint64_t off = 0, bytes = 0;
        const void* dev = nullptr;   ///< set: the tensor lives here instead of in dense_ (an 8-bit copy)
    };
    std::vector<Tensor> tensors_;
    static bool parse_index(const FileRange& index, std::vector<Tensor>& out, std::string& err);
    static bool is_hc8_matrix(const Tensor& t);
    uint8_t* dense_ = nullptr;
    uint64_t dense_bytes_ = 0;
    uint8_t* experts_ = nullptr;
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;
    // per-sequence K/V (set_stores): the stores' pool arrays, the sequence st_ holds, the cells each sequence has
    // written, and the range of st_ not yet saved to its store
    struct KvArr { void* p = nullptr; uint64_t page_bytes = 0; };
    int kv_layout(KvArr* out) const;
    void* store_ptr(int id, int i, const KvArr* lay) const { return id == cur_ ? lay[i].p : stores_[(size_t) id][(size_t) i]; }
    std::vector<std::array<void*, 4>> stores_;
    int n_stores_ = 0, cur_ = 0;
    std::vector<int64_t> store_hi_;
    std::vector<int64_t> store_cells_;   ///< set_stores: the slots' contexts
    std::vector<int64_t> store_pages_;   ///< pages each store holds (the main store: all of st_'s)
    int64_t dirty_lo_ = 0, dirty_hi_ = 0;   // empty when hi <= lo
    static constexpr int kKeepRows = 16;
    float* rkeep_ = nullptr;    ///< a window's rows, kept while the slots draft one after another
    float* rstage_ = nullptr;   ///< draft_seq's copy of the rows
    // the window rows live on another GPU than the drafter (a layer split's last stage): `relay_src_` is where
    // they are, `own_R_` takes the rows a round reads and `relay_` the blocks of a prompt's rows
    const float* relay_src_ = nullptr;
    float* own_R_ = nullptr;
    float* relay_ = nullptr;
    static constexpr int kRelayRows = 256;
    bool draft_rows(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err, float* probs,
                    int* n_drafts);   ///< `draft` with the rows already in the window buffer
    static constexpr int kPendRows = 32;
    std::vector<float*> pend_R_;                  ///< per sequence: the deferred rows
    std::vector<std::vector<int32_t>> pend_tok_;  ///< their next tokens
    std::vector<int64_t> pend_cell0_;              ///< the first one's cell
    void note_cells(int64_t lo, int64_t hi);
    bool prefill_cur(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float cum_th_ = -1.0f;   // < 0: the rule's own threshold
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    float* diag_ = nullptr;                         ///< spec_log: the draft head's logits per chain step (pinned)
    int64_t diag_stride_ = 0;
    std::vector<int32_t> diag_map_;                 ///< spec_log: the draft head's row -> token id
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    // the subset's leading base rows (n_tier1_) serve the argmax drafts unless the text just held a
    // token of the added rows; tier_ 0 = base rows only, 1 = every row (the graphs are captured per tier)
    int64_t n_tier1_ = 0;
    int tier_ = 1;
    int tier2_recent_ = 0;
    std::vector<uint8_t> tier2_;   ///< token id -> 1 when only the added rows hold it
    // context rows: n_xcap_ spare rows after the base rows, filled per request with the head rows of
    // tokens the subset lacks that the prompt or the text so far holds; an unused row has id -1 (never drafted)
    int64_t n_xcap_ = 0, xbase_ = 0, n_xused_ = 0;
    std::vector<uint8_t> in_sub_;   ///< token id -> 1 when the subset holds it or a context row does
    std::vector<int32_t> xpend_;    ///< tokens waiting for a context row
    std::vector<int32_t> xlist_;    ///< the tokens of the used context rows
    int32_t* dxids_ = nullptr;      ///< staging for a fill (n_xcap_ ids)
    void ctx_note(int64_t t);
    void ctx_flush(cudaStream_t cs);
    void* top_scratch_ = nullptr;   ///< the head's fused argmax partials
    uint8_t* proxy_ = nullptr;      ///< a 2-bit copy of the draft head's leading rows (n_tier1_), for the two-stage argmax
    void* proxy_scratch_ = nullptr;
    unsigned long long* gstamp_ = nullptr;   ///< --profile: the round's first and the draft's last GPU time
    FileRange draft_vocab_;   ///< the draft head's token subset, when the files have one
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
    float* probs_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace basalt::core
