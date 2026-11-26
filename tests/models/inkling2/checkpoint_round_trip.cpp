// tests/models/inkling2/checkpoint_round_trip.cpp - Inkling 2's parked conversation round-trip through save_cells /
// restore_cells (positional full-attention caches) and running_state / running_restored (target SWA + conv rings and
// the embedded DFlash 2 or DSpark drafter's context rings + window bounds).
//
// Saves a conversation both inside the 2048-cell window (at = 784) and past ring wraparound (at = 2304), clobbers the
// session with an unrelated shorter conversation from position 0, restores the parked conversation, and checks:
//   1. re-saved running_state and save_cells match byte-for-byte, and mismatched KV vectors / corrupt heads are rejected;
//   2. on DFlash 2 packs, draft_begin / draft_end immediately at the restored position (before any new tokens) yields
//      all 7 draft token ids and confidences bit-for-bit identical to the uninterrupted conversation;
//   3. after folding a 6-token follow-up turn at the restored position, both DFlash 2 and DSpark yield all 7 draft
//      token ids and confidences bit-for-bit identical to the uninterrupted conversation (proving the drafter's window
//      spans the full restored history rather than resetting to the new turn), followed by identical greedy tokens.
//
// Needs an Inkling 2 pack (argument or BASALT_INKLING2_PACK) and a GPU; exits 77 without either.
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "models/inkling2/adapter.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace basalt;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool kv_equal(const std::vector<core::ConversationKv>& a, const std::vector<core::ConversationKv>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto &x = a[i], &y = b[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.k != y.k || x.v != y.v || x.k_scale != y.k_scale || x.v_scale != y.v_scale ||
            x.pooled != y.pooled)
            return false;
    }
    return true;
}

std::vector<int32_t> greedy(model::ModelSession& s, int32_t tok, int64_t pos, int n) {
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        int32_t next = -1;
        model::DecodeRequest r;
        r.tokens = &tok;
        r.n_tokens = 1;
        r.pos = pos + i;
        r.next_token = &next;
        if (const model::Result e = s.decode_window(r); !e.ok()) {
            std::fprintf(stderr, "decode: %s\n", e.message.c_str());
            std::exit(1);
        }
        out.push_back(next);
        tok = next;
    }
    return out;
}

int32_t fold_turn(model::ModelSession& s, const std::vector<int32_t>& turn, int64_t pos) {
    int32_t next = -1;
    model::DecodeRequest r;
    r.tokens = turn.data();
    r.n_tokens = (int) turn.size();
    r.pos = pos;
    r.next_token = &next;
    if (const model::Result e = s.decode_window(r); !e.ok()) {
        std::fprintf(stderr, "fold_turn: %s\n", e.message.c_str());
        std::exit(1);
    }
    return next;
}

struct DraftOut {
    std::vector<int32_t> ids;
    std::vector<float> conf;
    bool same_bits(const DraftOut& o) const {
        return ids == o.ids && conf.size() == o.conf.size() &&
               (conf.empty() || std::memcmp(conf.data(), o.conf.data(), conf.size() * sizeof(float)) == 0);
    }
};

DraftOut draft_at(model::ModelSession& s, int32_t anchor, int64_t pos, int want) {
    if (const model::Result e = s.draft_begin(anchor, pos, want); !e.ok()) {
        std::fprintf(stderr, "draft_begin: %s\n", e.message.c_str());
        std::exit(1);
    }
    int32_t ids[16] = {};
    float conf[16] = {};
    int n = 0;
    const auto all = [](const float*, int, int) { return true; };
    if (const model::Result e = s.draft_end(ids, conf, want, n, all); !e.ok()) {
        std::fprintf(stderr, "draft_end: %s\n", e.message.c_str());
        std::exit(1);
    }
    DraftOut d;
    d.ids.assign(ids, ids + n);
    d.conf.assign(conf, conf + n);
    return d;
}

std::vector<uint8_t> save_running(model::ModelSession& s, int64_t pos, size_t* head_bytes = nullptr) {
    core::RunningState rs;
    if (const model::Result e = s.running_state(pos, true, rs); !e.ok()) {
        std::fprintf(stderr, "running_state: %s\n", e.message.c_str());
        std::exit(1);
    }
    if (head_bytes) *head_bytes = rs.head.size();
    std::vector<uint8_t> bytes(rs.bytes());
    if (!rs.head.empty()) std::memcpy(bytes.data(), rs.head.data(), rs.head.size());
    std::string err;
    if (!core::running_copy_out(rs, bytes.data(), err)) {
        std::fprintf(stderr, "running_copy_out: %s\n", err.c_str());
        std::exit(1);
    }
    return bytes;
}

bool restore_running(model::ModelSession& s, const std::vector<uint8_t>& bytes, int64_t pos) {
    core::RunningState rs;
    if (!s.running_state(pos, false, rs).ok() || rs.bytes() != bytes.size()) return false;
    std::string err;
    if (!core::running_copy_in(rs, bytes.data(), err)) return false;
    return s.running_restored(pos, bytes.data()).ok();
}

void clobber_with_short_conversation(model::ModelSession& s) {
    if (const model::Result e = s.set_history(nullptr, 0); !e.ok()) {
        std::fprintf(stderr, "set_history(0): %s\n", e.message.c_str());
        std::exit(1);
    }
    std::vector<int32_t> other(32);
    for (size_t i = 0; i < other.size(); ++i) other[i] = 5000 + (int32_t) ((i * 977) % 20000);
    model::PrefillRequest p;
    p.tokens = other.data();
    p.n_tokens = (int) other.size() - 1;
    p.total_tokens = p.n_tokens;
    if (const model::Result e = s.prefill_chunk(p); !e.ok()) {
        std::fprintf(stderr, "clobber prefill: %s\n", e.message.c_str());
        std::exit(1);
    }
    (void) greedy(s, other.back(), p.n_tokens, 4);
}

void exercise_length(model::ModelSession& s, int n_prompt, bool draft) {
    const int64_t at = (n_prompt - 1) + 8;
    const std::string tag = (draft ? " (drafter on, at=" : " (plain, at=") + std::to_string(at) + ")";
    const bool has_drafter = draft && s.max_window() > 1;

    if (const model::Result e = s.set_history(nullptr, 0); !e.ok()) {
        std::fprintf(stderr, "set_history: %s\n", e.message.c_str());
        std::exit(1);
    }
    std::vector<int32_t> prompt((size_t) n_prompt);
    for (int i = 0; i < n_prompt; ++i) prompt[(size_t) i] = 1000 + (int32_t) ((i * 7919) % 30000);
    model::PrefillRequest p;
    p.tokens = prompt.data();
    p.n_tokens = n_prompt - 1;
    p.total_tokens = n_prompt - 1;
    if (const model::Result e = s.prefill_chunk(p); !e.ok()) {
        std::fprintf(stderr, "prefill: %s\n", e.message.c_str());
        std::exit(1);
    }
    const std::vector<int32_t> lead = greedy(s, prompt.back(), n_prompt - 1, 8);
    std::vector<int32_t> hist_a(prompt.begin(), prompt.end());
    hist_a.insert(hist_a.end(), lead.begin(), lead.end() - 1);   // positions [0, at)
    const int32_t anchor = lead.back();

    size_t head_bytes = 0;
    const std::vector<uint8_t> state = save_running(s, at, &head_bytes);
    check(state.size() == s.running_bytes(), "running_state matches running_bytes()" + tag);

    std::vector<core::ConversationKv> kv;
    check(s.save_cells(kv, at).ok(), "save_cells succeeds" + tag);
    size_t kv_payload = 0;
    for (const auto& k : kv) kv_payload += k.k.size() + k.v.size() + k.k_scale.size() + k.v_scale.size();
    check(kv_payload == s.cells_bytes(at), "save_cells payload matches cells_bytes()" + tag);

    DraftOut ref_d0, ref_d1;
    if (has_drafter && head_bytes > 0) {
        ref_d0 = draft_at(s, anchor, at, 7);
        check(ref_d0.ids.size() == 7, "DFlash 2 drafts 7 tokens at checkpoint" + tag);
        // draft_begin writes speculative block keys at [at .. at + 7]; restore exact checkpoint before continuing
        check(restore_running(s, state, at), "checkpoint restores cleanly after draft probe" + tag);
    }

    const std::vector<int32_t> turn2 = {anchor, 2011, 2027, 2039, 2053, 2063};
    const int32_t turn2_tok = fold_turn(s, turn2, at);
    const int64_t at2 = at + (int64_t) turn2.size();
    if (has_drafter) {
        ref_d1 = draft_at(s, turn2_tok, at2, 7);
        check(ref_d1.ids.size() == 7, "drafter drafts 7 tokens after folded follow-up turn" + tag);
    }
    const std::vector<int32_t> first = greedy(s, turn2_tok, at2, 24);

    // Park -> run an unrelated shorter conversation from position 0 -> restore Conversation A.
    clobber_with_short_conversation(s);

    {
        std::vector<core::ConversationKv> extra = kv;
        extra.push_back(kv.front());
        check(!s.restore_cells(extra, at).ok(), "restore_cells rejects extra KV entries" + tag);
    }
    if (head_bytes >= sizeof(int64_t)) {
        std::vector<uint8_t> bad = state;
        const int64_t wrong_pos = at + 1;
        std::memcpy(bad.data(), &wrong_pos, sizeof(wrong_pos));
        check(!s.running_restored(at, bad.data()).ok(), "running_restored rejects mismatched head position" + tag);
    }

    check(s.restore_cells(kv, at).ok(), "restore_cells restores positional caches" + tag);
    check(restore_running(s, state, at), "running_restored restores rings and drafter window" + tag);
    check(s.set_history(hist_a.data(), at).ok(), "set_history restores token history" + tag);

    check(save_running(s, at) == state, "re-saved running_state is byte-identical" + tag);
    std::vector<core::ConversationKv> kv_again;
    check(s.save_cells(kv_again, at).ok() && kv_equal(kv, kv_again), "re-saved save_cells is byte-identical" + tag);

    if (has_drafter && head_bytes > 0) {
        const DraftOut got_d0 = draft_at(s, anchor, at, 7);
        check(got_d0.same_bits(ref_d0), "DFlash 2 drafts & confidences are bit-identical immediately after restore" + tag);
        check(restore_running(s, state, at), "running_restored re-mounts checkpoint after draft check" + tag);
    }

    const int32_t turn2_again = fold_turn(s, turn2, at);
    check(turn2_again == turn2_tok, "folded follow-up turn picks the same token after restore" + tag);
    if (has_drafter) {
        const DraftOut got_d1 = draft_at(s, turn2_again, at2, 7);
        check(got_d1.same_bits(ref_d1), "drafter window drafts & confidences are bit-identical on resumed turn" + tag);
    }
    const std::vector<int32_t> again = greedy(s, turn2_again, at2, 24);
    check(again == first, "greedy continuation is identical after restore" + tag);
}

void round_trip(const model::ModelAdapter& a, const std::string& pack, bool draft) {
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 4096;
    req.draft = draft;
    req.spec = draft ? 8 : 0;
    model::Started started;
    if (const model::Result e = a.start(req, started); !e.ok()) {
        std::fprintf(stderr, "start: %s\n", e.message.c_str());
        std::exit(1);
    }
    std::unique_ptr<model::ModelSession>& s = started.sessions.front();
    if (const model::Result e = s->prepare(); !e.ok()) {
        std::fprintf(stderr, "session: %s\n", e.message.c_str());
        std::exit(1);
    }
    exercise_length(*s, 777, draft);
    if (draft) exercise_length(*s, 2297, true);   // past the 2048-cell sliding window (ring wraparound)
}

}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("BASALT_INKLING2_PACK");
    const std::string pack = argc > 1 ? argv[1] : env ? env : "";
    int devices = 0;
    if (pack.empty() || cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("skip: no Inkling 2 pack (argument or BASALT_INKLING2_PACK) or no GPU\n");
        return 77;
    }
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) {
        std::fprintf(stderr, "%s: %s\n", pack.c_str(), err.c_str());
        return 1;
    }
    const std::unique_ptr<model::ModelAdapter> a = models::inkling2::make_adapter();
    round_trip(*a, pack, false);
    round_trip(*a, pack, true);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
