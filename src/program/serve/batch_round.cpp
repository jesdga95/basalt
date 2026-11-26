// src/program/serve/batch_round.cpp - the batch slots' rounds, for every model (docs/design/batching.md): each active slot's
// window, the rows the windows share, the groups that run as one batch through the model's sessions, then each slot's
// accepted drafts, grammar, logprobs, tokens out and next drafts.
#include "batch_plan.hpp"
#include "logprobs.hpp"
#include "state.hpp"

#include "basalt/protocol.hpp"

#include <iterator>

namespace basalt::program::serve {

// A slot's window of one round: what it feeds, what it drew, and the penalty history rows of its draws.
struct SlotWin {
    int T = 1;
    bool from_sfx = false;   // the drafts are a lookup proposal behind a match of `match` tokens
    int match = 0;
    std::vector<int32_t> win, out, history;
    float chance[basalt::spec::DraftPolicy::kMaxT] = {};
    Clock::time_point round0;
    basalt::model::DecodeRequest d;
    // set by slot_window_accept: the accepted drafts, the rows the slot keeps, the tokens' line, the finish reason, the
    // drafts to start next
    int a = 0;
    int kept = 0;
    std::string line;
    const char* fin = nullptr;
    int want = 0;
    bool drafting = false;
    bool solo = true;
};

// The slot's next window: its anchor and the drafts its drafter left (as many as the draft policy takes, or as its drafter's
// own length rule keeps).
void Runtime::slot_window_begin(int b, SlotWin& w, int limit) {
    BSlot& sl = bs[(size_t) b];
    basalt::model::ModelSession& s = slot_session(b);
    const int own_limit = s.window_limit(sl.p);
    const int most = (int) std::min<int64_t>({(int64_t) s.max_window(), slot_context(b) - sl.p, (int64_t) std::size(sl.dr) + 1, (int64_t) limit,
                                              own_limit > 0 ? (int64_t) own_limit : INT64_MAX});
    const int own = s.own_window(sl.dprob, sl.n_dr);
    w.T = std::max(1, std::min(own > 0 ? own : bpolicy->window(sl.dprob, sl.n_dr), most));
    w.from_sfx = false;
    w.match = 0;
    if (sl.sfx && most >= 2) {   // a lookup proposal, as in the solo loop
        const int k = sl.sfx->propose(most - 1, sl.sbuf);
        w.match = sl.sfx->last_match();
        if (k > 0 && (sl.n_dr == 0 || sl.sbuf[0] == sl.dr[0])) {
            const basalt::spec::DraftPolicy::Pick pk = own > 0 ? bpolicy->choose(w.T, k, w.match)
                                                           : bpolicy->choose_window(sl.dprob, std::min(sl.n_dr, most - 1), k, w.match);
            if (pk.lookup) {
                w.T = std::min(pk.t, most);
                w.from_sfx = true;
            }
        }
    }
    const int T = w.T;
    std::copy(sl.dprob, sl.dprob + (T - 1), w.chance);
    w.win.assign((size_t) T, 0);
    w.out.assign((size_t) T, -1);
    w.win[0] = sl.x;
    std::copy(w.from_sfx ? sl.sbuf : sl.dr, (w.from_sfx ? sl.sbuf : sl.dr) + (T - 1), w.win.begin() + 1);
    basalt::kernels::SamplerParams& sp = slot_sampling[(size_t) b];
    w.d = {};
    w.d.tokens = w.win.data();
    w.d.n_tokens = T;
    w.d.pos = sl.p;
    (T > 1 ? w.d.row_tokens : w.d.next_token) = w.out.data();
    w.d.sampling = &sp;
    w.d.drawn_drafts = !w.from_sfx && T > 1;
    if (sl.pen_n > 0) {
        w.history.resize((size_t) T * sl.pen_n);
        basalt::kernels::penalty_rows(sl.ids.data(), (int64_t) sl.ids.size(), w.win.data(), T, sl.pen_n, w.history.data());
        w.d.history = w.history.data();
        w.d.history_len = sl.pen_n;
    }
    sp.counter = (uint64_t) sl.produced;
}

// The slot's window is drawn: the accepted drafts, the grammar's mask, the logprobs, the tokens out, the rows it keeps, the
// slot's next drafts (made while the tokens go out), and its end.
bool Runtime::slot_window_finish(int b, SlotWin& w) {
    if (!slot_window_accept(b, w)) return false;
    basalt::model::ModelSession* s = &slot_session(b);
    if (const basalt::model::Result r = s->decode_kept(&s, &w.kept, 1); !r.ok()) {
        std::printf("ERR %s\n", r.message.c_str());
        return false;
    }
    if (w.want > 0) {
        const basalt::model::Result r = s->draft_begin(bs[(size_t) b].x, bs[(size_t) b].p, w.want);
        if (!r.ok() && r.status != basalt::model::Status::NotImplemented) {
            std::printf("ERR %s\n", r.message.c_str());
            return false;
        }
        w.drafting = r.ok();
    }
    return slot_window_end(b, w);
}

// The window's accepted drafts, grammar, logprobs and tokens; the rows it keeps (w.kept); whether drafts follow (w.want).
bool Runtime::slot_window_accept(int b, SlotWin& w) {
    std::vector<float> lp_rows;   // a logprobs slot's kept rows (row_logprobs)
    bool grammar_dead = false;    // the grammar allowed no token: the slot ends as an error after this window
    const auto failed = [&](const std::string& m) {
        std::printf("ERR %s\n", m.c_str());
        return false;
    };
    BSlot& sl = bs[(size_t) b];
    basalt::model::ModelSession& s = slot_session(b);
    const int T = w.T;
    std::vector<int32_t>& win_ = w.win;
    std::vector<int32_t>& out_ = w.out;
    int a = 0;
    while (a < T - 1 && win_[(size_t) a + 1] == out_[(size_t) a]) ++a;
    if (sl.guard != nullptr && sl.guard->on()) {
        // the first pick the grammar refuses is drawn again under its mask, and the window ends there; with no token
        // allowed at all the slot ends after it, as an error
        const int miss = sl.guard->scan(out_.data(), a + 1, 0);
        if (miss >= 0) {
            int32_t y = -1;
            std::string why;
            const basalt::program::grammar::Redraw verdict = sl.guard->redraw(
                {},
                [&](const std::vector<float>& bias, int32_t& t) {
                    const basalt::model::Result r = s.resample_row(miss, bias.data(), t);
                    if (!r.ok()) why = r.message;
                    return r.ok();
                },
                [&](std::vector<float>& lg) {
                    lg.resize((size_t) n_vocab);
                    const basalt::model::Result r = s.copy_logits(miss, lg.data());
                    if (!r.ok()) why = r.message;
                    return r.ok();
                },
                y);
            if (verdict == basalt::program::grammar::Redraw::failed) return failed(why.empty() ? "the masked draw failed" : why);
            if (verdict == basalt::program::grammar::Redraw::dead) grammar_dead = true;
            else out_[(size_t) miss] = y;
            a = miss;
        }
    }
    sl.drafts_offered += T - 1;
    sl.drafts_accepted += a;
    ++sl.windows;
    std::string line = "BT " + std::to_string(b);
    const char* fin = nullptr;
    const int64_t sent0 = sl.produced;
    if (sl.lp_top >= 0) {   // the kept rows' logprobs, read before the drafts touch the head
        lp_rows.resize((size_t) (a + 1) * basalt::kernels::kLogprobStride);
        if (const basalt::model::Result r = s.row_logprobs(0, a + 1, out_.data(), sl.lp_top, lp_rows.data()); !r.ok())
            return failed(r.message);
    }
    const int64_t ctx = slot_context(b);
    for (int i = 0; i <= a && fin == nullptr; ++i) {
        const int32_t y = out_[(size_t) i];
        if (sl.lp_top >= 0)   // ahead of the BT line
            basalt::protocol::emit_text("BP " + std::to_string(b) +
                                        logprob_fields(lp_rows.data() + (size_t) i * basalt::kernels::kLogprobStride, y, sl.lp_top) + "\n");
        sl.ids.push_back(win_[(size_t) i]);
        ++sl.produced;
        ++sl.p;
        sl.x = y;
        if (sl.sfx) sl.sfx->append(y);
        line += " " + std::to_string(y);
        const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
        fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new || sl.p + 2 > ctx ? "length" : nullptr;
    }
    if (grammar_dead) fin = "error";
    w.kept = (int) (sl.produced - sent0);
    stats_window(b, T - 1, a, w.kept);
    sl.n_dr = 0;
    w.a = a;
    w.line = line;
    w.fin = fin;
    w.want = fin == nullptr && s.max_window() > 1 ? drafts_ahead(T, s.max_window()) : 0;
    w.drafting = false;
    return true;
}

// The tokens go out while the drafts run; then the drafts are read, and the slot ends or goes on.
bool Runtime::slot_window_end(int b, SlotWin& w) {
    BSlot& sl = bs[(size_t) b];
    basalt::model::ModelSession& s = slot_session(b);
    const int T = w.T, a = w.a;
    const char* fin = w.fin;
    basalt::protocol::emit_text(w.line + "\n");
    basalt::core::progress_beat();
    if (w.drafting) {
        const int max_n = std::min(s.max_window() - 1, (int) std::size(sl.dr));
        const auto need = [&](const float* c, int j, int m) { return (j == 0 && sl.sfx) || bpolicy->may_verify(c, j, m); };
        if (const basalt::model::Result r = s.draft_end(sl.dr, sl.dprob, max_n, sl.n_dr, need); !r.ok()) {
            std::printf("ERR %s\n", r.message.c_str());
            return false;
        }
        if (w.solo) {
            const double ms = s.window_cold() ? 0.0 : std::chrono::duration<double, std::milli>(Clock::now() - w.round0).count();
            if (w.from_sfx) bpolicy->observe(true, T, a, w.match, ms);
            else bpolicy->observe_window(w.chance, T, a, ms);
        }
    }
    if (fin != nullptr) end_slot(b, fin);
    std::fflush(stdout);
    return true;
}

void Runtime::end_slot(int b, const char* fin) {
    BSlot& sl = bs[(size_t) b];
    std::printf("BDONE %d %lld %s %.1f %lld %lld %lld\n", b, (long long) sl.produced, fin,
                std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count(),
                (long long) sl.drafts_accepted, (long long) sl.drafts_offered, (long long) sl.windows);
    sl.active = false;
    sl.cached = o.prompt_cache > 0;
    sl.idle_busy = busy_now();
    sl.guard.reset();
}

// One window of every ready active slot.  A model that does not batch decodes them one session at a time: the slot's drafts
// as a solo window takes them, then the drafts of its next window, which run while its tokens go out.  A model that does
// packs the windows into groups of at most batch_rows() rows (first fit); a group of several runs as one batch through the
// weights, a lone window as a solo one; each group's sessions learn the rows they keep, then draft as one pass, and every
// slot then finishes as above.
bool Runtime::batch_round() {
    std::vector<int> act;
    for (int b = 0; b < (int) bs.size(); ++b) {
        // a stopped request ends before another window feeds a token: the solo loop ends at the window boundary the same
        // way, so what the slot keeps is the tokens its client saw (the client continues the conversation from there)
        if (bs[(size_t) b].active && bs[(size_t) b].stop) {
            end_slot(b, "cancel");
            std::fflush(stdout);
            continue;
        }
        if (bs[(size_t) b].active && slot_ready(b)) act.push_back(b);
    }
    if (act.empty()) return true;
    const auto failed = [&](const std::string& m) {
        std::printf("ERR %s\n", m.c_str());
        return false;
    };
    const int cap = slot_session(act[0]).batch_rows();
    if (cap <= 0) {
        SlotWin w;
        for (const int b : act) {
            w.round0 = Clock::now();
            slot_window_begin(b, w, 1 << 20);
            stats_lookups_begin();
            if (const basalt::model::Result r = slot_session(b).decode_window(w.d); !r.ok()) return failed(r.message);
            if (!slot_window_finish(b, w)) return false;
            stats_lookups_end();
        }
        return true;
    }
    // the windows of a round share the batch's rows.  Drafts with stated chances: with no more slots than rows one pass
    // serves all, the rows going to the drafts likeliest kept.  Drafters that set their own length: each window as it would
    // run alone, shared max-min fairly when they do not fit (a window held to fewer takes its drafts again within its share)
    constexpr int kShortSpecCap = 16;
    constexpr int kLongSpecCap = 96;
    const int n_act = (int) act.size();
    int64_t max_pos = 0;
    for (int i = 0; i < n_act; ++i) max_pos = std::max(max_pos, bs[(size_t) act[(size_t) i]].p);
    const int spec_cap = max_pos >= 4096 ? std::min(cap, kLongSpecCap) : std::min(cap, kShortSpecCap);
    const int row_cap = std::min(cap, std::max(n_act, spec_cap));
    const int slot_cap_by_n = n_act >= 24 ? 2 : n_act >= 6 ? 3 : n_act >= 2 ? 5 : row_cap;
    const int per_slot_cap = std::min(row_cap, slot_cap_by_n);
    std::vector<SlotWin> wins(act.size());
    std::vector<int> rows(act.size());
    const BSlot& s0 = bs[(size_t) act[0]];
    if (slot_session(act[0]).own_window(s0.dprob, s0.n_dr) > 0) {
        for (int i = 0; i < n_act; ++i) {
            slot_window_begin(act[(size_t) i], wins[(size_t) i], per_slot_cap);
            rows[(size_t) i] = wins[(size_t) i].T;
        }
        const std::vector<int> share = batch_fair_shares(rows, row_cap);
        for (int i = 0; i < n_act; ++i)
            if (share[(size_t) i] < rows[(size_t) i]) {
                slot_window_begin(act[(size_t) i], wins[(size_t) i], share[(size_t) i]);
                rows[(size_t) i] = wins[(size_t) i].T;
            }
    } else {
        std::vector<std::vector<double>> joint(act.size());
        for (int i = 0; i < n_act; ++i) {
            const BSlot& sl = bs[(size_t) act[(size_t) i]];
            double all = 1.0;
            const int max_k = std::min(sl.n_dr, std::max(0, per_slot_cap - 1));
            for (int k = 0; k < max_k; ++k) joint[(size_t) i].push_back(all *= bpolicy->chance_rate(sl.dprob[k]));
        }
        const std::vector<int> share = batch_shares(joint, row_cap);
        for (int i = 0; i < n_act; ++i) {
            slot_window_begin(act[(size_t) i], wins[(size_t) i], std::min(share[(size_t) i], per_slot_cap));
            rows[(size_t) i] = wins[(size_t) i].T;
        }
    }
    for (const std::vector<size_t>& group : batch_groups(rows, cap)) {
        const Clock::time_point t0 = Clock::now();
        stats_lookups_begin();
        std::vector<basalt::model::ModelSession*> ss;
        for (const size_t i : group) ss.push_back(&slot_session(act[i]));
        if (group.size() == 1) {
            if (const basalt::model::Result r = ss[0]->decode_window(wins[group[0]].d); !r.ok()) return failed(r.message);
        } else {
            std::vector<basalt::model::DecodeRequest> rq;
            for (const size_t i : group) rq.push_back(wins[i].d);
            if (const basalt::model::Result r = ss[0]->decode_batch(ss.data(), rq.data(), (int) group.size()); !r.ok()) return failed(r.message);
        }
        // every slot's tokens and the rows each keeps, then the group's drafts as one pass, then each slot's tokens out
        // and drafts read
        std::vector<int> kept;
        std::vector<basalt::model::ModelSession*> ds;
        std::vector<int32_t> anchors;
        std::vector<int64_t> dpos;
        std::vector<int> dn;
        for (const size_t i : group) {
            wins[i].round0 = t0;
            wins[i].solo = group.size() == 1;
            if (!slot_window_accept(act[i], wins[i])) return false;
            kept.push_back(wins[i].kept);
            // bound drafted tokens to what a slot's share of the batch window can verify next round
            wins[i].want = std::min(wins[i].want, std::min(per_slot_cap - 1, std::max(0, (spec_cap + n_act - 1) / n_act - 1)));
            if (wins[i].want > 0) {
                const BSlot& sl = bs[(size_t) act[i]];
                ds.push_back(&slot_session(act[i]));
                anchors.push_back(sl.x);
                dpos.push_back(sl.p);
                dn.push_back(wins[i].want);
            }
        }
        if (const basalt::model::Result r = ss[0]->decode_kept(ss.data(), kept.data(), (int) ss.size()); !r.ok()) return failed(r.message);
        if (!ds.empty()) {
            const basalt::model::Result r = ds[0]->draft_batch(ds.data(), anchors.data(), dpos.data(), dn.data(), (int) ds.size());
            if (!r.ok() && r.status != basalt::model::Status::NotImplemented) return failed(r.message);
            for (const size_t i : group) wins[i].drafting = r.ok() && wins[i].want > 0;
        }
        stats_lookups_end();
        for (const size_t i : group)
            if (!slot_window_end(act[i], wins[i])) return false;
    }
    return true;
}

}  // namespace basalt::program::serve
