// src/model/batch_check.cpp - the batch conformance check (batch_check.hpp): solo windows against the same windows batched.
#include "basalt/model/batch_check.hpp"

#include <algorithm>
#include <cstring>

namespace basalt::model {

namespace {

struct Win {
    std::vector<int32_t> tokens, picks;
    std::vector<float> logits;
    DecodeRequest req() {
        DecodeRequest r;
        r.tokens = tokens.data();
        r.n_tokens = (int) tokens.size();
        r.row_tokens = picks.data();
        r.logits = logits.data();
        return r;
    }
};

}  // namespace

Result check_batch(ModelSession* const* ss, int n, const std::vector<int32_t>& prompt, int steps, int64_t n_vocab, BatchCheck& out) {
    if (n < 2) return {Status::Error, "the batch check needs at least two sessions (--batch 2)"};
    const int cap = ss[0]->batch_rows();
    if (cap <= 0) return {Status::Error, "this model's sessions do not batch (batch_rows 0)"};
    const int skip = 7;
    if ((int64_t) prompt.size() < (int64_t) skip * n + 8) return {Status::Error, "the batch check needs a longer prompt"};
    // every session ready to run, as the serve engine readies its slots, then each session's prompt from its own start
    for (int i = 0; i < n; ++i)
        if (Result r = ss[i]->prepare(); !r.ok()) return r;
    std::vector<std::vector<int32_t>> seq((size_t) n);
    for (int i = 0; i < n; ++i) {
        std::vector<int32_t>& s = seq[(size_t) i];
        s.assign(prompt.begin() + (ptrdiff_t) i * skip, prompt.end());
        PrefillRequest p;
        p.tokens = s.data();
        p.n_tokens = (int) s.size() - 1;
        p.total_tokens = p.n_tokens;
        if (Result r = ss[i]->prefill_chunk(p); !r.ok()) return r;
        ss[i]->end_prompt();
    }
    const auto same = [&](const Win& a, const Win& b) {
        return a.picks == b.picks && std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0;
    };
    std::vector<std::vector<int>> orders;   // which sessions batch, the leader first
    std::vector<int> all((size_t) n);
    for (int i = 0; i < n; ++i) all[(size_t) i] = i;
    orders.push_back(all);
    std::reverse(all.begin(), all.end());
    orders.push_back(all);
    for (int i = 0; i + 1 < n; ++i) orders.push_back({i, i + 1});
    // each round, each order: every session's window (its last token, then drafts from the prompt, mostly rejected) alone,
    // then the same windows as one batch at the same positions; then each session keeps its first pick, as a verify that
    // rejects the drafts does
    for (int round = 0; round < steps; ++round) {
        for (const std::vector<int>& order : orders) {
            std::vector<int> rows;   // 1, 2 or 3 a session by round and place, at most the cap in all
            int sum = 0;
            for (size_t k = 0; k < order.size(); ++k) {
                const int r = std::min(1 + (int) (((size_t) round + (size_t) order[k] + k) % 3), cap - sum - (int) (order.size() - k - 1));
                if (r < 1) break;
                rows.push_back(r);
                sum += r;
            }
            if (rows.size() < 2) continue;
            std::vector<Win> solo(rows.size()), batch(rows.size());
            std::vector<DecodeRequest> reqs;
            std::vector<ModelSession*> peers;
            for (size_t k = 0; k < rows.size(); ++k) {
                const std::vector<int32_t>& s = seq[(size_t) order[k]];
                Win& w = solo[k];
                w.tokens.push_back(s.back());
                for (int j = 1; j < rows[k]; ++j) w.tokens.push_back(prompt[(size_t) (round * 5 + j * 3 + (int) k) % prompt.size()]);
                w.picks.assign((size_t) rows[k], -1);
                w.logits.assign((size_t) rows[k] * (size_t) n_vocab, 0.f);   // room for every row: a model may write them all
                batch[k] = w;
                DecodeRequest r = w.req();
                r.pos = (int64_t) s.size() - 1;
                if (Result e = ss[order[k]]->decode_window(r); !e.ok()) return e;
                peers.push_back(ss[order[k]]);
            }
            for (size_t k = 0; k < rows.size(); ++k) {
                DecodeRequest r = batch[k].req();
                r.pos = (int64_t) seq[(size_t) order[k]].size() - 1;
                reqs.push_back(r);
            }
            if (Result e = peers[0]->decode_batch(peers.data(), reqs.data(), (int) peers.size()); !e.ok()) return e;
            const std::vector<int> ones(peers.size(), 1);   // each keeps its first row
            if (Result e = peers[0]->decode_kept(peers.data(), ones.data(), (int) peers.size()); !e.ok()) return e;
            std::string line = "round " + std::to_string(round) + ", session x rows:";
            int bad = 0;
            for (size_t k = 0; k < rows.size(); ++k) {
                line += " " + std::to_string(order[k]) + "x" + std::to_string(rows[k]);
                ++out.compared;
                if (!same(solo[k], batch[k])) {
                    ++bad;
                    ++out.differ;
                    line += "(DIFFERS)";
                }
                seq[(size_t) order[k]].push_back(solo[k].picks[0]);
            }
            out.report += line + (bad ? ": DIFFERS\n" : ": identical\n");
        }
    }
    return {};
}

}  // namespace basalt::model
