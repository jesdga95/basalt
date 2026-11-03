// basalt/program/serve/reply_rows.hpp - which rows of a verify window the session keeps.
//
// Row i of a window is fed by window[i]: row 0 by the token carried over (the last prompt token or the previous
// window's last pick), row i > 0 by the draft the target confirmed as pick i - 1.  Picks e0..a are the reply's; the
// request sends them in order until max_new or an end token.  The session keeps the rows fed by tokens already in the
// conversation (prompt tokens and every pick sent but the last); the last pick sent is the next window's row 0.  So
// the kept rows, the token carried over and its position always agree, a window that ends the reply keeps no row
// past its last pick, and a batch slot's segment keeps the same rows.  Plain values in, decisions out; no engine state.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace basalt::program::serve {

struct ReplyRows {
    int emit = 0;        ///< picks the window sends
    bool ends = false;   ///< the reply ends in this window (max_new reached or an end token sent)
    int keep = 0;        ///< rows the session keeps: commit `keep`, then the next token is picks[keep - 1] at p + keep
};

/// `picks`: the window's picks; rows before `e0` are prompt rows (their picks are not sent); `a`: the last accepted
/// row; `produced` picks sent before this window, at most `max_new` in all; `scored`: a teacher-forced (PPL) window,
/// which keeps every row and ends nothing.
inline ReplyRows reply_rows(const int32_t* picks, int e0, int a, int64_t produced, int64_t max_new,
                            const std::vector<int64_t>& eos_ids, bool scored) {
    ReplyRows r;
    bool eos = false;
    for (int i = e0; i <= a && produced + r.emit < max_new && !eos; ++i) {
        ++r.emit;
        eos = !scored && std::find(eos_ids.begin(), eos_ids.end(), (int64_t) picks[i]) != eos_ids.end();
    }
    r.ends = !scored && r.emit > 0 && (eos || produced + r.emit >= max_new);
    r.keep = r.ends ? e0 + r.emit : a + 1;
    return r;
}

}  // namespace basalt::program::serve
