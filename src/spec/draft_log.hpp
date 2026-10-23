// src/spec/draft_log.hpp - the reader for --draft-log's window lines.
//
// `--draft-log FILE` writes one `W` line per decode window (generate.cpp), and until this reader existed nothing read
// them back.  The format is positional and append-only:
//
//     W <T> <accepted> <from_sfx> <ms_round> <ms_wait> <ms_draft> [wprob 0..16] [sfx_match]
//
// The format now has FOUR shapes, and presence alone cannot tell them apart - a lookup window may end after the six
// values (a log written before the match-length field) or carry only the match length (a new one):
//
//   line                        block  field        has_probs  has_match()  replay
//   old lookup window           no     no           false      false        skipped (no input for observe)
//   new lookup window           no     yes (>= 0)   false      true         REPLAYED
//   old MTP window              yes    no           true       false        replayed
//   new MTP window              yes    yes (-1)     true       false        replayed
//
// The writer emits the field on every line it writes, and `-1` there is not "missing": it is the writer's explicit
// "this window was MTP, it has no match length" (generate.cpp).  That is why has_match() is `>= 0` and not a separate
// flag, and why shape 3 exists only in logs written before the field.
//
// The reader decides by WHAT IT PARSED, not by what follows.  Two traps, both learned the hard way:
//
//   * `strtod` accepts a bare integer, so a greedy float loop swallows the match length and leaves nothing behind.
//     Tokens are therefore CLASSIFIED before being parsed: a `.` or `e`/`E` means a float (the writer emits wprob
//     with `%.4f`, always with a decimal point), anything else is the integer (`%d`, never with one).
//   * The block is `min(S_mtp, 16)` floats, NOT a fixed 16, so a partial block is legal and says nothing about the
//     shape.  Only "zero floats or some floats" distinguishes a lookup window from an MTP one.
//
// Anything else is malformed: more than 16 floats, a second integer, a float after the integer, or trailing text.
// The field was APPENDED after the block, never inserted before it, because inserting would have shifted every
// parse of every existing log.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace basalt::spec {

struct DraftLogWindow {
    int T = 0;              ///< verify window width
    int accepted = 0;       ///< drafts the window accepted (0..T-1)
    bool from_sfx = false;  ///< the window's drafts came from the suffix lookup
    double ms_round = 0, ms_wait = 0, ms_draft = 0;
    bool has_probs = false; ///< the MTP drafted: `wprob` follows
    /// The suffix match length a lookup window decided from; -1 when the line predates the field, or was MTP.
    int sfx_match = -1;
    bool has_match() const { return sfx_match >= 0; }
};

/// Parses the `W` lines of a draft log, in file order.  Lines that are not `W` lines (the `R` header, comments) and
/// malformed `W` lines are skipped, and `skipped` counts them, so a truncated file replays what it has.
inline std::vector<DraftLogWindow> draft_log_windows(const std::string& text, size_t* skipped = nullptr) {
    std::vector<DraftLogWindow> out;
    size_t bad = 0;
    for (size_t at = 0; at < text.size();) {
        const size_t end = text.find('\n', at);
        const std::string line = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        at = end == std::string::npos ? text.size() : end + 1;
        if (line.size() < 2 || line[0] != 'W' || line[1] != ' ') continue;   // not a window line
        // Fields: W and six values, then the optional probability block.
        const char* p = line.c_str() + 2;
        auto field = [&](const char* what, bool& ok) -> double {
            while (*p == ' ') ++p;
            if (*p == '\0') { ok = false; return 0.0; }
            char* stop = nullptr;
            const double v = std::strtod(p, &stop);
            if (stop == p) { ok = false; return 0.0; }
            p = stop;
            (void) what;
            return v;
        };
        bool ok = true;
        DraftLogWindow w;
        w.T = (int) field("T", ok);
        w.accepted = (int) field("accepted", ok);
        w.from_sfx = field("from_sfx", ok) != 0.0;
        w.ms_round = field("ms_round", ok);
        w.ms_wait = field("ms_wait", ok);
        w.ms_draft = field("ms_draft", ok);
        if (!ok || w.T < 1 || w.T > 64 || w.accepted < 0 || w.accepted >= w.T) { ++bad; continue; }
        // What follows the six values, decided by CLASSIFYING each token before parsing it: strtod() accepts a bare
        // integer, so a greedy float loop would swallow the match length of a new lookup window and leave none behind.
        // The writer's formats are what make the classification sound - wprob uses %.4f, always with a decimal point,
        // and the match length uses %d, never with one (generate.cpp:8888-8891).
        int floats = 0;
        bool saw_int = false, malformed = false;
        while (!malformed) {
            while (*p == ' ') ++p;
            if (*p == '\0') break;
            const char* tok = p;
            while (*p != '\0' && *p != ' ') ++p;
            const std::string token(tok, (size_t) (p - tok));
            const bool is_float = token.find_first_of(".eE") != std::string::npos;
            if (is_float) {
                if (saw_int || floats >= 16) malformed = true;   // a float after the integer, or a block over 16
                else ++floats;
                continue;
            }
            if (saw_int) { malformed = true; continue; }          // a second integer
            char* stop = nullptr;
            const long v = std::strtol(token.c_str(), &stop, 10);
            if (stop == nullptr || *stop != '\0') { malformed = true; continue; }   // not an integer at all
            w.sfx_match = (int) v;
            saw_int = true;
        }
        if (malformed) { ++bad; continue; }
        // A partial probability block is LEGAL: the writer emits `wprob_n = min(S_mtp, 16)` floats, so the count
        // depends on the run's speculation depth, not on a fixed width.  What tells the two line shapes apart is
        // simply whether any floats came at all.
        w.has_probs = floats > 0;
        out.push_back(w);
    }
    if (skipped != nullptr) *skipped = bad;
    return out;
}

}  // namespace basalt::spec
