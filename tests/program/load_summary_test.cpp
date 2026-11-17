// tests/program/load_summary_test.cpp - the load summary basalt-bench prints from the engine's `basalt load:` lines:
// the one weights line, the total, a total that moves, an error and a warning line, the throttling and the terminal
// mode. CPU only; the clock is passed in.
#include "basalt/program/load_summary.hpp"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const std::string& what, const std::string& got = "") {
    std::printf("load_summary %-62s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) std::printf("  got: [%s]\n", got.c_str());
    failures |= !ok;
}
void is(const std::string& got, const std::string& want, const std::string& what) { check(got == want, what, got); }

constexpr unsigned long long G = 1ull << 30;
std::string ld(double done_gib, double total_gib) {
    return "basalt load: weights " + std::to_string((unsigned long long) (done_gib * G)) + " " +
           std::to_string((unsigned long long) (total_gib * G));
}
std::string gb(const char* kind, double v) { return std::string("basalt load: ") + kind + " " + std::to_string((unsigned long long) (v * G)); }
}  // namespace

int main() {
    using basalt::program::LoadSummary;

    {   // off a terminal
        LoadSummary s("basalt-bench", false, 0.0);
        is(s.feed("basalt: some other line", 0.1), "", "an unrelated line prints nothing");
        is(s.feed(gb("total", 52.5), 0.2), "", "the total alone prints nothing");
        is(s.feed("basalt load: kv cuda0 1000 4096 1000", 0.3), "", "a kv line is skipped");
        is(s.feed("basalt load: ngram 1000 3000", 0.3), "", "the n-gram table lock is not the weights");
        is(s.feed(ld(0, 52.5), 0.5), "basalt-bench: loading 52.5 GiB\n",
           "the first weights line prints the header with the total");
        is(s.feed(ld(7.0, 52.5), 1.0), "", "progress prints nothing by itself");
        is(s.tick(5.0), "", "no periodic line before the period");
        is(s.tick(10.6), "basalt-bench: loading 7.0 / 52.5 GiB (13%)\n", "a periodic line after 10 s gives the percent");
        is(s.tick(15.0), "", "and then waits another period");
        is(s.feed(ld(52.5, 52.5), 21.0), "basalt-bench: loading weights 52.5 / 52.5 GiB done (21 s)\n",
           "the end prints a done line with the duration");
        is(s.feed(ld(52.5, 52.5), 21.1), "", "a repeated final line prints nothing more");
        is(s.tick(40.0), "", "no periodic line once done");
        is(s.feed("basalt error: out of memory on cuda0", 34.0), "basalt-bench: engine: out of memory on cuda0\n",
           "an error line is shown at once without its prefix");
        is(s.feed("basalt generate: error: the weight arena failed", 34.5), "basalt-bench: engine: the weight arena failed\n",
           "a line of any area marked error is shown the same way");
        is(s.feed("basalt generate: warning: the cache is short", 34.6), "basalt-bench: engine: warning: the cache is short\n",
           "a warning line is shown at once with its marker");
        is(s.feed("basalt generate: no error here", 34.7), "", "the word error in a message is not a marker");
        is(s.finish(80.0, true), "basalt-bench: engine ready in 80 s\n", "READY prints the ready line");
    }
    {   // lines that read done == total while the total grows (a phase decided later): the end is reported once
        LoadSummary s("basalt-bench", false, 0.0);
        s.feed(ld(0, 10.0), 0.0);
        is(s.feed(ld(10.0, 10.0), 3.0), "basalt-bench: loading weights 10.0 / 10.0 GiB done (3 s)\n", "the first end");
        is(s.feed(ld(10.5, 10.5), 3.25), "", "an equal line after a raised total prints nothing");
        is(s.feed(ld(11.0, 11.0), 3.5), "", "nor the next");
        is(s.feed(ld(11.0, 11.0), 3.75), "", "nor a repeat");
        is(s.tick(30.0), "", "and no periodic line follows");
    }
    {   // a failure prints no ready line
        LoadSummary s("basalt-probe", false, 0.0);
        is(s.finish(5.0, false), "", "a failed start prints no ready line");
        is(s.finish(5.0, true), "basalt-probe: engine ready in 5 s\n", "and the label is the caller's");
    }
    {   // a terminal
        LoadSummary s("basalt-bench", true, 0.0);
        s.feed(gb("total", 52.0), 0.0);
        const std::string a = s.feed(ld(0, 52.0), 0.1);
        is(a, "\r\x1b[2Kbasalt-bench: loading weights 0.0/52.0 GiB \xC2\xB7 0% \xC2\xB7 0 s", "the first line is drawn in place");
        is(s.feed(ld(14.0, 52.0), 0.3), "", "a redraw closer than the interval is held back");
        is(s.tick(1.2), "\r\x1b[2Kbasalt-bench: loading weights 14.0/52.0 GiB \xC2\xB7 26% \xC2\xB7 1 s",
           "the clock redraws with time passing");
        is(s.feed("basalt error: bad", 1.3), "\r\x1b[2Kbasalt-bench: engine: bad\n", "an error clears the line and prints its own");
        is(s.feed(ld(30.0, 52.0), 1.4), "\r\x1b[2Kbasalt-bench: loading weights 30.0/52.0 GiB \xC2\xB7 57% \xC2\xB7 1 s",
           "the line is drawn again after it");
        is(s.feed("basalt: warning: slow", 1.5), "\r\x1b[2Kbasalt-bench: engine: warning: slow\n", "so does a warning");
        s.feed(ld(52.0, 52.0), 2.0);
        is(s.finish(41.0, true),
           "\r\x1b[2Kbasalt-bench: loading weights 52.0/52.0 GiB \xC2\xB7 100% \xC2\xB7 41 s\n"
           "basalt-bench: engine ready in 41 s\n",
           "READY leaves the last status line and adds the ready line");
    }
    return failures ? 1 : 0;
}
