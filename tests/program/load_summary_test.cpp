// tests/program/load_summary_test.cpp - the load summary basalt-bench prints from the engine's `basalt load:` lines:
// every line kind, a skipped() re-print, a copy phase, an error line, the throttling and the terminal mode. CPU only;
// the clock is passed in.
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
std::string ld(const std::string& t, double done_gib, double total_gib, bool copy = false) {
    return std::string("basalt load: ") + (copy ? "copy " : "") + t + " " + std::to_string((unsigned long long) (done_gib * G)) +
           " " + std::to_string((unsigned long long) (total_gib * G));
}
std::string gb(const char* kind, double v) { return std::string("basalt load: ") + kind + " " + std::to_string((unsigned long long) (v * G)); }
}  // namespace

int main() {
    using basalt::program::LoadSummary;

    {   // off a terminal
        LoadSummary s("basalt-bench", false, 0.0);
        is(s.feed("basalt: some other line", 0.1), "", "an unrelated line prints nothing");
        is(s.feed(gb("total", 52.5), 0.2), "", "the total alone prints nothing");
        is(s.feed(gb("plan cuda0", 14.0), 0.2), "", "a plan alone prints nothing");
        is(s.feed(gb("plan ram", 38.0), 0.2), "", "a second plan prints nothing");
        is(s.feed(gb("plan copy cuda1", 5.0), 0.2), "", "a copy plan prints nothing");
        is(s.feed("basalt load: kv cuda0 1000 4096 1000", 0.3), "", "a kv line is skipped");
        is(s.feed(ld("cuda0", 0, 14.0), 0.5),
           "basalt-bench: loading 52.5 GiB: cuda0 0.0 / 14.0 GiB, ram 0.0 / 38.0 GiB, cuda1 copy 0.0 / 5.0 GiB\n",
           "the first phase line prints the header with every plan");
        is(s.feed(ld("cuda0", 7.0, 14.0), 1.0), "", "progress inside a phase prints nothing by itself");
        is(s.feed(ld("cuda0", 14.0, 14.0), 6.5), "basalt-bench: loading cuda0 14.0 / 14.0 GiB done (6 s)\n",
           "a finished phase prints a done line with its duration");
        is(s.feed(ld("cuda0", 14.0, 14.0), 6.6), "", "a repeated final line prints nothing more");
        is(s.feed(ld("ram", 0, 38.0), 7.0), "", "a new target starts silently");
        is(s.tick(8.0), "", "no periodic line before the period");
        is(s.feed(ld("ram", 10.0, 38.0), 15.0), "", "progress is not printed on its own");
        is(s.tick(16.6), "basalt-bench: loading ram 10.0 / 38.0 GiB (26%)\n", "a periodic line after 10 s lists the running target");
        is(s.tick(20.0), "", "and then waits another period");
        is(s.feed(ld("ram", 38.0, 38.0), 21.0), "basalt-bench: loading ram 38.0 / 38.0 GiB done (14 s)\n", "the copy to ram finishes");
        is(s.feed(gb("total", 40.0), 21.1), "", "a skipped() re-print only changes the total");
        is(s.feed(ld("cuda1", 0, 5.0, true), 21.2), "", "a copy phase starts silently");
        is(s.tick(31.5), "basalt-bench: loading cuda1 copy 0.0 / 5.0 GiB (0%)\n", "a copy phase is named as one");
        is(s.feed(ld("cuda1", 5.0, 5.0, true), 33.0), "basalt-bench: loading cuda1 copy 5.0 / 5.0 GiB done (12 s)\n",
           "and finishes");
        is(s.feed("basalt error: out of memory on cuda0", 34.0), "basalt-bench: engine: out of memory on cuda0\n",
           "an error line is shown at once without its prefix");
        is(s.finish(80.0, true), "basalt-bench: engine ready in 80 s\n", "READY prints the ready line");
    }
    {   // several phases of one target add up against the plan
        LoadSummary s("basalt-bench", false, 0.0);
        s.feed(gb("plan cuda0", 10.0), 0.0);
        is(s.feed(ld("cuda0", 0, 4.0), 0.0), "basalt-bench: loading cuda0 0.0 / 10.0 GiB\n", "the header without a total");
        is(s.feed(ld("cuda0", 4.0, 4.0), 1.0), "", "a phase that ends below the plan is not done yet");
        s.feed(ld("cuda0", 0, 6.0), 2.0);
        is(s.feed(ld("cuda0", 6.0, 6.0), 3.0), "basalt-bench: loading cuda0 10.0 / 10.0 GiB done (3 s)\n",
           "the next phase adds on and completes the plan");
    }
    {   // a failure prints no ready line
        LoadSummary s("basalt-probe", false, 0.0);
        is(s.finish(5.0, false), "", "a failed start prints no ready line");
        is(s.finish(5.0, true), "basalt-probe: engine ready in 5 s\n", "and the label is the caller's");
    }
    {   // a terminal
        LoadSummary s("basalt-bench", true, 0.0);
        s.feed(gb("total", 52.0), 0.0);
        s.feed(gb("plan cuda0", 14.0), 0.0);
        s.feed(gb("plan ram", 38.0), 0.0);
        const std::string a = s.feed(ld("cuda0", 0, 14.0), 0.1);
        is(a, "\r\x1b[2Kbasalt-bench: loading cuda0 0.0/14.0 GiB \xC2\xB7 0% \xC2\xB7 0 s", "the first line is drawn in place");
        is(s.feed(ld("cuda0", 7.0, 14.0), 0.3), "", "a redraw closer than the interval is held back");
        is(s.feed(ld("cuda0", 14.0, 14.0), 0.4),
           "\r\x1b[2Kbasalt-bench: loading cuda0 14.0/14.0 GiB \xC2\xB7 26% \xC2\xB7 0 s", "a finished phase redraws at once");
        s.feed(ld("ram", 0, 38.0), 0.5);
        is(s.tick(1.2), "\r\x1b[2Kbasalt-bench: loading cuda0 14.0/14.0 GiB \xC2\xB7 ram 0.0/38.0 GiB \xC2\xB7 26% \xC2\xB7 1 s",
           "the clock redraws with time passing");
        is(s.feed("basalt error: bad", 1.3), "\r\x1b[2Kbasalt-bench: engine: bad\n", "an error clears the line and prints its own");
        s.feed(ld("ram", 21.0, 38.0), 2.0);
        is(s.finish(41.0, true),
           "\r\x1b[2Kbasalt-bench: loading cuda0 14.0/14.0 GiB \xC2\xB7 ram 21.0/38.0 GiB \xC2\xB7 67% \xC2\xB7 41 s\n"
           "basalt-bench: engine ready in 41 s\n",
           "READY leaves the last status line and adds the ready line");
    }
    return failures ? 1 : 0;
}
