// include/basalt/log.hpp: the line each level renders. The writes themselves are one call each to stderr.
#include "basalt/log.hpp"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void is(const std::string& got, const std::string& want, const char* what) {
    const bool ok = got == want;
    std::printf("log %-52s %s\n", what, ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) {
        std::printf("   got  %s\n   want %s\n", got.c_str(), want.c_str());
        ++failures;
    }
}

}  // namespace

int main() {
    using basalt::log::Level;
    using basalt::log::render;
    is(render(Level::info, "basalt generate", "session is up (engine %s)", "1.2"),
       "basalt generate: session is up (engine 1.2)\n", "info keeps the tag and has no marker");
    is(render(Level::warning, "basalt serve", "RAM %d MiB low", 512), "basalt serve: warning: RAM 512 MiB low\n",
       "warning marker after the tag");
    is(render(Level::error, "basalt generate", "cannot open %s", "x.basalt"),
       "basalt generate: error: cannot open x.basalt\n", "error marker after the tag");
    is(render(Level::error, "basalt", "kernel: %s", "bad"), "basalt: error: kernel: bad\n", "a line of no area");
    is(render(Level::warning, "basalt-bench", "NVML is unavailable"), "basalt-bench: warning: NVML is unavailable\n",
       "a tool's own tag");
    is(render(Level::info, "basalt load", "plan %s%s %llu", "copy ", "cuda1", 4096ull),
       "basalt load: plan copy cuda1 4096\n", "a parsed progress line is unchanged");
    is(render(Level::info, "basalt serve", "%.0f%% of %lld", 12.0, 7ll), "basalt serve: 12% of 7\n", "percent signs");
    is(render(Level::info, "basalt", "%s", ""), "basalt: \n", "an empty message");
    const std::string longer(1500, 'x');
    is(render(Level::error, "basalt generate", "%s", longer.c_str()), "basalt generate: error: " + longer + "\n",
       "a message longer than the first buffer");
    is(render(Level::info, "basalt serve", "two\nlines"), "basalt serve: two\nlines\n", "an inner newline is kept");
    is(basalt::log::line(Level::warning, "basalt mtp", "text"), "basalt mtp: warning: text", "line() has no newline");

    // parse() reads back what render() wrote, and the older forms
    const auto parsed = [](const std::string& l) {
        basalt::log::Parsed p;
        if (!basalt::log::parse(l, p)) return std::string("not a log line");
        return std::string(p.level == Level::error ? "error" : p.level == Level::warning ? "warning" : "info") + "|" +
               p.tag + "|" + p.text;
    };
    is(parsed("basalt generate: error: no GPU"), "error|basalt generate|no GPU", "parse: error");
    is(parsed("basalt serve: warning: low RAM"), "warning|basalt serve|low RAM", "parse: warning");
    is(parsed("basalt load: cuda0 1 2"), "info|basalt load|cuda0 1 2", "parse: info");
    is(parsed("basalt: error: kernel: bad"), "error|basalt|kernel: bad", "parse: no area");
    is(parsed("basalt-bench: warning: NVML"), "warning|basalt-bench|NVML", "parse: a tool's tag");
    is(parsed("basalt error: old form"), "error|basalt|old form", "parse: older error form");
    is(parsed("basalt warning: old form"), "warning|basalt|old form", "parse: older warning form");
    is(parsed("basalt serve: no error here"), "info|basalt serve|no error here", "parse: the word error is no marker");
    is(parsed("CUDA error: out of memory"), "not a log line", "parse: a line of another writer");
    is(parsed("basalt (ms/window): x"), "not a log line", "parse: odd characters in the tag");
    return failures == 0 ? 0 : 1;
}
