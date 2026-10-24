// include/basalt/log.hpp - the engine's one way to write a log line to stderr.
//
// Every line carries its level, so the server and the terminal read it from the line and never guess from words:
//
//     info:     <tag>: <text>
//     warning:  <tag>: warning: <text>
//     error:    <tag>: error: <text>
//
// `tag` is the line's source, "basalt <area>" ("basalt generate", "basalt load"). The format is printf's; the line
// is built whole and written with one call, so lines from different threads do not interleave. `raw` writes text
// as given (tables, progress redraws) and counts as info. tools/check_stderr_logging.sh keeps every other write to
// stderr out of the engine sources.
#pragma once

#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>

#include "basalt/platform/format_check.hpp"

namespace basalt::log {

enum class Level { info, warning, error };

namespace detail {

inline void write_all(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stderr);
    std::fflush(stderr);
}

inline std::string format(const char* fmt, std::va_list ap) {
    std::va_list again;
    va_copy(again, ap);
    char stack_buf[512];
    const int n = std::vsnprintf(stack_buf, sizeof stack_buf, fmt, ap);
    std::string out;
    if (n < 0) {
        out = fmt;
    } else if ((size_t) n < sizeof stack_buf) {
        out.assign(stack_buf, (size_t) n);
    } else {
        out.resize((size_t) n + 1);
        std::vsnprintf(&out[0], out.size(), fmt, again);
        out.resize((size_t) n);
    }
    va_end(again);
    return out;
}

}  // namespace detail

/// The line `<tag>: [warning: |error: ]<text>` without its newline.
inline std::string line(Level level, const char* tag, const std::string& text) {
    std::string out = tag;
    out += level == Level::warning ? ": warning: " : level == Level::error ? ": error: " : ": ";
    return out + text;
}

/// The line `info`, `warn` and `error` write, with its newline.
inline std::string render_v(Level level, const char* tag, const char* fmt, std::va_list ap) {
    return line(level, tag, detail::format(fmt, ap)) + "\n";
}

inline std::string render(Level level, const char* tag, const char* fmt, ...) BASALT_PRINTF_FORMAT(3, 4);
inline std::string render(Level level, const char* tag, const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    std::string out = render_v(level, tag, fmt, ap);
    va_end(ap);
    return out;
}

inline void emit_v(Level level, const char* tag, const char* fmt, std::va_list ap) {
    detail::write_all(render_v(level, tag, fmt, ap));
}

inline void info(const char* tag, const char* fmt, ...) BASALT_PRINTF_FORMAT(2, 3);
inline void warn(const char* tag, const char* fmt, ...) BASALT_PRINTF_FORMAT(2, 3);
inline void error(const char* tag, const char* fmt, ...) BASALT_PRINTF_FORMAT(2, 3);
inline void raw(const char* fmt, ...) BASALT_PRINTF_FORMAT(1, 2);

inline void info(const char* tag, const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    emit_v(Level::info, tag, fmt, ap);
    va_end(ap);
}

inline void warn(const char* tag, const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    emit_v(Level::warning, tag, fmt, ap);
    va_end(ap);
}

inline void error(const char* tag, const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    emit_v(Level::error, tag, fmt, ap);
    va_end(ap);
}

inline void raw(const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    detail::write_all(detail::format(fmt, ap));
    va_end(ap);
}

/// A log line read back: its tag, level and text. Older engines wrote "basalt error: <text>" and
/// "basalt warning: <text>"; both read as a line of tag "basalt". False for a line that is not an engine log line.
struct Parsed {
    Level level = Level::info;
    std::string tag;
    std::string text;
};

inline bool parse(const std::string& line, Parsed& out) {
    if (line.compare(0, 6, "basalt") != 0) return false;
    for (const char* legacy : {"basalt error: ", "basalt warning: "}) {
        const std::string head = legacy;
        if (line.compare(0, head.size(), head) == 0) {
            out = {head[7] == 'e' ? Level::error : Level::warning, "basalt", line.substr(head.size())};
            return true;
        }
    }
    const size_t colon = line.find(": ");
    if (colon == std::string::npos) return false;
    for (size_t i = 6; i < colon; ++i) {
        const char c = line[i];
        const bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (!word && c != ' ' && c != '-') return false;
    }
    std::string rest = line.substr(colon + 2);
    out.level = Level::info;
    out.tag = line.substr(0, colon);
    for (const auto& m : {std::make_pair("warning: ", Level::warning), std::make_pair("error: ", Level::error)}) {
        const std::string marker = m.first;
        if (rest.compare(0, marker.size(), marker) == 0) {
            out.level = m.second;
            rest.erase(0, marker.size());
            break;
        }
    }
    out.text = rest;
    return true;
}

/// The stream for diagnostic dumpers that take a FILE* and print their own unprefixed lines.
inline std::FILE* stream() { return stderr; }

/// Writes `text` as given (no tag, no newline added).
inline void raw_text(const std::string& text) { detail::write_all(text); }

}  // namespace basalt::log
