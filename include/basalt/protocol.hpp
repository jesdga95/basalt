// include/basalt/protocol.hpp - the engine's one way to write a protocol line to stdout.
//
// The server reads stdout line by line. A line is built whole and written with one call, so lines from different
// threads never interleave and a line is never split by another. A line made of several pieces (a token list) is
// built into a string first and passed to `emit_text`.
#pragma once

#include <cstdarg>
#include <cstdio>
#include <string>

#include "basalt/log.hpp"
#include "basalt/platform/format_check.hpp"

namespace basalt::protocol {

/// Writes `line` (its newline included) to stdout in one call and flushes.
inline void emit_text(const std::string& line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);
}

inline void emit(const char* fmt, ...) BASALT_PRINTF_FORMAT(1, 2);
inline void emit(const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    emit_text(basalt::log::detail::format(fmt, ap));
    va_end(ap);
}

}  // namespace basalt::protocol
