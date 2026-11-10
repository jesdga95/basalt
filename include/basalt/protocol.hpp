// include/basalt/protocol.hpp - the engine's one way to write a protocol line to stdout.
//
// The server reads stdout line by line. A line is built whole and written with one call, so lines from different
// threads never interleave and a line is never split by another. A line made of several pieces (a token list) is
// built into a string first and passed to `emit_text`.
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstddef>
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

/// `ERR context_overflow prompt=N max_new=M context=C[ slot=S]: <message>`: the prompt and its answer do not fit the
/// context (a slot's own, when `slot` >= 0). The fields are the facts the server reads; the message is for people.
inline std::string context_overflow_line(long long prompt, long long max_new, long long context, int slot = -1) {
    std::string l = "ERR context_overflow prompt=" + std::to_string(prompt) + " max_new=" + std::to_string(max_new) +
                    " context=" + std::to_string(context);
    if (slot >= 0) l += " slot=" + std::to_string(slot);
    l += ": prompt (" + std::to_string(prompt) + " tokens) + max_new (" + std::to_string(max_new) + ") exceeds ";
    l += slot >= 0 ? "slot " + std::to_string(slot) + "'s context (" : "the context (";
    return l + std::to_string(context) + ")\n";
}

/// `ERR context_no_room prompt=N context=C: <message>`: the prompt leaves no room to answer.
inline std::string context_no_room_line(long long prompt, long long context) {
    return "ERR context_no_room prompt=" + std::to_string(prompt) + " context=" + std::to_string(context) +
           ": prompt (" + std::to_string(prompt) + " tokens) leaves no room to answer in the context (" +
           std::to_string(context) + ")\n";
}

/// `CONV parked=N bytes=N evictions=N[ restored=N restored_ms=X]`: the conversation cache's RAM tier, after a park
/// or (with `restored` >= 0, the tokens a request got back and the ms it took) a restore.
inline std::string conversation_line(size_t parked, size_t bytes, size_t evictions, long long restored = -1,
                                     double restored_ms = 0) {
    std::string l = "CONV parked=" + std::to_string(parked) + " bytes=" + std::to_string(bytes) +
                    " evictions=" + std::to_string(evictions);
    if (restored >= 0) {
        char ms[32];
        std::snprintf(ms, sizeof ms, "%.1f", restored_ms);
        l += " restored=" + std::to_string(restored) + " restored_ms=" + ms;
    }
    return l + "\n";
}

}  // namespace basalt::protocol
