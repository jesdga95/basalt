// tests/program/protocol_lines_test.cpp - the protocol lines the server reads for the conversation cache and for a
// prompt that does not fit the context (include/basalt/protocol.hpp): their exact form, one line each.
#include "basalt/protocol.hpp"

#include <cstdio>
#include <string>

static int g_fail = 0;
static void expect(const std::string& got, const std::string& want, int line) {
    if (got != want) {
        std::printf("FAIL line %d:\n  got  %s  want %s", line, got.c_str(), want.c_str());
        ++g_fail;
    }
}
#define EXPECT(got, want) expect((got), (want), __LINE__)

int main() {
    using namespace basalt::protocol;

    // a park reports the RAM tier's figures
    EXPECT(conversation_line(6, 13314398618ull, 3), "CONV parked=6 bytes=13314398618 evictions=3\n");
    EXPECT(conversation_line(0, 0, 0), "CONV parked=0 bytes=0 evictions=0\n");
    // a restore adds the tokens it returned and its ms
    EXPECT(conversation_line(5, 12000000000ull, 1, 187342, 912.34), "CONV parked=5 bytes=12000000000 evictions=1 restored=187342 restored_ms=912.3\n");
    EXPECT(conversation_line(1, 2, 3, 0, 0.0), "CONV parked=1 bytes=2 evictions=3 restored=0 restored_ms=0.0\n");

    // the context-overflow ERR: fields for the server, the sentence for people
    EXPECT(context_overflow_line(3000, 500, 3200),
           "ERR context_overflow prompt=3000 max_new=500 context=3200: prompt (3000 tokens) + max_new (500) exceeds the context (3200)\n");
    EXPECT(context_overflow_line(3000, 500, 3200, 2),
           "ERR context_overflow prompt=3000 max_new=500 context=3200 slot=2: prompt (3000 tokens) + max_new (500) exceeds slot 2's context (3200)\n");
    EXPECT(context_no_room_line(4000, 4000),
           "ERR context_no_room prompt=4000 context=4000: prompt (4000 tokens) leaves no room to answer in the context (4000)\n");

    if (g_fail == 0) std::printf("protocol lines OK\n");
    return g_fail == 0 ? 0 : 1;
}
