// src/models/inkling2/pretok_test.cpp - the inkling2 pre-tokenizer on strings whose pieces the expression fixes (the
// expected piece ends come from Python's `regex` module running the checkpoint's own expression).
#include "models/inkling2/adapter.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    struct Case {
        const char* text;
        std::vector<size_t> ends;
    };
    const Case cases[] = {
        {"Hello world", {5, 11}},
        {"HelloWorld camelCase", {5, 10, 16, 20}},
        {"I'm sure they'll do it; WE'VE won", {3, 8, 16, 19, 22, 23, 29, 33}},
        {"12345 3.14159 x2", {3, 5, 6, 7, 8, 11, 13, 15, 16}},
        {"  leading\n\n  trailing  ", {1, 9, 11, 12, 21, 23}},
        {"a\r\nb", {1, 3, 4}},
        {"Gr\xc3\xbc\xc3\x9f" "e aus M\xc3\xbc" "nchen", {7, 11, 20}},
        {"\xce\x91\xce\x98\xce\x97\xce\x9d\xce\x91 \xce\x91\xce\xb8\xce\xae\xce\xbd\xce\xb1", {10, 21}},
        {"Hello\xd0\x9c\xd0\xb8\xd1\x80", {5, 11}},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81", {6, 15, 18}},
        {"e\xcc\x81\xcc\x81 combining", {5, 15}},
        {"path/to/file.txt", {4, 7, 12, 16}},
        {"<|tag|> !!! ???", {2, 5, 7, 11, 15}},
        {"MixedCASE_and-dashes", {5, 9, 13, 20}},
        {"\xc2\xa0nbsp\xc2\xa0", {6, 8}},
        {"tabs\tand\tspaces   ", {4, 8, 15, 18}},
        {"D\xc5\xbe d\xc5\xbe D\xc5\xbd", {3, 7, 11}},
        {"\xc7\x85ungla", {7}},
    };
    int failures = 0;
    for (const Case& c : cases) {
        std::vector<size_t> ends;
        const std::string s(c.text);
        basalt::models::inkling2::pretok_inkling2(s.data(), s.size(), ends);
        if (ends != c.ends) {
            ++failures;
            std::printf("FAIL %s:", c.text);
            for (size_t e : ends) std::printf(" %zu", e);
            std::printf("\n");
        }
    }
    std::printf("inkling2_pretok_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
