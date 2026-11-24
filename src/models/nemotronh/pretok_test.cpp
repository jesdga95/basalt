// src/models/nemotronh/pretok_test.cpp - the nemotronh pre-tokenizer on strings whose pieces the expression fixes (the
// expected piece ends come from Python's `regex` module running the checkpoint's own expression).
#include "models/nemotronh/adapter.hpp"

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
        {"I'm sure they'll do it; WE'VE won", {1, 3, 8, 13, 16, 19, 22, 23, 26, 29, 33}},
        {"12345 3.14159 x2", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}},
        {"  leading\n\n  trailing  ", {1, 9, 11, 12, 21, 23}},
        {"a\r\nb", {1, 3, 4}},
        {"Grüße aus München", {7, 11, 20}},
        {"ΑΘΗΝΑ Αθήνα", {10, 21}},
        {"HelloМир", {5, 11}},
        {"你好，世界！", {6, 15, 18}},
        {"é́ combining", {4, 14}},
        {"path/to/file.txt", {4, 7, 12, 16}},
        {"<|tag|> !!! ???", {2, 5, 7, 11, 15}},
        {"MixedCASE_and-dashes", {5, 9, 13, 20}},
        {" nbsp ", {6, 8}},
        {"tabs\tand\tspaces   ", {4, 8, 15, 18}},
        {"Dž dž DŽ", {3, 7, 11}},
        {"ǅungla", {7}},
        {"def f(x):\n    return x**2  # 2024-11-23", {3, 5, 7, 10, 13, 20, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}},
        {"don't STOP'n go", {3, 5, 10, 12, 15}},
    };
    int failures = 0;
    for (const Case& c : cases) {
        std::vector<size_t> ends;
        const std::string s(c.text);
        basalt::models::nemotronh::pretok_nemotronh(s.data(), s.size(), ends);
        if (ends != c.ends) {
            ++failures;
            std::printf("FAIL %s:", c.text);
            for (size_t e : ends) std::printf(" %zu", e);
            std::printf("\n");
        }
    }
    std::printf("nemotronh_pretok_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
