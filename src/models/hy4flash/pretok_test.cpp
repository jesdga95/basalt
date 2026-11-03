// src/models/hy4flash/pretok_test.cpp - the hy_v4 pre-tokenizer on strings whose pieces the expression fixes.
#include "models/hy4flash/adapter.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    using basalt::models::hy4flash::pretok_hy4;
    struct Case { const char* text; std::vector<std::string> pieces; };
    const std::vector<Case> cases = {
        {"Hello world", {"Hello", " world"}},
        {"x = 12345", {"x", " =", " ", "123", "45"}},
        {"a.b", {"a", ".b"}},
        {"end.\n\nNext", {"end", ".\n\n", "Next"}},
        {"two  spaces", {"two", " ", " spaces"}},
        {"\xe4\xbd\xa0\xe5\xa5\xbd world", {"\xe4\xbd\xa0\xe5\xa5\xbd", " world"}},
    };
    int failures = 0;
    for (const Case& c : cases) {
        std::vector<size_t> ends;
        pretok_hy4(c.text, std::string(c.text).size(), ends);
        std::vector<std::string> got;
        size_t from = 0;
        for (size_t e : ends) { got.push_back(std::string(c.text).substr(from, e - from)); from = e; }
        const bool ok = got == c.pieces;
        if (!ok) {
            std::printf("FAIL '%s':", c.text);
            for (const std::string& p : got) std::printf(" [%s]", p.c_str());
            std::printf("\n");
        }
        failures += !ok;
    }
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
