#include "basalt/models/deepseek4/tokenizer.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool expect(const std::string& text, const std::vector<std::string>& expected, const char* label) {
    std::vector<size_t> ends;
    basalt::models::deepseek4::pretok_deepseek4(text.data(), text.size(), ends);
    std::vector<std::string> actual;
    size_t begin = 0;
    for (size_t end : ends) {
        if (end <= begin || end > text.size()) {
            std::printf("%s: invalid boundary %zu after %zu for %zu-byte input\n", label, end, begin, text.size());
            return false;
        }
        actual.emplace_back(text.substr(begin, end - begin));
        begin = end;
    }
    if (begin != text.size() || actual != expected) {
        std::printf("%s: pre-token boundaries differ\n", label);
        for (const auto& piece : actual) std::printf("  actual [%s]\n", piece.c_str());
        for (const auto& piece : expected) std::printf("  expect [%s]\n", piece.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!expect("abc1234567中文カナ can't x", {"abc", "123", "456", "7", "中文カナ", " can", "'t", " x"},
                "numbers CJK and contractions")) return 1;
    if (!expect("hello   world", {"hello", "  ", " world"}, "whitespace before a word")) return 1;
    if (!expect(".abc!\r\n\n", {".abc", "!\r\n\n"}, "punctuation and line endings")) return 1;
    if (!expect(" café", {" café"}, "leading space on a word")) return 1;
    if (!expect("x中文 y", {"x", "中文", " y"}, "CJK tokens stay isolated")) return 1;
    if (!expect("١٢٣٤", {"١٢٣", "٤"}, "Unicode numbers")) return 1;
    if (!expect("", {}, "empty input")) return 1;
    std::puts("deepseek4 tokenizer pre-split ok");
    return 0;
}
