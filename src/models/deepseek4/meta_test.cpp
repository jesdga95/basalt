#include "models/deepseek4/pack.hpp"
#include "models/deepseek4/meta.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

basalt::core::Json integer(uint64_t value) {
    basalt::core::Json out;
    out.type = basalt::core::Json::Type::Int;
    out.u = value;
    return out;
}

basalt::core::Json floating(double value) {
    basalt::core::Json out;
    out.type = basalt::core::Json::Type::Float;
    out.f = value;
    return out;
}

basalt::core::Json array(std::initializer_list<basalt::core::Json> values) {
    basalt::core::Json out;
    out.type = basalt::core::Json::Type::Array;
    out.items = values;
    out.count = out.items.size();
    return out;
}

}  // namespace

int main() {
    using basalt::core::Json;
    using basalt::models::deepseek4::parse_compression_ratios;
    int failures = 0;
    std::string err;
    std::vector<int> ratios{9};
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::fprintf(stderr, "deepseek4 metadata test: %s\n", name);
            ++failures;
        }
    };

    check(parse_compression_ratios(array({integer(0), integer(4), integer(128)}), ratios, err) &&
          ratios == std::vector<int>({0, 4, 128}), "valid integer schedule");
    for (const Json& invalid : {floating(4.5), floating(std::numeric_limits<double>::infinity()),
                                floating(std::numeric_limits<double>::quiet_NaN()), integer(uint64_t{1} << 40)}) {
        const std::vector<int> before = ratios;
        check(!parse_compression_ratios(array({integer(0), invalid}), ratios, err) && ratios == before,
              "invalid schedule item must be refused without changing output");
    }
    Json incomplete = array({integer(0)});
    incomplete.count = 2;
    check(!parse_compression_ratios(incomplete, ratios, err), "truncated metadata array must be refused");
    check(!parse_compression_ratios(integer(4), ratios, err), "non-array schedule must be refused");

    basalt::MetaValue schedule;
    schedule.type = basalt::MetaType::ARRAY;
    schedule.elem = basalt::MetaType::I32;
    schedule.count = 5;
    for (uint64_t ratio : {0, 4, 0, 0, 0}) {
        basalt::MetaValue item;
        item.type = schedule.elem;
        item.u = ratio;
        schedule.items.push_back(item);
    }
    check(basalt::models::deepseek4::validate_compression_ratios(&schedule, 2).empty(),
          "valid integer GGUF schedule");
    schedule.elem = basalt::MetaType::F32;
    check(!basalt::models::deepseek4::validate_compression_ratios(&schedule, 2).empty(),
          "floating-point GGUF schedule must be refused");
    schedule.elem = basalt::MetaType::I32;
    schedule.items[0].type = basalt::MetaType::F32;
    check(!basalt::models::deepseek4::validate_compression_ratios(&schedule, 2).empty(),
          "inconsistent GGUF array item type must be refused");

    if (failures) return 1;
    std::puts("deepseek4 metadata validation: ok");
    return 0;
}
