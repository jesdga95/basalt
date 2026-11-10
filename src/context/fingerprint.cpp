// src/context/fingerprint.cpp - see include/basalt/context/fingerprint.hpp.
#include "basalt/context/fingerprint.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>

namespace basalt::context {
namespace {

constexpr const char* kFingerprintDomain = "basalt/context/model-fingerprint/v1";
constexpr const char* kPrefixDomain = "basalt/context/prefix/v1";

bool fail(std::string& error, const char* message) {
    error = std::string("context identity: ") + message;
    return false;
}

void put_params(DomainHash& h, const FingerprintParams& params) {
    h.u64(params.size());
    for (const auto& [key, value] : params) h.str(key).str(value);
}

bool keys_named(const FingerprintParams& params) {
    return std::none_of(params.begin(), params.end(), [](const auto& kv) { return kv.first.empty(); });
}

}  // namespace

DomainHash::DomainHash(std::string_view domain) { str(domain); }

DomainHash& DomainHash::u8(uint8_t v) {
    sha_.update(&v, 1);
    return *this;
}

DomainHash& DomainHash::u32(uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; ++i) b[i] = (uint8_t) (v >> (8 * i));
    sha_.update(b, sizeof b);
    return *this;
}

DomainHash& DomainHash::u64(uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t) (v >> (8 * i));
    sha_.update(b, sizeof b);
    return *this;
}

DomainHash& DomainHash::i64(int64_t v) { return u64((uint64_t) v); }

DomainHash& DomainHash::str(std::string_view s) {
    u64(s.size());
    sha_.update(s.data(), s.size());
    return *this;
}

DomainHash& DomainHash::digest(const Digest& d) {
    sha_.update(d.data(), d.size());
    return *this;
}

DomainHash& DomainHash::bytes(const void* p, size_t n) {
    sha_.update(p, n);
    return *this;
}

Digest DomainHash::finish() { return sha_.finish(); }

bool is_zero(const Digest& d) {
    return std::all_of(d.begin(), d.end(), [](uint8_t b) { return b == 0; });
}

std::string canonical_int(int64_t v) {
    char b[24];
    const auto r = std::to_chars(b, b + sizeof b, v);
    return std::string(b, r.ptr);
}

std::string canonical_real(double v) {
    if (std::isnan(v)) return "nan";   // one spelling whatever the sign and payload
    if (v == 0) v = 0;                 // -0 is the same setting as 0
    char b[64];
    const auto r = std::to_chars(b, b + sizeof b, v);
    return std::string(b, r.ptr);
}

bool model_fingerprint(const FingerprintInputs& in, Digest& out, std::string& error) {
    out = {};
    if (in.weights.empty()) return fail(error, "no weights identifier");
    if (is_zero(in.weights_digest)) return fail(error, "no weights digest");
    if (is_zero(in.tokenizer_digest)) return fail(error, "no tokenizer digest");
    if (is_zero(in.chat_template_digest)) return fail(error, "no chat template digest");
    if (in.state_schema == 0) return fail(error, "no state schema version");
    if (in.geometry.empty()) return fail(error, "no model geometry");
    if (in.kv_precision.empty()) return fail(error, "no K/V precision");
    if (in.state_precision.empty()) return fail(error, "no state precision");
    if (in.positional.empty()) return fail(error, "no positional configuration");
    if (!keys_named(in.positional) || !keys_named(in.inputs) || !keys_named(in.settings))
        return fail(error, "a fingerprint parameter without a name");
    DomainHash h(kFingerprintDomain);
    h.str(in.weights).digest(in.weights_digest).digest(in.tokenizer_digest).digest(in.chat_template_digest);
    h.u32(in.state_schema).u64(in.geometry.size());
    for (int64_t g : in.geometry) h.i64(g);
    h.str(in.kv_precision).str(in.state_precision);
    put_params(h, in.positional);
    put_params(h, in.inputs);
    put_params(h, in.settings);
    out = h.finish();
    return true;
}

bool prefix_hash(std::span<const int32_t> tokens, std::span<const PrefixInput> inputs, bool steering, Digest& out,
                 std::string& error) {
    out = {};
    if (tokens.empty()) return fail(error, "an empty prefix");
    if (std::any_of(tokens.begin(), tokens.end(), [](int32_t t) { return t < 0; }))
        return fail(error, "a negative token ID");
    int64_t previous = -1;
    uint64_t inside = 0;
    for (const auto& x : inputs) {
        if (x.kind != InputKind::image) return fail(error, "an input of unknown kind");
        if (x.position <= previous) return fail(error, "input positions not strictly increasing from 0");
        if (is_zero(x.digest)) return fail(error, "an input without a content digest");
        previous = x.position;
        if ((uint64_t) x.position < tokens.size()) ++inside;
    }
    DomainHash h(kPrefixDomain);
    h.u64(tokens.size());
    if constexpr (std::endian::native == std::endian::little) {
        h.bytes(tokens.data(), tokens.size_bytes());
    } else {
        for (int32_t t : tokens) h.u32((uint32_t) t);
    }
    h.u64(inside);
    for (const auto& x : inputs)
        if ((uint64_t) x.position < tokens.size()) h.u8((uint8_t) x.kind).i64(x.position).digest(x.digest);
    h.u8(steering ? 1 : 0);
    out = h.finish();
    return true;
}

}  // namespace basalt::context
