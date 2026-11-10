// The identities context snapshots are keyed by (fingerprint.hpp): SHA-256 against the FIPS 180 examples and every
// padding edge, fed in any split; hex round trips; domain separation and length framing; the model fingerprint's
// required fields, its change under every field and its indifference to the order parameters are set in and to the
// spelling of a real; the prefix hash's change under tokens, length, inputs and steering, inputs past the cut left
// out, and its refusals. The pinned digests come from an independent implementation of the documented encodings.
#include "basalt/context/fingerprint.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace basalt::context;

namespace {
int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

std::string hex_of(const std::string& s) { return to_hex(sha256(s.data(), s.size())); }
Digest digest_of(const std::string& s) { return sha256(s.data(), s.size()); }

void sha256_vectors() {
    CHECK(hex_of("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex_of("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(hex_of("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmn"
                 "opqrstnopqrstu") == "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    CHECK(hex_of(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // the lengths where the padding takes one more block, or just fits
    const struct {
        size_t n;
        const char* hex;
    } edges[] = {
        {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
        {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
        {57, "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6"},
        {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
        {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
        {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
        {111, "6374f73208854473827f6f6a3f43b1f53eaa3b82c21c1a6d69a2110b2a79baad"},
        {112, "f54353008a2553262ecdc4a34749563ba0950e8b0fc8652780b0a614b99683c1"},
        {119, "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb"},
        {120, "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c"},
        {127, "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a"},
        {128, "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e"},
        {129, "c12cb024a2e5551cca0e08fce8f1c5e314555cc3fef6329ee994a3db752166ae"},
    };
    for (const auto& e : edges) CHECK(hex_of(std::string(e.n, 'a')) == e.hex);
}

void sha256_streaming() {
    std::mt19937_64 rng(11);
    std::vector<uint8_t> data(100003);
    for (auto& b : data) b = (uint8_t) rng();
    const Digest whole = sha256(data.data(), data.size());
    for (int trial = 0; trial < 64; ++trial) {
        Sha256 h;
        size_t at = 0;
        while (at < data.size()) {
            const size_t n = std::min<size_t>(rng() % (trial < 32 ? 70 : 9000), data.size() - at);
            h.update(data.data() + at, n);   // zero-length pieces included
            at += n;
        }
        CHECK(h.finish() == whole);
    }
    Sha256 h;
    h.update("abc", 3);
    (void) h.finish();
    h.update("abc", 3);   // finish starts the hasher over
    CHECK(h.finish() == digest_of("abc"));
    h.reset();
    CHECK(h.finish() == digest_of(""));
}

void hex_forms() {
    const Digest d = digest_of("x");
    Digest back{};
    CHECK(to_hex(d).size() == 64);
    CHECK(from_hex(to_hex(d), back) && back == d);
    std::string upper = to_hex(d);
    for (auto& c : upper) c = (char) std::toupper((unsigned char) c);
    back = {};
    CHECK(from_hex(upper, back) && back == d);
    CHECK(!from_hex(to_hex(d).substr(1), back));
    CHECK(!from_hex(to_hex(d) + "0", back));
    std::string bad = to_hex(d);
    bad[10] = 'g';
    back = {};
    CHECK(!from_hex(bad, back) && is_zero(back));   // a refused string leaves `out` alone
    CHECK(is_zero(Digest{}));
    CHECK(!is_zero(d));
}

void domain_framing() {
    CHECK(DomainHash("a").str("x").finish() != DomainHash("b").str("x").finish());
    CHECK(DomainHash("d").str("ab").str("c").finish() != DomainHash("d").str("a").str("bc").finish());
    CHECK(DomainHash("ab").finish() != DomainHash("a").bytes("b", 1).finish());
    CHECK(DomainHash("d").u32(1).finish() != DomainHash("d").u64(1).finish());
    CHECK(DomainHash("d").finish() != sha256("d", 1));
    const Digest pinned = DomainHash("basalt/test").u8(1).u32(7).u64(9).i64(-2).str("xyz").digest(digest_of("abc"))
                              .bytes("raw", 3).finish();
    CHECK(to_hex(pinned) == "f980bea0dfa8714576683b17254a1931d78111d922beae4ad9dcdc0ffde6cc5c");
}

void canonical_values() {
    CHECK(canonical_real(1e7) == "1e+07");
    CHECK(canonical_real(262144) == "262144");
    CHECK(canonical_real(0.1) == "0.1");
    CHECK(canonical_real(1.0000000000000002) == "1.0000000000000002");
    CHECK(canonical_real(-0.0) == canonical_real(0.0));
    CHECK(canonical_real(std::nan("")) == canonical_real(-std::nan("")));
    CHECK(canonical_real(-1.0) == "-1");
    CHECK(canonical_int(32768) == "32768");
    CHECK(canonical_int(-1) == "-1");
}

FingerprintInputs golden_inputs() {
    FingerprintInputs in;
    in.weights = "qwen3.8-flash-next q4_k_m";
    in.weights_digest = digest_of("weights");
    in.tokenizer_digest = digest_of("tokenizer");
    in.chat_template_digest = digest_of("template");
    in.state_schema = 1;
    in.geometry = {2048, 48, 4, 128, 16, 32, 4, 8192, 128, 16, 2, 256, 32, 128, 4, 1, 512, 1536};
    in.kv_precision = "int8";
    in.state_precision = "f32";
    in.positional = {{"rope_scaling", "yarn"},
                     {"freq_base", canonical_real(1e7)},
                     {"yarn_orig_ctx", canonical_real(262144)}};
    in.inputs = {{"vision", to_hex(digest_of("mmproj"))}};
    in.settings = {{"mtp", to_hex(digest_of("mtp"))}, {"mtp_window", canonical_int(32768)}};
    return in;
}

Digest fingerprint_of(const FingerprintInputs& in) {
    Digest d{};
    std::string error;
    CHECK(model_fingerprint(in, d, error));
    return d;
}

void fingerprint_pinned() {
    const Digest d = fingerprint_of(golden_inputs());
    CHECK(to_hex(d) == "771c2ce20630fa8351743098ceccc4f549b58f76b9edbd329018caecb1ee0531");
    CHECK(fingerprint_of(golden_inputs()) == d);   // deterministic
    // parameters set in another order give the same map, the same fingerprint
    FingerprintInputs in = golden_inputs();
    in.positional.clear();
    in.positional["yarn_orig_ctx"] = canonical_real(262144.0);
    in.positional["freq_base"] = canonical_real(10000000.0);
    in.positional["rope_scaling"] = "yarn";
    CHECK(fingerprint_of(in) == d);
}

void fingerprint_sensitivity() {
    const Digest base = fingerprint_of(golden_inputs());
    const std::vector<std::function<void(FingerprintInputs&)>> changes = {
        [](FingerprintInputs& in) { in.weights += " "; },
        [](FingerprintInputs& in) { in.weights_digest[31] ^= 1; },
        [](FingerprintInputs& in) { in.tokenizer_digest[0] ^= 1; },
        [](FingerprintInputs& in) { in.chat_template_digest[5] ^= 0x80; },
        [](FingerprintInputs& in) { in.state_schema = 2; },
        [](FingerprintInputs& in) { in.geometry[3] += 1; },
        [](FingerprintInputs& in) { in.geometry.push_back(0); },
        [](FingerprintInputs& in) { in.kv_precision = "fp16"; },
        [](FingerprintInputs& in) { in.state_precision = "bf16"; },
        [](FingerprintInputs& in) { in.positional["freq_base"] = canonical_real(1e6); },
        [](FingerprintInputs& in) { in.positional["yarn_ext_factor"] = canonical_real(1); },
        [](FingerprintInputs& in) { in.inputs.clear(); },
        [](FingerprintInputs& in) { in.inputs["cvec.0"] = to_hex(digest_of("cvec")) + " " + canonical_real(0.5); },
        [](FingerprintInputs& in) { in.settings["mtp_window"] = canonical_int(0); },
        [](FingerprintInputs& in) { in.settings.erase("mtp"); },
        // a value moved between categories is another configuration
        [](FingerprintInputs& in) { in.inputs["mtp"] = in.settings["mtp"]; in.settings.erase("mtp"); },
        // a key/value split differently does not collide
        [](FingerprintInputs& in) { in.settings.erase("mtp_window"); in.settings["mtp_windo"] = "w32768"; },
    };
    std::vector<Digest> seen{base};
    for (const auto& change : changes) {
        FingerprintInputs in = golden_inputs();
        change(in);
        const Digest d = fingerprint_of(in);
        CHECK(std::find(seen.begin(), seen.end(), d) == seen.end());
        seen.push_back(d);
    }
}

void fingerprint_required() {
    const std::vector<std::function<void(FingerprintInputs&)>> missing = {
        [](FingerprintInputs& in) { in.weights.clear(); },
        [](FingerprintInputs& in) { in.weights_digest = {}; },
        [](FingerprintInputs& in) { in.tokenizer_digest = {}; },
        [](FingerprintInputs& in) { in.chat_template_digest = {}; },
        [](FingerprintInputs& in) { in.state_schema = 0; },
        [](FingerprintInputs& in) { in.geometry.clear(); },
        [](FingerprintInputs& in) { in.kv_precision.clear(); },
        [](FingerprintInputs& in) { in.state_precision.clear(); },
        [](FingerprintInputs& in) { in.positional.clear(); },
        [](FingerprintInputs& in) { in.settings[""] = "x"; },
        [](FingerprintInputs& in) { in.inputs[""] = "x"; },
    };
    for (const auto& change : missing) {
        FingerprintInputs in = golden_inputs();
        change(in);
        Digest d = digest_of("left");
        std::string error;
        CHECK(!model_fingerprint(in, d, error));
        CHECK(!error.empty());
        CHECK(is_zero(d));
    }
    // a model without image encoder, control vector or drafter has empty optional categories
    FingerprintInputs in = golden_inputs();
    in.inputs.clear();
    in.settings.clear();
    Digest d{};
    std::string error;
    CHECK(model_fingerprint(in, d, error) && !is_zero(d));
}

std::vector<int32_t> golden_tokens() {
    std::vector<int32_t> t;
    for (int i = 0; i < 10; ++i) t.push_back(i * 1000 + 3);
    return t;
}

std::vector<PrefixInput> golden_images() {
    return {{InputKind::image, 2, digest_of("image-a")}, {InputKind::image, 7, digest_of("image-b")}};
}

Digest prefix_of(std::span<const int32_t> tokens, const std::vector<PrefixInput>& inputs, bool steering) {
    Digest d{};
    std::string error;
    CHECK(prefix_hash(tokens, inputs, steering, d, error));
    return d;
}

void prefix_pinned() {
    const auto tokens = golden_tokens();
    const auto images = golden_images();
    CHECK(to_hex(prefix_of(tokens, images, true)) == "5a3f0f779a48b1c4a3570d51baf44f3b3a0451d073c91122e36babad16f1131b");
    // the first six tokens with the whole sequence's inputs: the image at 7 is outside the prefix
    const std::span<const int32_t> six(tokens.data(), 6);
    const Digest cut = prefix_of(six, images, true);
    CHECK(to_hex(cut) == "04f3cb5c373167f85621107fcecbb0d96a4df7ffd0aa87ab486dcc12a74f7a1d");
    CHECK(prefix_of(six, {images[0]}, true) == cut);
}

void prefix_sensitivity() {
    const auto tokens = golden_tokens();
    const auto images = golden_images();
    const Digest base = prefix_of(tokens, images, true);
    std::vector<Digest> seen{base};
    auto distinct = [&](const Digest& d) {
        CHECK(std::find(seen.begin(), seen.end(), d) == seen.end());
        seen.push_back(d);
    };
    auto t = tokens;
    t[4] += 1;
    distinct(prefix_of(t, images, true));
    t = tokens;
    std::swap(t[0], t[1]);
    distinct(prefix_of(t, images, true));
    distinct(prefix_of(std::span<const int32_t>(tokens.data(), 9), images, true));
    t = tokens;
    t.push_back(0);
    distinct(prefix_of(t, images, true));
    distinct(prefix_of(tokens, images, false));
    distinct(prefix_of(tokens, {}, true));
    auto moved = images;
    moved[0].position = 3;
    distinct(prefix_of(tokens, moved, true));
    auto other = images;
    other[1].digest[0] ^= 1;
    distinct(prefix_of(tokens, other, true));
    // the same tokens with and without a picture at their place
    distinct(prefix_of(tokens, {images[1]}, true));
}

void prefix_refusals() {
    const auto tokens = golden_tokens();
    Digest d = digest_of("left");
    std::string error;
    CHECK(!prefix_hash({}, {}, false, d, error) && is_zero(d));
    auto t = tokens;
    t[3] = -1;
    CHECK(!prefix_hash(t, {}, false, d, error));
    auto images = golden_images();
    std::swap(images[0], images[1]);
    CHECK(!prefix_hash(tokens, images, false, d, error));   // out of order
    images = golden_images();
    images[1].position = images[0].position;
    CHECK(!prefix_hash(tokens, images, false, d, error));   // two at one position
    images = golden_images();
    images[0].position = -1;
    CHECK(!prefix_hash(tokens, images, false, d, error));
    images = golden_images();
    images[1].digest = {};
    CHECK(!prefix_hash(tokens, images, false, d, error));
    images = golden_images();
    images[0].kind = (InputKind) 9;
    CHECK(!prefix_hash(tokens, images, false, d, error));
    // inputs past the cut are still checked
    images = golden_images();
    images.push_back({InputKind::image, 5, digest_of("late")});
    CHECK(!prefix_hash(std::span<const int32_t>(tokens.data(), 4), images, false, d, error));
}

}  // namespace

int main() {
    sha256_vectors();
    sha256_streaming();
    hex_forms();
    domain_framing();
    canonical_values();
    fingerprint_pinned();
    fingerprint_sensitivity();
    fingerprint_required();
    prefix_pinned();
    prefix_sensitivity();
    prefix_refusals();
    if (failures) {
        std::fprintf(stderr, "fingerprint_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("fingerprint_test: ok\n");
    return 0;
}
