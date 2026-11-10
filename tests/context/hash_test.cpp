// tests/context/hash_test.cpp - SHA-256 known-answer vectors and streaming/hex round-trips.
#include "basalt/context/hash.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", what); }
}

std::string hex_of(const std::string& s) { return basalt::context::to_hex(basalt::context::Sha256::hash(s.data(), s.size())); }

void known_answers() {
    expect(hex_of("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    expect(hex_of("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    expect(hex_of(two) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "two blocks minus tail");
    const std::string million(1000000, 'a');
    expect(hex_of(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "one megabyte of a");
}

void streaming_matches_one_shot() {
    const std::string text(1000, 'x');
    basalt::context::Sha256 s;
    for (size_t off = 0; off < text.size(); off += 37)
        s.update(text.data() + off, std::min<size_t>(37, text.size() - off));
    expect(basalt::context::to_hex(s.finish()) == hex_of(text), "streaming 37-byte chunks");
    basalt::context::Sha256 d;
    d.update("domain", 6);
    d.update_be32(7);
    d.update_be64(11);
    basalt::context::Digest dd = d.finish();
    basalt::context::Sha256 e;
    e.update("domain", 6);
    const uint8_t be32v[4] = {0, 0, 0, 7}, be64v[8] = {0, 0, 0, 0, 0, 0, 0, 11};
    e.update(be32v, 4);
    e.update(be64v, 8);
    expect(e.finish() == dd, "big-endian helpers equal raw bytes");
}

void hex_round_trip() {
    const std::string h = hex_of("round");
    basalt::context::Digest d{};
    expect(basalt::context::from_hex(h, d) && basalt::context::to_hex(d) == h, "hex round trip");
    expect(!basalt::context::from_hex("nope", d), "short hex rejected");
    std::string bad = h;
    bad[10] = 'z';
    expect(!basalt::context::from_hex(bad, d), "non-hex rejected");
}
}  // namespace

int main() {
    known_answers();
    streaming_matches_one_shot();
    hex_round_trip();
    if (failures) { std::printf("%d hash test failures\n", failures); return 1; }
    std::printf("hash tests ok\n");
    return 0;
}
