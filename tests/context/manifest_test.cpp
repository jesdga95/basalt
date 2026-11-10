// The snapshot manifest (manifest.hpp): its bytes against an independent implementation of the documented layout;
// round trips (with and without a parent, the longest id, the most entries); every single-bit flip and every
// truncation refused; tampering re-signed with a valid digest still refused by the field and order rules; the
// structural rules on the struct; completeness against the model's requirements, K/V alone never enough; entry
// digests bound to kind, part and length, and the payload digest to every entry field.
#include "basalt/context/manifest.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
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

constexpr StateKind kKv = StateKind::kv, kGdn = StateKind::gdn, kPle = StateKind::ple, kMtp = StateKind::mtp_draft,
                    kTail = StateKind::tail;

Digest digest_of(const std::string& s) { return sha256(s.data(), s.size()); }

// the fixture bytes the independent implementation uses too
std::vector<uint8_t> fixture(StateKind kind, uint32_t part, size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t) ((i * 7 + (size_t) kind * 31 + part) & 255);
    return v;
}

Digest entry_digest(StateKind kind, uint32_t part, const std::vector<uint8_t>& bytes) {
    StateDigest d(kind, part, bytes.size());
    d.update(bytes.data(), bytes.size());
    return d.finish();
}

InventoryEntry stored(StateKind kind, uint32_t part, size_t n) {
    return {kind, EntryMode::stored, part, n, entry_digest(kind, part, fixture(kind, part, n))};
}
InventoryEntry recompute(StateKind kind) { return {kind, EntryMode::recompute, 0, 0, {}}; }

Manifest golden() {
    Manifest m;
    m.context_id = "ctx_golden-01";
    m.model_fingerprint = digest_of("fingerprint");
    m.prefix_hash = digest_of("prefix");
    m.prefix_tokens = 4096;
    m.parent = ParentPrefix{digest_of("parent"), 1024};
    m.generation = 7;
    m.inventory = {stored(kKv, 0, 1000), stored(kKv, 1, 1), stored(kGdn, 0, 4096), stored(kPle, 0, 0), recompute(kMtp),
                   stored(kTail, 0, 77)};
    m.payload_digest = payload_digest(m.inventory);
    return m;
}

std::vector<uint8_t> encode(const Manifest& m) {
    std::vector<uint8_t> out;
    std::string error;
    CHECK(serialize_manifest(m, out, error));
    return out;
}

ManifestParse parse(const std::vector<uint8_t>& b, Manifest& out) {
    std::string error;
    const ManifestParse p = parse_manifest(b.data(), b.size(), out, error);
    CHECK(p == ManifestParse::ok || !error.empty());   // a refusal says why
    return p;
}

void put32(std::vector<uint8_t>& b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + (size_t) i] = (uint8_t) (v >> (8 * i));
}
void put64(std::vector<uint8_t>& b, size_t at, uint64_t v) {
    for (int i = 0; i < 8; ++i) b[at + (size_t) i] = (uint8_t) (v >> (8 * i));
}
uint32_t get32(const std::vector<uint8_t>& b, size_t at) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) b[at + (size_t) i] << (8 * i);
    return v;
}

// The trailing digest recomputed as the format documents it: tampering that only a field rule can catch.
void resign(std::vector<uint8_t>& b) {
    const size_t n = b.size() - 32;
    const Digest d = DomainHash("basalt/context/manifest/v1").u64(n).bytes(b.data(), n).finish();
    std::memcpy(b.data() + n, d.data(), 32);
}

// The payload digest recomputed from the entry table as it stands, so an entry edit is the only defect left.
void repay(std::vector<uint8_t>& b) {
    const uint32_t count = get32(b, 20);
    DomainHash h("basalt/context/payload/v1");
    h.u64(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = b.data() + kManifestHeaderBytes + i * kManifestEntryBytes;
        uint64_t bytes = 0;
        for (int k = 0; k < 8; ++k) bytes |= (uint64_t) e[8 + k] << (8 * k);
        h.u8(e[0]).u8(e[1]).u32(get32(b, kManifestHeaderBytes + i * kManifestEntryBytes + 4)).u64(bytes).bytes(e + 16, 32);
    }
    const Digest d = h.finish();
    std::memcpy(b.data() + 144, d.data(), 32);
}

void golden_bytes() {
    const Manifest m = golden();
    const auto b = encode(m);
    CHECK(manifest_bytes(6) == 632);
    CHECK(b.size() == 632);
    CHECK(std::memcmp(b.data(), "BSLTCTXM", 8) == 0);
    CHECK(to_hex(m.inventory[0].digest) == "f28649824117ab18194942926c798279241c88d0d11851502c1db642e4435d71");
    CHECK(to_hex(m.payload_digest) == "700d250e7a5dfe4f46b70150e2494e11886599487f73dd5b86dbe0c3b47bec94");
    Digest tail{};
    std::memcpy(tail.data(), b.data() + b.size() - 32, 32);
    CHECK(to_hex(tail) == "f1359bff1c46dfac1988f5c2703d1bd949dead2018652526b442ccdf68843f4d");
    CHECK(to_hex(sha256(b.data(), b.size())) == "764603906ad8b2a1c8a9b8e2f758874040fd9ac4428a812284b672a66756a945");
    CHECK(std::string(state_kind_name(kMtp)) == "mtp_draft");
    CHECK(std::string(state_kind_name((StateKind) 0)) == "unknown");
}

void round_trips() {
    const Manifest m = golden();
    auto b = encode(m);
    Manifest back;
    CHECK(parse(b, back) == ManifestParse::ok && back == m);
    // the bytes after a manifest (its padding in a snapshot file) are not its business
    auto padded = b;
    padded.resize(b.size() + 4096, 0xAB);
    CHECK(parse(padded, back) == ManifestParse::ok && back == m);
    uint64_t bytes = 0;
    std::string error;
    CHECK(manifest_extent(b.data(), b.size(), bytes, error) == ManifestParse::ok && bytes == b.size());

    Manifest root = m;
    root.parent.reset();
    CHECK(parse(encode(root), back) == ManifestParse::ok && back == root);

    Manifest same = m;   // a fork that has not diverged: its parent's prefix is its own
    same.parent = ParentPrefix{m.prefix_hash, m.prefix_tokens};
    CHECK(parse(encode(same), back) == ManifestParse::ok && back == same);

    Manifest big = m;
    big.context_id = std::string(128, 'z');
    big.prefix_tokens = kMaxPrefixTokens;
    big.generation = ~0ull;
    big.inventory.clear();
    for (uint32_t i = 0; i + 4 < kMaxInventoryEntries; ++i) big.inventory.push_back(stored(kKv, i, i % 3));
    big.inventory.push_back(stored(kGdn, 0, 10));
    big.inventory.push_back(stored(kPle, 0, 10));
    big.inventory.push_back(recompute(kMtp));
    big.inventory.push_back({kTail, EntryMode::stored, 0, kMaxEntryBytes, digest_of("large")});
    big.payload_digest = payload_digest(big.inventory);
    b = encode(big);
    CHECK(b.size() == manifest_bytes(kMaxInventoryEntries));
    CHECK(parse(b, back) == ManifestParse::ok && back == big);
}

void every_bit_flip() {
    const auto b = encode(golden());
    int wrong = 0;
    for (size_t i = 0; i < b.size(); ++i)
        for (int bit = 0; bit < 8; ++bit) {
            auto c = b;
            c[i] ^= (uint8_t) (1u << bit);
            Manifest out;
            const ManifestParse p = parse(c, out);
            // the magic, the version and the size fields are read before the digest; any other byte breaks the digest
            const ManifestParse expect = i < 8 ? ManifestParse::not_manifest
                                         : i < 12 ? ManifestParse::unsupported_version
                                         : i < 16 || (i >= 20 && i < 24) ? ManifestParse::invalid
                                                                         : ManifestParse::digest_mismatch;
            if (p != expect) ++wrong;
            if (!(out == Manifest{})) ++wrong;   // nothing decoded from a refused manifest
        }
    CHECK(wrong == 0);
}

void every_truncation() {
    const auto b = encode(golden());
    for (size_t n = 0; n < b.size(); ++n) {
        Manifest out;
        std::string error;
        CHECK(parse_manifest(b.data(), n, out, error) == ManifestParse::truncated);
    }
}

void resigned_tampering() {
    const auto b = encode(golden());
    const size_t entry0 = kManifestHeaderBytes;
    const struct {
        const char* what;
        ManifestParse expect;
        std::function<void(std::vector<uint8_t>&)> edit;
    } cases[] = {
        {"flag bit outside the format", ManifestParse::invalid, [](auto& c) { c[16] |= 2; }},
        {"parent fields without the flag", ManifestParse::invalid, [](auto& c) { c[16] &= (uint8_t) ~1u; }},
        {"reserved byte after the id length", ManifestParse::invalid, [](auto& c) { c[177] = 1; }},
        {"id padding", ManifestParse::invalid, [](auto& c) { c[184 + 13] = 'x'; }},
        {"id length 0", ManifestParse::invalid, [](auto& c) { c[176] = 0; }},
        {"id length past the field", ManifestParse::invalid, [](auto& c) { c[176] = 129; }},
        {"id character outside the alphabet", ManifestParse::invalid, [](auto& c) { c[184] = '/'; }},
        {"prefix of 0 tokens", ManifestParse::invalid, [](auto& c) { put64(c, 24, 0); }},
        {"prefix too long", ManifestParse::invalid, [](auto& c) { put64(c, 24, kMaxPrefixTokens + 1); }},
        {"generation 0", ManifestParse::invalid, [](auto& c) { put64(c, 32, 0); }},
        {"parent longer than the prefix", ManifestParse::invalid, [](auto& c) { put64(c, 40, 4097); }},
        {"zero fingerprint", ManifestParse::invalid, [](auto& c) { std::fill(c.begin() + 48, c.begin() + 80, 0); }},
        {"zero prefix hash", ManifestParse::invalid, [](auto& c) { std::fill(c.begin() + 80, c.begin() + 112, 0); }},
        {"stale payload digest", ManifestParse::invalid, [](auto& c) { c[150] ^= 1; }},
        {"format version 2", ManifestParse::unsupported_version, [](auto& c) { put32(c, 8, 2); }},
        {"magic", ManifestParse::not_manifest, [](auto& c) { c[0] = 'X'; }},
        {"entry count and size disagree", ManifestParse::invalid, [](auto& c) { put32(c, 20, 5); }},
        {"manifest size field", ManifestParse::invalid, [](auto& c) { put32(c, 12, 633); }},
        // entry edits keep the payload digest consistent, so only the rule they break is left
        {"kind 0", ManifestParse::invalid, [=](auto& c) { c[entry0] = 0; repay(c); }},
        {"kind past tail", ManifestParse::invalid, [=](auto& c) { c[entry0 + 5 * 48] = 6; repay(c); }},
        {"mode 0", ManifestParse::invalid, [=](auto& c) { c[entry0 + 1] = 0; repay(c); }},
        {"mode 3", ManifestParse::invalid, [=](auto& c) { c[entry0 + 1] = 3; repay(c); }},
        {"entry reserved bytes", ManifestParse::invalid, [=](auto& c) { c[entry0 + 2] = 1; repay(c); }},
        {"part gap", ManifestParse::invalid, [=](auto& c) { put32(c, entry0 + 48 + 4, 2); repay(c); }},
        {"parts from 1", ManifestParse::invalid, [=](auto& c) { put32(c, entry0 + 4, 1); repay(c); }},
        {"kinds out of order", ManifestParse::invalid, [=](auto& c) { c[entry0 + 2 * 48] = 3; c[entry0 + 3 * 48] = 2; repay(c); }},
        {"recompute with bytes", ManifestParse::invalid, [=](auto& c) { put64(c, entry0 + 4 * 48 + 8, 1); repay(c); }},
        {"recompute with a digest", ManifestParse::invalid, [=](auto& c) { c[entry0 + 4 * 48 + 16] = 1; repay(c); }},
        {"stored without a digest", ManifestParse::invalid,
         [=](auto& c) { std::fill(c.begin() + entry0 + 16, c.begin() + entry0 + 48, 0); repay(c); }},
        {"entry too large", ManifestParse::invalid, [=](auto& c) { put64(c, entry0 + 8, kMaxEntryBytes + 1); repay(c); }},
        // a consistent forgery of an entry's size is a valid manifest: the payload check catches it (snapshot IO)
        {"entry size, payload digest kept consistent", ManifestParse::ok,
         [=](auto& c) { put64(c, entry0 + 8, 1001); repay(c); }},
    };
    for (const auto& t : cases) {
        auto c = b;
        t.edit(c);
        resign(c);
        Manifest out;
        const ManifestParse p = parse(c, out);
        if (p != t.expect) std::fprintf(stderr, "case \"%s\": parse result %d\n", t.what, (int) p);
        CHECK(p == t.expect);
    }
}

void structural_rules() {
    auto refused = [](const Manifest& m) {
        std::vector<uint8_t> out{1};
        std::string error;
        const bool ok = serialize_manifest(m, out, error);
        return !ok && out.empty() && !error.empty();
    };
    auto sealed = [](Manifest m) {
        m.payload_digest = payload_digest(m.inventory);
        return m;
    };
    const std::vector<std::function<void(Manifest&)>> breaks = {
        [](Manifest& m) { m.format_version = 2; },
        [](Manifest& m) { m.context_id.clear(); },
        [](Manifest& m) { m.context_id = std::string(129, 'a'); },
        [](Manifest& m) { m.context_id = "two words"; },
        [](Manifest& m) { m.model_fingerprint = {}; },
        [](Manifest& m) { m.prefix_hash = {}; },
        [](Manifest& m) { m.prefix_tokens = 0; },
        [](Manifest& m) { m.prefix_tokens = kMaxPrefixTokens + 1; },
        [](Manifest& m) { m.generation = 0; },
        [](Manifest& m) { m.parent->tokens = 0; },
        [](Manifest& m) { m.parent->tokens = m.prefix_tokens + 1; },
        [](Manifest& m) { m.parent->hash = {}; },
        [](Manifest& m) { m.inventory.clear(); },
        [](Manifest& m) {
            const InventoryEntry first = m.inventory[0];
            m.inventory.resize(kMaxInventoryEntries + 1, first);
        },
        [](Manifest& m) { m.inventory[0].kind = (StateKind) 0; },
        [](Manifest& m) { m.inventory[5].kind = (StateKind) 6; },
        [](Manifest& m) { m.inventory[2].mode = (EntryMode) 0; },
        [](Manifest& m) { std::swap(m.inventory[2], m.inventory[5]); },
        [](Manifest& m) { m.inventory[1].part = 0; },
        [](Manifest& m) { m.inventory[0].part = 1; m.inventory[1].part = 2; },
        [](Manifest& m) { m.inventory[1].part = 2; },
        [](Manifest& m) { m.inventory[4].bytes = 8; },
        [](Manifest& m) { m.inventory[4].digest = digest_of("x"); },
        [](Manifest& m) { m.inventory.insert(m.inventory.begin() + 5, stored(kMtp, 1, 8)); },
        [](Manifest& m) { m.inventory[2].digest = {}; },
        [](Manifest& m) { m.inventory[2].bytes = kMaxEntryBytes + 1; },
    };
    for (const auto& change : breaks) {
        Manifest m = golden();
        change(m);
        CHECK(refused(sealed(m)));
    }
    Manifest m = golden();
    m.inventory[2].bytes += 1;   // the payload digest no longer the inventory's
    CHECK(refused(m));

    // a draft may leave the digests a publish computes at zero, but not break any other rule
    Manifest draft = golden();
    for (auto& e : draft.inventory) e.digest = {};
    draft.payload_digest = {};
    std::string error;
    CHECK(validate_manifest_draft(draft, error));
    CHECK(!validate_manifest(draft, error));
    draft.inventory[4].digest = digest_of("x");   // still a recompute entry with a digest
    CHECK(!validate_manifest_draft(draft, error));
    draft = golden();
    std::swap(draft.inventory[0], draft.inventory[5]);
    CHECK(!validate_manifest_draft(draft, error));
    m = golden();
    m.parent->tokens = m.prefix_tokens;   // a parent as long as the prefix is allowed
    CHECK(validate_manifest(m, error));
    m.prefix_tokens = kMaxPrefixTokens;
    CHECK(validate_manifest(m, error));
}

Manifest with(std::vector<InventoryEntry> inventory) {
    Manifest m = golden();
    m.inventory = std::move(inventory);
    m.payload_digest = payload_digest(m.inventory);
    std::string error;
    CHECK(validate_manifest(m, error));
    return m;
}

void completeness() {
    std::string error;
    const uint32_t kv = state_bit(kKv), gdn = state_bit(kGdn), ple = state_bit(kPle), mtp = state_bit(kMtp),
                   tail = state_bit(kTail);
    const auto kv0 = stored(kKv, 0, 64), kv1 = stored(kKv, 1, 64), g = stored(kGdn, 0, 64), p = stored(kPle, 0, 64),
               d = stored(kMtp, 0, 64), t = stored(kTail, 0, 64);
    CHECK(manifest_complete(golden(), kHybridState, error));
    CHECK(manifest_complete(with({kv0, kv1, g, p, d, t}), kHybridState, error));
    // K/V alone does not restore the hybrid model
    CHECK(!manifest_complete(with({kv0, kv1}), kHybridState, error));
    CHECK(error.find("gdn") != std::string::npos);
    CHECK(!manifest_complete(with({kv0, p, d, t}), kHybridState, error));   // no GDN state
    CHECK(!manifest_complete(with({kv0, g, d, t}), kHybridState, error));   // no PLE history
    CHECK(!manifest_complete(with({kv0, g, p, t}), kHybridState, error));   // no draft state, not even marked
    CHECK(!manifest_complete(with({kv0, g, p, d}), kHybridState, error));   // no indexer tail
    CHECK(!manifest_complete(with({g, p, d, t}), kHybridState, error));     // no K/V
    // only what the model can rebuild may be marked recompute
    CHECK(manifest_complete(with({kv0, g, p, recompute(kMtp), t}), kHybridState, error));
    CHECK(!manifest_complete(with({kv0, recompute(kGdn), p, d, t}), kHybridState, error));
    CHECK(!manifest_complete(with({kv0, g, recompute(kPle), d, t}), kHybridState, error));
    CHECK(!manifest_complete(with({kv0, g, p, d, recompute(kTail)}), kHybridState, error));
    CHECK(!manifest_complete(with({recompute(kKv), g, p, d, t}), kHybridState, error));
    // a model without PLE or drafter: state it does not hold is refused, not ignored
    const StateRequirements lean{kv | gdn | tail, 0};
    CHECK(manifest_complete(with({kv0, g, t}), lean, error));
    CHECK(!manifest_complete(with({kv0, g, p, t}), lean, error));
    CHECK(!manifest_complete(with({kv0, g, recompute(kMtp), t}), lean, error));
    CHECK(manifest_complete(with({kv0, kv1}), StateRequirements{kv, 0}, error));
    // requirements themselves
    CHECK(valid_requirements(kHybridState));
    CHECK(valid_requirements(lean));
    CHECK(!valid_requirements({0, 0}));
    CHECK(!valid_requirements({gdn | tail, 0}));
    CHECK(!valid_requirements({kv, kv}));
    CHECK(!valid_requirements({kv, gdn}));
    CHECK(!valid_requirements({kv | (1u << 9), 0}));
    CHECK(!valid_requirements({kv | 1u, 0}));
    CHECK(!manifest_complete(golden(), {kv | ple | mtp, kv}, error));
    CHECK(error.find("requirements") != std::string::npos);
}

void entry_digests() {
    const auto bytes = fixture(kKv, 0, 5000);
    const Digest whole = entry_digest(kKv, 0, bytes);
    StateDigest pieces(kKv, 0, bytes.size());
    pieces.update(bytes.data(), 1);
    pieces.update(bytes.data() + 1, 0);
    pieces.update(bytes.data() + 1, 4000);
    pieces.update(bytes.data() + 4001, 999);
    CHECK(pieces.finish() == whole);
    CHECK(entry_digest(kGdn, 0, bytes) != whole);   // bound to its kind
    CHECK(entry_digest(kKv, 1, bytes) != whole);    // and to its part
    StateDigest under(kKv, 0, bytes.size());
    under.update(bytes.data(), bytes.size() - 1);
    CHECK(is_zero(under.finish()));
    StateDigest over(kKv, 0, 10);
    over.update(bytes.data(), 11);
    CHECK(is_zero(over.finish()));
    StateDigest empty(kPle, 0, 0);
    const Digest none = empty.finish();
    CHECK(!is_zero(none));
    CHECK(none != entry_digest(kTail, 0, {}));

    const auto inventory = golden().inventory;
    const Digest base = payload_digest(inventory);
    const std::vector<std::function<void(std::vector<InventoryEntry>&)>> changes = {
        [](auto& v) { v[0].kind = kGdn; },
        [](auto& v) { v[2].mode = EntryMode::recompute; },
        [](auto& v) { v[1].part = 7; },
        [](auto& v) { v[5].bytes += 1; },
        [](auto& v) { v[3].digest[0] ^= 1; },
        [](auto& v) { v.pop_back(); },
        [](auto& v) { std::swap(v[0], v[1]); },
    };
    for (const auto& change : changes) {
        auto v = inventory;
        change(v);
        CHECK(payload_digest(v) != base);
    }
}

}  // namespace

int main() {
    golden_bytes();
    round_trips();
    every_bit_flip();
    every_truncation();
    resigned_tampering();
    structural_rules();
    completeness();
    entry_digests();
    if (failures) {
        std::fprintf(stderr, "manifest_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("manifest_test: ok\n");
    return 0;
}
