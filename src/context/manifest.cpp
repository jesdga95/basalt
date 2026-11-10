// src/context/manifest.cpp - see include/basalt/context/manifest.hpp.
#include "basalt/context/manifest.hpp"

#include <cstring>

namespace basalt::context {
namespace {

constexpr char kMagic[8] = {'B', 'S', 'L', 'T', 'C', 'T', 'X', 'M'};
constexpr const char* kManifestDomain = "basalt/context/manifest/v1";
constexpr const char* kEntryDomain = "basalt/context/state-entry/v1";
constexpr const char* kPayloadDomain = "basalt/context/payload/v1";
constexpr uint32_t kFlagParent = 1;
constexpr size_t kIdBytes = 128;
constexpr size_t kDigestBytes = 32;

enum : size_t {
    kAtVersion = 8, kAtBytes = 12, kAtFlags = 16, kAtCount = 20, kAtTokens = 24, kAtGeneration = 32,
    kAtParentTokens = 40, kAtFingerprint = 48, kAtPrefix = 80, kAtParent = 112, kAtPayload = 144, kAtIdLength = 176,
    kAtId = 184,
};
static_assert(kAtId + kIdBytes == kManifestHeaderBytes, "the context id closes the header");

void put32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t) (v >> (8 * i));
}
void put64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t) (v >> (8 * i));
}
uint32_t get32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) p[i] << (8 * i);
    return v;
}
uint64_t get64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (8 * i);
    return v;
}
Digest get_digest(const uint8_t* p) {
    Digest d;
    std::memcpy(d.data(), p, d.size());
    return d;
}

bool fail(std::string& error, const std::string& message) {
    error = "context manifest: " + message;
    return false;
}

ManifestParse refuse(std::string& error, ManifestParse why, const std::string& message) {
    fail(error, message);
    return why;
}

bool known_kind(StateKind k) { return (uint8_t) k >= (uint8_t) StateKind::kv && (uint8_t) k <= (uint8_t) StateKind::tail; }
bool known_mode(EntryMode m) { return m == EntryMode::stored || m == EntryMode::recompute; }

std::string entry_name(const InventoryEntry& e) {
    return std::string(state_kind_name(e.kind)) + " part " + std::to_string(e.part);
}

Digest manifest_digest(const uint8_t* p, size_t n) { return DomainHash(kManifestDomain).u64(n).bytes(p, n).finish(); }

// `sealed`: the digests a publish computes are present and consistent.
bool check(const Manifest& m, bool sealed, std::string& error) {
    if (m.format_version != kManifestFormat) return fail(error, "format version " + std::to_string(m.format_version));
    if (!valid_context_id(m.context_id)) return fail(error, "invalid context id");
    if (is_zero(m.model_fingerprint)) return fail(error, "no model fingerprint");
    if (is_zero(m.prefix_hash)) return fail(error, "no prefix hash");
    if (m.prefix_tokens == 0 || m.prefix_tokens > kMaxPrefixTokens) return fail(error, "prefix length out of range");
    if (m.generation == 0) return fail(error, "no generation");
    if (m.parent && (is_zero(m.parent->hash) || m.parent->tokens == 0 || m.parent->tokens > m.prefix_tokens))
        return fail(error, "the parent prefix is not a prefix of this one");
    const auto& inv = m.inventory;
    if (inv.empty() || inv.size() > kMaxInventoryEntries) return fail(error, "inventory size out of range");
    for (size_t i = 0; i < inv.size(); ++i) {
        const InventoryEntry& e = inv[i];
        if (!known_kind(e.kind)) return fail(error, "unknown state kind " + std::to_string((int) e.kind));
        if (!known_mode(e.mode)) return fail(error, entry_name(e) + ": unknown entry mode");
        if (i > 0 && inv[i - 1].kind > e.kind) return fail(error, "inventory not sorted by kind");
        const bool first = i == 0 || inv[i - 1].kind != e.kind;
        const bool last = i + 1 == inv.size() || inv[i + 1].kind != e.kind;
        if (e.part != (first ? 0 : inv[i - 1].part + 1))
            return fail(error, entry_name(e) + ": parts not numbered from 0 without gaps");
        if (e.mode == EntryMode::recompute) {
            if (e.bytes != 0 || !is_zero(e.digest)) return fail(error, entry_name(e) + ": recompute with bytes or a digest");
            if (!first || !last) return fail(error, entry_name(e) + ": recompute beside other parts of its kind");
        } else {
            if (e.bytes > kMaxEntryBytes) return fail(error, entry_name(e) + ": too large");
            if (sealed && is_zero(e.digest)) return fail(error, entry_name(e) + ": no digest");
        }
    }
    if (sealed && m.payload_digest != payload_digest(inv)) return fail(error, "payload digest differs from the inventory's");
    return true;
}

// The canonical bytes of a manifest `check` has accepted.
void encode(const Manifest& m, std::vector<uint8_t>& out) {
    const size_t n = m.inventory.size();
    out.assign((size_t) manifest_bytes(n), 0);
    uint8_t* p = out.data();
    std::memcpy(p, kMagic, sizeof kMagic);
    put32(p + kAtVersion, m.format_version);
    put32(p + kAtBytes, (uint32_t) out.size());
    put32(p + kAtFlags, m.parent ? kFlagParent : 0);
    put32(p + kAtCount, (uint32_t) n);
    put64(p + kAtTokens, m.prefix_tokens);
    put64(p + kAtGeneration, m.generation);
    put64(p + kAtParentTokens, m.parent ? m.parent->tokens : 0);
    std::memcpy(p + kAtFingerprint, m.model_fingerprint.data(), kDigestBytes);
    std::memcpy(p + kAtPrefix, m.prefix_hash.data(), kDigestBytes);
    if (m.parent) std::memcpy(p + kAtParent, m.parent->hash.data(), kDigestBytes);
    std::memcpy(p + kAtPayload, m.payload_digest.data(), kDigestBytes);
    p[kAtIdLength] = (uint8_t) m.context_id.size();
    std::memcpy(p + kAtId, m.context_id.data(), m.context_id.size());
    uint8_t* e = p + kManifestHeaderBytes;
    for (const auto& x : m.inventory) {
        e[0] = (uint8_t) x.kind;
        e[1] = (uint8_t) x.mode;
        put32(e + 4, x.part);
        put64(e + 8, x.bytes);
        std::memcpy(e + 16, x.digest.data(), kDigestBytes);
        e += kManifestEntryBytes;
    }
    const Digest d = manifest_digest(p, out.size() - kDigestBytes);
    std::memcpy(e, d.data(), kDigestBytes);
}

}  // namespace

const char* state_kind_name(StateKind k) {
    switch (k) {
        case StateKind::kv: return "kv";
        case StateKind::gdn: return "gdn";
        case StateKind::ple: return "ple";
        case StateKind::mtp_draft: return "mtp_draft";
        case StateKind::tail: return "tail";
    }
    return "unknown";
}

StateDigest::StateDigest(StateKind kind, uint32_t part, uint64_t bytes) : hash_(kEntryDomain), want_(bytes) {
    hash_.u8((uint8_t) kind).u32(part).u64(bytes);
}

void StateDigest::update(const void* data, size_t n) {
    hash_.bytes(data, n);
    fed_ += n;
}

Digest StateDigest::finish() {
    const Digest d = hash_.finish();
    return fed_ == want_ ? d : Digest{};
}

Digest payload_digest(const std::vector<InventoryEntry>& inventory) {
    DomainHash h(kPayloadDomain);
    h.u64(inventory.size());
    for (const auto& e : inventory) h.u8((uint8_t) e.kind).u8((uint8_t) e.mode).u32(e.part).u64(e.bytes).digest(e.digest);
    return h.finish();
}

bool validate_manifest(const Manifest& m, std::string& error) { return check(m, true, error); }

bool validate_manifest_draft(const Manifest& m, std::string& error) { return check(m, false, error); }

bool valid_requirements(const StateRequirements& req) {
    uint32_t every = 0;
    for (StateKind k : kStateKinds) every |= state_bit(k);
    const uint32_t kv = state_bit(StateKind::kv);
    return !(req.needed & ~every) && !(req.recomputable & ~req.needed) && (req.needed & kv) && !(req.recomputable & kv);
}

bool manifest_complete(const Manifest& m, const StateRequirements& req, std::string& error) {
    if (!valid_requirements(req)) return fail(error, "invalid state requirements");
    uint32_t present = 0, recompute = 0;
    for (const auto& e : m.inventory) {
        if (!known_kind(e.kind)) return fail(error, "unknown state kind " + std::to_string((int) e.kind));
        present |= state_bit(e.kind);
        if (e.mode == EntryMode::recompute) recompute |= state_bit(e.kind);
    }
    for (StateKind k : kStateKinds) {
        const uint32_t b = state_bit(k);
        const std::string name = state_kind_name(k);
        if ((req.needed & b) && !(present & b)) return fail(error, "no " + name + " state, which the model needs to resume");
        if (!(req.needed & b) && (present & b)) return fail(error, name + " state, which the model does not hold");
        if ((recompute & b) && !(req.recomputable & b))
            return fail(error, name + " state marked recompute, which the model cannot rebuild");
    }
    return true;
}

bool serialize_manifest(const Manifest& m, std::vector<uint8_t>& out, std::string& error) {
    out.clear();
    if (!validate_manifest(m, error)) return false;
    encode(m, out);
    return true;
}

ManifestParse manifest_extent(const uint8_t* head, size_t n, uint64_t& bytes, std::string& error) {
    bytes = 0;
    if (n < kManifestHeaderBytes) return refuse(error, ManifestParse::truncated, "truncated header");
    if (std::memcmp(head, kMagic, sizeof kMagic) != 0)
        return refuse(error, ManifestParse::not_manifest, "not a context manifest");
    const uint32_t version = get32(head + kAtVersion);
    if (version != kManifestFormat)
        return refuse(error, ManifestParse::unsupported_version,
                      "format version " + std::to_string(version) + ", expected " + std::to_string(kManifestFormat));
    const uint32_t count = get32(head + kAtCount);
    if (count == 0 || count > kMaxInventoryEntries || get32(head + kAtBytes) != manifest_bytes(count))
        return refuse(error, ManifestParse::invalid, "inconsistent manifest size");
    bytes = manifest_bytes(count);
    return ManifestParse::ok;
}

ManifestParse parse_manifest(const uint8_t* data, size_t n, Manifest& out, std::string& error) {
    out = {};
    uint64_t bytes = 0;
    const ManifestParse extent = manifest_extent(data, n, bytes, error);
    if (extent != ManifestParse::ok) return extent;
    if (n < bytes) return refuse(error, ManifestParse::truncated, "truncated manifest");
    if (get_digest(data + bytes - kDigestBytes) != manifest_digest(data, (size_t) bytes - kDigestBytes))
        return refuse(error, ManifestParse::digest_mismatch, "manifest digest mismatch");
    Manifest m;
    m.format_version = get32(data + kAtVersion);
    m.prefix_tokens = get64(data + kAtTokens);
    m.generation = get64(data + kAtGeneration);
    m.model_fingerprint = get_digest(data + kAtFingerprint);
    m.prefix_hash = get_digest(data + kAtPrefix);
    m.payload_digest = get_digest(data + kAtPayload);
    if (get32(data + kAtFlags) & kFlagParent)
        m.parent = ParentPrefix{get_digest(data + kAtParent), get64(data + kAtParentTokens)};
    const size_t id_bytes = data[kAtIdLength];
    if (id_bytes > kIdBytes) return refuse(error, ManifestParse::invalid, "context id length out of range");
    m.context_id.assign(reinterpret_cast<const char*>(data + kAtId), id_bytes);
    m.inventory.resize(get32(data + kAtCount));
    for (size_t i = 0; i < m.inventory.size(); ++i) {
        const uint8_t* e = data + kManifestHeaderBytes + i * kManifestEntryBytes;
        m.inventory[i] = InventoryEntry{(StateKind) e[0], (EntryMode) e[1], get32(e + 4), get64(e + 8), get_digest(e + 16)};
    }
    if (!validate_manifest(m, error)) return ManifestParse::invalid;
    // the fields re-encoded must give these bytes: reserved bytes, padding and flag bits outside the format are zero
    std::vector<uint8_t> again;
    encode(m, again);
    if (again.size() != bytes || std::memcmp(again.data(), data, (size_t) bytes) != 0)
        return refuse(error, ManifestParse::invalid, "non-canonical encoding");
    out = std::move(m);
    return ManifestParse::ok;
}

}  // namespace basalt::context
