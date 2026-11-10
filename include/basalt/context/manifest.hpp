// include/basalt/context/manifest.hpp - the record that makes a context snapshot restorable.
//
// A manifest names what a snapshot is (its context, the model fingerprint, the prefix hash and length, the fencing
// generation) and every piece of running state in it: one inventory entry per part of each state kind, with its bytes
// and digest. K/V alone does not restore a hybrid model, so completeness is checked per kind against what the
// consumer's model holds; state the consumer can rebuild itself may be marked `recompute` instead of stored.
//
// Binary form (format 1), little-endian, fixed offsets:
//   0    magic "BSLTCTXM"             8    format version          12   manifest bytes (header, entries, digest)
//   16   flags (bit 0: has parent)    20   entry count             24   prefix tokens
//   32   generation                   40   parent prefix tokens    48   model fingerprint (32)
//   80   prefix hash (32)             112  parent prefix hash (32) 144  payload digest (32)
//   176  context id length            177  zero (7)                184  context id, zero padded (128)
//   312  entries, 48 bytes each: kind, mode, zero (2), part (u32), bytes (u64), digest (32)
//   end  SHA-256 under "basalt/context/manifest/v1" of the u64 count of bytes before it, then those bytes
// Every byte is determined by the fields: reserved bytes and padding are zero, so one manifest has one encoding.
//
// Digests: an entry's is SHA-256 under "basalt/context/state-entry/v1" of kind (u8), part (u32), bytes (u64) and the
// entry's bytes; the payload digest is SHA-256 under "basalt/context/payload/v1" of the entry count (u64) and, per
// entry, kind (u8), mode (u8), part (u32), bytes (u64) and digest.
#pragma once

#include "basalt/context/fingerprint.hpp"
#include "basalt/context/ids.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace basalt::context {

inline constexpr uint32_t kManifestFormat = 1;
inline constexpr size_t kManifestHeaderBytes = 312;
inline constexpr size_t kManifestEntryBytes = 48;
inline constexpr size_t kMaxInventoryEntries = 4096;
inline constexpr uint64_t kMaxPrefixTokens = (1ull << 31) - 1;
inline constexpr uint64_t kMaxEntryBytes = 1ull << 40;

/// The running state of a sequence at a token position, by what a parked conversation holds of it:
///   kv         the K/V and indexer pooled keys of the QSA layers
///   gdn        the GDN layers' recurrence and convolution state
///   ple        the PLE history
///   mtp_draft  the MTP draft layer's K/V
///   tail       the QSA indexer's running state: the block being filled, the spare key, the block position
/// A kind's bytes may come in parts (one per layer, say); their layout is the engine's, versioned by the state schema
/// in the model fingerprint.
enum class StateKind : uint8_t { kv = 1, gdn = 2, ple = 3, mtp_draft = 4, tail = 5 };
inline constexpr StateKind kStateKinds[] = {StateKind::kv, StateKind::gdn, StateKind::ple, StateKind::mtp_draft,
                                            StateKind::tail};
const char* state_kind_name(StateKind k);
constexpr uint32_t state_bit(StateKind k) { return 1u << (uint32_t) k; }

/// `stored`: the bytes are in the snapshot. `recompute`: they are not, and the consumer rebuilds that state itself
/// (the entry has no bytes and a zero digest, and is the only entry of its kind).
enum class EntryMode : uint8_t { stored = 1, recompute = 2 };

struct InventoryEntry {
    StateKind kind = StateKind::kv;
    EntryMode mode = EntryMode::stored;
    uint32_t part = 0;
    uint64_t bytes = 0;
    Digest digest{};
    bool operator==(const InventoryEntry&) const = default;
};

/// The prefix this context was forked from: a prefix of its own, at most as long.
struct ParentPrefix {
    Digest hash{};
    uint64_t tokens = 0;
    bool operator==(const ParentPrefix&) const = default;
};

/// Inventory entries are sorted by kind, then part; a kind's parts are numbered from 0 without gaps.
struct Manifest {
    uint32_t format_version = kManifestFormat;
    ContextId context_id;
    Digest model_fingerprint{};
    Digest prefix_hash{};
    uint64_t prefix_tokens = 0;
    std::optional<ParentPrefix> parent;
    Generation generation = 0;
    std::vector<InventoryEntry> inventory;
    Digest payload_digest{};
    bool operator==(const Manifest&) const = default;
};

/// The digest of one stored entry, fed its bytes in order. A count fed other than `bytes` finishes to the zero
/// digest, which no stored entry may carry.
class StateDigest {
public:
    StateDigest(StateKind kind, uint32_t part, uint64_t bytes);
    void update(const void* data, size_t n);
    Digest finish();

private:
    DomainHash hash_;
    uint64_t want_ = 0, fed_ = 0;
};

Digest payload_digest(const std::vector<InventoryEntry>& inventory);

/// Every field in range, the inventory in canonical order, every stored entry with a digest and the payload digest
/// the inventory gives.
bool validate_manifest(const Manifest& m, std::string& error);
/// The same for a manifest about to be published: the digests a publish computes (stored entries', the payload's)
/// may still be zero; a stored entry's nonzero digest is one its bytes must match.
bool validate_manifest_draft(const Manifest& m, std::string& error);

/// The state kinds a consumer's model holds (`needed`), and those of them it can rebuild without the snapshot
/// (`recomputable`). K/V is always needed and never recomputable.
struct StateRequirements {
    uint32_t needed = 0;
    uint32_t recomputable = 0;
};
/// The hybrid model with every module present. The draft layer's K/V only steers drafting, never the output (the
/// verify window decides every emitted token), so it may be rebuilt rather than restored.
inline constexpr StateRequirements kHybridState{
    state_bit(StateKind::kv) | state_bit(StateKind::gdn) | state_bit(StateKind::ple) |
        state_bit(StateKind::mtp_draft) | state_bit(StateKind::tail),
    state_bit(StateKind::mtp_draft)};

/// Known kinds only, K/V needed, and recomputable kinds a subset of the needed ones other than K/V.
bool valid_requirements(const StateRequirements& req);

/// Whether `m` restores a model with `req`: every needed kind has entries, no kind outside it does, and only
/// recomputable kinds are marked `recompute`.
bool manifest_complete(const Manifest& m, const StateRequirements& req, std::string& error);

/// The encoded size of a manifest with `entries` inventory entries.
constexpr uint64_t manifest_bytes(size_t entries) {
    return kManifestHeaderBytes + kManifestEntryBytes * (uint64_t) entries + 32;
}

/// The binary form of a valid manifest; false with `error` (and `out` empty) for an invalid one.
bool serialize_manifest(const Manifest& m, std::vector<uint8_t>& out, std::string& error);

enum class ManifestParse : uint8_t { ok, truncated, not_manifest, unsupported_version, digest_mismatch, invalid };

/// The manifest's encoded size from its first `n` bytes (`n` >= kManifestHeaderBytes): magic, format version and
/// entry count are checked, nothing else.
ManifestParse manifest_extent(const uint8_t* head, size_t n, uint64_t& bytes, std::string& error);

/// Decodes a manifest from the start of `data` (`n` may run past its end). `ok` only for the canonical encoding of a
/// valid manifest with its digest intact.
ManifestParse parse_manifest(const uint8_t* data, size_t n, Manifest& out, std::string& error);

}  // namespace basalt::context
