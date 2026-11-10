// Shared SHA-256 for the agent context library (digests in ids.hpp). Host-only; no CUDA, no OS calls.
// Contract: docs/architecture/agent-context-contract.md. Producers domain-separate by hashing a label first.
#pragma once
#include "basalt/context/ids.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::context {

/// FIPS 180-4 SHA-256, streaming.
class Sha256 {
public:
    Sha256();
    /// Back to the empty state; usable again as if freshly constructed.
    void reset();
    void update(const void* data, size_t bytes);
    /// Fixed-width big-endian: unambiguous over integers.
    void update_be32(uint32_t v);
    void update_be64(uint64_t v);
    /// The digest of everything fed so far; the hasher starts over afterwards, so one object serves many digests.
    Digest finish();

    static Digest hash(const void* data, size_t bytes);

private:
    void compress(const uint8_t* block);
    uint32_t h_[8];
    uint64_t len_bits_ = 0;
    uint8_t buf_[64];
    size_t buf_len_ = 0;
};

inline void update(Sha256& s, const std::string& text) { s.update(text.data(), text.size()); }

/// One-shot SHA-256.
inline Digest sha256(const void* data, size_t bytes) { return Sha256::hash(data, bytes); }

std::string to_hex(const Digest& d);
/// Hex to Digest, either case; false when `hex` is not 64 hex characters, leaving `out` untouched.
bool from_hex(const std::string& hex, Digest& out);
}  // namespace basalt::context
