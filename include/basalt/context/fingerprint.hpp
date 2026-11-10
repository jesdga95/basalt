// include/basalt/context/fingerprint.hpp - the identities a context snapshot is keyed by.
//
// A snapshot's running state is reusable only by an engine that computes the same bytes for the same input. The model
// fingerprint names everything on the engine's side that decides those bytes; the prefix hash names the input:
// canonical token IDs and the non-text inputs placed among them. Both are SHA-256 over a canonical encoding behind a
// domain tag, so a digest of one kind never equals one of another, and every variable-length field carries its
// length, so two different inputs never encode to the same bytes.
//
// Encoding: integers are fixed-width little-endian, a string or a byte run is its u64 length then its bytes, a digest
// is its 32 bytes. The domain tag is encoded as a string first.
#pragma once

#include "basalt/context/hash.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace basalt::context {

/// SHA-256 of one domain: the tag is absorbed first, then the fields in the encoding above.
class DomainHash {
public:
    explicit DomainHash(std::string_view domain);
    DomainHash& u8(uint8_t v);
    DomainHash& u32(uint32_t v);
    DomainHash& u64(uint64_t v);
    DomainHash& i64(int64_t v);
    DomainHash& str(std::string_view s);
    DomainHash& digest(const Digest& d);
    /// Bytes as given: the caller has encoded their length.
    DomainHash& bytes(const void* p, size_t n);
    Digest finish();

private:
    Sha256 sha_;
};

/// The all-zero digest stands for "none": no real SHA-256 output is expected to be zero.
bool is_zero(const Digest& d);

/// Canonical text for a fingerprint parameter value. Reals print as the shortest decimal that reads back to the same
/// double, with -0 as 0, so one setting has one spelling however it was written.
std::string canonical_int(int64_t v);
std::string canonical_real(double v);

/// Named parameters of one fingerprint category, each value in canonical text. Keys are sorted by the map, so the
/// fingerprint does not depend on the order they were set in.
using FingerprintParams = std::map<std::string, std::string>;

/// Everything that decides the bytes of a context's running state. Two engines with equal fingerprints compute the
/// same state for the same prefix; any difference makes binary reuse unsafe and the context is rebuilt from its
/// transcript instead. Values are the ones the run resolved (an option left at its default enters with the value it
/// resolved to), so two spellings of one configuration agree.
struct FingerprintInputs {
    std::string weights;              ///< the weight set and its quantization, as the engine names them
    Digest weights_digest{};          ///< content digest of the weights
    Digest tokenizer_digest{};        ///< vocabulary, merges and special tokens
    Digest chat_template_digest{};    ///< the chat template transcripts are rendered with
    uint32_t state_schema = 0;        ///< layout version of the running state the engine saves and restores
    std::vector<int64_t> geometry;    ///< the model geometry the running state is laid out by
    std::string kv_precision;         ///< K/V storage format
    std::string state_precision;      ///< element format of the recurrent, indexer and PLE state
    FingerprintParams positional;     ///< rope and context extension
    FingerprintParams inputs;         ///< non-text model inputs: the image encoder, control vectors and their layers
    FingerprintParams settings;       ///< every other setting that changes the state's bytes (the draft layer, its window)
};

/// SHA-256 of `in` under the model-fingerprint domain. False with `error` when a required field is missing: every
/// field but `inputs` and `settings` is required (a model without an image encoder or control vector has none).
bool model_fingerprint(const FingerprintInputs& in, Digest& out, std::string& error);

enum class InputKind : uint8_t { image = 1 };

/// A non-text input placed in the token stream: its kind, the position of its first token, the digest of its content.
struct PrefixInput {
    InputKind kind = InputKind::image;
    int64_t position = 0;
    Digest digest{};
};

/// SHA-256 of a token prefix: the token IDs, the inputs that start inside it (`inputs` sorted by position, strictly
/// increasing; those at or past the end are not part of this prefix), and whether a control vector steered it.
/// Hashing a sequence's first n tokens with the whole sequence's inputs gives that n-token prefix's hash. False with
/// `error` for an empty prefix, a negative token ID, or inputs out of order or without a digest.
bool prefix_hash(std::span<const int32_t> tokens, std::span<const PrefixInput> inputs, bool steering, Digest& out,
                 std::string& error);

}  // namespace basalt::context
