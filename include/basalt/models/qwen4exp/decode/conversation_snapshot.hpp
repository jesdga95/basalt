#pragma once

#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"

#include <string>
#include <thread>

namespace basalt::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// The K/V up to `upto` of `from` into `to` (the same layout) without a host copy: what a save of `from` and a
// restore of it into `to` leave there.
bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g, int64_t upto,
                          bool include_index, std::string& error);
// The pools a K/V image of cells [0, upto) is copied from, as (pointer, bytes) in image order.  `check` false: the
// sizes only, without the buffer and backing checks a copy needs.
bool conversation_kv_regions(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                             std::vector<std::pair<void*, size_t>>& out, bool check, std::string& error);
// After the pools were written in place: the resident copies refill from them.
bool conversation_kv_settle(const QsaState& state, const ModelGeometry& g, int64_t upto, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve: `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
/// Overwrites the running state of layers [layer_lo, layer_hi) in a whole-model checkpoint with `part`'s, a session
/// that holds those layers (another card's share of a prompt read).  The caller has synchronized `part`'s device.
bool conversation_checkpoint_save_layers(ConversationCheckpoint& checkpoint, const SessionState& part,
                                         const ModelGeometry& g, int64_t layer_lo, int64_t layer_hi, std::string& error);
/// The bytes of one checkpoint of this session's carve: what its host parts add up to, and the size of the device buffer
/// a deferred save snapshots into.
size_t conversation_checkpoint_bytes(const SessionState& session, const ModelGeometry& g);
/// The same save with the host copy off the caller's critical path (basalt/core/checkpoint_transfer.hpp): the checkpoint's
/// bytes are valid only once the deferral has finished.  Without a deferral buffer it is a plain save.
bool conversation_checkpoint_save_deferred(CheckpointDeferral& deferral, ConversationCheckpoint& checkpoint,
                                           const SessionState& session, const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error);

/// A parked conversation kept in VRAM (--kv-pool): its running state and the K/V of its cells, every owned layer's
/// and the draft layer's, copied device to device into one allocation.  Its checkpoints stay in RAM.  Restored only
/// into a session with the layout it was saved from.
struct DeviceConversation {
    ConversationCheckpoint live;   ///< ids and images; the running state is in `mem`
    std::vector<ConversationCheckpoint> checkpoints;
    bool cvec = true;
    bool from_slot = false;   ///< always false: the cache's slot entries are RAM images
    void* mem = nullptr;
    std::vector<size_t> sizes;   ///< the regions in `mem`, in save order
    size_t mem_bytes = 0;
    int64_t layer_lo = 0, layer_hi = 0;
    /// the VRAM it holds (the pool's budget counts VRAM only)
    size_t bytes() const { return mem_bytes; }
    DeviceConversation() = default;
    DeviceConversation(DeviceConversation&& o) noexcept { *this = std::move(o); }
    DeviceConversation& operator=(DeviceConversation&& o) noexcept;
    DeviceConversation(const DeviceConversation&) = delete;
    DeviceConversation& operator=(const DeviceConversation&) = delete;
    ~DeviceConversation();
};
/// The VRAM a DeviceConversation of `tokens` cells takes.  0 on an invalid session.
size_t conversation_device_bytes(const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                 int64_t tokens);
/// Allocates `image` and copies the session's state for the view's tokens into it (the view's checkpoints are
/// checked, not copied: the caller moves them in).  The session is not changed.
bool conversation_device_save(DeviceConversation& image, const ConversationView& view, const SessionState& session,
                              const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_device_restore(const DeviceConversation& image, SessionState& session,
                                                const ModelGeometry& g, const QsaState* draft, std::string& error);

} // namespace basalt::core
