// include/basalt/core/conversation_store.hpp - parked conversations on disk, the tier under the RAM cache.
//
// A conversation leaves the session (a switch to another one, an eviction from the RAM cache, the end of the process)
// as a SavedConversation; the store writes it to its directory and keeps an index of every conversation it may
// restore: the live tokens, the pictures in them and the checkpoint lengths.  A later prompt that starts with one of
// those prefixes reads it back cut at that prefix (the K/V up to it, the running state there, the root checkpoint)
// instead of reading the tokens again - after a restart too, since the index is rebuilt from the files.
//
// A conversation is one file for its tokens, live state and the K/V past its last whole block, plus shared files: one
// per block of K/V cells and one per checkpoint state, each named by its token prefix and its bytes.  The same
// conversation written again a few turns on writes only its new blocks and states; files no conversation refers to
// are removed.  Every name carries the identity it was written under (the model file and every setting
// that changes the state's bytes).  Reads are fail-closed: header, section table and every 1 MiB block of payload carry
// hashes, a shared file must match its name, and any mismatch drops the conversation.  The directory keeps to a byte
// budget, least recently used conversations first.  A read can leave the K/V in its files until it is copied to where
// it is used (load() with a ConversationKvRead); such a read checks each block as it copies it, so a bad block is
// found after part of the K/V was delivered.  A file such a read holds open is removed once the read has closed it.
// A conversation can also be written where it is, a write buffer at a time, with no image of it in RAM (write()): the
// same files, a new block's bytes copied out twice (once to name its file by their hashes, once to write it).
#pragma once

#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/conversation_stream.hpp"
#include "basalt/platform/direct_file.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace basalt::core {

/// The stored-conversation format, written in every file header; a file of another format is dropped.  Change it
/// whenever the stored bytes of a conversation change: the file layouts, the snapshot layout, the K/V or GDN state
/// layout, a checkpoint state.
inline constexpr uint32_t kConversationFormat = 2;

/// One aligned buffer of a read or a write, freed with it; `p` is null when it could not be allocated.
struct ConversationStoreBuffer {
    void* p = nullptr;
    explicit ConversationStoreBuffer(size_t n) : p(basalt::platform::DirectFile::alloc_aligned(n)) {}
    ~ConversationStoreBuffer() { basalt::platform::DirectFile::free_aligned(p); }
    ConversationStoreBuffer(const ConversationStoreBuffer&) = delete;
    ConversationStoreBuffer& operator=(const ConversationStoreBuffer&) = delete;
    uint8_t* data() const { return static_cast<uint8_t*>(p); }
};

class ConversationStore {
public:
    /// Cells behind one pooled row of a K/V layer's index (before open).
    void set_row_cells(int64_t cells) { row_cells_ = cells; }
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        uint64_t entry_id = 0;
    };
    struct Stats {
        size_t files = 0, written = 0, restored = 0, evicted = 0, dropped = 0;
        uint64_t bytes = 0, written_bytes = 0, read_bytes = 0;
        uint64_t whole_bytes = 0;   ///< what the written conversations hold, their shared files counted in full
    };

    ConversationStore();
    ~ConversationStore();
    ConversationStore(const ConversationStore&) = delete;
    ConversationStore& operator=(const ConversationStore&) = delete;

    /// Creates `dir` if needed and indexes the files written under `identity`.  `budget` bytes of files at most (every
    /// identity's files count).  False with `error` when the directory cannot be used.
    bool open(const std::string& dir, uint64_t identity, uint64_t budget, std::string& error);
    bool enabled() const { return open_; }
    uint64_t budget() const { return budget_; }

    /// The longest stored prefix of `prompt` (live tokens or a checkpoint, with the same pictures and control-vector
    /// state), at most prompt.size() - 1 tokens; tokens == 0 when there is none.
    Match best(const int32_t* prompt, size_t n, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// Bytes `load` allocates for `m` (the RAM admission's figure); with `kv` false, those of a load that leaves the
    /// K/V in its files.
    uint64_t load_bytes(const Match& m, bool kv = true) const;
    /// Makes the entry of `m` the most recently used, so the budget evicts it last: what a caller does before the
    /// writes that make room to load it.
    void touch(const Match& m);

    /// The image of `m` cut at m.tokens: `out.live` is the state there, `out.kv` the K/V up to it, `out.checkpoints`
    /// the chain's root (its shortest checkpoint) and that state once more when it is a checkpoint.  A file that fails
    /// any check is deleted and false returned with `error`; the file stays otherwise (a later prompt may match it),
    /// and so does one whose read stopped for want of a read buffer.  False when the entry matched has left the index
    /// since (an eviction or a later turn of it); one that only moved in it is read where it is.
    /// With `kv`, the K/V is not copied: `out.kv` holds the layers' shapes with empty buffers and `*kv` reads their
    /// bytes later (the store outlives it; one load or K/V read runs at a time).
    bool load(const Match& m, SavedConversation& out, std::string& error,
              std::unique_ptr<ConversationKvRead>* kv = nullptr);

    /// Queues `image` for writing; the writer thread owns it from here.  Files it supersedes (the same conversation
    /// a turn back) are dropped.  False when the queue is over its byte cap (nothing queued).
    bool put(SavedConversation&& image, uint64_t queue_cap);

    /// Writes `src` now, on the caller's thread, a write buffer at a time, after a write in progress on the writer
    /// thread and before the queued ones: the files put() writes of its image.  Until it has succeeded the index and
    /// the files it names stay as they were; an entry indexed or dropped during the write refuses it too.  False with
    /// `error` when the conversation is not stored once this returns: not written (one larger than the budget is not
    /// begun, and one whose files and the queued images do not fit it together is not put in place, so it evicts
    /// nothing), or evicted by the budget at once.  `progress` is called while it waits for the writer thread, about
    /// once a second, and for every piece it copies from `src`.
    bool write(const ConversationSource& src, std::string& error, const std::function<void()>& progress = {});
    /// The RAM a write() of `tokens` tokens takes beside its source.
    static uint64_t write_bytes(size_t tokens);
    /// Whether a conversation of `bytes` could be written now: what write() and the writer refuse before they begin
    /// (one larger than the budget, or one fits() finds no room for on the volume), so a caller can tell before it
    /// makes room for the write.
    bool may_take(uint64_t bytes) const;

    /// Whether an image with these live tokens is stored or queued.
    bool holds(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// Waits until every queued image is written and its RAM given back; `waiting` is called about once a second
    /// meanwhile.
    void flush(const std::function<void()>& waiting = {});

    Stats stats() const;
    uint64_t queued_bytes() const;
    /// Whether an image of `bytes` would be written: the volume keeps its floor free after the queue and the image. A
    /// caller asks before it copies a conversation off the card, so a full disk costs no copy the writer then drops.
    bool fits(uint64_t bytes) const;

private:
    friend struct ConversationStoreFixture;   // loses the read buffers for the test of a read that has none
    struct Entry;
    struct Job;
    struct PartFile {                        ///< a block or state file: its size and the conversations that use it
        uint64_t bytes = 0;
        int refs = 0;
        int64_t mtime = 0;
    };
    class KvFileRead;
    class KvQueuedRead;
    void writer();
    std::unique_ptr<Entry> entry_of(const ConversationSource& c, uint64_t bytes) const;
    std::vector<const Entry*> superseded_locked(const Entry& e) const;
    Entry& place_locked(std::unique_ptr<Entry> e);
    void supersede_locked(std::vector<const Entry*>& superseded, const Entry& e, bool own_file);
    bool write_now(Job& job, const ConversationSource& src, uint64_t& wrote, uint64_t& whole, std::string& why,
                   std::string* staged);
    bool write_job(Job& job, const ConversationSource& src, uint64_t& wrote, uint64_t& whole, std::string& error,
                   std::string* staged);
    Entry* find_entry_locked(const Match& m) const;
    void evict_locked();
    void drop_locked(size_t index, bool keep_file = false);
    void drop_read(const std::string& name, uint64_t entry_id, const std::string& part, std::string& error);
    void unlink_unread_locked();             ///< removes the dropped files no read holds open any more
    void release_locked(const std::vector<std::string>& refs);
    void gc_locked(int64_t min_age);         ///< removes unused block and state files at least `min_age` s old

    std::string dir_;
    mutable std::atomic<bool> full_said_{false};   ///< fits() said the volume is full, once until it has room again
    uint64_t identity_ = 0, budget_ = 0;
    bool open_ = false;
    mutable std::mutex mu_;
    std::shared_mutex publish_mu_;            ///< shared by a load opening its files, held by a write() to publish
    std::timed_mutex write_mu_;               ///< held while a conversation's files are written
    int direct_waiting_ = 0;                  ///< write() calls waiting for write_mu_
    std::condition_variable cv_, idle_;
    std::map<std::string, int> reading_;      ///< files a ConversationKvRead holds open: removed once it closed them
    std::vector<std::string> unlinked_;       ///< conversation files dropped while open for such a read
    std::vector<std::unique_ptr<Entry>> entries_;
    std::map<std::string, PartFile> parts_;   ///< block and state files of this identity, by name
    std::deque<std::shared_ptr<Job>> queue_;
    uint64_t queue_bytes_ = 0, clock_ = 0, next_id_ = 0;
    uint64_t revision_ = 0;                   ///< counts the entries indexed and dropped
    bool stop_ = false, busy_ = false;
    Stats stats_;
    std::vector<std::unique_ptr<ConversationStoreBuffer>> read_bufs_;   ///< the loads' aligned read buffers, kept
    std::thread thread_;
    int64_t row_cells_ = 1;
};

/// What conversation_cache_to_store moved: the conversations that left the cache and their bytes, those of them it
/// wrote, and whether the caller's condition held at the end.
struct ConversationsToStore {
    size_t left = 0, bytes = 0, written = 0;
    bool admitted = false;
};

/// Conversations parked in `cache` leave for `store`, the longest parked first, until `admitted()` holds or none is
/// left to go.  One that `stored` says the store holds already only leaves the cache; the others are written first
/// (write(), with `waiting` as its progress), one of them at most per call: an image's RAM is free once it is written.  One shorter than `shortest`
/// tokens, larger than the store's budget or with a layer split's stage images (the store takes none) stays parked;
/// one the store does not hold after its write goes back to the cache as its most recent entry, and no more are given
/// up after it.
ConversationsToStore conversation_cache_to_store(ConversationCache& cache, ConversationStore& store, int64_t shortest,
                                                 const std::function<bool(const SavedConversation&)>& stored,
                                                 const std::function<bool()>& admitted,
                                                 const std::function<void()>& waiting = {});

/// The 64-bit hash the store's identity and file checks use (XXH64's rounds).
uint64_t conversation_store_hash(const void* data, size_t n, uint64_t seed);
/// `seed` hashed on with a file's size and samples of its bytes: what names the model file in a store's identity.
uint64_t conversation_store_hash_file(const std::string& path, uint64_t seed);

}  // namespace basalt::core
