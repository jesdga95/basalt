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
// budget, least recently used conversations first.  A conversation too large to copy into RAM whole is written from
// the session itself a write buffer at a time (write()), and read back with its K/V left in the files until the restore
// copies it into the session (load() with a ConversationKvRead): the same files either way.  Such a read checks each
// block as it copies it, so a bad block fails the restore itself (the caller reads the prompt again); such a write
// copies a new block file's bytes out twice, once to name the file by its hashes and once to write it.  A file a read
// holds open is removed only once the read has closed it.
#pragma once

#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/conversation_stream.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
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

struct ConversationStoreBuffer;

class ConversationStore {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
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

    /// The longest stored prefix of `prompt` (live tokens or a checkpoint, with the same pictures and control-vector
    /// state), at most prompt.size() - 1 tokens; tokens == 0 when there is none.
    Match best(const int32_t* prompt, size_t n, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// Bytes `load` allocates for `m` (the RAM admission's figure); with `kv` false, those of a load that leaves the
    /// K/V in its files.
    uint64_t load_bytes(const Match& m, bool kv = true) const;

    /// The image of `m` cut at m.tokens: `out.live` is the state there, `out.kv` the K/V up to it, `out.checkpoints`
    /// the chain's root (its shortest checkpoint) and that state once more when it is a checkpoint.  A file that fails
    /// any check is deleted and false returned with `error`; the file stays otherwise (a later prompt may match it).
    /// With `kv`, the K/V stays in its files: `out.kv` holds the layers' shapes with empty buffers and `*kv` reads
    /// their bytes later (the store outlives it).
    bool load(const Match& m, SavedConversation& out, std::string& error,
              std::unique_ptr<ConversationKvRead>* kv = nullptr);

    /// Queues `image` for writing; the writer thread owns it from here.  Files it supersedes (the same conversation
    /// a turn back) are dropped.  False when the queue is over its byte cap (nothing queued).
    bool put(SavedConversation&& image, uint64_t queue_cap);

    /// Writes `src` now, on the caller's thread, a write buffer at a time (no copy of its state in RAM), after a write
    /// in progress on the writer thread. A failed write leaves the existing index and files intact; an index change
    /// during the write also refuses publication. False with `error` when it was not written.
    bool write(const ConversationSource& src, std::string& error);
    /// The RAM a write() of `tokens` tokens takes beside its source.
    static uint64_t write_bytes(size_t tokens);

    /// Whether an image with these live tokens is stored or queued.
    bool holds(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// Waits until every queued image is written.
    void flush();

    Stats stats() const;
    uint64_t queued_bytes() const;

private:
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
    using CheckRefs = std::vector<const ConversationCheckpoint*>;   ///< the checkpoints a write stores
    std::shared_ptr<Job> job_locked(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                                    const CheckRefs& checks, bool cvec, uint64_t bytes, bool keep_file = false);
    bool write_now(Job& job, const ConversationSource& src, const CheckRefs& checks, uint64_t& wrote, uint64_t& whole);
    bool write_job(Job& job, const ConversationSource& src, const CheckRefs& checks, uint64_t& wrote, uint64_t& whole,
                   std::string& error);
    void finish_locked(const std::shared_ptr<Job>& job, bool ok, uint64_t wrote, uint64_t whole);
    void drop_read(const std::string& name, uint64_t generation, const std::string& part, std::string& error);
    void unlink_unread_locked();             ///< removes the dropped files no read holds open any more
    void evict_locked();
    void drop_locked(size_t index, bool keep_file = false);
    void release_locked(const std::vector<std::string>& refs);
    void gc_locked(int64_t min_age);         ///< removes unused block and state files at least `min_age` s old

    std::string dir_;
    uint64_t identity_ = 0, budget_ = 0;
    bool open_ = false;
    mutable std::mutex mu_;
    mutable std::shared_mutex publish_mu_;    ///< readers open a consistent set before publication
    std::mutex write_mu_;                     ///< held while a conversation's files are written
    int direct_waiting_ = 0;                  ///< write() calls waiting for write_mu_: the writer lets them go first
    std::map<std::string, int> reading_;      ///< files a ConversationKvRead holds open: not removed until it closes them
    std::vector<std::string> unlinked_;       ///< conversation files dropped while open for a read
    std::condition_variable cv_, idle_;
    std::vector<std::unique_ptr<Entry>> entries_;
    std::map<std::string, PartFile> parts_;   ///< block and state files of this identity, by name
    std::deque<std::shared_ptr<Job>> queue_;
    uint64_t queue_bytes_ = 0, clock_ = 0, revision_ = 0, generation_ = 0;
    bool stop_ = false, busy_ = false;
    Stats stats_;
    std::vector<std::unique_ptr<ConversationStoreBuffer>> read_bufs_;   ///< the loads' aligned read buffers, kept
    std::thread thread_;
};

/// The 64-bit hash the store's identity and file checks use (XXH64's rounds).
uint64_t conversation_store_hash(const void* data, size_t n, uint64_t seed);

}  // namespace basalt::core
