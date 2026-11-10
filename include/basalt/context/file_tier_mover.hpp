// HOST <-> DISK data path for TierManager: a context's pages move between a PageStore (the pre-reserved host pool) and
// one file per context generation. Host-only, no CUDA. GPU <-> HOST copies need the device and belong to another mover.
//
// File: magic(8) | page_bytes(be64) | pages(be64) | generation(be64) | pages*page_bytes | digest(32).
// Written to a temporary name, fsynced, renamed, then the directory is fsynced: a reader sees a whole file or none.
// The digest covers the page count and every page's content digest, so it equals the PageStore view of the pages.
#pragma once

#include "basalt/context/page_store.hpp"
#include "basalt/context/tier_manager.hpp"

#include <string>

namespace basalt::context {

class FileTierMover : public TierMover {
public:
    /// `store` and `dir` must outlive the mover; `dir` must exist.
    FileTierMover(PageStore& store, std::string dir) : store_(store), dir_(std::move(dir)) {}
    Outcome copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest, std::string& error) override;
    /// Frees the host copy (PageStore::remove) or deletes the context's files.
    void release(const ContextId& context, Tier tier) override;

    /// Removes temporary files an earlier process left behind (call once at startup, before any copy). Returns the count.
    static size_t sweep(const std::string& dir);
    std::string path_of(const ContextId& context, Generation g) const;
    /// Digest of the pages as the file format defines it.
    static Digest combine(const std::vector<Digest>& pages);

private:
    PageStore& store_;
    std::string dir_;
};

}  // namespace basalt::context
