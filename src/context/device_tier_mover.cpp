#include "basalt/context/device_tier_mover.hpp"

#include <vector>

namespace basalt::context {

TierMover::Outcome DeviceTierMover::copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest,
                                         std::string& error) {
    const size_t pb = store_.page_bytes();
    if (pb == 0) { error = "page store is not open"; return Outcome::failed; }
    if (move.bytes == 0 || move.bytes > (uint64_t) 1 << 48) { error = "bytes out of range"; return Outcome::failed; }
        if (move.bytes == 0 || move.bytes > ((uint64_t) 1 << 48)) { error = "bytes out of range"; return Outcome::failed; }
    const size_t pages = (size_t) ((move.bytes + pb - 1) / pb);
    std::vector<uint8_t> buf(pb);
    std::vector<Digest> digests;
    digests.reserve(pages);
    if (move.from == Tier::gpu && move.to == Tier::host) {
        PageResult c = store_.create(move.context, pages);
        if (!c) { error = "host create: " + c.reason; return Outcome::failed; }
        auto drop = [&](Outcome o, std::string m) { store_.remove(move.context); error = std::move(m); return o; };
        for (size_t i = 0; i < pages; ++i) {
            if (cancel.load()) return drop(Outcome::canceled, "canceled");
            const uint64_t off = (uint64_t) i * pb;
            const size_t n = (size_t) std::min<uint64_t>(pb, move.bytes - off);
            std::fill(buf.begin(), buf.end(), 0);   // the tail of the last page is defined, so the digest is stable
            std::string e;
            if (!device_.download(move.context, off, buf.data(), n, e)) return drop(Outcome::failed, "device read: " + e);
            digests.push_back(page_digest(buf.data(), pb));
            PageResult w = store_.write(move.context, i, 0, buf.data(), pb);
            if (!w) return drop(Outcome::failed, "host write: " + w.reason);
        }
        digest = FileTierMover::combine(digests);
        return Outcome::done;
    }
    if (move.from == Tier::host && move.to == Tier::gpu) {
        // verify the host copy first: nothing reaches the device from bytes that fail their digest
        for (size_t i = 0; i < pages; ++i) {
            PageResult r = store_.read(move.context, i, 0, buf.data(), pb);
            if (!r) { error = "host read: " + r.reason; return r.status == PageStatus::digest_mismatch ? Outcome::corrupt : Outcome::failed; }
            digests.push_back(page_digest(buf.data(), pb));
        }
        const Digest total = FileTierMover::combine(digests);
        if (move.verify && total != move.digest) { error = "host copy fails its digest"; return Outcome::corrupt; }
        std::string e;
        if (!device_.allocate(move.context, move.bytes, e)) { error = "device allocate: " + e; return Outcome::failed; }
        for (size_t i = 0; i < pages; ++i) {
            if (cancel.load()) { device_.free(move.context); error = "canceled"; return Outcome::canceled; }
            const uint64_t off = (uint64_t) i * pb;
            const size_t n = (size_t) std::min<uint64_t>(pb, move.bytes - off);
            PageResult r = store_.read(move.context, i, 0, buf.data(), pb);
            if (!r || !device_.upload(move.context, off, buf.data(), n, e)) {
                device_.free(move.context);
                error = r ? "device write: " + e : "host read: " + r.reason;
                return Outcome::failed;
            }
        }
        digest = total;
        return Outcome::done;
    }
    error = std::string("unsupported move ") + tier_name(move.from) + "->" + tier_name(move.to);
    return Outcome::failed;
}

void DeviceTierMover::release(const ContextId& context, Tier tier) {
    if (tier == Tier::gpu) device_.free(context);
    else if (tier == Tier::host) store_.remove(context);
}

TierMover::Outcome CompositeTierMover::copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest,
                                            std::string& error) {
    const bool gpu = move.from == Tier::gpu || move.to == Tier::gpu;
    const bool disk = move.from == Tier::disk || move.to == Tier::disk;
    if (gpu && disk) { error = "gpu<->disk moves go through host"; return Outcome::failed; }
    return (gpu ? device_ : file_).copy(move, cancel, digest, error);
}

void CompositeTierMover::release(const ContextId& context, Tier tier) {
    (tier == Tier::disk ? file_ : device_).release(context, tier);
}

}  // namespace basalt::context
