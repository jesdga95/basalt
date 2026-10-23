#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/core/kv_arena.hpp"
#include "conversation_checked.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace basalt::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = basalt::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode != 0 || st.kv_q4 || st.kv_int8)) {
        error = "conversation snapshot: hybrid K8V4 requires an identity layout and distinct format flags";
        return false;
    }
    const auto s = basalt::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        error = "conversation snapshot: invalid K/V geometry";
        return false;
    }
    // The draft layer's image is the cells its key names, whole pages only: the page holding the boundary cell is not
    // published, and a restore invalidates the pages it did not publish.  Everything else is page-rounded up, since
    // its last page is entirely the key's own.
    const int64_t cells = index ? ((upto + s.page_size - 1) / s.page_size) * s.page_size
                                : draft_published_cells(upto, s.page_size);
    const size_t per = st.kv_q4 ? (size_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint
    // restore reconstructs that row when rewinding to an earlier prefix.
    const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V
    // (KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    l = {(st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated, cells, pooled, s.page_size, 0, 0, 0, 0, 0};
    using conversation_detail::product;
    if (!product(l.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(l.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                           int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(l.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        error = "conversation snapshot: K/V byte count overflow";
        return false;
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if (st.kv_hybrid) {
        if (!product(l.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                   (uint64_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            error = "conversation snapshot: hybrid V byte count overflow";
            return false;
        }
        l.value_scales = 0;
    }
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid) return {st.k_q, st.v_q4, st.k_scale, nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

// The pages the token count names.  A state's attention reads every cell it has seen, published or not, so these
// bound what a restore has to leave readable.
int64_t token_pages(const Layout& l, int64_t upto) {
    return upto > 0 ? (upto + l.page_size - 1) / l.page_size : 0;
}

// The cells a restore writes: the image's own extent, plus - for a draft image, whose extent is the whole pages its
// key publishes - the tail pages the restore invalidates below it.  The bounds that keep a write inside the
// allocation are these, not the image's.
int64_t written_cells(const Layout& l, int64_t upto) {
    return std::max(l.cells, token_pages(l, upto) * l.page_size);
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        token_pages(l, upto) > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                    : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    return true;
}

// An elastic state backs only a prefix of its cells; a copy past it would fault inside the driver.  `cells` is the
// extent the caller is about to write, which for a restore is wider than the image.
bool backed(const QsaState& st, int64_t cells, std::string& error) {
    if (!st.kv_arena || st.kv_arena->cells() >= cells) return true;
    error = "conversation snapshot: " + std::to_string((long long) cells) + " cells to copy, " +
            std::to_string((long long) st.kv_arena->cells()) + " backed";
    return false;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDefault);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
} // namespace

// The five buffers a state's K/V payload lives in: the authoritative side when the mode streams, the resident pool
// otherwise.  Exposed so a caller - a test included - reads the same buffers the engine moves data through, rather
// than re-deriving the three-way format branch and risking a different one.
std::array<void*, 5> conversation_kv_pools(const QsaState& st, bool resident) { return pools(st, resident); }

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error) || !backed(st, l.cells, error)) return false;
    // The retained prefix a caller claims is measured in TOKENS; an image's extent is measured in CELLS, and the two
    // are the same number only for the main layers.  The draft's extent is whole pages of the cells its key names, so
    // its claim is capped by the cells it actually holds - the tokens past its last published cell were never in it.
    if (unchanged_tokens < 0 || unchanged_tokens > upto ||
        (index && unchanged_tokens > image.cells)) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t unchanged = index ? unchanged_tokens
                                    : std::min<int64_t>(draft_cells(unchanged_tokens),
                                                        std::min<int64_t>(l.cells, image.cells));
    const int64_t whole_cells = (unchanged / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged / basalt::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        dst[i]->resize(sizes[i]);
        if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error);
            })) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !backed(st, written_cells(l, upto), error)) return false;
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (!src[i]->visit(0, src[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error);
            })) return false;
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.kv_mode == 1) basalt::kernels::kv_stream_reset(st.map, nullptr);
    // The image's own extent, in pages: the draft's is rounded down to the pages it publishes.
    const int64_t end = l.cells / l.page_size;
    const int64_t pages = token_pages(l, upto);
    // A page the image does not publish must not stay readable, in every residency mode: draft attention reads
    // every cell the layer has seen, and the cells below this key are still the outgoing conversation's.  Zero them
    // in the authoritative side - the resident pools where mode 0 keeps everything on the device, the host copy
    // where modes 1 and 2 keep it there - so every later refill, and a later park of a longer prefix, carries zeros
    // instead of those bytes.  A main-layer image's extent rounds up to the token count, so this is a no-op there.
    if (pages > end) {
        // Per-cell bytes from the GEOMETRY, not from the image: a short prefix publishes zero cells, so dividing
        // the image's payload by l.cells would be a division by zero - and it is exactly that case, where the gap
        // is the whole first page, that must be invalidated.  Each buffer gets the width layout() gave it, which for
        // hybrid K8V4 is not the same for K and V.
        const bool int8_keys = st.kv_int8 || st.kv_hybrid;
        const size_t cell = st.kv_q4 ? (size_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                     : (size_t) g.head_dim * (int8_keys ? 1 : 2);
        const size_t v_cell = st.kv_hybrid
                                      ? (size_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim) : cell;
        const size_t scale_cell = (size_t) g.n_head_kv *
                (int8_keys && !st.kv_q4 ? (size_t) (g.head_dim / 64) * 2 : 0);
        const std::array<size_t,5> per = {(size_t) g.n_head_kv * cell, (size_t) g.n_head_kv * v_cell,
                                          scale_cell, st.kv_hybrid ? 0 : scale_cell, 0};
        const size_t gap_pages = (size_t) (pages - end);
        for (size_t i = 0; i < per.size(); ++i) {
            if (per[i] == 0 || dst[i] == nullptr) continue;
            // One page at a time through the same transfer the restore uses: the authoritative side is a device pool
            // in mode 0 and a host array in the streaming modes, and a device memset has nothing to address in the
            // second case.  Pages are grouped so a long gap does not become many small calls or a large allocation.
            const size_t row = (size_t) l.page_size * per[i];
            const size_t chunk = std::max<size_t>(1, std::min(gap_pages, (size_t) 1 << 20) / std::max(row, (size_t) 1));
            const std::vector<uint8_t> zeros(chunk * row, 0);
            uint8_t* at = static_cast<uint8_t*>(dst[i]) + (size_t) end * row;
            for (size_t done = 0; done < gap_pages; done += chunk) {
                const size_t n = std::min(chunk, gap_pages - done);
                if (!transfer(at + done * row, zeros.data(), n * row, error)) return false;
            }
        }
    }
    // The ring is mode 2's own.  It is refilled over the whole page range the token count names, so the gap pages
    // above reach it too; a page merely left out of the refill would keep another conversation's K/V.
    if (st.kv_mode == 2 && pages > 0) {
        auto shapes = basalt::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        basalt::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                         std::max<int64_t>(0, pages - st.n_slots), pages, st.n_slots,
                                         shapes, nullptr);
    }
    const auto status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
    return false;
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto authoritative = pools(st);
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return compare(static_cast<const uint8_t*>(authoritative[i]) + at, p, n, true);
            })) return false;
    if (st.kv_mode == 2 && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        // The ring holds the last n_slots pages the TOKEN COUNT names, and that range includes the unpublished gap
        // page the restore zeroes.  Comparing from `end - n_slots` would read the slot the gap page evicted, so the
        // lower bound follows the refill, not the image.
        const int64_t token_pages = (upto + image.page_size - 1) / image.page_size;
        const int64_t begin = std::max<int64_t>(0, token_pages - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

} // namespace basalt::core
