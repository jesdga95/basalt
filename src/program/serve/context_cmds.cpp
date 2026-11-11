// src/program/serve/context_cmds.cpp - the serve protocol's CONTEXT_* commands: the agent-context snapshot
// lifecycle over the conversation snapshot machinery (docs/architecture/agent-context-contract.md).
//
//   CONTEXT_FINGERPRINT <chat_template_hex|->  -> CONTEXT_OK FINGERPRINT <fingerprint_hex> <epoch_hex>
//   CONTEXT_SNAP <context_id> <generation>     -> CONTEXT_OK SNAP <id> <gen> <bytes> <tokens> <prefix_hash_hex> <meta>
//   CONTEXT_RESTORE <context_id> <gen> <ids_file> <meta>  -> CONTEXT_OK RESTORE <id> <gen> <tokens>
//   CONTEXT_DROP <context_id>                  -> CONTEXT_OK DROP <id> <removed>
//
// A snapshot is the solo session's conversation at its current position: the manifest names it (fingerprint,
// prefix hash and length, generation), the payload holds every state kind's bytes - one inventory part per
// non-empty K/V buffer, plus the running state's gdn/ple/tail bytes and the draft layer's K/V (a `recompute`
// part when the engine has none). The K/V metadata a restore must stamp back onto the image travels as the
// `meta` string: the registry stores it verbatim as the context's locator and hands it back on resume.
//
// The commands run between requests, on the solo session only; every refusal is `ERR CONTEXT: <reason>`, which
// tells the caller to rebuild the context from its transcript instead.
#include "request.hpp"

#include "basalt/context/fingerprint.hpp"
#include "basalt/context/hash.hpp"
#include "basalt/context/manifest.hpp"
#include "basalt/context/snapshot_io.hpp"
#include "basalt/core/conversation_snapshot.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace basalt::program::serve {
namespace {

namespace ctx = basalt::context;
using basalt::core::CheckpointBytes;
using basalt::core::ConversationBuffer;
using basalt::core::SavedConversation;

void reply_err(const std::string& why) {
    std::printf("ERR CONTEXT: %s\n", why.c_str());
    std::fflush(stdout);
}

// The five K/V buffers of one layer's image, in the order their parts take.
constexpr int kKvBuffers = 5;

struct KvGroupMeta {
    int64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    uint64_t size[kKvBuffers] = {0, 0, 0, 0, 0};
    bool operator==(const KvGroupMeta& o) const {
        return format == o.format && cells == o.cells && heads == o.heads && head_dim == o.head_dim &&
               page_size == o.page_size && pooled_rows == o.pooled_rows && idx_dim == o.idx_dim &&
               std::equal(std::begin(size), std::end(size), std::begin(o.size));
    }
};

/// Everything a restore must know that the payload's bytes do not carry: the K/V group metadata, the running-state
/// part sizes and whether a control vector steered the conversation. Text: `<has_draft>,<gdn>,<ple>,<tails>,
/// <dead>,<block_pos>,<cvec>` then one
/// `<format>,<cells>,<heads>,<head_dim>,<page_size>,<pooled_rows>,<idx_dim>,<k>,<v>,<k_scale>,<v_scale>,<pooled>`
/// group per main layer, then the draft's when present; groups separated by `;`. A run of identical groups - the
/// common case, one layer repeated - collapses to `<count>x<group>` so the meta fits the registry's locator.
struct SnapshotMeta {
    bool has_draft = false;
    int cvec = 0;
    uint64_t gdn = 0, ple = 0, tails = 0, dead = 0, block_pos = 0;
    std::vector<KvGroupMeta> groups;   // main layers, then the draft when present

    std::string encode() const {
        std::ostringstream out;
        out << (has_draft ? 1 : 0) << ',' << gdn << ',' << ple << ',' << tails << ',' << dead << ',' << block_pos
            << ',' << cvec;
        for (size_t j = 0; j < groups.size();) {
            size_t run = 1;
            while (j + run < groups.size() && groups[j + run] == groups[j]) ++run;
            out << ';' << (run > 1 ? std::to_string(run) + "x" : "")
                << groups[j].format << ',' << groups[j].cells << ',' << groups[j].heads << ','
                << groups[j].head_dim << ',' << groups[j].page_size << ',' << groups[j].pooled_rows
                << ',' << groups[j].idx_dim;
            for (int b = 0; b < kKvBuffers; ++b) out << ',' << groups[j].size[b];
            j += run;
        }
        return out.str();
    }
    static bool decode(const std::string& text, SnapshotMeta& out, std::string& error) {
        out = {};
        // encode() separates every number with a comma and every group with a semicolon; the numbers land on the
        // same whitespace-token path here, so the meta a SNAP published is exactly the meta a RESTORE decodes.
        std::string flat;
        flat.reserve(text.size());
        for (const char c : text) flat += (c == ',') ? ' ' : c;
        std::istringstream in(flat);
        std::string head;
        if (!std::getline(in, head, ';')) { error = "meta: empty"; return false; }
        std::istringstream h(head);
        int draft = 0;
        long long v[6];
        if (!(h >> draft >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]) || (draft != 0 && draft != 1)) {
            error = "meta: header";
            return false;
        }
        out.has_draft = draft == 1;
        out.cvec = v[5] != 0 ? 1 : 0;
        out.gdn = (uint64_t) v[0]; out.ple = (uint64_t) v[1]; out.tails = (uint64_t) v[2];
        out.dead = (uint64_t) v[3]; out.block_pos = (uint64_t) v[4];
        std::string group;
        while (std::getline(in, group, ';')) {
            if (group.empty()) continue;
            long long repeat = 1;
            const auto x = group.find('x');
            const auto comma = group.find(',');
            if (x != std::string::npos && (comma == std::string::npos || x < comma)) {
                char* end = nullptr;
                const long long n = std::strtoll(group.c_str(), &end, 10);
                if (end != group.c_str() + x || n < 2 || n > (1LL << 20)) { error = "meta: repeat count"; return false; }
                repeat = n;
                group = group.substr(x + 1);
            }
            std::istringstream g(group);
            KvGroupMeta q;
            long long f[7];
            long long s[kKvBuffers];
            if (!(g >> f[0] >> f[1] >> f[2] >> f[3] >> f[4] >> f[5] >> f[6])) { error = "meta: group"; return false; }
            for (int b = 0; b < kKvBuffers; ++b) {
                if (!(g >> s[b])) { error = "meta: buffer sizes"; return false; }
            }
            q.format = f[0]; q.cells = f[1]; q.heads = f[2]; q.head_dim = f[3];
            q.page_size = f[4]; q.pooled_rows = f[5]; q.idx_dim = f[6];
            for (int b = 0; b < kKvBuffers; ++b) q.size[b] = (uint64_t) s[b];
            if (out.groups.size() + (size_t) repeat > (size_t) (1 << 20)) { error = "meta: group count"; return false; }
            for (long long r = 0; r < repeat; ++r) out.groups.push_back(q);
        }
        if (out.groups.size() > (size_t) (1 << 20) || (out.has_draft == (out.groups.empty()))) {
            error = "meta: group count";
            return false;
        }
        return true;
    }
};

/// The inventory both sides enumerate identically: kv/mtp_draft parts walk the meta's groups and their non-empty
/// buffers in order, then gdn, ple and the non-empty running-state parts.
struct InventoryPlan {
    struct Item {
        ctx::StateKind kind;
        uint32_t part;
        uint64_t bytes;
        const ConversationBuffer* buf = nullptr;      // a K/V buffer
        const CheckpointBytes* raw = nullptr;         // running-state bytes
    };
    std::vector<Item> items;
};

void plan_from_image(SavedConversation& image, const SnapshotMeta& meta, InventoryPlan& plan) {
    for (size_t j = 0; j < image.kv.size(); ++j) {
        const bool draft_part = meta.has_draft && j == image.kv.size() - 1;
        const auto kind = draft_part ? ctx::StateKind::mtp_draft : ctx::StateKind::kv;
        const ConversationBuffer* bufs[kKvBuffers] = {&image.kv[j].k, &image.kv[j].v, &image.kv[j].k_scale,
                                                      &image.kv[j].v_scale, &image.kv[j].pooled};
        for (int b = 0; b < kKvBuffers; ++b) {
            if (bufs[b]->empty()) continue;
            plan.items.push_back({kind, 0, (uint64_t) bufs[b]->size(), bufs[b], nullptr});
        }
    }
    const CheckpointBytes* running[5] = {&image.live.gdn, &image.live.ple, &image.live.tails, &image.live.dead,
                                         &image.live.block_pos};
    const ctx::StateKind running_kind[5] = {ctx::StateKind::gdn, ctx::StateKind::ple, ctx::StateKind::tail,
                                            ctx::StateKind::tail, ctx::StateKind::tail};
    for (int i = 0; i < 5; ++i) {
        if (running[i]->empty()) continue;
        plan.items.push_back({running_kind[i], 0, (uint64_t) running[i]->size(), nullptr, running[i]});
    }
    // parts are numbered per kind, without gaps
    uint32_t next[6] = {0, 0, 0, 0, 0, 0};
    for (auto& item : plan.items) item.part = next[(int) item.kind]++;
    // the manifest demands the inventory sorted by kind, then by part
    std::sort(plan.items.begin(), plan.items.end(), [](const InventoryPlan::Item& a, const InventoryPlan::Item& b) {
        return a.kind != b.kind ? a.kind < b.kind : a.part < b.part;
    });
}

class ImageSource final : public ctx::StateSource {
public:
    explicit ImageSource(const InventoryPlan& plan) : plan_(plan) {}
    bool read(const ctx::InventoryEntry& entry, uint64_t offset, void* dst, size_t n) override {
        for (const auto& item : plan_.items) {
            if (item.kind != entry.kind || item.part != entry.part) continue;
            if (offset + n > item.bytes) return false;
            if (item.buf != nullptr) return item.buf->read(dst, offset, n);
            std::memcpy(dst, item.raw->data() + offset, n);
            return true;
        }
        return false;
    }

private:
    const InventoryPlan& plan_;
};

class ImageSink final : public ctx::StateSink {
public:
    ImageSink(SavedConversation& image, const SnapshotMeta& meta) : image_(image), meta_(meta) {}

    bool begin(const ctx::InventoryEntry& entry) override {
        target_buf_ = nullptr;
        target_raw_ = nullptr;
        if (entry.mode != ctx::EntryMode::stored) return false;
        const size_t ordinal = seen_[(int) entry.kind]++;
        // Walk the same enumeration the plan made: the ordinal within a kind picks the buffer or checkpoint field.
        if (entry.kind == ctx::StateKind::kv || entry.kind == ctx::StateKind::mtp_draft) {
            const bool draft_kind = entry.kind == ctx::StateKind::mtp_draft;
            size_t want = 0;
            for (size_t j = 0; j < meta_.groups.size(); ++j) {
                const bool is_draft_group = meta_.has_draft && j + 1 == meta_.groups.size();
                if (is_draft_group != draft_kind) continue;
                const uint64_t* sizes = meta_.groups[j].size;
                for (int b = 0; b < kKvBuffers; ++b) {
                    if (sizes[b] == 0) continue;
                    if (want++ != ordinal) continue;
                    const size_t group = j;   // the draft group is the last one, and `j` already is it
                    if (group >= image_.kv.size()) return false;
                    if (entry.bytes != sizes[b]) return false;
                    ConversationBuffer* buffers[kKvBuffers] = {&image_.kv[group].k, &image_.kv[group].v,
                                                              &image_.kv[group].k_scale, &image_.kv[group].v_scale,
                                                              &image_.kv[group].pooled};
                    target_buf_ = buffers[b];
                    target_buf_->resize_for_overwrite((size_t) sizes[b]);
                    return true;
                }
            }
            return false;
        }
        CheckpointBytes* fields[3] = {nullptr, nullptr, nullptr};
        if (entry.kind == ctx::StateKind::gdn) fields[0] = &image_.live.gdn;
        else if (entry.kind == ctx::StateKind::ple) fields[0] = &image_.live.ple;
        else if (entry.kind == ctx::StateKind::tail) {
            fields[0] = &image_.live.tails; fields[1] = &image_.live.dead; fields[2] = &image_.live.block_pos;
        }
        const uint64_t sizes[3] = {entry.kind == ctx::StateKind::gdn    ? meta_.gdn
                                   : entry.kind == ctx::StateKind::ple ? meta_.ple
                                                                       : meta_.tails,
                                   entry.kind == ctx::StateKind::tail ? meta_.dead : 0,
                                   entry.kind == ctx::StateKind::tail ? meta_.block_pos : 0};
        size_t want = 0;
        for (int i = 0; i < 3; ++i) {
            if (fields[i] == nullptr || sizes[i] == 0) continue;
            if (want++ != ordinal) continue;
            if (entry.bytes != sizes[i]) return false;
            target_raw_ = fields[i];
            target_raw_->resize((size_t) sizes[i]);
            return true;
        }
        return false;
    }

    bool write(const ctx::InventoryEntry&, uint64_t offset, const void* src, size_t n) override {
        if (target_buf_ != nullptr) {
            return target_buf_->visit((size_t) offset, n, [&](uint8_t* p, size_t count, size_t) {
                std::memcpy(p, static_cast<const uint8_t*>(src), count);
                return true;
            });
        }
        if (target_raw_ == nullptr || offset + n > target_raw_->size()) return false;
        std::memcpy(target_raw_->data() + offset, src, n);
        return true;
    }

private:
    SavedConversation& image_;
    const SnapshotMeta& meta_;
    size_t seen_[6] = {0, 0, 0, 0, 0, 0};
    ConversationBuffer* target_buf_ = nullptr;
    CheckpointBytes* target_raw_ = nullptr;
};

/// The engine's own identity contributions; the chat-template digest comes from the caller, which owns the template.
void fingerprint_inputs(const Options& o, const basalt::core::ModelGeometry& g, const std::string& template_hex,
                        ctx::FingerprintInputs& in) {
    const std::string base = [&] {
        const auto slash = o.model.find_last_of("/\\");
        return slash == std::string::npos ? o.model : o.model.substr(slash + 1);
    }();
    in.weights = base + " " + o.kv;
    // The content digest samples the model file the way the conversation cache keys do: size, then sixteen
    // 64 KiB blocks across the file. The tokenizer and chat template live in that file, so their digests derive
    // from the same content under their own domains.
    uint64_t size = 0;
    basalt::platform::file_size(o.model, size);
    ctx::Sha256 sha;
    const uint64_t size_field = size;
    sha.update(&size_field, sizeof size_field);
    const int fd = basalt::platform::open_fd_utf8(o.model);
    std::vector<uint8_t> buf((size_t) 64 << 10);
    const uint64_t n = std::min<uint64_t>(buf.size(), size);
    for (int i = 0; i < 16 && fd >= 0 && n > 0; ++i) {
        const uint64_t off = (size - n) / 15 * (uint64_t) i;
        if (!basalt::platform::pread_all(fd, buf.data(), n, off)) break;
        sha.update(buf.data(), (size_t) n);
    }
    if (fd >= 0) basalt::platform::close_fd(fd);
    in.weights_digest = sha.finish();
    in.tokenizer_digest = ctx::DomainHash("basalt/context/tokenizer/v1").digest(in.weights_digest).finish();
    ctx::Digest tmpl{};
    if (template_hex.empty() || template_hex == "-" || !ctx::from_hex(template_hex, tmpl))
        tmpl = ctx::DomainHash("basalt/context/chat-template/v1").str("none").finish();
    in.chat_template_digest = tmpl;
    in.state_schema = 1;
    for (int64_t v : basalt::core::conversation_geometry_key(g)) in.geometry.push_back(v);
    in.kv_precision = o.kv;
    in.state_precision = "f32";
    auto num = [](double v) { return ctx::canonical_real(v); };
    in.positional["rope_scaling"] = o.rope_scaling;
    in.positional["rope_scale"] = num(o.rope_scale);
    in.positional["rope_freq_base"] = num(o.rope_freq_base);
    in.positional["rope_freq_scale"] = num(o.rope_freq_scale);
    in.positional["yarn_orig_ctx"] = num(o.yarn_orig_ctx);
    in.positional["yarn_ext_factor"] = num(o.yarn_ext_factor);
    in.positional["yarn_attn_factor"] = num(o.yarn_attn_factor);
    in.positional["yarn_beta_fast"] = num(o.yarn_beta_fast);
    in.positional["yarn_beta_slow"] = num(o.yarn_beta_slow);
    if (!o.vision_mmproj.empty()) in.inputs["mmproj"] = o.vision_mmproj;
    for (const auto& [path, scale] : o.cvec_files)
        in.inputs["cvec " + path] = std::to_string(scale);
    in.settings["mtp"] = o.mtp;
}

bool read_ids_file(const std::string& path, std::vector<int32_t>& ids,
                   std::vector<basalt::core::ConversationImageKey>& imgs, std::string& error) {
    std::ifstream in(path);
    if (!in) { error = "ids file unreadable"; return false; }
    std::string line;
    if (!std::getline(in, line)) { error = "ids file empty"; return false; }
    std::istringstream tokens(line);
    std::string piece;
    while (std::getline(tokens, piece, ',')) {
        if (piece.empty()) continue;
        char* end = nullptr;
        const long long v = std::strtoll(piece.c_str(), &end, 10);
        if (end == piece.c_str() || *end != '\0' || v < 0 || v > INT32_MAX) { error = "ids file: bad token"; return false; }
        ids.push_back((int32_t) v);
    }
    if (ids.empty()) { error = "ids file: no tokens"; return false; }
    if (std::getline(in, line)) {
        std::istringstream images(line);
        while (std::getline(images, piece, ',')) {
            const auto colon = piece.find(':');
            if (colon == std::string::npos) continue;
            basalt::core::ConversationImageKey key;
            key.start = std::strtoll(piece.c_str(), nullptr, 10);
            key.hash = (uint64_t) std::strtoull(piece.c_str() + colon + 1, nullptr, 10);
            imgs.push_back(key);
        }
    }
    return true;
}

ctx::Digest image_input_digest(uint64_t hash) {
    return ctx::sha256(&hash, sizeof hash);
}

}  // namespace

Flow Runtime::context_dispatch() {
    std::istringstream in(line);
    std::string command;
    in >> command;
    if (command == "CONTEXT_FINGERPRINT") {
        std::string tmpl;
        in >> tmpl;
        if (!ctx_fingerprint) {
            ctx::FingerprintInputs inputs;
            fingerprint_inputs(o, g, tmpl, inputs);
            ctx::Digest fp{};
            std::string error;
            if (!ctx::model_fingerprint(inputs, fp, error)) { reply_err(error); return Flow::next; }
            ctx_fingerprint = fp;
            std::random_device rd;
            std::ostringstream epoch;
            // 32 hex characters: the registry stores the epoch as a UUID, which parses this simple form.
            for (int i = 0; i < 4; ++i) epoch << std::hex << std::setw(8) << std::setfill('0') << rd();
            ctx_epoch = epoch.str();
        }
        std::printf("CONTEXT_OK FINGERPRINT %s %s\n", ctx::to_hex(*ctx_fingerprint).c_str(), ctx_epoch.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (o.context_store_dir.empty()) { reply_err("no --context-store-dir: the context commands are off"); return Flow::next; }
    if (!ctx_fingerprint) { reply_err("CONTEXT_FINGERPRINT must come first"); return Flow::next; }
    static ctx::PlatformSnapshotFiles files;

    if (command == "CONTEXT_SNAP") {
        std::string id;
        unsigned long long generation = 0;
        if (!(in >> id >> generation) || !ctx::valid_context_id(id) || generation == 0) {
            reply_err("snap: bad context id or generation");
            return Flow::next;
        }
        if (batch_on()) { reply_err("snap: batch slots hold their own conversations"); return Flow::next; }
        if (!live_ok || live.empty()) { reply_err("snap: the session holds no conversation"); return Flow::next; }
        std::string error;
        const basalt::core::ConversationView view{live, live_imgs, {}, cvec_cached};
        size_t estimate = 0;
        if (!basalt::core::conversation_snapshot_bytes(view, ss, g, mtp_kv, estimate, error)) {
            reply_err("snap: " + error);
            return Flow::next;
        }
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
        if (!capture_admit(estimate, floor)) { reply_err("snap: physical RAM admission refused the capture"); return Flow::next; }
        SavedConversation image;
        if (!basalt::core::conversation_snapshot_save(image, view, ss, g, mtp_kv, error)) {
            reply_err("snap: " + error);
            return Flow::next;
        }
        SnapshotMeta meta;
        meta.has_draft = mtp_kv != nullptr && !image.kv.empty();
        meta.cvec = cvec_cached ? 1 : 0;
        meta.gdn = image.live.gdn.size(); meta.ple = image.live.ple.size();
        meta.tails = image.live.tails.size(); meta.dead = image.live.dead.size();
        meta.block_pos = image.live.block_pos.size();
        for (auto& kv : image.kv) {
            KvGroupMeta q;
            q.format = kv.format; q.cells = kv.cells; q.heads = kv.heads; q.head_dim = kv.head_dim;
            q.page_size = kv.page_size; q.pooled_rows = kv.pooled_rows; q.idx_dim = kv.idx_dim;
            const ConversationBuffer* bufs[kKvBuffers] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
            for (int b = 0; b < kKvBuffers; ++b) q.size[b] = (uint64_t) bufs[b]->size();
            meta.groups.push_back(q);
        }
        InventoryPlan plan;
        plan_from_image(image, meta, plan);
        ctx::Manifest draft;
        draft.context_id = id;
        draft.model_fingerprint = *ctx_fingerprint;
        // The prefix identity the registry records: the session's canonical token IDs, the images inside them and
        // whether a control vector steered the conversation.
        std::vector<ctx::PrefixInput> inputs;
        for (const auto& img : live_imgs)
            inputs.push_back({ctx::InputKind::image, img.start, image_input_digest(img.hash)});
        ctx::Digest prefix_hash{};
        if (!ctx::prefix_hash(live, inputs, cvec_cached, prefix_hash, error)) { reply_err("snap: " + error); return Flow::next; }
        draft.prefix_hash = prefix_hash;
        draft.prefix_tokens = live.size();
        draft.generation = generation;
        for (const ctx::StateKind kind : ctx::kStateKinds) {
            if (kind == ctx::StateKind::mtp_draft && !meta.has_draft)
                draft.inventory.push_back({kind, ctx::EntryMode::recompute, 0, 0, {}});
            for (const auto& item : plan.items)
                if (item.kind == kind)
                    draft.inventory.push_back({item.kind, ctx::EntryMode::stored, item.part, item.bytes, {}});
        }
        ctx::SnapshotStore store(o.context_store_dir + "/" + id, files);
        ImageSource source(plan);
        const auto result = store.publish(draft, source);
        if (!result.ok()) {
            reply_err(std::string("snap: ") + ctx::publish_status_name(result.status) +
                      (result.error.empty() ? "" : ": " + result.error));
            return Flow::next;
        }
        uint64_t payload_bytes = 0;
        for (const auto& item : plan.items) payload_bytes += item.bytes;
        std::printf("CONTEXT_OK SNAP %s %llu %llu %zu %s %s\n", id.c_str(), generation,
                    (unsigned long long) payload_bytes, live.size(), ctx::to_hex(prefix_hash).c_str(),
                    meta.encode().c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (command == "CONTEXT_RESTORE") {
        std::string id, ids_path, meta_text;
        unsigned long long generation = 0;
        if (!(in >> id >> generation >> ids_path >> meta_text) || !ctx::valid_context_id(id) || generation == 0) {
            reply_err("restore: bad arguments");
            return Flow::next;
        }
        if (batch_on()) { reply_err("restore: batch slots hold their own conversations"); return Flow::next; }
        SnapshotMeta meta;
        std::string error;
        if (!SnapshotMeta::decode(meta_text, meta, error)) { reply_err(error); return Flow::next; }
        std::vector<int32_t> ids;
        std::vector<basalt::core::ConversationImageKey> imgs;
        if (!read_ids_file(ids_path, ids, imgs, error)) { reply_err(error); return Flow::next; }
        std::vector<ctx::PrefixInput> inputs;
        for (const auto& img : imgs) inputs.push_back({ctx::InputKind::image, img.start, image_input_digest(img.hash)});
        ctx::Digest prefix_hash{};
        if (!ctx::prefix_hash(ids, inputs, meta.cvec == 1, prefix_hash, error)) { reply_err(error); return Flow::next; }
        // The session may still hold a live conversation, and this restore overwrites its cells: park it first
        // (unparked it would be lost) and with it the K/V retained for it, which the restored conversation
        // replaces below. A park that cannot run leaves the session untouched and the caller replays.
        if (!park_current(0, 0)) {
            reply_err("restore: parking the outgoing conversation failed: " + err);
            err.clear();
            return Flow::next;
        }
        ctx::SnapshotStore store(o.context_store_dir + "/" + id, files);
        SavedConversation image;
        image.kv.resize(meta.groups.size());
        ImageSink sink(image, meta);
        const ctx::RestoreRequest want{id, *ctx_fingerprint, prefix_hash, ids.size(), generation, ctx::kHybridState};
        const auto result = store.restore(want, sink);
        if (result.status != ctx::RestoreStatus::restored) {
            reply_err(std::string("restore: ") + ctx::restore_status_name(result.status) +
                      (result.error.empty() ? "" : ": " + result.error));
            return Flow::next;
        }
        for (size_t j = 0; j < meta.groups.size(); ++j) {
            auto& kv = image.kv[j];
            const auto& q = meta.groups[j];
            kv.format = (int) q.format; kv.cells = q.cells; kv.heads = q.heads; kv.head_dim = q.head_dim;
            kv.page_size = q.page_size; kv.pooled_rows = q.pooled_rows; kv.idx_dim = q.idx_dim;
        }
        image.geometry = basalt::core::conversation_geometry_key(g);
        image.layer_lo = ss.layer_lo; image.layer_hi = ss.layer_hi;
        image.live.ids = ids;
        image.live.imgs = imgs;
        image.cvec = meta.cvec == 1;
        auto restored = basalt::core::conversation_snapshot_restore(image, ss, g, mtp_kv, error);
        if (restored != basalt::core::ConversationRestore::restored) {
            reply_err("restore: " + error);
            return Flow::next;
        }
        live = std::move(ids);
        live_imgs = std::move(imgs);
        checks.clear();
        cvec_cached = meta.cvec == 1;
        live_ok = true;
        // The image's buffers are this conversation's authoritative K/V: retain them for its next park, exactly
        // as the conversation cache's restore does. retain() replaces whatever was held for the conversation
        // that gave way, so no stale prefix can ride into this conversation's image.
        conversations->retain(std::move(image.kv), int64_t(live.size()), {}, rsp);
        std::printf("CONTEXT_OK RESTORE %s %llu %zu\n", id.c_str(), generation, live.size());
        std::fflush(stdout);
        return Flow::next;
    }
    if (command == "CONTEXT_DROP") {
        std::string id;
        if (!(in >> id) || !ctx::valid_context_id(id)) { reply_err("drop: bad context id"); return Flow::next; }
        ctx::SnapshotStore store(o.context_store_dir + "/" + id, files);
        size_t removed = 0;
        for (const auto generation : store.generations())
            if (store.discard(generation)) ++removed;
        removed += store.remove_stale_temporaries(0);
        std::printf("CONTEXT_OK DROP %s %zu\n", id.c_str(), removed);
        std::fflush(stdout);
        return Flow::next;
    }
    reply_err("unknown or unsupported context command");
    return Flow::next;
}

}  // namespace basalt::program::serve
