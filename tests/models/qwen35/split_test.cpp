// tests/models/qwen35/split_test.cpp - the layer split (layers [0, cut) on the main card, [cut, n) on a second card) gives the
// main card's output to the bit: after a prompt (read in chunks pipelined over the cards, once stopped part-way and then
// whole) the running state has the same bytes (the main card's recurrent state followed by the second card's), and the
// windows that follow (one row, drafted windows of several rows kept in part or
// whole, two sessions batched) give the same picks and logits; a checkpoint taken, decoded past and restored gives them
// again.  Cuts 56, 48 and 44, fp16 and int8 KV, the block drafter on and off; with a picture's path as the second argument,
// a picture read across the cards too.  Needs a 27B pack (argument or BASALT_QWEN35_PACK) and two GPUs; exits 77 without them.
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "models/qwen35/adapter.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace basalt;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
[[noreturn]] void die(const char* what, const std::string& msg) {
    std::fprintf(stderr, "%s: %s\n", what, msg.c_str());
    std::exit(1);
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

struct Setup {
    int cut = 0;          // 0: one card
    bool draft = false, int8 = false;
    int prompt = 0;
};

// what a run gives: the running state after the prompt (main regions, then the second card's), then per window its picks and
// its last row's logits, hashed
struct Out {
    std::vector<uint8_t> state;
    std::vector<int32_t> picks;
    std::vector<uint64_t> logits;
    std::vector<uint64_t> batch;   // two sessions in one batch
    std::vector<int32_t> again;    // after a restore
    int stopped = 0;               // reports a stopped read gave
};

std::vector<uint8_t> save(model::ModelSession& s, int64_t pos) {
    core::RunningState rs;
    if (const model::Result e = s.running_state(pos, true, rs); !e.ok()) die("running_state", e.message);
    std::vector<uint8_t> bytes(rs.bytes());
    std::memcpy(bytes.data(), rs.head.data(), rs.head.size());
    std::string err;
    if (!core::running_copy_out(rs, bytes.data(), err)) die("copy out", err);
    return bytes;
}
void restore(model::ModelSession& s, int64_t pos, const std::vector<uint8_t>& bytes) {
    core::RunningState rs;
    std::string err;
    if (!s.running_state(pos, false, rs).ok() || rs.bytes() != bytes.size() || !core::running_copy_in(rs, bytes.data(), err) ||
        !s.running_restored(pos, bytes.data()).ok())
        die("restore", err);
}
// the GDN state in layer order: the main card's region 0, then the second card's (a split), then the other regions
std::vector<uint8_t> canonical(model::ModelSession& s, int64_t pos, const std::vector<uint8_t>& bytes) {
    core::RunningState rs;
    if (!s.running_state(pos, false, rs).ok()) die("running_state", "");
    std::vector<uint8_t> out(bytes.begin(), bytes.begin() + (long) rs.head.size());   // head: rope delta, position, tail, valid
    const size_t r0 = rs.regions.empty() ? 0 : rs.regions[0].bytes, rest = rs.regions_bytes() - r0;
    const uint8_t* at = bytes.data() + rs.head.size();
    out.insert(out.end(), at, at + r0);
    out.insert(out.end(), at + rs.regions_bytes(), bytes.data() + bytes.size());
    out.insert(out.end(), at + r0, at + r0 + rest);
    return out;
}

int32_t window(model::ModelSession& s, const int32_t* tok, int T, int64_t pos, Out& o) {
    std::vector<int32_t> rows((size_t) T);
    std::vector<float> lg(262144);
    model::DecodeRequest r;
    r.tokens = tok;
    r.n_tokens = T;
    r.pos = pos;
    r.row_tokens = rows.data();
    r.logits = lg.data();
    if (const model::Result e = s.decode_window(r); !e.ok()) die("decode", e.message);
    o.picks.insert(o.picks.end(), rows.begin(), rows.end());
    o.logits.push_back(fnv(lg.data(), 248320 * sizeof(float)));
    return rows.back();
}

Out run(const model::ModelAdapter& a, const std::string& pack, const Setup& su) {
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = su.prompt + 1024;
    req.sessions = 2;
    req.draft = su.draft;
    req.spec = su.draft ? 16 : 0;
    if (su.int8) req.kv = "int8";
    if (su.cut) {
        req.peer_devices = {1};
        req.diag_split_at = su.cut;
    }
    req.sessions = 2;
    model::Started started;
    if (const model::Result e = a.start(req, started); !e.ok()) die("start", e.message);
    std::unique_ptr<model::ModelSession>& s = started.sessions[0];
    std::unique_ptr<model::ModelSession>& s2 = started.sessions[1];
    if (const model::Result e = s->prepare(); !e.ok()) die("session", e.message);
    if (const model::Result e = s2->prepare(); !e.ok()) die("session", e.message);
    std::vector<int32_t> prompt((size_t) su.prompt);
    uint32_t x = 12345;
    for (int32_t& t : prompt) {
        x = x * 1664525u + 1013904223u;
        t = 1000 + (int32_t) ((x >> 8) % 30000);
    }
    Out o;
    const int n = su.prompt - 1;
    {   // a read stopped after its second report, then the whole prompt again from the start
        int reports = 0;
        model::PrefillRequest p;
        p.tokens = prompt.data();
        p.n_tokens = n;
        p.total_tokens = n;
        p.progress = [&](int) { ++reports; };
        p.stop = [&] { return reports >= 2; };
        if (const model::Result e = s->prefill_chunk(p); !e.ok()) die("stopped prefill", e.message);
        o.stopped = reports;
    }
    for (model::ModelSession* ss : {s.get(), s2.get()}) {
        model::PrefillRequest p;
        p.tokens = prompt.data();
        p.n_tokens = n;
        p.total_tokens = n;
        if (const model::Result e = ss->prefill_chunk(p); !e.ok()) die("prefill", e.message);
    }
    o.state = canonical(*s, n, save(*s, n));
    // one-row windows (greedy), then windows of 4, 8 and 16 rows of fixed tokens kept in part or whole
    int32_t tok = prompt.back();
    int64_t pos = n;
    for (int i = 0; i < 8; ++i) tok = window(*s, &tok, 1, pos++, o);
    const std::vector<uint8_t> mid = save(*s, pos);
    const int64_t mid_pos = pos;
    std::vector<int32_t> fixed(16);
    for (int32_t& t : fixed) {
        x = x * 1664525u + 1013904223u;
        t = 1000 + (int32_t) ((x >> 8) % 30000);
    }
    const int widths[] = {4, 8, 16, 8, 1, 16}, keeps[] = {4, 3, 16, 1, 1, 9};
    for (size_t w = 0; w < sizeof widths / sizeof *widths; ++w) {
        fixed[0] = tok;
        tok = window(*s, fixed.data(), widths[w], pos, o);
        pos += keeps[w];
        tok = fixed[(size_t) keeps[w] % (size_t) widths[w]];
    }
    // the checkpoint after the greedy rows, restored: the same windows again
    restore(*s, mid_pos, mid);
    Out again;
    int32_t t2 = o.picks[7];
    window(*s, &t2, 1, mid_pos, again);
    o.again = again.picks;
    // two sessions batched: the first at its restored position, the second right after its prompt
    {
        int32_t a1 = again.picks.back(), a2 = prompt.back();
        std::vector<int32_t> rows1(8), rows2(4);
        std::vector<float> l1(262144), l2(262144);
        fixed[0] = a1;
        std::vector<int32_t> f2(fixed.begin() + 4, fixed.begin() + 8);
        f2[0] = a2;
        model::DecodeRequest r[2];
        r[0].tokens = fixed.data();
        r[0].n_tokens = 8;
        r[0].pos = mid_pos + 1;
        r[0].row_tokens = rows1.data();
        r[0].logits = l1.data();
        r[1].tokens = f2.data();
        r[1].n_tokens = 4;
        r[1].pos = n;
        r[1].row_tokens = rows2.data();
        r[1].logits = l2.data();
        model::ModelSession* peers[2] = {s.get(), s2.get()};
        if (const model::Result e = s->decode_batch(peers, r, 2); !e.ok()) die("batch", e.message);
        o.batch = {fnv(rows1.data(), 32), fnv(rows2.data(), 16), fnv(l1.data(), 248320 * 4), fnv(l2.data(), 248320 * 4)};
    }
    return o;
}

// a picture's rows read after a short text, then one-row windows: the logits of each, hashed
std::vector<uint64_t> vision_run(const model::ModelAdapter& a, const std::string& pack, const std::vector<uint8_t>& image, int cut) {
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 8192;
    req.vision = true;
    if (cut) {
        req.peer_devices = {1};
        req.diag_split_at = cut;
    }
    model::Started started;
    if (const model::Result e = a.start(req, started); !e.ok()) die("start", e.message);
    std::unique_ptr<model::ModelSession>& s = started.sessions.front();
    if (const model::Result e = s->prepare(); !e.ok()) die("session", e.message);
    std::vector<int32_t> text(64);
    for (size_t i = 0; i < text.size(); ++i) text[i] = 1000 + (int32_t) (i * 37 % 20000);
    model::PrefillRequest p;
    p.tokens = text.data();
    p.n_tokens = (int) text.size();
    p.total_tokens = p.n_tokens;
    if (const model::Result e = s->prefill_chunk(p); !e.ok()) die("prefill", e.message);
    std::vector<float> rows;
    int n_rows = 0;
    model::ImageGrid grid;
    if (const model::Result e = s->encode_image(image.data(), image.size(), rows, n_rows, grid); !e.ok()) die("encode", e.message);
    if (const model::Result e = s->prefill_embeddings(rows.data(), n_rows, p.n_tokens, grid); !e.ok()) die("image", e.message);
    s->images_done();
    Out o;
    int32_t tok = 1000;
    for (int i = 0; i < 8; ++i) tok = window(*s, &tok, 1, p.n_tokens + n_rows + i, o);
    o.logits.push_back(fnv(o.picks.data(), o.picks.size() * 4));
    return o.logits;
}

void compare(const model::ModelAdapter& a, const std::string& pack, Setup su, const std::vector<int>& cuts) {
    const std::string tag = std::string(su.draft ? "drafter on, " : "plain, ") + (su.int8 ? "int8 KV, " : "fp16 KV, ") +
                            std::to_string(su.prompt) + "-token prompt";
    const Out one = run(a, pack, su);
    for (int c : cuts) {
        su.cut = c;
        const Out two = run(a, pack, su);
        const std::string at = " (cut " + std::to_string(c) + ", " + tag + ")";
        check(two.stopped >= 2 && two.stopped <= (su.prompt + 4095) / 4096, "a stopped read reports its chunks (" + std::to_string(two.stopped) + ")" + at);
        check(one.state == two.state, "the state after the prompt (read again after a stopped read) has the same bytes" + at);
        check(one.picks == two.picks && one.logits == two.logits, "the same picks and logits in windows of 1 to 16 rows" + at);
        check(one.again == two.again, "the same again after a checkpoint is restored" + at);
        check(one.batch == two.batch, "the same for two sessions in one batch" + at);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("BASALT_QWEN35_PACK");
    const std::string pack = argc > 1 ? argv[1] : env ? env : "";
    int devices = 0;
    if (pack.empty() || cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::printf("skip: no 27B pack (argument or BASALT_QWEN35_PACK) or fewer than two GPUs\n");
        return 77;
    }
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) die(pack.c_str(), err);
    const std::unique_ptr<model::ModelAdapter> a = models::qwen35::make_adapter();
    compare(*a, pack, {0, true, false, 8200}, {56, 48, 44});
    compare(*a, pack, {0, true, true, 16400}, {48});
    compare(*a, pack, {0, false, true, 8200}, {44});
    if (argc > 2) {   // a picture (its path the second argument)
        std::vector<uint8_t> image;
        if (FILE* f = std::fopen(argv[2], "rb")) {
            int c;
            while ((c = std::fgetc(f)) != EOF) image.push_back((uint8_t) c);
            std::fclose(f);
        }
        if (image.empty()) die(argv[2], "cannot read the picture");
        check(vision_run(*a, pack, image, 0) == vision_run(*a, pack, image, 48), "a picture and the windows after it give the same logits (cut 48)");
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
