// tests/models/qwen35/prompt_line_test.cpp - the prompt line (a second card reading a long prompt's last layers) leaves the
// session as a read on the main card alone does: after a long prompt the running state (recurrent state, the drafter's
// context) has the same bytes and the greedy tokens that follow, which read every KV cell, are the same; a read stopped part-way ends
// within a few chunks and the same session then reads the prompt again to the same bytes.  Needs a 27B pack (argument or
// BASALT_QWEN35_PACK) and two GPUs; exits 77 without them.  Runs with the pack's drafter on and the KV in fp16, and plain
// with the KV in int8.
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "models/qwen35/adapter.hpp"

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
    if (!ok) ++failures;
}
[[noreturn]] void die(const char* what, const std::string& msg) {
    std::fprintf(stderr, "%s: %s\n", what, msg.c_str());
    std::exit(1);
}

std::vector<int32_t> greedy(model::ModelSession& s, int32_t tok, int64_t pos, int n) {
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        int32_t next = -1;
        model::DecodeRequest r;
        r.tokens = &tok;
        r.n_tokens = 1;
        r.pos = pos + i;
        r.next_token = &next;
        if (const model::Result e = s.decode_window(r); !e.ok()) die("decode", e.message);
        out.push_back(next);
        tok = next;
    }
    return out;
}

std::vector<uint8_t> save(model::ModelSession& s, int64_t pos) {
    core::RunningState rs;
    if (const model::Result e = s.running_state(pos, true, rs); !e.ok()) die("running_state", e.message);
    std::vector<uint8_t> bytes(rs.bytes());
    std::memcpy(bytes.data(), rs.head.data(), rs.head.size());
    std::string err;
    if (!core::checkpoint_copy_out({{bytes.data() + rs.head.size(), rs.regions}}, (cudaStream_t) rs.stream, err)) die("copy out", err);
    return bytes;
}

struct Read {
    std::vector<uint8_t> state;      // the running state after the whole prompt
    std::vector<int32_t> tokens;     // greedy tokens after it
    std::vector<uint8_t> state_again;   // after a stopped read and a second whole one (the line's session only)
    int stopped_at = 0;              // tokens the stopped read had read when it ended
};

// A model with (or without) the second card, a session, `n` prompt tokens read; with `stop_test`, a read that is stopped
// after its second progress report, then the whole prompt again.
Read run(const model::ModelAdapter& a, const std::string& pack, bool line, bool draft, bool int8, int n, bool stop_test) {
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = n + 256;
    req.draft = draft;
    req.spec = draft ? 16 : 0;
    if (int8) req.kv = "int8";
    if (line) req.peer_devices = {1};
    std::unique_ptr<model::LoadedModel> loaded;
    if (const model::Result e = a.load(req, loaded); !e.ok()) die("load", e.message);
    std::unique_ptr<model::ModelSession> s;
    if (const model::Result e = loaded->make_session(s); !e.ok() || !s->prepare().ok()) die("session", e.message);
    std::vector<int32_t> prompt((size_t) n);
    uint32_t x = 12345;
    for (int i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        prompt[(size_t) i] = 1000 + (int32_t) ((x >> 8) % 30000);
    }
    Read r;
    const auto read_all = [&] {
        model::PrefillRequest p;
        p.tokens = prompt.data();
        p.n_tokens = n - 1;
        p.total_tokens = n - 1;
        if (const model::Result e = s->prefill_chunk(p); !e.ok()) die("prefill", e.message);
    };
    if (stop_test) {
        int reports = 0, seen = 0;
        model::PrefillRequest p;
        p.tokens = prompt.data();
        p.n_tokens = n - 1;
        p.total_tokens = n - 1;
        p.progress = [&](int k) { seen = k; ++reports; };
        p.stop = [&] { return reports >= 2; };
        if (const model::Result e = s->prefill_chunk(p); !e.ok()) die("stopped prefill", e.message);
        r.stopped_at = seen;
    }
    read_all();
    r.state = save(*s, n - 1);
    r.tokens = greedy(*s, prompt.back(), n - 1, 24);
    return r;
}

void compare(const model::ModelAdapter& a, const std::string& pack, bool draft, bool int8) {
    const std::string tag = std::string(draft ? " (drafter on, " : " (plain, ") + (int8 ? "int8 KV)" : "fp16 KV)");
    const int n = 30000;   // 3 whole chunks short of 4: the line splits it and the main card reads the last chunks
    const Read one = run(a, pack, false, draft, int8, n, false);
    const Read two = run(a, pack, true, draft, int8, n, false);
    check(one.state.size() == two.state.size() && one.state == two.state, "the state after a long prompt has the same bytes on two cards" + tag);
    check(one.tokens == two.tokens, "the same greedy tokens after it" + tag);
    const Read stopped = run(a, pack, true, draft, int8, n, true);
    check(stopped.stopped_at > 0 && stopped.stopped_at < n - 1 - 8192, "a stopped read ends well before the prompt does (" + std::to_string(stopped.stopped_at) + " tokens)" + tag);
    check(stopped.state == one.state, "the whole prompt read again after a stopped read has the same bytes" + tag);
    check(stopped.tokens == one.tokens, "and the same greedy tokens" + tag);
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
    compare(*a, pack, true, false);
    compare(*a, pack, false, true);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
