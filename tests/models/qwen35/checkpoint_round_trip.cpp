// tests/models/qwen35/checkpoint_round_trip.cpp - the dense session's running state through the shared checkpoint
// transfer: saved after a prompt and a few decoded tokens, changed by decoding on, written back, the same greedy tokens
// again and the same bytes when saved a second time. Needs a 27B pack (argument or BASALT_QWEN35_PACK) and its GPU;
// exits 77 without either. Runs once with the pack's drafter on (its waiting row or block context in the state) and once
// plain.
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

// greedy tokens one window of one row at a time from `tok` at `pos`
std::vector<int32_t> greedy(model::ModelSession& s, int32_t tok, int64_t pos, int n) {
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        int32_t next = -1;
        model::DecodeRequest r;
        r.tokens = &tok;
        r.n_tokens = 1;
        r.pos = pos + i;
        r.next_token = &next;
        if (const model::Result e = s.decode_window(r); !e.ok()) {
            std::fprintf(stderr, "decode: %s\n", e.message.c_str());
            std::exit(1);
        }
        out.push_back(next);
        tok = next;
    }
    return out;
}

std::vector<uint8_t> save(model::ModelSession& s, int64_t pos) {
    core::RunningState rs;
    if (const model::Result e = s.running_state(pos, true, rs); !e.ok()) {
        std::fprintf(stderr, "running_state: %s\n", e.message.c_str());
        std::exit(1);
    }
    std::vector<uint8_t> bytes(rs.bytes());
    std::memcpy(bytes.data(), rs.head.data(), rs.head.size());
    std::string err;
    if (!core::checkpoint_copy_out({{bytes.data() + rs.head.size(), rs.regions}}, (cudaStream_t) rs.stream, err)) {
        std::fprintf(stderr, "copy out: %s\n", err.c_str());
        std::exit(1);
    }
    return bytes;
}

bool restore(model::ModelSession& s, const std::vector<uint8_t>& bytes, int64_t pos) {
    core::RunningState rs;
    if (!s.running_state(pos, false, rs).ok() || rs.bytes() != bytes.size()) return false;
    std::string err;
    if (!core::checkpoint_copy_in({{bytes.data() + rs.head.size(), rs.regions}}, (cudaStream_t) rs.stream, err)) return false;
    return s.running_restored(pos, bytes.data()).ok();
}

void round_trip(const model::ModelAdapter& a, const std::string& pack, bool draft) {
    const std::string tag = draft ? " (drafter on)" : " (plain)";
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 4096;
    req.draft = draft;
    req.spec = draft ? 16 : 0;
    std::unique_ptr<model::LoadedModel> loaded;
    if (const model::Result e = a.load(req, loaded); !e.ok()) {
        std::fprintf(stderr, "load: %s\n", e.message.c_str());
        std::exit(1);
    }
    std::unique_ptr<model::ModelSession> s;
    if (const model::Result e = loaded->make_session(s); !e.ok() || !s->prepare().ok()) {
        std::fprintf(stderr, "session: %s\n", e.message.c_str());
        std::exit(1);
    }
    // a prompt of ordinary token ids, all but its last read as a prompt
    const int n = 777;
    std::vector<int32_t> prompt(n);
    for (int i = 0; i < n; ++i) prompt[(size_t) i] = 1000 + (int32_t) ((i * 7919) % 30000);
    model::PrefillRequest p;
    p.tokens = prompt.data();
    p.n_tokens = n - 1;
    p.total_tokens = n - 1;
    if (const model::Result e = s->prefill_chunk(p); !e.ok()) {
        std::fprintf(stderr, "prefill: %s\n", e.message.c_str());
        std::exit(1);
    }
    // a few decoded tokens first, so the save settles a pending window
    const std::vector<int32_t> lead = greedy(*s, prompt.back(), n - 1, 8);
    const int64_t at = n - 1 + 8;
    const std::vector<uint8_t> state = save(*s, at);
    check(state.size() == s->running_bytes(), "the saved state is running_bytes() long" + tag);
    const std::vector<int32_t> first = greedy(*s, lead.back(), at, 24);
    check(restore(*s, state, at), "the state goes back" + tag);
    check(save(*s, at) == state, "saved again it has the same bytes" + tag);
    const std::vector<int32_t> again = greedy(*s, lead.back(), at, 24);
    check(again == first, "the same greedy tokens after the restore" + tag);
}
}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("BASALT_QWEN35_PACK");
    const std::string pack = argc > 1 ? argv[1] : env ? env : "";
    int devices = 0;
    if (pack.empty() || cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("skip: no 27B pack (argument or BASALT_QWEN35_PACK) or no GPU\n");
        return 77;
    }
    // the pack's GGUF view, as the program opens it
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) {
        std::fprintf(stderr, "%s: %s\n", pack.c_str(), err.c_str());
        return 1;
    }
    const std::unique_ptr<model::ModelAdapter> a = models::qwen35::make_adapter();
    round_trip(*a, pack, false);
    round_trip(*a, pack, true);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
