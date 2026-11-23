// src/models/hy4flash/dfly_real.cpp - the drafter's block pass on the real weights against a reference's outputs: the target's last
// hidden state, the target's KV rows of layers 25, 29, .. and the expected block (tools: ~/poc/hy4-ref/dfly_ref.py, vLLM's forward
// in fp32). usage: hy4flash_dfly_real DIR [DFLY.safetensors DENSE.safetensors]
//        hy4flash_dfly_real --pack PACK.basalt [DFLY.safetensors]: the pack's dfly section as the loader places it, byte for byte against the file
#include "basalt/core/loader.hpp"
#include "basalt/platform/file.hpp"
#include "models/hy4flash/dfly.hpp"
#include "models/hy4flash/pack.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace basalt::models::hy4flash;
using basalt::core::Json;

namespace {

struct St {
    std::string path;
    std::map<std::string, DflyTensor> t;   // absolute offsets
};

bool open_st(const std::string& path, St& s, const std::string& prefix) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n = 0;
    f.read((char*) &n, 8);
    std::string h(n, 0);
    f.read(h.data(), (std::streamsize) n);
    Json j;
    std::string err;
    if (!basalt::core::parse_json(h.data(), h.size(), j, err)) return false;
    s.path = path;
    for (const auto& [name, v] : j.members) {
        if (name == "__metadata__" || name.rfind(prefix, 0) != 0) continue;
        DflyTensor t;
        const Json* off = v.get("data_offsets");
        t.offset = 8 + n + off->items[0].u64();
        t.bytes = off->items[1].u64() - off->items[0].u64();
        for (const Json& d : v.get("shape")->items) t.shape.push_back(d.u64());
        s.t[name.substr(prefix.size())] = std::move(t);
    }
    return true;
}

std::vector<uint8_t> read_at(const std::string& path, uint64_t off, uint64_t n) {
    std::vector<uint8_t> b(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    std::fseek(f, (long) off, SEEK_SET);
    if (std::fread(b.data(), 1, n, f) != n) b.clear();
    std::fclose(f);
    return b;
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

float bf(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

// The drafter as the engine loads it (dfly_plan and the loader) against the safetensors file it came from.
static int check_pack(const std::string& pack_path, const std::string& dfly_path) {
    basalt::models::hy4flash::Pack pack;
    std::string err;
    if (!basalt::models::hy4flash::open_pack(pack_path, pack, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    cudaFree(nullptr);
    basalt::core::LoadPlan plan;
    const int fd = basalt::platform::open_fd_utf8(pack_path);
    std::unique_ptr<DflyWeights> w;
    const basalt::model::Result r = dfly_plan(pack, plan, plan.file_fd(fd), 0, w);
    if (!r.ok() || !w) { std::fprintf(stderr, "dfly_plan: %s\n", r.message.c_str()); return 1; }
    basalt::core::Loader loader(8);
    if (!loader.prepare({0}, err) || !loader.run(plan, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    cudaDeviceSynchronize();
    St dfly;
    open_st(dfly_path, dfly, "model.dfly_layers.");
    int bad = 0;
    for (const DflySpec& s : dfly_specs(w->dims)) {
        const DflyTensor& t = dfly.t.at(s.name);
        const std::vector<uint8_t> raw = read_at(dfly_path, t.offset, t.bytes);
        size_t diff = 0, n = (size_t) s.elems();
        if (s.f32) {
            std::vector<float> got(n);
            cudaMemcpy(got.data(), w->slot(s.name), n * 4, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < n; ++i) diff += got[i] != bf(((const uint16_t*) raw.data())[i]);
        } else {   // as stored, or a projection as the loader rounds it to Q8_0
            std::vector<uint8_t> want = raw;
            if (s.q8) {
                want.resize(s.bytes());
                basalt::core::quantize_bf16_q8_0((const uint16_t*) raw.data(), n, want.data());
            }
            std::vector<uint8_t> got(want.size());
            cudaMemcpy(got.data(), w->slot(s.name), got.size(), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < got.size(); ++i) diff += got[i] != want[i];
        }
        if (diff) { std::printf("MISMATCH %s: %zu of %zu differ\n", s.name.c_str(), diff, n); ++bad; }
    }
    std::printf("%s: %d of 87 tensors differ from %s\n", pack_path.c_str(), bad, dfly_path.c_str());
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--pack")
        return check_pack(argv[2], argc > 3 ? argv[3] : "/mnt/nvme/models/hy4-bf16-extra/hy4-dfly.safetensors");
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s DIR [DFLY.safetensors DENSE.safetensors]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string dfly_path = argc > 2 ? argv[2] : "/mnt/nvme/models/hy4-bf16-extra/hy4-dfly.safetensors";
    const std::string dense_path = argc > 3 ? argv[3] : "/mnt/nvme/models/hy4-bf16-extra/dense-bf16.safetensors";
    St dfly, dense;
    if (!open_st(dfly_path, dfly, "model.dfly_layers.") || !open_st(dense_path, dense, "")) return 1;
    DflyDims d;
    d.layers = 5; d.vocab = 120832; d.mask_id = 120009; d.rms_eps = 1e-5f; d.rope_base = 1e7f; d.topk = 2048;
    d.kv_layers = d.idx_layers = {25, 29, 33, 37, 41};
    std::string err;
    if (!dfly_check_shapes(dfly.t, d, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    DflyWeights w;
    if (!w.allocate(d, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (const DflySpec& s : dfly_specs(d)) {
        const DflyTensor& t = dfly.t.at(s.name);
        const std::vector<uint8_t> raw = read_at(dfly_path, t.offset, t.bytes);
        if (s.f32) {
            std::vector<float> v(raw.size() / 2);
            for (size_t i = 0; i < v.size(); ++i) v[i] = bf(((const uint16_t*) raw.data())[i]);
            cudaMemcpy(w.slot(s.name), v.data(), v.size() * 4, cudaMemcpyHostToDevice);
        } else if (s.q8 || s.q4) {
            std::vector<uint8_t> q(s.bytes());
            if (s.q4) basalt::core::quantize_bf16_q4_0((const uint16_t*) raw.data(), s.elems(), q.data());
            else basalt::core::quantize_bf16_q8_0((const uint16_t*) raw.data(), s.elems(), q.data());
            cudaMemcpy(w.slot(s.name), q.data(), q.size(), cudaMemcpyHostToDevice);
        } else cudaMemcpy(w.slot(s.name), raw.data(), raw.size(), cudaMemcpyHostToDevice);
    }
    const DflyTensor& hw = dense.t.at("lm_head.weight");
    void* head = nullptr;   // the head as the engine loads it: BF16 rounded to Q8_0
    {
        const std::vector<uint8_t> raw = read_at(dense_path, hw.offset, hw.bytes);
        std::vector<uint8_t> q(raw.size() / 2 / 32 * 34);
        basalt::core::quantize_bf16_q8_0((const uint16_t*) raw.data(), raw.size() / 2, q.data());
        cudaMalloc(&head, q.size());
        cudaMemcpy(head, q.data(), q.size(), cudaMemcpyHostToDevice);
    }
    w.head_q8 = head;
    if (!w.pack_screen(nullptr, err)) { std::printf("%s\n", err.c_str()); return 1; }
    const DflyTensor& ew = dense.t.at("model.embed_tokens.weight");
    const int E = d.E;

    const std::vector<uint8_t> hidb = slurp(dir + "/hid.f32");
    std::vector<uint8_t> anc = slurp(dir + "/anchor.npy");
    const int32_t anchor = (int32_t) *(const int64_t*) (anc.data() + anc.size() - 8);
    std::vector<uint8_t> ids = slurp(dir + "/ids.npy");
    const int T = (int) ((ids.size() - 128) / 8);
    std::vector<uint8_t> et = slurp(dir + "/exp_tok.npy"), ec = slurp(dir + "/exp_conf.npy");
    const int n = (int) (et.size() - 128) / 8;
    std::vector<float> ez(slurp(dir + "/exp_z.f32").size() / 4);
    {
        const std::vector<uint8_t> b = slurp(dir + "/exp_z.f32");
        std::memcpy(ez.data(), b.data(), b.size());
    }

    float* dhid = nullptr;
    cudaMalloc(&dhid, (size_t) E * 4);
    cudaMemcpy(dhid, hidb.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
    auto tap = DflyTap::create(d, 2056, err);
    tap->begin_window(T - 1);
    tap->hidden(0, dhid, nullptr);
    for (int l : d.idx_layers) tap->selection(l, 0, nullptr, -1, nullptr);   // below 2,048 positions the indexer keeps every one
    std::vector<uint16_t*> dkv;
    for (int i = 0; i < 5; ++i) {
        const std::vector<uint8_t> b = slurp(dir + "/kv" + std::to_string(i) + ".u16");
        uint16_t* p = nullptr;
        cudaMalloc(&p, b.size());
        cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice);
        dkv.push_back(p);
    }
    DflyRunner::Env env;
    env.w = &w;
    env.embed = [&](int32_t t, float* o) {
        const std::vector<uint8_t> raw = read_at(dense_path, ew.offset + (uint64_t) t * E * 2, (uint64_t) E * 2);
        for (int i = 0; i < E; ++i) o[i] = bf(((const uint16_t*) raw.data())[i]);
        return !raw.empty();
    };
    auto run = DflyRunner::create(env, err);
    if (!run) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<const void*> caches(dkv.begin(), dkv.end());
    int bad = 0;
    cudaStream_t st = nullptr;
    if (argc > 4 && std::string(argv[4]) == "stream") cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);   // as the session's
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run->begin(anchor, T, n, *tap, 0, caches.data(), st, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const auto t1 = std::chrono::steady_clock::now();
        int32_t tok[8];
        float conf[8];
        int made = 0;
        if (!run->finish(tok, conf, made, st, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const auto t2 = std::chrono::steady_clock::now();
        if (rep < 2) continue;
        std::printf("block %.2f ms (enqueue), heads %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count());
        std::vector<float> z((size_t) (n + 1) * E);
        run->copy_block(z.data(), n + 1, st);
        for (int r = 0; r <= n; ++r) {
            double num = 0, den = 0, mx = 0;
            for (int i = 0; i < E; ++i) {
                const double a = z[(size_t) r * E + i], b = ez[(size_t) r * E + i];
                num += (a - b) * (a - b);
                den += b * b;
                mx = std::max(mx, std::fabs(a - b));
            }
            std::printf("z row %d: relative rms error %.3e, max abs %.3e\n", r, std::sqrt(num / den), mx);
        }
        for (int j = 0; j < n; ++j) {
            const int64_t want = ((const int64_t*) (et.data() + 128))[j];
            const double wc = ((const double*) (ec.data() + 128))[j];
            std::printf("draft %d: token %d (reference %lld) confidence %.4f (reference %.4f)%s\n", j, tok[j], (long long) want, conf[j], wc,
                        tok[j] == want ? "" : "  DIFFERS");
            bad += tok[j] != want;
        }
    }
    return bad ? 1 : 0;
}
