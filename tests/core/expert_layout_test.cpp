// tests/core/expert_layout_test.cpp - the expert layout text (a .basalt's `experts.layout` section), without a model
// or a GPU.
//
//   1. The layouts of real models (tests/data/native_experts: Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M, a second IQ2_XS)
//      parse to the fields of each line, and the totals agree with the header's.
//   2. A newer version, an empty text, a gap between layers, a malformed line and a blob size that does not match
//      the formats are refused; columns past the fifth are ignored before v7.
//   3. v7's sixth column, the up rows' type: a mixed layer's offsets and sizes, a missing column and a mix that
//      cannot share a layer refused.
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using basalt::kernels::cpu::ExpertLayout;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool load(const std::string& text, int64_t n_layers, int64_t n_expert, std::string& err) {
    err.clear();
    return basalt::kernels::cpu::expert_layout_load_text(text, n_layers, n_expert, err, 2560, 640);
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

// ---- 1. the real models' layouts
void real_layouts(const std::string& data) {
    for (const char* name : {"q2_0", "iq2_xs", "iq3_xxs", "iq3_s", "coder-iq1_m", "swift-iq2_xs"}) {
        const std::string text = read_file(data + "/" + name + ".txt");
        std::string err;
        const bool ok = load(text, 48, 512, err);
        check(ok, std::string(name) + ": loads" + (ok ? "" : ": " + err));
        if (!ok) continue;
        const ExpertLayout& L = basalt::kernels::cpu::expert_layout();
        std::istringstream in(text);
        std::string line;
        long long hdr_n = -1, hdr_total = -1;
        bool same = L.native && L.n_layers == 48;
        uint64_t max_blob = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.find("(n_expert ") == std::string::npos) continue;
                std::istringstream h(line.substr(line.find("(n_expert ") + 10));
                char comma = 0;
                std::string word;
                h >> hdr_n >> comma >> word >> hdr_total;   // "(n_expert N, total T)"
                continue;
            }
            std::istringstream ss(line);
            long long l = 0, gt = 0, dt = 0;
            unsigned long long off = 0, blob = 0;
            ss >> l >> gt >> dt >> off >> blob;
            same = same && L.fmt[(size_t) l].gu_type == gt && L.fmt[(size_t) l].d_type == dt &&
                   L.offset[(size_t) l] == off && L.bytes[(size_t) l] == blob;
            if (blob > max_blob) max_blob = blob;
        }
        same = same && L.n_expert == hdr_n && (long long) L.total == hdr_total && L.max_blob == max_blob;
        check(same, std::string(name) + ": every field as its line, totals as the header (v" +
                        std::to_string(L.version) + ", n_expert " + std::to_string(L.n_expert) + ")");
    }
}

// ---- 2. the refusals
std::string layout(int version, int64_t blob, const std::string& extra_column, int n_expert = 2,
                   const std::string& types = "12 7") {
    std::string s = "# experts v" + std::to_string(version) + ": layer gu_type d_type offset blob_bytes (n_expert " +
                    std::to_string(n_expert) + ", total " + std::to_string(2 * blob * n_expert) + ")\n";
    s += "0 " + types + " 0 " + std::to_string(blob) + extra_column + "\n";
    s += "1 " + types + " " + std::to_string(blob * n_expert) + " " + std::to_string(blob) + extra_column + "\n";
    return s;
}

void refusals() {
    basalt::kernels::cpu::NativeFmt f;
    std::string err;
    if (!basalt::kernels::cpu::native_fmt(12, 7, 2560, 640, f, err)) {
        check(false, "Q4_K/Q5_1 native format: " + err);
        return;
    }
    const int64_t blob = (int64_t) f.bytes;
    check(load(layout(5, blob, ""), 2, 512, err), "a v5 layout loads" + (err.empty() ? "" : ": " + err));
    check(load(layout(3, blob, " 100 200 300 shard.gguf"), 2, 512, err),
          "columns past the fifth are ignored" + (err.empty() ? "" : ": " + err));
    check(!load(layout(8, blob, ""), 2, 512, err) && err.find("v8") != std::string::npos, "v8 is refused");
    // NVFP4: the blob carries a 16-byte FP32 scale tail after down, counted in the blob size
    basalt::kernels::cpu::NativeFmt nv;
    check(basalt::kernels::cpu::native_fmt(40, 40, 2560, 640, nv, err) && nv.tail == 16 &&
              nv.bytes == 2 * 640 * (2560 / 64 * 36) + 2560 * (640 / 64 * 36) + 16,
          "NVFP4 blob is 2,764,816 B with its scale tail" + (err.empty() ? "" : ": " + err));
    check(load(layout(6, (int64_t) nv.bytes, "", 2, "40 40"), 2, 512, err),
          "a v6 NVFP4 layout loads" + (err.empty() ? "" : ": " + err));
    check(!load(layout(6, (int64_t) nv.bytes - 16, "", 2, "40 40"), 2, 512, err),
          "an NVFP4 blob without its tail is refused");
    check(!basalt::kernels::cpu::native_fmt(40, 7, 2560, 640, nv, err), "NVFP4 gate/up with a Q5_1 down is refused");
    check(!load(std::string(), 2, 512, err), "an empty layout is refused");
    check(!load(layout(5, blob, "").substr(0, layout(5, blob, "").rfind("1 12 7")), 2, 512, err),
          "a missing layer is refused");
    check(!load(layout(5, blob + 8, ""), 2, 512, err), "a blob size that does not match the formats is refused");
    check(!load("# experts v5: x\n0 12 seven 0 1\n", 1, 512, err), "a malformed line is refused");
}

// ---- 3. v7: gate and up of their own types
void mixed() {
    namespace c = basalt::kernels::cpu;
    std::string err;
    c::NativeFmt m, s, x;
    const bool ok = c::native_fmt(22, 42, 2560, 640, m, err, 0.f, 16) && c::native_fmt(22, 42, 2560, 640, s, err) &&
                    c::native_fmt(16, 42, 2560, 640, x, err);
    check(ok && m.up_type == 16 && m.gu_row == s.gu_row && m.up_row == x.gu_row && m.up_off == s.up_off &&
              m.down_off == 640 * (s.gu_row + x.gu_row) && m.bytes == m.down_off + 2560 * s.d_row && m.gu_act == x.gu_act,
          "IQ2_S gate with IQ2_XXS up: rows, offsets and blob from both types" + (err.empty() ? "" : ": " + err));
    check(s.up_type == 22 && s.up_row == s.gu_row && s.down_off == 2 * s.up_off, "a uniform layer's up is its gate's");
    const int64_t blob = (int64_t) m.bytes;
    check(load(layout(7, blob, " 16", 2, "22 42"), 2, 512, err) && c::expert_layout().fmt[1].up_type == 16 &&
              c::expert_layout().fmt[1].up_off == m.up_off && c::expert_layout().total == (uint64_t) (4 * blob),
          "a v7 layout with a mixed layer loads" + (err.empty() ? "" : ": " + err));
    check(load(layout(7, (int64_t) s.bytes, " 22", 2, "22 42"), 2, 512, err) && c::expert_layout().fmt[0].up_type == 22,
          "a v7 uniform line loads" + (err.empty() ? "" : ": " + err));
    check(!load(layout(7, blob, "", 2, "22 42"), 2, 512, err), "a v7 line without its up type is refused");
    check(!load(layout(5, blob, " 16", 2, "22 42"), 2, 512, err), "a v5 layout never reads a sixth column");
    check(!c::native_fmt(22, 42, 2560, 640, x, err, 0.f, 42) && !c::native_fmt(40, 40, 2560, 640, x, err, 0.f, 22) &&
              !c::native_fmt(22, 42, 2560, 640, x, err, 0.f, 8),
          "Q2_0, NVFP4 or another activation format beside a different type is refused");
}
}  // namespace

int main(int argc, char** argv) {
    std::printf("expert_layout_test\n");
    if (argc < 2) {
        std::printf("usage: expert_layout_test <tests/data/native_experts>\n");
        return 2;
    }
    real_layouts(argv[1]);
    refusals();
    mixed();
    std::printf(g_fail ? "expert_layout_test: %d FAILED\n" : "expert_layout_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
