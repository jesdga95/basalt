// src/models/inkling2/walk_test.cpp - the walk against the forward it replaces: a forward that copies each kept input into
// a buffer of its own. Both are run over symbols (each visit's output a distinct value); every input, block and bank entry
// a step reads must hold the value that forward reads there.
#include "models/inkling2/walk.hpp"

#include <cstdio>
#include <vector>

using namespace basalt::models::inkling2;

namespace {

int failures = 0;
void expect(bool ok, const char* what, int visit) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s at visit %d\n", what, visit);
        ++failures;
    }
}

constexpr int kEmb = -1;   // the embedding's symbol; visit i's output is i

struct Reads {
    int hid = 0;
    std::vector<int> blocks;
    bool msgs = false;
    int bank[2] = {0, 0};
};

// The copying forward: the prelude's output in its own buffer, the bank's two entries, the kept inputs.
std::vector<Reads> reference(const Geometry& g, int& final_hid, std::vector<int>& final_blocks) {
    std::vector<Reads> out;
    std::vector<int> blocks, pre_blocks;
    int hid = kEmb, pre = 0, bank_out = 0, i12[2] = {0, 0}, bank = 0;
    for (size_t i = 0; i < g.visits.size(); ++i) {
        const Visit& v = g.visits[i];
        const int l = v.layer;
        if (l == g.loop_start) {
            if (v.pass == 0) {
                pre = hid;
                pre_blocks = blocks;
            } else {
                bank_out = hid;
                bank = i12[(v.pass - 1) % 2];
                blocks = pre_blocks;
            }
            hid = pre;
        }
        Reads r;
        r.hid = hid;
        r.blocks = blocks;
        if (v.pass > 0) {
            r.msgs = true;
            r.bank[0] = bank;
            r.bank[1] = bank_out;
        }
        out.push_back(r);
        if (g.boundary(l)) {
            if (g.looped(l) && l != g.loop_start) i12[v.pass % 2] = hid;
            blocks.push_back(hid);
        }
        hid = (int) i;
    }
    final_hid = hid;
    final_blocks = blocks;
    return out;
}

void check(int layers, int block, int passes) {
    Geometry g;
    g.n_layers = layers;
    g.block = block;
    g.loop_start = block;
    g.loop_end = 3 * block;
    g.loop_passes = passes;
    g.visits = loop_visits(layers, g.loop_start, g.loop_end, passes);
    Walk w;
    std::string err;
    if (!walk(g, w, err)) {
        std::fprintf(stderr, "FAIL walk: %s\n", err.c_str());
        ++failures;
        return;
    }
    int ref_final = 0;
    std::vector<int> ref_blocks;
    const std::vector<Reads> ref = reference(g, ref_final, ref_blocks);
    expect(w.steps.size() == g.visits.size(), "a step per visit", 0);
    int buf[kBufs];
    for (int& b : buf) b = -2;   // nothing written yet
    buf[(int) Buf::X0] = kEmb;
    for (size_t i = 0; i < w.steps.size() && i < ref.size(); ++i) {
        const Step& s = w.steps[i];
        const int vi = (int) i;
        expect(s.visit == vi, "visit order", vi);
        expect(buf[(int) s.hid] == ref[i].hid, "input", vi);
        expect(s.n_blocks == (int) ref[i].blocks.size(), "block count", vi);
        for (int j = 0; j < s.n_blocks && j < (int) ref[i].blocks.size(); ++j)
            expect(buf[(int) s.blocks[j]] == ref[i].blocks[(size_t) j], "block", vi);
        expect(s.msgs == ref[i].msgs, "messages", vi);
        if (s.msgs) {
            expect(buf[(int) s.bank[0]] == ref[i].bank[0], "bank entry 0", vi);
            expect(buf[(int) s.bank[1]] == ref[i].bank[1], "bank entry 1", vi);
        }
        expect(s.out != s.hid, "output over its own input", vi);
        buf[(int) s.out] = vi;
    }
    expect(buf[(int) w.final_hid] == ref_final, "the head's input", -1);
    expect(w.n_blocks == (int) ref_blocks.size(), "the head's block count", -1);
    for (int j = 0; j < w.n_blocks && j < (int) ref_blocks.size(); ++j)
        expect(buf[(int) w.blocks[j]] == ref_blocks[(size_t) j], "the head's block", -1);
}

}  // namespace

int main() {
    check(24, 6, 3);   // Inkling 2 Mini
    check(24, 6, 1);
    check(24, 6, 2);
    check(24, 6, 4);
    check(8, 2, 5);
    {   // a layout the walk does not cover
        Geometry g;
        g.n_layers = 24;
        g.block = 6;
        g.loop_start = 6;
        g.loop_end = 12;
        g.loop_passes = 2;
        g.visits = loop_visits(24, 6, 12, 2);
        Walk w;
        std::string err;
        expect(!walk(g, w, err), "refuses a loop of one block", -1);
    }
    if (failures) return 1;
    std::printf("inkling2 walk: ok\n");
    return 0;
}
