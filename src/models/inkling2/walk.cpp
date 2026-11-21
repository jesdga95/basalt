// src/models/inkling2/walk.cpp - the row buffers of each visit of one forward (see walk.hpp).
#include "models/inkling2/walk.hpp"

#include <algorithm>

namespace basalt::models::inkling2 {

bool walk(const Geometry& g, Walk& w, std::string& err) {
    if (g.block <= 0 || g.loop_start != g.block || g.loop_end - g.loop_start != 2 * g.block ||
        g.n_layers - g.loop_end != g.block || g.loop_passes < 1) {
        err = "the forward takes a prelude of one block, a loop of two blocks and a coda of one block";
        return false;
    }
    w = Walk{};
    std::vector<Buf> blocks, pre_blocks;
    Buf hid = Buf::X0;
    const auto mid = [](int pass) { return pass % 2 ? Buf::I12b : Buf::I12a; };
    for (size_t i = 0; i < g.visits.size(); ++i) {
        const Visit& v = g.visits[i];
        const int l = v.layer;
        if (l == g.loop_start) {
            if (v.pass == 0) pre_blocks = blocks;   // every pass starts from the blocks the prelude finished
            else blocks = pre_blocks;
        }
        Step s;
        s.visit = (int) i;
        s.hid = hid;
        s.n_blocks = (int) blocks.size();
        std::copy(blocks.begin(), blocks.end(), s.blocks);
        if (v.pass > 0) {
            s.msgs = true;
            s.bank[0] = mid(v.pass - 1);
            s.bank[1] = Buf::BankOut;
        }
        // the output goes where the next visit keeps it
        const Visit* nx = i + 1 < g.visits.size() ? &g.visits[i + 1] : nullptr;
        if (nx && nx->layer == g.loop_start) s.out = nx->pass == 0 ? Buf::Pre : Buf::BankOut;
        else if (nx && nx->layer == g.loop_start + g.block) s.out = mid(nx->pass);
        else if (nx && nx->layer == g.loop_end) s.out = Buf::I18;
        else s.out = hid == Buf::H0 ? Buf::H1 : Buf::H0;
        w.steps.push_back(s);
        if (g.boundary(l)) blocks.push_back(hid);   // the layer's input joins the finished blocks
        // a later pass reads the prelude's output again, not the previous pass's
        hid = nx && nx->layer == g.loop_start ? Buf::Pre : s.out;
    }
    if (blocks.size() + 1 > (size_t) kMaxMix) {
        err = "more finished blocks than a mix reads";
        return false;
    }
    w.final_hid = hid;
    w.n_blocks = (int) blocks.size();
    std::copy(blocks.begin(), blocks.end(), w.blocks);
    return true;
}

}  // namespace basalt::models::inkling2
