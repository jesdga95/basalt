// tests/models/glm5next/forward_test.cpp - the tiny runnable pack through the whole forward pass (§5.5).
//
// What this proves: every layer of the port runs on a real pack - the embedding off disk, mHC around a KDA layer, an
// MLA layer with its DSA indexer, a dense MLP and a routed MoE with shared experts, the head - and it produces finite
// numbers, the same numbers in a fresh session, and caches that move when positions advance. It also proves the bounds
// hold (a window past the cache is refused, not wrapped).
//
// What it does NOT prove is output quality. The fixture's weights are random, so a correct pass and a subtly wrong one
// both give finite, stable logits. Quality needs the reference's activations, which is §5.6 against a real pack.
//
// Positions have to be fed in order: the DSA cache scores pools of `kpool` tokens, and a pool is only complete once
// every one of its cells has been written. Asking for position 2 without having written 0 and 1 would ask the model to
// attend cells that were never computed.
#include "models/glm5next/forward.hpp"
#include "models/glm5next/kernels/cell_expand.hpp"
#include "models/glm5next/kernels/mla_sparse.hpp"
#include "models/glm5next/model.hpp"
#include "models/glm5next/prefill_plan.hpp"
#include "models/glm5next/session.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

/// The runnable fixture: geometry-true F32 weights, so a forward pass can multiply them.
std::string fixture() {
    if (const char* e = std::getenv("GLM5NEXT_TINY_PACK"); e && *e && file_exists(e)) return e;
#ifdef GLM_SRC_DIR
    const std::string out = "/tmp/glm5next_runnable.basalt";
    const std::string cmd = "python3 '" GLM_SRC_DIR "/tools/convert/glm5next_tiny_pack.py' '" + out + "' >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0 && file_exists(out)) return out;
#endif
    return "";
}

bool have_sm120() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) return true;
    }
    return false;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int) v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

}  // namespace

int main() {
    const std::string path = fixture();
    if (path.empty()) {
        std::printf("SKIP: no runnable tiny pack (set GLM5NEXT_TINY_PACK or provide python + the converter)\n");
        return 0;
    }
    if (!have_sm120()) {
        std::printf("SKIP: no sm_120 device\n");
        return 77;
    }

    basalt::models::glm5next::GlmModel model;
    basalt::core::LoadPlan reads;
    std::vector<std::unique_ptr<basalt::model::ModelSession>> sessions;
    basalt::model::LoadRequest req;
    req.paths = {path};
    req.spec = 2;
    req.draft = true;   // the drafted window is what this section is about (--spec is what asks for one)
    const auto pr = model.plan(req, reads, sessions);
    if (!pr.ok()) {
        std::printf("FAIL: plan: %s\n", pr.message.c_str());
        return 1;
    }
    std::string err;
    basalt::core::Loader loader;
    check(loader.run(reads, err), "the loader places the plan");
    if (!err.empty()) std::printf("loader: %s\n", err.c_str());
    check(model.attach(req, sessions).ok(), "attach succeeds");
    check(sessions.size() == 1, "plan made one session");
    if (sessions.empty() || g_failures) return 1;
    auto* sess = static_cast<basalt::models::glm5next::GlmSession*>(sessions[0].get());
    const int vocab = model.desc().n_vocab;

    // --- one token at position 0: the shortest path through every layer, including the pool-less first token
    const std::vector<int32_t> one = {0};
    std::vector<float> first(vocab, 0.f), again(vocab, 0.f);
    int32_t next = -1;
    {
        basalt::model::DecodeRequest dr;
        dr.tokens = one.data();
        dr.n_tokens = 1;
        dr.pos = 0;
        dr.logits = first.data();
        dr.next_token = &next;
        const auto r = sess->decode_window(dr);
        check(r.ok(), "one token at position 0 runs");
        if (!r.ok()) std::printf("  decode: %s\n", r.message.c_str());
    }
    check(all_finite(first), "the logits are finite");
    check(next >= 0 && next < vocab, "the greedy pick is a token id in the vocab");

    // --- the drafter's block. The fixture carries a next-n block whose indexer is the model's own and whose experts
    // have a tier of their own, so the session offers a drafted window, and the block can say what follows a token the
    // model has not run: its input is the hidden the pass left beside that token's embedding.
    check(sess->max_window() == 2, "the session offers a drafted window");
    {
        int32_t draft = -1, repeat = -1;
        float conf = 0.f, repeat_conf = 0.f;
        int n = -1;
        const auto b = sess->draft_begin(next, 1, 1);
        check(b.ok(), "the drafter runs on the window the pass left behind");
        if (!b.ok()) std::printf("  draft: %s\n", b.message.c_str());
        check(sess->draft_end(&draft, &conf, 1, n, {}).ok() && n == 1, "the drafter returns one draft");
        check(draft >= 0 && draft < vocab, "the draft is a token id in the vocab");
        check(conf > 0.f && conf <= 1.f, "the draft carries the chance the engine weighs it on");
        // The block is a function of the state the pass left, so the same anchor twice is the same guess twice.
        sess->draft_begin(next, 1, 1);
        sess->draft_end(&repeat, &repeat_conf, 1, n, {});
        check(repeat == draft && repeat_conf == conf, "the same anchor drafts the same token");
    }

    // --- the same window on a fresh session gives the same logits, bit for bit. A session's state only moves
    // forward inside a conversation: the KDA recurrent state and the conv windows advance with every window and are
    // not indexed by position, so a window further in is not the same input twice. Position 0 is - it starts a
    // conversation, and starting one is what clears the state (see the block after this one).
    {
        std::string err2;
        basalt::models::glm5next::GlmModel second_model;
        basalt::core::LoadPlan second_reads;
        std::vector<std::unique_ptr<basalt::model::ModelSession>> second_sessions;
        basalt::model::LoadRequest second_req;
        second_req.paths = {path};
        check(second_model.plan(second_req, second_reads, second_sessions).ok(), "the second session plans");
        basalt::core::Loader second_loader;
        check(second_loader.run(second_reads, err2), "the loader places the second plan");
        if (!err2.empty()) std::printf("loader: %s\n", err2.c_str());
        check(second_model.attach(second_req, second_sessions).ok(), "the second session attaches");
        if (!second_sessions.empty()) {
            check(second_sessions[0]->max_window() == 1, "without --spec the session drafts no window");
            int32_t next2 = -1;
            basalt::model::DecodeRequest dr;
            dr.tokens = one.data();
            dr.n_tokens = 1;
            dr.pos = 0;
            dr.logits = again.data();
            dr.next_token = &next2;
            check(second_sessions[0]->decode_window(dr).ok(), "the fresh session runs the same window");
            check(all_finite(again), "the second run's logits are finite");
            check(std::memcmp(first.data(), again.data(), first.size() * sizeof(float)) == 0,
                  "a fresh session gives the same logits for the same window");
            check(next2 == argmax(again), "the fresh session's greedy pick is the logits' argmax");
        }
    }

    // --- the same session, asked for position 0 twice, must answer the same. A session outlives the conversations it
    // serves and this model keeps no checkpoint to restore, so position 0 is the only signal that the KDA state and
    // the conv window have to start over: a conversation that inherits the previous one's recurrent state is decoded
    // as if it were its continuation. Something else has to run in between, or the rerun would say nothing.
    {
        const std::vector<int32_t> other = {next, one[0], next};
        std::vector<float> scratch(vocab, 0.f);
        basalt::model::DecodeRequest d;
        d.tokens = other.data();
        d.n_tokens = (int) other.size();
        d.pos = 0;
        d.logits = scratch.data();
        check(sess->decode_window(d).ok(), "the session runs another conversation in between");
        std::fill(scratch.begin(), scratch.end(), 0.f);

        int32_t again_in_place = -1;
        std::vector<float> third(vocab, 0.f);
        d.tokens = one.data();
        d.n_tokens = 1;
        d.pos = 0;
        d.logits = third.data();
        d.next_token = &again_in_place;
        check(sess->decode_window(d).ok(), "the same session runs position 0 again");
        check(all_finite(third), "the rerun's logits are finite");
        check(std::memcmp(first.data(), third.data(), first.size() * sizeof(float)) == 0,
              "position 0 starts a conversation on a clean state");
        check(again_in_place == next, "and picks what a fresh conversation picks");
    }

    // --- a drafted window's rows. The engine verifies a draft by running it beside the token it came from, which is
    // sound only if a row picks the same token whatever stands around it; a fresh session reads its first row here,
    // the same one the single-token window above computed alone.
    {
        std::string werr;
        basalt::models::glm5next::GlmModel drafted_model;
        basalt::core::LoadPlan drafted_reads;
        std::vector<std::unique_ptr<basalt::model::ModelSession>> drafted_sessions;
        basalt::model::LoadRequest drafted_req;
        drafted_req.paths = {path};
        const bool opened = drafted_model.plan(drafted_req, drafted_reads, drafted_sessions).ok() &&
                            basalt::core::Loader{}.run(drafted_reads, werr) &&
                            drafted_model.attach(drafted_req, drafted_sessions).ok() && !drafted_sessions.empty();
        check(opened, "the drafted window's session opens");
        if (opened) {
            const std::vector<int32_t> two = {one[0], next};
            std::vector<int32_t> picks(2, -1);
            basalt::model::DecodeRequest w;
            w.tokens = two.data();
            w.n_tokens = 2;
            w.pos = 0;
            w.row_tokens = picks.data();
            const auto r = drafted_sessions[0]->decode_window(w);
            check(r.ok(), "a two-row drafted window runs");
            if (!r.ok()) std::printf("  drafted window: %s\n", r.message.c_str());
            check(picks[0] == next, "a row picks the same token in a drafted window as alone");
            check(picks[1] >= 0 && picks[1] < vocab, "the drafted row picks as well");
        }
    }

    // --- the logit bias. The engine calls set_logit_bias on every request, with null when the request carries none,
    // so refusing that call stops a chat before its first token; and a -inf bias has to ban its token on both the
    // greedy argmax and the sampler. Fresh sessions, for the reason the block above gives.
    {
        const auto nrb = sess->set_logit_bias(nullptr);
        check(nrb.ok(), "a request without a logit bias is accepted");
        if (!nrb.ok()) std::printf("  logit_bias: %s\n", nrb.message.c_str());

        const int32_t banned = argmax(first);
        int32_t second = -1;
        float best = -INFINITY;
        for (int v = 0; v < vocab; ++v)
            if (v != banned && first[v] > best) {
                best = first[v];
                second = v;
            }
        std::vector<float> bias((size_t) vocab, 0.0f);
        bias[banned] = -INFINITY;

        auto fresh = [&](const char* what, basalt::models::glm5next::GlmModel& m,
                         std::vector<std::unique_ptr<basalt::model::ModelSession>>& ss) {
            basalt::core::LoadPlan reads;
            basalt::model::LoadRequest rq;
            rq.paths = {path};
            std::string e;
            check(m.plan(rq, reads, ss).ok(), (std::string(what) + " plans").c_str());
            basalt::core::Loader ld;
            check(ld.run(reads, e), (std::string(what) + " loads").c_str());
            if (!e.empty()) std::printf("loader: %s\n", e.c_str());
            check(m.attach(rq, ss).ok(), (std::string(what) + " attaches").c_str());
            if (!ss.empty())
                check(ss[0]->set_logit_bias(bias.data()).ok(), (std::string(what) + " takes the bias").c_str());
        };
        {
            basalt::models::glm5next::GlmModel banned_model;
            std::vector<std::unique_ptr<basalt::model::ModelSession>> banned_sessions;
            fresh("the banned greedy session", banned_model, banned_sessions);
            if (!banned_sessions.empty()) {
                int32_t pick = -1;
                basalt::model::DecodeRequest dr;
                dr.tokens = one.data();
                dr.n_tokens = 1;
                dr.pos = 0;
                dr.next_token = &pick;
                check(banned_sessions[0]->decode_window(dr).ok(), "the banned window runs");
                check(pick == second, "the greedy pick skips the banned token");
            }
        }
        {
            basalt::models::glm5next::GlmModel drawn_model;
            std::vector<std::unique_ptr<basalt::model::ModelSession>> drawn_sessions;
            fresh("the banned drawn session", drawn_model, drawn_sessions);
            if (!drawn_sessions.empty()) {
                basalt::kernels::SamplerParams sp;
                sp.top_k = 1;  // one kept token: the draw is that token, so the ban is visible
                sp.greedy = false;
                int32_t pick = -1;
                basalt::model::DecodeRequest dr;
                dr.tokens = one.data();
                dr.n_tokens = 1;
                dr.pos = 0;
                dr.sampling = &sp;
                dr.next_token = &pick;
                check(drawn_sessions[0]->decode_window(dr).ok(), "the banned window runs sampled");
                check(pick == second, "the sampled pick skips the banned token");
            }
        }
    }

    // --- the prefill plan's bound covers what the pass really allocated: the two are not allowed to drift apart
    {
        std::string ferr;
        auto* fwd = sess->forward(ferr);
        check(fwd != nullptr, "the session built its forward pass");
        if (!ferr.empty()) std::printf("  forward: %s\n", ferr.c_str());
        if (fwd) {
            // the planner counts the rows the pass allocates; the sparse attention's workspace is the kernel's own,
            // so the two together have to cover what the pass really asked the device for
            const auto& geo = model.geo();
            const uint64_t n_pool = sess->context() / (uint64_t) geo.kpool;
            const int n_top = std::max(1, std::min((int) n_pool, geo.idx_top_k / geo.kpool));
            const int n_sel = basalt::models::glm5next::cell_expand_n_sel(geo.kpool, n_top, true);
            const uint64_t bound =
                basalt::models::glm5next::prefillp::window_units(geo, sess->context(), fwd->window_cap()) +
                basalt::models::glm5next::mla_sparse_workspace_floats(fwd->window_cap(), geo.n_head, n_sel);
            check(fwd->scratch_units() <= bound, "the scratch stays inside the prefill plan's bound");
            check(fwd->scratch_units() * 2 > bound, "the bound is close to the real allocation, not a fudge factor");
        }
    }

    // --- the footprint the window cap and the tier reserve charge the pass for, against what it ends up holding
    //     outside its window. Both spend the card's free memory after this number: under-price it and a request dies
    //     part way through a block on a full card, which is the failure that ended every real chat once.
    {
        std::string ferr;
        if (auto* fwd = sess->forward(ferr)) {
            const uint64_t held = fwd->memo_bytes();
            const uint64_t charged = basalt::models::glm5next::forward_footprint_bytes(model.geo(), model.pack());
            check(held <= charged, "what the pass keeps outside a window stays inside what it is charged for");
            std::printf("  unquantized pack: charged %llu KiB, the pass holds %llu KiB\n",
                        (unsigned long long) (charged >> 10), (unsigned long long) (held >> 10));
        }
    }
    if (const char* twin = std::getenv("GLM5NEXT_TINY_PACK_PACKED"); twin && *twin) {
        // The packed pack is the one that makes those copies, and a wide window is what makes the padded ones, so
        // charge and hold are compared there where they can disagree.
        basalt::models::glm5next::GlmModel m;
        basalt::core::LoadPlan plan;
        std::vector<std::unique_ptr<basalt::model::ModelSession>> ss;
        basalt::model::LoadRequest lr;
        lr.paths = {std::string(twin)};
        std::string werr;
        const bool good = m.plan(lr, plan, ss).ok() && basalt::core::Loader{}.run(plan, werr) &&
                          m.attach(lr, ss).ok() && !ss.empty();
        check(good, "the packed pack opens for the footprint comparison");
        if (good) {
            auto* ws = static_cast<basalt::models::glm5next::GlmSession*>(ss[0].get());
            const std::vector<int32_t> toks = {0, 1, 2, 0, 1, 2, 0, 1, 2};  // past the narrow product's columns
            basalt::model::PrefillRequest pre;
            pre.tokens = toks.data();
            pre.n_tokens = (int) toks.size();
            pre.pos = 0;
            check(ws->prefill_chunk(pre).ok(), "the packed pack reads a wide window");
            std::string ferr;
            if (auto* fwd = ws->forward(ferr)) {
                const uint64_t held = fwd->memo_bytes();
                const uint64_t charged = basalt::models::glm5next::forward_footprint_bytes(m.geo(), m.pack());
                check(held > 0, "the packed pack leaves the pass weight copies to compare");
                check(held <= charged, "the packed pack's copies stay inside the footprint it is charged for");
                std::printf("  packed pack: charged %llu KiB, the pass holds %llu KiB\n",
                            (unsigned long long) (charged >> 10), (unsigned long long) (held >> 10));
            } else {
                check(false, "the packed session built its forward pass");
            }
        }
    }

    // --- a multi-token window: positions 0 and 1 together, which is also what completes the first DSA pool
    {
        const std::vector<int32_t> two = {0, 1};
        basalt::model::PrefillRequest pre;
        pre.tokens = two.data();
        pre.n_tokens = 2;
        pre.pos = 0;
        const auto r = sess->prefill_chunk(pre);
        check(r.ok(), "a two-token prompt chunk runs");
        if (!r.ok()) std::printf("  prefill: %s\n", r.message.c_str());
    }

    // --- decode at position 2, where the indexer now has a completed pool to score
    std::vector<float> at2(vocab, 0.f);
    {
        const std::vector<int32_t> tok = {2};
        std::vector<float> l(vocab, 0.f);
        basalt::model::DecodeRequest dr;
        dr.tokens = tok.data();
        dr.n_tokens = 1;
        dr.pos = 2;
        dr.logits = l.data();
        const auto r = sess->decode_window(dr);
        check(r.ok(), "a token at position 2 runs, with a pool complete");
        if (!r.ok()) std::printf("  decode pos 2: %s\n", r.message.c_str());
        at2 = l;
    }
    check(all_finite(at2), "the logits at position 2 are finite");
    check(std::memcmp(at2.data(), first.data(), vocab * sizeof(float)) != 0,
          "position 2 answers differently from position 0, so the caches are read, not ignored");

    // --- the caches themselves moved: the KDA recurrent state is zeroed on alloc, so a run has to leave it non-zero
    {
        auto& caches = sess->caches();
        std::vector<float> state((size_t) model.geo().n_head * model.geo().kda_head_dim *
                                 model.geo().kda_head_dim);
        check(cudaMemcpy(state.data(), caches.kda_state(0, 0), state.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
                  cudaSuccess,
              "read back the KDA state");
        bool any = false;
        for (float v : state) if (v != 0.f) any = true;
        check(any, "the KDA recurrent state holds what the run wrote");
        bool finite = true;
        for (float v : state) if (!std::isfinite(v)) finite = false;
        check(finite, "the KDA state is finite");
    }

    // --- the same pack a second time, with a card that can home nothing beyond the token's own picks: every other
    //     expert of every layer is then computed by the CPU pool from its pinned place, and the block's sum reads the
    //     pool's rows beside the card's. Two independent implementations of the same dot products (the pool's vec_dots,
    //     the card's grouped kernels over the packed rows), so the same window has to give the same answer to within
    //     the quantized activation's tolerance - and a row wired to the wrong expert shows up here as a wrong logit.
    {
        basalt::models::glm5next::GlmModel split_model;
        basalt::core::LoadPlan split_reads;
        std::vector<std::unique_ptr<basalt::model::ModelSession>> split_sessions;
        basalt::model::LoadRequest split_req = req;
        split_req.diag_tier_vram = model.geo().n_used;   // the fewest slots the tier accepts; the rest live in RAM
        const auto sr = split_model.plan(split_req, split_reads, split_sessions);
        check(sr.ok(), "a start with a capped card plans");
        if (!sr.ok()) std::printf("  capped plan: %s\n", sr.message.c_str());
        check(loader.run(split_reads, err), "the loader places the capped plan");
        check(split_model.attach(split_req, split_sessions).ok(), "the capped start attaches");
        if (!split_sessions.empty()) {
            auto* ssess = static_cast<basalt::models::glm5next::GlmSession*>(split_sessions[0].get());
            std::vector<float> sfirst(vocab, 0.f), sat2(vocab, 0.f);
            {
                basalt::model::DecodeRequest dr;
                dr.tokens = one.data();
                dr.n_tokens = 1;
                dr.pos = 0;
                dr.logits = sfirst.data();
                check(ssess->decode_window(dr).ok(), "the capped card runs the same first token");
            }
            {
                const std::vector<int32_t> two = {0, 1};
                basalt::model::PrefillRequest pre;
                pre.tokens = two.data();
                pre.n_tokens = 2;
                pre.pos = 0;
                check(ssess->prefill_chunk(pre).ok(), "the capped card runs the same prompt chunk");
            }
            {
                const std::vector<int32_t> tok = {2};
                basalt::model::DecodeRequest dr;
                dr.tokens = tok.data();
                dr.n_tokens = 1;
                dr.pos = 2;
                dr.logits = sat2.data();
                check(ssess->decode_window(dr).ok(), "the capped card runs the same token at position 2");
            }
            check(all_finite(sfirst) && all_finite(sat2), "the capped card's logits are finite");
            // the card's experts go through a q8_1 activation and the pool's through its own, so the two agree to
            // within that, not bit for bit; the pick they lead to has to be the same token. The split's measured
            // spread is 0.4% at the first token and 2.5% two positions in, which the FFN branch only started to show
            // once it landed in the residual stream: before that, whatever the two paths made of an expert never
            // reached the logits at all, and this comparison passed on nothing.
            auto worst = [&](const std::vector<float>& a, const std::vector<float>& b) {
                double scale = 0.0, diff = 0.0;
                for (size_t i = 0; i < a.size(); ++i) {
                    scale = std::max(scale, (double) std::fabs(a[i]));
                    diff = std::max(diff, (double) std::fabs(a[i] - b[i]));
                }
                return scale > 0.0 ? diff / scale : 0.0;
            };
            const double w0 = worst(first, sfirst), w2 = worst(at2, sat2);
            std::printf("      tier split: logits differ by %.3g (first token) and %.3g (position 2)\n", w0, w2);
            check(w0 < 0.04 && w2 < 0.04, "the pool's rows and the card's rows give the same logits");
            check(argmax(sfirst) == argmax(first) && argmax(sat2) == argmax(at2),
                  "both tier arrangements pick the same token");
        }
    }

    // --- bounds: a window past the cache is refused, and an empty one is refused
    {
        basalt::model::DecodeRequest dr;
        dr.tokens = one.data();
        dr.n_tokens = 1;
        dr.pos = (int64_t) sess->context();
        check(!sess->decode_window(dr).ok(), "a window past the cache is refused");
        dr.pos = 0;
        dr.n_tokens = 0;
        check(!sess->decode_window(dr).ok(), "an empty window is refused");
        // an id outside the pack's vocab cannot address an embedding row
        const std::vector<int32_t> bad = {vocab};
        dr.tokens = bad.data();
        dr.n_tokens = 1;
        check(!sess->decode_window(dr).ok(), "a token id outside the vocab is refused");
    }

    // --- the same tokens as one window and one token at a time, in both formats: a causal model cannot tell the two
    //     shapes apart, so what is left over is the kernels. Up to the narrow product's eight columns a window runs
    //     that very product, whose multi-column mode is pinned bitwise equal to a single column, so the answers are
    //     identical to the bit. Past those columns a packed window runs the tensor-core MMQ product instead: the same
    //     pack bytes and a different accumulation, and the router's top-k turns the rounding into a different expert
    //     set, so the two shapes drift by a fraction of the logit scale instead of not at all. The fp32 pack runs one
    //     product at either shape, so it drifts by nothing - and a stride or a layout mistake moves exactly that.
    //     Each comparison runs at two window lengths, either side of that eight-column line.
    auto compare = [&](const std::string& twin, const std::vector<int32_t>& wide, bool same_kernel) {
        std::vector<float> a(vocab, 0.f), b(vocab, 0.f);
        bool both = true;
        int which = 0;
        for (const std::string& p : {path, std::string(twin)}) {
            basalt::models::glm5next::GlmModel m;
            basalt::core::LoadPlan plan;
            std::vector<std::unique_ptr<basalt::model::ModelSession>> ss;
            basalt::model::LoadRequest lr;
            lr.paths = {p};
            std::string werr;
            const bool good = m.plan(lr, plan, ss).ok() && basalt::core::Loader{}.run(plan, werr) &&
                              m.attach(lr, ss).ok() && !ss.empty();
            if (!good) {
                both = false;
                std::printf("  %s: %s\n", p.c_str(), werr.empty() ? "could not open" : werr.c_str());
                continue;
            }
            auto* ws = static_cast<basalt::models::glm5next::GlmSession*>(ss[0].get());
            basalt::model::PrefillRequest pre;
            pre.tokens = wide.data();
            pre.n_tokens = (int) wide.size();
            pre.pos = 0;
            const auto r = ws->prefill_chunk(pre);
            if (!r.ok()) {
                both = false;
                std::printf("  %s: %s\n", p.c_str(), r.message.c_str());
                continue;
            }
            float* dst = which++ == 0 ? a.data() : b.data();
            auto* fwd = ws->forward(werr);
            both = both && fwd && fwd->copy_logits(dst, werr);
        }
        check(both, (std::to_string(wide.size()) + " tokens run through each of the two formats").c_str());
        double sq = 0.0, sd = 0.0, worst = 0.0;
        for (int i = 0; i < vocab; ++i) {
            const double d = (double) a[i] - b[i];
            sq += (double) a[i] * a[i];
            sd += d * d;
            worst = std::max(worst, std::fabs(d));
        }
        const double rms = std::sqrt(sq / vocab), drift = std::sqrt(sd / vocab);
        std::printf("  fp32 vs packed, %d tokens in one window: max |diff| = %.3e, RMS %.3e, logit RMS %.3e\n",
                    (int) wide.size(), worst, drift, rms);
        check(argmax(a) == argmax(b), "the wide packed product picks the same token as the fp32 one");

        // The formats differ by their quantization error, which random weights amplify and nothing here is calibrated
        // to. The same tokens one at a time go through the narrow product (mmvq, up to eight columns), and a
        // causal model cannot tell a window of twelve from twelve windows of one. Run for both formats, so the two axes
        // separate: one format across the two window shapes is the shapes alone, one shape across the formats is the
        // quantization alone. The fp32 pack runs the same fp32 product at either shape, so its two runs differ by the
        // state a window leaves and nothing else - which is the axis the packed comparison cannot separate on its own.
        {
            std::vector<std::vector<float>> narrow;
            for (int which = 0; which < 2; ++which) {
                const std::string pack = which ? std::string(twin) : path;
                basalt::models::glm5next::GlmModel m;
                basalt::core::LoadPlan plan;
                std::vector<std::unique_ptr<basalt::model::ModelSession>> ss;
                basalt::model::LoadRequest lr;
                lr.paths = {pack};
                std::string werr;
                const bool good = m.plan(lr, plan, ss).ok() && basalt::core::Loader{}.run(plan, werr) &&
                                  m.attach(lr, ss).ok() && !ss.empty();
                check(good, which ? "the packed twin opens again for the narrow comparison"
                                  : "the fp32 pack opens again for the narrow comparison");
                if (!good) continue;
                auto* ws = static_cast<basalt::models::glm5next::GlmSession*>(ss[0].get());
                std::vector<float> out(vocab, 0.f);
                for (int t = 0; t < (int) wide.size(); ++t) {
                    const int32_t tok = wide[(size_t) t];
                    basalt::model::DecodeRequest dr;
                    dr.tokens = &tok;
                    dr.n_tokens = 1;
                    dr.pos = t;
                    dr.logits = out.data();  // the last window's logits land here each time
                    const auto r = ws->decode_window(dr);
                    if (!r.ok()) {
                        std::printf("  narrow token %d: %s\n", t, r.message.c_str());
                        break;
                    }
                }
                narrow.push_back(std::move(out));
            }
            if (narrow.size() == 2u) {
                auto drift = [&](const char* what, const std::vector<float>& u, const std::vector<float>& v) {
                    double sd = 0.0, worst = 0.0;
                    for (size_t i = 0; i < u.size(); ++i) {
                        const double d = (double) u[i] - v[i];
                        sd += d * d;
                        worst = std::max(worst, std::fabs(d));
                    }
                    const double r = std::sqrt(sd / u.size());
                    std::printf("  %s: max |diff| = %.3e (RMS %.3e)\n", what, worst, r);
                    return r;
                };
                const std::string n = std::to_string(wide.size());
                const double d_fp32 = drift((n + " tokens, fp32, one window vs one at a time").c_str(), a, narrow[0]);
                const double d_pack = drift((n + " tokens, packed, one window vs one at a time").c_str(), b, narrow[1]);
                drift((n + " tokens, one at a time, fp32 vs packed").c_str(), narrow[0], narrow[1]);
                check(d_fp32 <= 1e-3 * rms + 1e-6,
                      (n + " tokens answer as one window like they do one at a time, unquantized").c_str());
                // One product at both shapes answers to the bit; two products answer within their rounding of each
                // other, which the router's top-k then amplifies into a different expert set.
                check(same_kernel ? d_pack == 0.0 : d_pack < 0.25 * rms + 1e-3,
                      (n + " tokens answer as one window like they do one at a time, packed" +
                       (same_kernel ? std::string(", to the bit")
                                    : std::string(", within the two products' rounding")))
                          .c_str());
            }
        }
    };
    if (const char* twin = std::getenv("GLM5NEXT_TINY_PACK_PACKED"); twin && *twin) {
        const std::vector<int32_t> wide = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};  // past the narrow product's columns
        compare(twin, wide, false);
        compare(twin, std::vector<int32_t>(wide.begin(), wide.begin() + 8), true);  // within them: one kernel, both shapes
    } else {
        std::printf("  SKIP: no GLM5NEXT_TINY_PACK_PACKED twin, so the wide and narrow products are not compared\n");
    }

    // A drafted window is checked one row at a time, so every row owes a pick. A causal model answers row i of a window
    // exactly as it answers the same prefix one token at a time, and that is the only assertion which can tell the rows
    // were read in order: an off-by-one row or a wrong logit stride still answers with a plausible token.
    {
        const std::vector<int32_t> toks = {0, 1, 2};
        std::vector<int32_t> rows(toks.size(), -1);
        int32_t last_pick = -1;
        basalt::model::LoadRequest lr;
        lr.paths = {path};
        basalt::models::glm5next::GlmModel wm;
        basalt::core::LoadPlan wplan;
        std::vector<std::unique_ptr<basalt::model::ModelSession>> wss;
        std::string werr;
        const bool wgood = wm.plan(lr, wplan, wss).ok() && basalt::core::Loader{}.run(wplan, werr) &&
                           wm.attach(lr, wss).ok() && !wss.empty();
        check(wgood, "the pack opens again for the row picks");
        if (wgood) {
            basalt::model::DecodeRequest dr;
            dr.tokens = toks.data();
            dr.n_tokens = (int) toks.size();
            dr.pos = 0;
            dr.row_tokens = rows.data();
            dr.next_token = &last_pick;
            check(wss[0]->decode_window(dr).ok(), "a decode window answers with a pick for each of its rows");
            check(rows.back() == last_pick, "a window's last row's pick is also its next token");

            std::vector<int32_t> steps;
            basalt::models::glm5next::GlmModel sm;
            basalt::core::LoadPlan splan;
            std::vector<std::unique_ptr<basalt::model::ModelSession>> sss;
            std::string serr;
            const bool sgood = sm.plan(lr, splan, sss).ok() && basalt::core::Loader{}.run(splan, serr) &&
                               sm.attach(lr, sss).ok() && !sss.empty();
            check(sgood, "the pack opens again for the same prefixes one at a time");
            if (sgood) {
                for (size_t t = 0; t < toks.size(); ++t) {
                    basalt::model::DecodeRequest sr;
                    sr.tokens = &toks[t];
                    sr.n_tokens = 1;
                    sr.pos = (int64_t) t;
                    int32_t one = -1;
                    sr.next_token = &one;
                    if (!sss[0]->decode_window(sr).ok()) break;
                    steps.push_back(one);
                }
                check(steps.size() == toks.size(), "each prefix ran on its own");
                for (size_t t = 0; t < steps.size() && t < rows.size(); ++t)
                    check(steps[t] == rows[t],
                          ("row " + std::to_string(t) + " of the window picks what the same prefix picks alone")
                              .c_str());
            }
        }
    }

    if (g_failures != 0) {
        std::printf("FORWARD: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("FORWARD: ok (%d vocab, structure and finiteness only - not output quality)\n", vocab);
    return 0;
}
