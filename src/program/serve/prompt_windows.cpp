// src/program/serve/prompt_windows.cpp - request prompt_windows operations.
#include "request.hpp"

namespace basalt::program::serve {

// A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
// header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path has
// a high fixed cost per run however few tokens it has (it streams every expert the chunk routes to that is
// not in VRAM over PCIe) and it borrows slots it must refill after; a window costs little per token, with
// the misses on the CPU.  Each part below is decided on its own, so a long first message is read batched
// and its header still goes through the windows.  Picture rows need the batched path.
bool Request::windows_ok(int64_t a, int64_t b) {
    if ((in_slot && !solo_req) || b - a > runtime_.o.short_read) return false;   // (the verifier is bound to this request's state)
    if (runtime_.sp.embd_rows != nullptr)
        for (int64_t i = a; i < b; ++i)
            if (runtime_.sp.embd_rows[i] != nullptr) return false;
    return true;
}

// tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
bool Request::read_windows(int64_t a, int64_t b, std::string& e) {
    basalt::core::progress_at("reading the prompt (verify windows), from token", a);   // not "batched"
    // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
    struct NoHeadSampling {
        basalt::core::Verifier& v;
        explicit NoHeadSampling(basalt::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
        ~NoHeadSampling() { v.set_head_sampling(true); }
    } no_head_sampling(runtime_.ver);
    std::vector<int32_t> win((size_t) runtime_.S), outw((size_t) runtime_.S), nxt((size_t) runtime_.S);
    for (int64_t q = a; q < b;) {
        if (runtime_.halted()) { e = "cancelled"; return false; }
        const int T = (int) std::min<int64_t>(runtime_.S, b - q);
        for (int t = 0; t < T; ++t) {
            win[(size_t) t] = (int32_t) runtime_.cur[(size_t) (q + t)];
            nxt[(size_t) t] = (int32_t) runtime_.cur[(size_t) (q + t + 1)];
        }
        runtime_.drive.d.layers = 0;
        runtime_.drive.d.experts = 0;
        runtime_.drive.d.failed = false;
        if (!runtime_.ver.run(T, win.data(), q, runtime_.win_pool_fn, runtime_.win_pool_user, outw.data(), e) || runtime_.drive.d.failed) {
            if (runtime_.drive.d.failed && runtime_.drive.d.fail) e = runtime_.drive.d.fail;
            return false;
        }
        if (!runtime_.ver.commit(T, e) || (runtime_.mtp_on && !runtime_.mtp.prefill(runtime_.ver.final_R_all(), nxt.data(), T, q, e))) return false;
        q += T;
        runtime_.pp_reached = q;
    }
    // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
    if (!runtime_.ver.wait_commit(e)) return false;
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - runtime_.pp_t0).count();
    std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) runtime_.pp_total, ms,
                ms > 0.0 ? 1000.0 * (double) (b - runtime_.pp_from) / ms : 0.0);
    basalt::core::progress_beat();
    std::fflush(stdout);
    return true;
}

}  // namespace basalt::program::serve
