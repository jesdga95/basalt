// src/program/serve/prompt_read.cpp - request prompt_read operations.
#include "request.hpp"

namespace basalt::program::serve {

bool Request::read_part(int64_t a0, int64_t b0, std::string& e) {
    // (a layer split reads its stages as a pipeline over one run's chunks: in pieces only beside slots)
    if (runtime_.o.batch <= 0 || runtime_.piped || (!runtime_.stages.empty() && !runtime_.batch_on()) || (b0 - a0 <= runtime_.kReadAlone && resumed_chunk <= 0)) {
        // no window runs meanwhile, so the CPU pool is free: it takes its share of the streamed experts
        const bool lend_pool = runtime_.o.batch > 0 && !runtime_.multi_gpu && !runtime_.o.no_pool;
        if (lend_pool) { runtime_.sp.set_cpu_pool(&runtime_.pool); runtime_.sp.set_cpu_share(runtime_.o.prefill_cpu_share); }
        const bool ok = runtime_.sp.run(ids.data() + a0, b0 - a0, a0, e);
        if (lend_pool) runtime_.sp.set_cpu_pool(nullptr);
        return ok;
    }
    if (runtime_.batch_on() && runtime_.stages.empty()) {
        // the whole range is read on a side thread, in the prompt path's own chunks, while this thread keeps
        // running the slots' windows; the work that touches the draft layer or the output runs here, between
        // two windows, when the prompt path asks for it
        if (!runtime_.xq_drain()) { e = "an adaptive refill failed"; return false; }
        runtime_.apply_pending(true);
        struct Mail {
            std::mutex mu;
            std::condition_variable cv;
            std::function<void()> job;
            bool job_done = false, fin = false;
        } mail;
        std::atomic<bool> abort_read{false};
        bool ok = false;
        std::string terr;
        int yield_req = -1;                 // a BYIELD <slot> not acted on yet
        int64_t yield_q = -1;               // the chunk boundary the read stopped at, to give way
        int yield_slot = -1;
        auto orig_chunk = runtime_.sp.on_chunk;
        auto orig_stop = runtime_.sp.should_stop;
        runtime_.sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
            bool r = false;
            std::unique_lock<std::mutex> lk(mail.mu);
            mail.job = [&] {
                r = orig_chunk ? orig_chunk(R_rows, T, p0, ce) : true;
                if (!r || yield_req < 0) return;
                // a shorter request waits: the read stops after this chunk (the draft layer has it) when the
                // slot can take what was read, or goes on
                const int ys = yield_req;
                yield_req = -1;
                if (can_yield(p0 + T, b0, ys, true)) {
                    yield_q = p0 + T;
                    yield_slot = ys;
                    abort_read.store(true);
                } else {
                    basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys,
                                 (long long) (p0 + T), (long long) n);
                }
            };
            mail.job_done = false;
            mail.cv.wait(lk, [&] { return mail.job_done; });
            return r;
        };
        runtime_.sp.should_stop = [&] { return abort_read.load() || (orig_stop && orig_stop()); };
        runtime_.ov_active = true;
        std::thread th([&] {
            const basalt::core::OnDevice on(0);
            ok = runtime_.sp.run(ids.data() + a0, b0 - a0, a0, terr);
            if (ok && runtime_.ov_post) {   // (the checkpoint's copy to the host beside the windows)
                ok = runtime_.ov_post();
                runtime_.ov_post_done = ok;
                if (!ok) terr = "saving a conversation checkpoint failed" + runtime_.ckpt_why;
            }
            std::lock_guard<std::mutex> lk(mail.mu);
            mail.fin = true;
            mail.cv.notify_all();
        });
        bool fatal = false;
        int64_t ov_jobs = 0, ov_wins = 0;
        double ov_job_ms = 0, ov_job_max = 0, ov_win_ms = 0, ov_win_max = 0;
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mail.mu);
                if (mail.job && !mail.job_done) job = mail.job;
                else if (mail.fin) break;
            }
            if (job) {
                const auto tj = Clock::now();
                job();
                const double jm = std::chrono::duration<double, std::milli>(Clock::now() - tj).count();
                ++ov_jobs; ov_job_ms += jm; ov_job_max = std::max(ov_job_max, jm);
                std::lock_guard<std::mutex> lk(mail.mu);
                mail.job = nullptr;
                mail.job_done = true;
                mail.cv.notify_all();
                continue;
            }
            {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                std::lock_guard<std::mutex> lk(runtime_.in_mu);
                for (auto it = runtime_.in_lines.begin(); it != runtime_.in_lines.end();) {
                    if (it->rfind("BSTOP ", 0) == 0) {
                        const int b = std::atoi(it->c_str() + 6);
                        if (b >= 0 && b < (int) runtime_.bs.size()) runtime_.bs[(size_t) b].stop = true;
                        it = runtime_.in_lines.erase(it);
                    } else if (it->rfind("BYIELD ", 0) == 0) {
                        yield_req = std::atoi(it->c_str() + 7);
                        it = runtime_.in_lines.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            if (!fatal && runtime_.batch_on()) {
                const auto tw = Clock::now();
                if (!runtime_.batch_step()) { fatal = true; abort_read.store(true); }
                const double wm = std::chrono::duration<double, std::milli>(Clock::now() - tw).count();
                ++ov_wins; ov_win_ms += wm; ov_win_max = std::max(ov_win_max, wm);
            } else {
                std::unique_lock<std::mutex> lk(mail.mu);
                mail.cv.wait_for(lk, std::chrono::milliseconds(2), [&] { return mail.fin || (mail.job && !mail.job_done); });
            }
        }
        th.join();
        runtime_.ov_active = false;
        basalt::log::info("basalt batch", "overlapped prompt read: %lld windows (%.1f ms avg, %.0f ms max) beside "
                             "%lld chunk jobs run between windows (%.1f ms avg, %.0f ms max)",
                     (long long) ov_wins, ov_wins ? ov_win_ms / (double) ov_wins : 0.0, ov_win_max,
                     (long long) ov_jobs, ov_jobs ? ov_job_ms / (double) ov_jobs : 0.0, ov_job_max);
        runtime_.sp.on_chunk = std::move(orig_chunk);
        runtime_.sp.should_stop = std::move(orig_stop);
        if (yield_q >= 0 && !fatal) {   // the read stopped at a chunk boundary to give way (BYIELD)
            std::string ye;
            const auto ty = Clock::now();
            if (yield_q >= 2) {   // a run that ends early leaves the token window of the PLE hash where it began
                runtime_.rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
                runtime_.rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
            }
            if (!park_read(yield_q, yield_slot, ye, runtime_.sp.chunk())) {
                e = "BYIELD: parking the prompt read failed: " + ye;
                return false;
            }
            basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                 "holds it (copied in %.1f ms)", (long long) yield_q, (long long) n, yield_slot,
                         std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
            e = "yield";
            return false;
        }
        if (fatal) { batch_fatal = true; e = "a batch window failed"; return false; }
        if (!ok) e = terr;
        return ok;
    }
    const int64_t C = std::max<int64_t>(runtime_.sp.chunk(), 1);
    // while slots are decoding the prompt goes in chunks of a fixed small size, so a long prompt holds the
    // slots' windows back for one short chunk at a time
    constexpr int64_t live_chunk = 1024;
    for (int64_t q = a0; q < b0;) {
        const int64_t r = std::min(b0, q + (runtime_.batch_on() ? std::min(C, live_chunk) : C));
        const auto tq = Clock::now();
        if (!runtime_.sp.run(ids.data() + q, r - q, q, e)) return false;
        q = r;
        if (q >= b0) continue;
        int ys = -1;
        {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
            std::lock_guard<std::mutex> lk(runtime_.in_mu);
            for (auto it = runtime_.in_lines.begin(); it != runtime_.in_lines.end();) {
                if (it->rfind("BSTOP ", 0) == 0) {
                    const int b = std::atoi(it->c_str() + 6);
                    if (b >= 0 && b < (int) runtime_.bs.size()) runtime_.bs[(size_t) b].stop = true;
                    it = runtime_.in_lines.erase(it);
                } else if (it->rfind("BYIELD ", 0) == 0) {
                    ys = std::atoi(it->c_str() + 7);
                    it = runtime_.in_lines.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (ys >= 0) {
            std::string ye;
            const auto ty = Clock::now();
            if (can_yield(q, b0, ys) && park_read(q, ys, ye)) {
                basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                     "holds it (copied in %.1f ms)", (long long) q, (long long) n, ys,
                             std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                e = "yield";
                return false;
            }
            basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld%s%s", ys, (long long) q,
                         (long long) n, ye.empty() ? "" : ": ", ye.c_str());
        }
        if (decode_share <= 0.0 || !runtime_.batch_on()) continue;
        const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
        const auto td = Clock::now();
        do {
            if (!runtime_.batch_step()) { batch_fatal = true; e = "a batch window failed"; return false; }
        } while (runtime_.batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
        ++il_parts;
        il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
    }
    return true;
}

}  // namespace basalt::program::serve
