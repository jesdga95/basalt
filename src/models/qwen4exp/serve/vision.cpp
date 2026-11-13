// src/models/qwen4exp/serve/vision.cpp - vision operations for the persistent engine.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_state.hpp"
#include "basalt/vision/video.hpp"

#include <future>

namespace basalt::models::qwen4exp::serve {

// ---- ENC <image> <out> [<image> <out> ...] and ENCV <video> <out> [k=v ...] (--vision), between requests and
// between batch windows.  The images (or the video's sampled frames) are decoded on the host first; then the
// encoder borrows the last expert-cache slots (the prompt path's way: their rows are marked missing for the time),
// uploads its weights from the pinned copy, encodes each picture (or each two-frame group) on its own stream, and
// the slots get their experts back from RAM before anything else reads the cache.  One reply line per image,
// `ENC OK <n> <nx> <ny> <ms>`; for a video one line, `ENC OK <groups> <nx> <ny> <ms> <time,time,...>`, and its file
// holds one record per group; or a single `ENC ERR <message>`.
void QfnRuntime::enc_command(const std::string& cmd) {
    auto reply_err = [&](const std::string& m, bool logged = false) {
        if (!logged) basalt::log::warn("basalt serve", "image encoder: refused: %s", m.c_str());
        std::printf("ENC ERR %s\n", m.c_str());
        std::fflush(stdout);
    };
    if (!o.vision_in_engine) { reply_err("this engine was started without --vision"); return; }
    const bool video = cmd.rfind("ENCV ", 0) == 0;
    if (pf_parts.size() != 1 || pf_parts[0].first < 0 || !stages.empty()) {
        reply_err("the in-engine image encoder needs one GPU's expert cache with a prompt lend region");
        return;
    }
    std::vector<std::string> words;
    if (!video) {
        size_t pos = 3;
        for (;;) {
            pos = cmd.find_first_not_of(' ', pos);
            if (pos == std::string::npos) break;
            const size_t end = cmd.find(' ', pos);
            words.push_back(cmd.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
            if (end == std::string::npos) break;
            pos = end;
        }
    }
    if (!video && (words.empty() || words.size() % 2 != 0)) { reply_err("expected: ENC <image> <output> [<image> <output> ...]"); return; }
    const auto t0 = Clock::now();
    // a picture is one group of 768-wide rows; a video is `groups` groups of 1536-wide rows, `times` their times
    struct Job { std::string out; basalt::vision::Size sz; std::vector<float> patches; int groups = 1; std::string times; };
    std::vector<Job> jobs;
    const basalt::vision::Geometry geo;
    int max_patches = 0;
    std::string decoder;   // where a video's frames were decoded
    std::string borrow_note;   // the expert-cache memory lent to NVDEC for the decode, when it was
    double borrow_ms = 0, refill_ms = 0;
    if (video) {
        std::string vpath, opath, verr;
        basalt::vision::VideoOptions vo;
        vo.group_tokens = geo.max_tokens;
        cudaGetDevice(&vo.gpu);   // NVDEC on the encoder's card, in the engine's context: no context of FFmpeg's own
        vo.cuda_context = basalt::vision::current_context();
        basalt::vision::Video v;
        if (!basalt::vision::parse_encv(cmd, vpath, opath, vo, verr)) { reply_err(verr); return; }
        // NVDEC takes no more than the card has free above the floor of what the engine allocates after loading
        // (for the seconds of a decode it may use the reserve kept for other programs). When the stream needs more,
        // the expert cache lends it the memory, as SLEEP takes it: the tail segments' experts go to RAM homes, the
        // segments are unmapped, and after the decode they are mapped again and refilled in place, so the placement
        // comes back the same
        const int64_t floor_b = (int64_t) basalt::program::serve::vram_floor_mib() << 20;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t need = basalt::vision::video_nvdec_bytes(vpath);
        bool borrowed = false;
        // (only between replies: the cache's segments are shared with the elastic KV, and the engine changes its
        // size only with no slot decoding, as for VRAM and SLEEP; beside a decoding reply the video decodes on the CPU)
        if (need > 0 && need > (int64_t) free_b - floor_b && !batch_on() && xcache.segmented() && pf_parts.size() == 1 &&
            pf_parts[0].lent.empty()) {
            const auto tb = Clock::now();
            std::string be, quiet;
            borrowed = vram_command("VRAM " + std::to_string((long long) ((need + floor_b) >> 20) + 1), be, true, &quiet);
            borrow_ms = std::chrono::duration<double, std::milli>(Clock::now() - tb).count();
            if (!borrowed) borrow_note = ", no NVDEC room: " + be;
            cudaMemGetInfo(&free_b, &total_b);
        } else if (need > 0 && need > (int64_t) free_b - floor_b) {
            borrow_note = batch_on() ? ", a reply is decoding" : !xcache.segmented() ? ", the expert cache cannot lend memory (not segmented)"
                                                                         : ", a prompt's loan is out";
        }
        vo.gpu_spare = std::max<int64_t>(0, (int64_t) free_b - floor_b);
        bool windows_ok = true;
        if (borrowed) {
            // the cache is short of its slots: the decode runs alone, then the cache takes its memory back
            const bool ok = basalt::vision::read_video(vpath, vo, v, verr);
            const auto tr = Clock::now();
            std::string re, quiet;
            const bool back = vram_command("VRAM", re, true, &quiet);
            refill_ms = std::chrono::duration<double, std::milli>(Clock::now() - tr).count();
            borrow_note = ", " + std::to_string((long long) (need >> 20)) + " MiB lent to NVDEC (shrink " +
                          std::to_string((long long) borrow_ms) + " ms, refill " + std::to_string((long long) refill_ms) + " ms)";
            if (!back) { reply_err("giving the expert cache its memory back failed: " + re); return; }
            if (!ok) { reply_err(verr); return; }
        } else {
            // the decode runs on its own thread while the decoding slots keep their windows (a minute of 4K takes
            // seconds on the CPU; the slots stop only for the encode below, as for an image)
            std::future<bool> dec = std::async(std::launch::async, [&] { return basalt::vision::read_video(vpath, vo, v, verr); });
            while (dec.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                if (batch_on() && windows_ok) windows_ok = next_batch_window();
                else dec.wait();
            }
            if (!dec.get()) { reply_err(verr); return; }
        }
        if (!windows_ok) { reply_err("a batch window failed while the video was decoded"); return; }
        decoder = v.decoder;
        Job j;
        j.out = opath;
        j.sz = v.canvas;
        j.groups = v.groups();
        j.times = basalt::vision::times_text(basalt::vision::group_times(v.indices, v.fps));
        basalt::vision::patchify_video(v.frames, (int) v.indices.size(), v.canvas, geo, j.patches);
        max_patches = j.sz.w / 16 * (j.sz.h / 16);
        jobs.push_back(std::move(j));
    }
    for (size_t i = 0; i < words.size(); i += 2) {
        std::ifstream f(words[i], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const bool read_ok = static_cast<bool>(f);
        f.close();   // the server removes the input once it has the reply
        basalt::vision::Image img;
        std::string ierr;
        if (!read_ok || bytes.empty() || !basalt::vision::decode_image(bytes.data(), bytes.size(), img, ierr)) {
            reply_err("cannot read the image " + words[i]);
            return;
        }
        Job j;
        j.out = words[i + 1];
        j.sz = basalt::vision::fit_size(img.w, img.h, geo);
        std::vector<float> planar;
        basalt::vision::normalize_planar(basalt::vision::to_canvas(img, j.sz), planar);
        basalt::vision::patchify(planar, j.sz, geo, j.patches);
        max_patches = std::max(max_patches, j.sz.w / 16 * (j.sz.h / 16));
        jobs.push_back(std::move(j));
    }
    const int tokens = std::max(max_patches / 4, geo.min_tokens);
    const auto t_dec = Clock::now();
    // the work that touches the cache starts from a quiet cache and device
    std::string e;
    if (ba_job) { ba_thr.join(); ba_job = false; }
    if (!ver.wait_commit(e)) { reply_err(e); return; }
    apply_pending(true);
    if (!xq_drain()) { reply_err("an adaptive swap failed"); return; }
    if (cudaDeviceSynchronize() != cudaSuccess) { reply_err(std::string("device: ") + cudaGetErrorString(cudaGetLastError())); return; }
    PfPart& p = pf_parts[0];
    if (!p.lent.empty()) { reply_err("a prompt's loan is still out"); return; }
    const int64_t k = serve_loans::cache_slots_for(*p.cache, vis_enc.arena_bytes(tokens));
    // the room: the cache's tail down to the first slot whose expert has no RAM copy to refill it from (never
    // less than the prompt path's loan region, which always has one)
    auto lendable = [&]() -> int64_t {
        std::vector<uint8_t> in_ram;
        if (srcp == &src && src.complement_ready()) {
            in_ram.resize(host_res.size());
            for (size_t i = 0; i < host_res.size(); ++i)
                in_ram[i] = src.resident_blob((int64_t) i / g.n_expert, (int64_t) i % g.n_expert) != nullptr;
        }
        const int64_t floor = basalt::program::image_lend_floor(host_res, in_ram, p.cache->slots());
        int64_t r = p.cache->slots() - std::min<int64_t>(floor, p.first);
        if (batch_on()) r = std::min<int64_t>(r, kLendCapDecoding);
        return r;
    };
    int64_t room = lendable();
    // short of room: the elastic KV gives back the cells no session holds (idle slots first), and the cache
    // maps that VRAM again; the sessions take cells back on demand after the encode
    int64_t kv_given = 0;
    if (k > room && kv_floor > 0 &&
        !kv_give_back((int64_t) live.size(), nullptr, -1, [&] { room = lendable(); return k <= room; }, kv_given, e)) {
        reply_err(e);
        return;
    }
    if (k <= 0 || k > room) {
        basalt::log::warn("basalt serve", "image encoder: refused, %lld cache slots needed, %lld lendable "
                             "(cache %lld slots, %lld KV cells given back, the rest held by the sessions)",
                     (long long) k, (long long) room, (long long) p.cache->slots(), (long long) kv_given);
        reply_err("the image needs " + std::to_string((long long) k) + " expert-cache slots and " +
                  std::to_string((long long) room) + " can be lent" +
                  (kv_floor > 0 ? ", with every spare KV cell given back" : ""), true);
        return;
    }
    if (kv_given > 0)
        basalt::log::info("basalt serve", "image encoder: %lld KV cells given back for %lld cache slots",
                     (long long) kv_given, (long long) k);
    const int32_t first = (int32_t) (p.cache->slots() - k);
    for (int64_t l = p.lb; l < p.le; ++l)
        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
            const size_t i = (size_t) (l * g.n_expert + ex);
            if (host_res[i] >= first) {
                p.lent.emplace_back((int32_t) i, host_res[i]);
                host_res[i] = basalt::core::kNotResident;
            }
        }
    res_upload();
    const auto t_lent = Clock::now();
    std::vector<std::string> done;   // the reply lines
    int groups = 0;
    std::string eerr;
    bool ok = vis_enc.attach(p.cache->device_slot(first), part_bytes(p, first), tokens, eerr);
    if (!ok && eerr.empty()) eerr = "the encoder could not be placed";
    const auto t_up = Clock::now();
    for (size_t i = 0; ok && i < jobs.size(); ++i) {
        const Job& j = jobs[i];
        const auto ti = Clock::now();
        const int nx = j.sz.w / 32, ny = j.sz.h / 32, n = nx * ny, n_embd = (int) g.n_embd;
        std::FILE* fo = std::fopen(j.out.c_str(), "wb");
        if (!fo) { ok = false; eerr = "cannot write " + j.out; break; }
        const size_t per = j.patches.size() / (size_t) j.groups;
        for (int gi = 0; ok && gi < j.groups; ++gi) {
            std::vector<float> out, part;
            if (j.groups > 1) part.assign(j.patches.begin() + (ptrdiff_t) (gi * per), j.patches.begin() + (ptrdiff_t) ((gi + 1) * per));
            if (!vis_enc.encode(j.groups > 1 ? part : j.patches, j.sz, out)) { ok = false; eerr = "the vision encoder failed: " + vis_enc.last_error(); break; }
            const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
            const bool wrote = (int) (out.size() / (size_t) n_embd) == n && std::fwrite(hdr, sizeof hdr, 1, fo) == 1 &&
                               std::fwrite(out.data(), sizeof(float) * (size_t) n_embd, (size_t) n, fo) == (size_t) n;
            if (!wrote) { ok = false; eerr = "short write to " + j.out; }
        }
        std::fclose(fo);
        if (!ok) break;
        groups += j.groups;
        char line[96];
        std::snprintf(line, sizeof line, "ENC OK %d %d %d %.0f", video ? j.groups : n, nx, ny,
                      std::chrono::duration<double, std::milli>(Clock::now() - ti).count());
        done.push_back(std::string(line) + (video ? " " + j.times : ""));
    }
    vis_enc.detach();
    const auto t_enc = Clock::now();
    // the slots get their experts back from RAM, in the same slots, then the table goes up again
    for (const auto& [i, slot] : p.lent) {
        const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
        const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
        if (b == nullptr || !p.cache->fill_slot_queued(slot, b, e, nb)) {
            basalt::log::error("basalt serve", "image encoder: refilling the expert cache failed: %s", e.c_str());
            reply_err("refilling the expert cache failed: " + e, true);
            return;
        }
        host_res[(size_t) i] = slot;
    }
    if (!p.cache->sync_queued(e)) {
        basalt::log::error("basalt serve", "image encoder: refilling the expert cache failed: %s", e.c_str());
        reply_err("refilling the expert cache failed: " + e, true);
        return;
    }
    const int64_t n_lent = (int64_t) p.lent.size();
    p.lent.clear();
    p.lent_chunk = 0;
    p.first_now = -1;   // the prompt path lays its buffers out again at its next loan
    res_upload();
    const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    if (video)
        basalt::log::info("basalt serve", "video encoder: %d groups of %d tokens in %lld cache slots (%.2f GiB), "
                             "%lld expert rows lent; decode %.0f ms (%s), lend %.0f ms, weights %.0f ms, encode %.0f ms, refill %.0f ms",
                     groups, tokens, (long long) k, (double) vis_enc.arena_bytes(tokens) / 1073741824.0, (long long) n_lent,
                     ms(t0, t_dec), (decoder + borrow_note).c_str(), ms(t_dec, t_lent), ms(t_lent, t_up), ms(t_up, t_enc), ms(t_enc, Clock::now()));
    else
        basalt::log::info("basalt serve", "image encoder: %zu image(s) up to %d tokens in %lld cache slots (%.2f GiB), "
                             "%lld expert rows lent; decode %.0f ms, lend %.0f ms, weights %.0f ms, encode %.0f ms, refill %.0f ms",
                     jobs.size(), tokens, (long long) k, (double) vis_enc.arena_bytes(tokens) / 1073741824.0, (long long) n_lent,
                     ms(t0, t_dec), ms(t_dec, t_lent), ms(t_lent, t_up), ms(t_up, t_enc), ms(t_enc, Clock::now()));
    if (!ok) { reply_err(eerr); return; }
    for (const std::string& d : done) std::printf("%s\n", d.c_str());
    std::fflush(stdout);
}

}  // namespace basalt::models::qwen4exp::serve
