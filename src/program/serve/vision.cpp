// src/program/serve/vision.cpp - vision operations for the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

// ---- ENC <image> <out> [<image> <out> ...] (--vision-mmproj), between requests and between batch windows.  The
// images are decoded on the host first; then the encoder borrows the last expert-cache slots (the prompt path's
// way: their rows are marked missing for the time), uploads its weights from the pinned copy, encodes each
// image on its own stream, and the slots get their experts back from RAM before anything else reads the cache.
// One reply line per image: `ENC OK <n> <nx> <ny> <ms>`, or a single `ENC ERR <message>`.
void Runtime::enc_command(const std::string& cmd) {
    auto reply_err = [&](const std::string& m, bool logged = false) {
        if (!logged) basalt::log::warn("basalt serve", "image encoder: refused: %s", m.c_str());
        std::printf("ENC ERR %s\n", m.c_str());
        std::fflush(stdout);
    };
    if (o.vision_mmproj.empty()) { reply_err("this engine was started without --vision-mmproj"); return; }
    if (pf_parts.size() != 1 || pf_parts[0].first < 0 || !stages.empty()) {
        reply_err("the in-engine image encoder needs one GPU's expert cache with a prompt lend region");
        return;
    }
    std::vector<std::string> words;
    {
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
    if (words.empty() || words.size() % 2 != 0) { reply_err("expected: ENC <image> <output> [<image> <output> ...]"); return; }
    const auto t0 = Clock::now();
    struct Job { std::string out; basalt::vision::Size sz; std::vector<float> patches; };
    std::vector<Job> jobs;
    const basalt::vision::Geometry geo;
    int max_patches = 0;
    for (size_t i = 0; i < words.size(); i += 2) {
        std::ifstream f(words[i], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        basalt::vision::Image img;
        std::string ierr;
        if (!f || bytes.empty() || !basalt::vision::decode_image(bytes.data(), bytes.size(), img, ierr)) {
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
    std::vector<std::array<double, 4>> done;   // n, nx, ny, ms
    std::string eerr;
    bool ok = vis_enc.attach(p.cache->device_slot(first), part_bytes(p, first), tokens, eerr);
    if (!ok && eerr.empty()) eerr = "the encoder could not be placed";
    const auto t_up = Clock::now();
    for (size_t i = 0; ok && i < jobs.size(); ++i) {
        const Job& j = jobs[i];
        const auto ti = Clock::now();
        std::vector<float> out;
        if (!vis_enc.encode(j.patches, j.sz, out)) { ok = false; eerr = "the vision encoder failed: " + vis_enc.last_error(); break; }
        const int nx = j.sz.w / 32, ny = j.sz.h / 32, n = nx * ny, n_embd = (int) g.n_embd;
        std::FILE* fo = std::fopen(j.out.c_str(), "wb");
        if (!fo) { ok = false; eerr = "cannot write " + j.out; break; }
        const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
        const bool wrote = (int) (out.size() / (size_t) n_embd) == n && std::fwrite(hdr, sizeof hdr, 1, fo) == 1 &&
                           std::fwrite(out.data(), sizeof(float) * (size_t) n_embd, (size_t) n, fo) == (size_t) n;
        std::fclose(fo);
        if (!wrote) { ok = false; eerr = "short write to " + j.out; break; }
        done.push_back({(double) n, (double) nx, (double) ny,
                        std::chrono::duration<double, std::milli>(Clock::now() - ti).count()});
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
    basalt::log::info("basalt serve", "image encoder: %zu image(s) up to %d tokens in %lld cache slots (%.2f GiB), "
                         "%lld expert rows lent; decode %.0f ms, lend %.0f ms, weights %.0f ms, encode %.0f ms, refill %.0f ms",
                 jobs.size(), tokens, (long long) k, (double) vis_enc.arena_bytes(tokens) / 1073741824.0, (long long) n_lent,
                 ms(t0, t_dec), ms(t_dec, t_lent), ms(t_lent, t_up), ms(t_up, t_enc), ms(t_enc, Clock::now()));
    if (!ok) { reply_err(eerr); return; }
    for (const auto& d : done) std::printf("ENC OK %d %d %d %.0f\n", (int) d[0], (int) d[1], (int) d[2], d[3]);
    std::fflush(stdout);
}

}  // namespace basalt::program::serve
