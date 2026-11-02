// src/program/serve/request_image.cpp - request prepare_image operations.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::prepare_image() {
    ppl = nullptr;
    if (req_ppl >= 0) {
        std::string pe2;
        basalt::program::PplGate::configure(runtime_.ppl_base, runtime_.ppl_out);
        ppl = &basalt::program::PplGate::get();
        if (geni || !ppl->begin(req_ppl, ids.data(), n, runtime_.n_vocab, pe2)) {
            std::printf("ERR %s\n", geni ? "ppl: text only" : pe2.c_str());
            return Flow::next;
        }
        req_temperature = 0.0f;
        req_ppl_t = std::clamp(req_ppl_t, 1, runtime_.S);
    }
    if (geni || !runtime_.mrope_identity) {
        // positions for every cell this request can reach; the identity again for a text request
        std::string ve;
        runtime_.row_ptr.assign((size_t) n, nullptr);
        const int64_t cells = (int64_t) runtime_.mrope_host.size() / 3;
        auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
            runtime_.mrope_host[(size_t) c * 3] = (int32_t) t;
            runtime_.mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
            runtime_.mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
        };
        if (!geni) {
            for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
        } else {
            struct Img { int64_t n, nx, ny; size_t off; };
            std::vector<Img> imgs;
            runtime_.img_rows.clear();
            std::FILE* f = std::fopen(emb_path.c_str(), "rb");
            if (!f) ve = "cannot open " + emb_path;
            while (f && ve.empty()) {
                int32_t hdr[5];
                const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                if (got == 0) break;
                if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                    (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) runtime_.g.n_embd) {
                    ve = "bad embeddings file (expected basalt-vision records of width " +
                         std::to_string((long long) runtime_.g.n_embd) + ")";
                    break;
                }
                const size_t off = runtime_.img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                runtime_.img_rows.resize(off + cnt);
                if (std::fread(runtime_.img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                imgs.push_back({hdr[1], hdr[2], hdr[3], off});
            }
            if (f) std::fclose(f);
            // the table holds one entry per cell: a longer prompt is refused before any is written
            if (ve.empty() && (n > cells || n > runtime_.o.max_context))
                ve = "prompt (" + std::to_string((long long) n) + " tokens) exceeds the context (" +
                     std::to_string((long long) runtime_.o.max_context) + ")";
            int64_t p = 0, i = 0;
            size_t k = 0;
            while (ve.empty() && i < n) {
                if (ids[(size_t) i] != runtime_.kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                const Img& im = imgs[k++];
                {   // what the conversation cache compares: a picture is its grid and its embeddings
                    const int64_t grid[3] = {im.n, im.nx, im.ny};
                    uint64_t h = fnv1a(grid, sizeof grid);
                    h = fnv1a(runtime_.img_rows.data() + im.off, (size_t) im.n * (size_t) runtime_.g.n_embd * sizeof(float), h);
                    runtime_.req_imgs.push_back({i, h});
                }
                for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                    if (i + j >= n || ids[(size_t) (i + j)] != runtime_.kImagePad)
                        ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                             " rows but fewer <|image_pad|> tokens";
                for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                    const int64_t y = j / im.nx, x = j % im.nx;
                    put(i + j, p, p + y, p + x);
                    runtime_.row_ptr[(size_t) (i + j)] = runtime_.img_rows.data() + im.off + (size_t) j * (size_t) runtime_.g.n_embd;
                }
                i += im.n;
                p += std::max(im.nx, im.ny);
            }
            if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
            if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == runtime_.kImagePad) ve = "the prompt cannot end in an image";
            for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
        }
        cudaDeviceSynchronize();
        // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
        auto upload_mrope = [&]() -> bool {
            bool ok = cudaMemcpy(runtime_.d_mrope, runtime_.mrope_host.data(), runtime_.mrope_host.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice) == cudaSuccess;
            for (auto& st : runtime_.stages) {
                const basalt::core::OnDevice on(st->dev);
                cudaDeviceSynchronize();
                ok = ok && cudaMemcpy(st->mrope, runtime_.mrope_host.data(), runtime_.mrope_host.size() * sizeof(int32_t),
                                      cudaMemcpyHostToDevice) == cudaSuccess;
            }
            return ok;
        };
        if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
        if (!ve.empty()) {
            // leave the table as the identity so the next text request is untouched
            for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
            upload_mrope();
            runtime_.mrope_identity = true;
            std::printf("ERR %s\n", ve.c_str());
            std::fflush(stdout);
            return Flow::next;
        }
        runtime_.mrope_identity = !geni;
    }
    runtime_.sp.embd_rows = geni ? runtime_.row_ptr.data() : nullptr;
    if (from_text && req_clip && runtime_.admit_slot < 0) {
        const long long room = (long long) (runtime_.o.max_context - 8 - n);
        if (room < 1) {
            std::printf("ERR prompt (%lld tokens) leaves no room to answer in the context (%lld)\n", (long long) n,
                        (long long) runtime_.o.max_context);
            std::fflush(stdout);
            return Flow::next;
        }
        max_new = std::min(max_new, room);
    }
    {
        const int64_t ctx = runtime_.admit_slot >= 0 ? runtime_.slot_cap[(size_t) runtime_.admit_slot] : runtime_.o.max_context;
        const long long want = runtime_.admit_slot >= 0 ? runtime_.admit_max_new : (long long) max_new;   // a slot's request: its own max_new
        if (n + want + 8 > ctx) {
            if (runtime_.admit_slot >= 0)
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds slot %d's context (%lld)\n", (long long) n,
                            want, runtime_.admit_slot, (long long) ctx);
            else
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            want, (long long) ctx);
            std::fflush(stdout);
            runtime_.admit_slot = -1;
            return Flow::next;
        }
    }
    if (from_text) { std::printf("PT %lld %lld\n", (long long) n, (long long) max_new); std::fflush(stdout); }
    bad = false;
    for (int64_t t : ids) bad = bad || t < 0 || t >= runtime_.n_vocab;
    if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); return Flow::next; }

    return Flow::proceed;
}

}  // namespace basalt::program::serve
