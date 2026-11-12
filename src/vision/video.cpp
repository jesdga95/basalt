// src/vision/video.cpp - see basalt/vision/video.hpp.
#include "basalt/vision/video.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include "basalt/vision/ffmpeg_api.hpp"

namespace basalt::vision {

std::vector<int64_t> sample_indices(int64_t total, double src_fps, const VideoOptions& o) {
    if (total <= 0) return {};
    if (!(src_fps > 0)) src_fps = 24.0;
    // int(total / fps * target): the reference truncates
    int64_t n = (int64_t) ((double) total / src_fps * o.fps);
    n = std::min<int64_t>({std::max<int64_t>(n, o.min_frames), o.max_frames, total});
    std::vector<int64_t> idx((size_t) n);
    if (n == 1) { idx[0] = 0; return idx; }
    // numpy linspace(0, total - 1, n).round(): start + i * step, the last one exactly the end; round ties to even
    const double step = (double) (total - 1) / (double) (n - 1);
    for (int64_t i = 0; i < n; ++i) idx[(size_t) i] = (int64_t) std::nearbyint(i == n - 1 ? (double) (total - 1) : (double) i * step);
    return idx;
}

Size fit_video(int frames, int w, int h, const VideoOptions& o) {
    if (frames <= 0 || w <= 0 || h <= 0) return {0, 0};
    constexpr int f = 32;
    double hh = h, ww = w;
    if (h < f || w < f) {   // a frame under one merged token is scaled up first, aspect kept
        const double scale = std::max((double) f / h, (double) f / w);
        hh = (double) (int) (h * scale);
        ww = (double) (int) (w * scale);
    }
    if (std::max(hh, ww) / std::min(hh, ww) > 200) return {0, 0};
    const int64_t max_px = (int64_t) o.max_tokens * 2048, min_px = o.min_volume;
    const int64_t t_bar = (int64_t) std::nearbyint(frames / 2.0) * 2;
    int64_t hb = (int64_t) std::nearbyint(hh / f) * f, wb = (int64_t) std::nearbyint(ww / f) * f;
    const double vol = (double) frames * hh * ww;
    if (t_bar * hb * wb > max_px) {
        const double beta = std::sqrt(vol / (double) max_px);
        hb = std::max<int64_t>(f, (int64_t) std::floor(hh / beta / f) * f);
        wb = std::max<int64_t>(f, (int64_t) std::floor(ww / beta / f) * f);
    } else if (t_bar * hb * wb < min_px) {
        const double beta = std::sqrt((double) min_px / vol);
        hb = (int64_t) std::ceil(hh * beta / f) * f;
        wb = (int64_t) std::ceil(ww * beta / f) * f;
    }
    // one group is one encoder picture: a frame past its size is shrunk alone, aspect kept
    while ((hb / f) * (wb / f) > o.group_tokens) {
        const double beta = std::sqrt((double) hb * (double) wb / ((double) o.group_tokens * f * f));
        const int64_t h2 = std::max<int64_t>(f, (int64_t) std::floor(hb / beta / f) * f);
        const int64_t w2 = std::max<int64_t>(f, (int64_t) std::floor(wb / beta / f) * f);
        if (h2 == hb && w2 == wb) {   // floor already at one token per side on one axis
            if (hb > wb) hb -= f; else wb -= f;
        } else {
            hb = h2;
            wb = w2;
        }
    }
    return {(int) wb, (int) hb};
}

std::vector<double> group_times(const std::vector<int64_t>& indices, double src_fps) {
    if (!(src_fps > 0)) src_fps = 24.0;
    std::vector<int64_t> idx = indices;
    if (idx.size() % 2 != 0) idx.push_back(idx.back());
    std::vector<double> out;
    for (size_t i = 0; i + 1 < idx.size(); i += 2) {
        const double a = (double) idx[i] / src_fps, b = (double) idx[i + 1] / src_fps;
        out.push_back((a + b) / 2);
    }
    return out;
}

void patchify_video(const std::vector<uint8_t>& frames, int n, Size canvas, const Geometry& g, std::vector<float>& out) {
    const int P = g.patch, m = g.merge, gx = canvas.w / P, gy = canvas.h / P, groups = (n + 1) / 2;
    const size_t frame = (size_t) canvas.w * canvas.h * 3, row = (size_t) 3 * 2 * P * P;
    out.resize((size_t) groups * gx * gy * row);
    size_t r = 0;
    for (int t = 0; t < groups; ++t) {
        const uint8_t* fr[2] = {&frames[(size_t) (2 * t) * frame], &frames[(size_t) std::min(2 * t + 1, n - 1) * frame]};
        for (int yb = 0; yb < gy; yb += m)
            for (int xb = 0; xb < gx; xb += m)
                for (int dy = 0; dy < m; ++dy)
                    for (int dx = 0; dx < m; ++dx, ++r) {
                        float* d = &out[r * row];
                        const int y0 = (yb + dy) * P, x0 = (xb + dx) * P;
                        for (int c = 0; c < 3; ++c)
                            for (int k = 0; k < 2; ++k)
                                for (int ky = 0; ky < P; ++ky) {
                                    const uint8_t* s = fr[k] + ((size_t) (y0 + ky) * canvas.w + x0) * 3 + c;
                                    for (int kx = 0; kx < P; ++kx)
                                        *d++ = ((float) s[(size_t) kx * 3] - 127.5f) / 127.5f;
                                }
                    }
    }
}

Image video_frame_to_canvas(const Image& frame, Size canvas) { return resize_bicubic_to(frame, canvas.w, canvas.h); }

namespace {

// One open video stream and its decoder; closes both when it goes.
struct Reader {
    const Ffmpeg& F;
    ff::AVFormatContext* fmt = nullptr;
    ff::AVCodecContext* ctx = nullptr;
    ff::AVPacket* pkt = nullptr;
    ff::AVFrame* frm = nullptr;
    ff::AVStream* st = nullptr;
    int index = -1;
    bool eof = false;

    explicit Reader(const Ffmpeg& f) : F(f) {}
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    ~Reader() {
        if (frm != nullptr) F.av_frame_free(&frm);
        if (pkt != nullptr) F.av_packet_free(&pkt);
        if (ctx != nullptr) F.avcodec_free_context(&ctx);
        if (fmt != nullptr) F.avformat_close_input(&fmt);
    }

    bool open(const std::string& path, std::string& err) {
        int r = F.avformat_open_input(&fmt, path.c_str(), nullptr, nullptr);
        if (r < 0) { fmt = nullptr; err = "FFmpeg cannot open the video: " + F.error_text(r); return false; }
        if ((r = F.avformat_find_stream_info(fmt, nullptr)) < 0) { err = "FFmpeg cannot read the video: " + F.error_text(r); return false; }
        const ff::AVCodec* dec = nullptr;
        index = F.av_find_best_stream(fmt, ff::kMediaTypeVideo, -1, -1, &dec, 0);
        if (index < 0 || dec == nullptr) { err = "the file has no video stream FFmpeg can decode"; return false; }
        // the layout check: a wrong offset table shows here as a stream index or time base out of range
        const unsigned nb = ff::field<unsigned>(fmt, ff::kFmtNbStreams);
        ff::AVStream* const* streams = ff::field<ff::AVStream**>(fmt, ff::kFmtStreams);
        if ((unsigned) index >= nb || streams == nullptr || (st = streams[index]) == nullptr ||
            ff::field<ff::AVRational>(st, F.abi.st_time_base).den <= 0) {
            err = F.version + ": layout check failed";
            return false;
        }
        ctx = F.avcodec_alloc_context3(dec);
        pkt = F.av_packet_alloc();
        frm = F.av_frame_alloc();
        if (ctx == nullptr || pkt == nullptr || frm == nullptr) { err = "FFmpeg is out of memory"; return false; }
        if ((r = F.avcodec_parameters_to_context(ctx, ff::field<ff::AVCodecParameters*>(st, F.abi.st_codecpar))) < 0) {
            err = "FFmpeg cannot set up the decoder: " + F.error_text(r);
            return false;
        }
        ff::AVDictionary* opts = nullptr;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        F.av_dict_set(&opts, "threads", std::to_string(std::min(8u, hw)).c_str(), 0);
        r = F.avcodec_open2(ctx, dec, &opts);
        F.av_dict_free(&opts);
        if (r < 0) { err = "FFmpeg cannot open the decoder: " + F.error_text(r); return false; }
        return true;
    }

    /// The next decoded frame in `frm` (1), the end of the stream (0) or an error (-1, `err`).
    int next(std::string& err) {
        F.av_frame_unref(frm);
        for (;;) {
            const int r = F.avcodec_receive_frame(ctx, frm);
            if (r >= 0) return 1;
            if (r == ff::kErrorEof) return 0;
            if (r != ff::kErrorEagain) { err = "FFmpeg cannot decode the video: " + F.error_text(r); return -1; }
            if (eof) return 0;
            const int rr = F.av_read_frame(fmt, pkt);
            if (rr < 0) {   // the end of the file: drain the decoder
                eof = true;
                F.avcodec_send_packet(ctx, nullptr);
                continue;
            }
            if (ff::field<int>(pkt, ff::kPacketStreamIndex) == index) F.avcodec_send_packet(ctx, pkt);   // a damaged packet is skipped
            F.av_packet_unref(pkt);
        }
    }
};

}  // namespace

bool read_video(const std::string& path, const VideoOptions& o, Video& out, std::string& err) {
    std::string why;
    const Ffmpeg* F = ffmpeg(why);
    if (F == nullptr) { err = "this process cannot read video: " + why; return false; }
    const ff::AbiRow& a = F->abi;
    Reader rd(*F);
    if (!rd.open(path, err)) return false;
    const ff::AVRational tb = ff::field<ff::AVRational>(rd.st, a.st_time_base);
    const ff::AVCodecParameters* par = ff::field<ff::AVCodecParameters*>(rd.st, a.st_codecpar);
    const int pw = ff::field<int>(par, a.par_width), ph = ff::field<int>(par, a.par_height);
    const int64_t nb_frames = ff::field<int64_t>(rd.st, a.st_nb_frames);
    const int64_t sdur = ff::field<int64_t>(rd.st, a.st_duration), fdur = ff::field<int64_t>(rd.fmt, a.fmt_duration);
    const double avg = ff::q2d(ff::field<ff::AVRational>(rd.st, a.st_avg_frame_rate));
    const double rfr = ff::q2d(ff::field<ff::AVRational>(rd.st, a.st_r_frame_rate));
    const double fps = avg > 0 ? avg : rfr > 0 ? rfr : 24.0;
    const double seconds = sdur > 0 && sdur != ff::kNoPts ? (double) sdur * ff::q2d(tb)
                         : fdur > 0 && fdur != ff::kNoPts ? (double) fdur / ff::kTimeBase
                         : nb_frames > 0 ? (double) nb_frames / fps : 0.0;
    if (seconds > o.max_seconds) {
        err = "the video is " + std::to_string((int64_t) std::ceil(seconds)) + " s long; the limit is " +
              std::to_string((int64_t) o.max_seconds) + " s";
        return false;
    }
    int64_t total = nb_frames > 0 ? nb_frames : seconds > 0 ? (int64_t) std::llround(seconds * fps) : 0;
    if (total <= 0) {   // neither the container nor the length gives a count: decode once to count
        Reader count(*F);
        if (!count.open(path, err)) return false;
        int r;
        while ((r = count.next(err)) == 1)
            if (++total > o.max_source_frames) break;
        if (r < 0) return false;
    }
    if (total <= 0) { err = "the video has no frames"; return false; }
    if (total > o.max_source_frames) {
        err = "the video has more than " + std::to_string((long long) o.max_source_frames) + " frames";
        return false;
    }
    out = Video{};
    out.total = total;
    out.fps = fps;
    out.indices = sample_indices(total, fps, o);
    const int64_t n = (int64_t) out.indices.size();
    if (pw > 0 && ph > 0 && n * pw * (int64_t) ph > o.max_sampled_pixels) {
        err = "the video's sampled frames are " + std::to_string((long long) (n * pw * (int64_t) ph >> 20)) +
              " megapixels; the limit is " + std::to_string((long long) (o.max_sampled_pixels >> 20));
        return false;
    }
    ff::SwsContext* sws = nullptr;
    int sws_w = 0, sws_h = 0, sws_fmt = -1, sws_cs = -1, sws_range = -1;
    size_t k = 0;
    int64_t i = 0;
    int r;
    while (k < (size_t) n && (r = rd.next(err)) == 1) {
        if (i++ != out.indices[k]) continue;
        const int fw = ff::field<int>(rd.frm, ff::kFrameWidth), fh = ff::field<int>(rd.frm, ff::kFrameHeight);
        if (k == 0 && pw > 0 && (fw != pw || fh != ph)) { err = F->version + ": layout check failed"; break; }
        F->av_frame_apply_cropping(rd.frm, 0);
        const int w = ff::field<int>(rd.frm, ff::kFrameWidth), h = ff::field<int>(rd.frm, ff::kFrameHeight);
        const int pf = ff::field<int>(rd.frm, ff::kFrameFormat);
        if (w <= 0 || h <= 0 || pf < 0) { err = "FFmpeg returned an empty frame"; break; }
        if (k == 0) {
            out.src_w = w;
            out.src_h = h;
            out.canvas = fit_video((int) n, w, h, o);
            if (out.canvas.w == 0) { err = "the video's frames are " + std::to_string(w) + "x" + std::to_string(h) + ", too narrow"; break; }
            out.frames.reserve((size_t) n * out.canvas.w * out.canvas.h * 3);
        }
        if (sws == nullptr || w != sws_w || h != sws_h || pf != sws_fmt) {
            if (sws != nullptr) F->sws_freeContext(sws);
            sws = F->sws_getContext(w, h, pf, w, h, ff::kPixFmtRgb24, ff::kSwsBilinear, nullptr, nullptr, nullptr);
            sws_w = w; sws_h = h; sws_fmt = pf; sws_cs = -1; sws_range = -1;
            if (sws == nullptr) { err = "FFmpeg cannot convert the video's pixel format"; break; }
        }
        // the frame's own colour matrix and range, as the reference's decoder converts (an unspecified matrix is
        // BT.601, an unspecified range the format's own)
        const int cs = ff::field<int>(rd.frm, a.fr_colorspace), range = ff::field<int>(rd.frm, a.fr_color_range);
        if (cs != sws_cs || range != sws_range) {
            int *inv = nullptr, *tbl = nullptr, src_range = 0, dst_range = 0, bri = 0, con = 0, sat = 0;
            if (F->sws_getColorspaceDetails(sws, &inv, &src_range, &tbl, &dst_range, &bri, &con, &sat) >= 0) {
                if (range == ff::kRangeMpeg) src_range = 0;
                else if (range == ff::kRangeJpeg) src_range = 1;
                F->sws_setColorspaceDetails(sws, F->sws_getCoefficients(cs), src_range, tbl, dst_range, bri, con, sat);
            }
            sws_cs = cs;
            sws_range = range;
        }
        Image img;
        img.w = w;
        img.h = h;
        img.rgb.resize((size_t) w * h * 3);
        uint8_t* dst[4] = {img.rgb.data(), nullptr, nullptr, nullptr};
        const int dst_stride[4] = {w * 3, 0, 0, 0};
        uint8_t* data[ff::kNumDataPointers];
        int linesize[ff::kNumDataPointers];
        std::memcpy(data, (const unsigned char*) rd.frm + ff::kFrameData, sizeof data);
        std::memcpy(linesize, (const unsigned char*) rd.frm + ff::kFrameLinesize, sizeof linesize);
        F->sws_scale(sws, data, linesize, 0, h, dst, dst_stride);
        const Image c = video_frame_to_canvas(img, out.canvas);
        out.frames.insert(out.frames.end(), c.rgb.begin(), c.rgb.end());
        ++k;
    }
    if (sws != nullptr) F->sws_freeContext(sws);
    if (!err.empty()) return false;
    if (k == 0) { err = "no frame of the video could be decoded"; return false; }
    out.indices.resize(k);   // a container that claims more frames than it holds: the tail is gone
    return true;
}

}  // namespace basalt::vision
