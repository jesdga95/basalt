// src/vision/video.cpp - see basalt/vision/video.hpp.
#include "basalt/vision/video.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

#include "basalt/vision/ffmpeg_api.hpp"

namespace basalt::vision {

namespace {
// What NVDEC takes for frames of `px` pixels: its surfaces (the reference frames and the output pool, about 40 frames
// of 1.5 bytes a pixel, counted at 16 bits a sample, as a 10-bit stream's) and its own state.
int64_t nvdec_need(int64_t px) { return px * 3 / 2 * 40 * 2 + (64ll << 20); }

}  // namespace

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

int64_t video_nvdec_bytes(const std::string& path) {
    std::string why;
    const Ffmpeg* F = ffmpeg(why);
    if (F == nullptr) return -1;
    ff::AVFormatContext* fmt = nullptr;
    if (F->avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) return -1;
    int64_t need = -1;
    const ff::AVCodec* dec = nullptr;
    if (F->avformat_find_stream_info(fmt, nullptr) >= 0) {
        const int index = F->av_find_best_stream(fmt, ff::kMediaTypeVideo, -1, -1, &dec, 0);
        ff::AVStream* const* streams = ff::field<ff::AVStream**>(fmt, ff::kFmtStreams);
        if (index >= 0 && (unsigned) index < ff::field<unsigned>(fmt, ff::kFmtNbStreams) && streams != nullptr) {
            const ff::AVCodecParameters* par = ff::field<ff::AVCodecParameters*>(streams[index], F->abi.st_codecpar);
            const int64_t px = (int64_t) ff::field<int>(par, F->abi.par_width) * ff::field<int>(par, F->abi.par_height);
            if (px > 0) need = nvdec_need(px);
        }
    }
    F->avformat_close_input(&fmt);
    return need;
}

bool parse_encv(const std::string& line, std::string& video, std::string& output, VideoOptions& o, std::string& err) {
    std::vector<std::string> words;
    size_t at = 0;
    while ((at = line.find_first_not_of(' ', at)) != std::string::npos) {
        const size_t end = std::min(line.find(' ', at), line.size());
        words.push_back(line.substr(at, end - at));
        at = end;
    }
    if (words.size() < 3 || words[0] != "ENCV") { err = "expected: ENCV <video> <output> [fps=F] [max_frames=N] [max_tokens=N]"; return false; }
    video = words[1];
    output = words[2];
    for (size_t i = 3; i < words.size(); ++i) {
        const size_t eq = words[i].find('=');
        const std::string k = words[i].substr(0, eq), v = eq == std::string::npos ? "" : words[i].substr(eq + 1);
        char* end = nullptr;
        const double x = std::strtod(v.c_str(), &end);
        if (v.empty() || end == nullptr || *end != 0 || !(x > 0)) { err = "ENCV: " + words[i] + " is not a positive number"; return false; }
        if (k == "fps" && x <= 60) o.fps = x;
        else if (k == "max_frames" && x == std::floor(x) && x >= o.min_frames && x <= 768) o.max_frames = (int) x;
        else if (k == "max_tokens" && x == std::floor(x) && x >= 16 && x <= 1 << 20) o.max_tokens = (int) x;
        else { err = "ENCV: " + words[i] + " is not a known option in range"; return false; }
    }
    return true;
}

std::string times_text(const std::vector<double>& times) {
    std::string s;
    char buf[32];
    for (size_t i = 0; i < times.size(); ++i) {
        // %.17g reads back as the same double, so the server formats exactly the reference's number
        std::snprintf(buf, sizeof buf, "%.17g", times[i]);
        s += (i ? "," : "") + std::string(buf);
    }
    return s;
}

namespace {

// The CUDA device NVDEC decodes on, one per GPU for the life of the process. On the caller's own CUDA context when
// it gives one: a context of FFmpeg's own costs a few hundred MiB the engine has handed to its expert cache, and
// FFmpeg will not take the primary context with scheduling flags other than its own. Without one (a program with no
// CUDA of its own), FFmpeg opens a context. Null with `why` when neither works (no driver, no CUDA support in this
// FFmpeg build, no memory left).
ff::AVBufferRef* nvdec_device(const Ffmpeg& F, int gpu, void* cuda_context, std::string& why) {
    static std::mutex m;
    static std::map<int, ff::AVBufferRef*> devices;
    static std::map<int, std::string> failed;
    std::lock_guard<std::mutex> l(m);
    if (auto it = devices.find(gpu); it != devices.end()) return it->second;
    if (auto it = failed.find(gpu); it != failed.end()) { why = it->second; return nullptr; }
    ff::AVBufferRef* d = nullptr;
    int r = 0;
    if (cuda_context != nullptr) {
        d = F.av_hwdevice_ctx_alloc(ff::kHwDeviceCuda);
        if (d == nullptr) {
            r = -12;   // AVERROR(ENOMEM)
        } else {
            unsigned char* dev = ff::field<unsigned char*>(d, ff::kBufferData);
            unsigned char* cuda = ff::field<unsigned char*>(dev, F.abi.hwdev_hwctx);
            std::memcpy(cuda, &cuda_context, sizeof cuda_context);   // AVCUDADeviceContext.cuda_ctx
            if ((r = F.av_hwdevice_ctx_init(d)) < 0) F.av_buffer_unref(&d);
        }
    } else {
        r = F.av_hwdevice_ctx_create(&d, ff::kHwDeviceCuda, std::to_string(gpu).c_str(), nullptr, 0);
    }
    if (r < 0 || d == nullptr) {
        failed[gpu] = why = "FFmpeg cannot open GPU " + std::to_string(gpu) + " for NVDEC: " + F.error_text(r);
        return nullptr;
    }
    devices[gpu] = d;
    return d;
}

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

    std::string nvdec_why;   // why NVDEC was not asked, when it was wanted

    bool open(const std::string& path, std::string& err, int gpu = -1, void* cuda_context = nullptr, int64_t spare = -1) {
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
        // NVDEC where the GPU and FFmpeg's build allow: FFmpeg's default format choice takes the device's frames
        // when a device is set, and decodes in software itself when NVDEC refuses the stream
        // a stream whose NVDEC surfaces do not fit in what the card can spare is decoded on the CPU rather than take
        // memory the model needs
        const ff::AVCodecParameters* par = ff::field<ff::AVCodecParameters*>(st, F.abi.st_codecpar);
        const int64_t need = nvdec_need((int64_t) ff::field<int>(par, F.abi.par_width) * ff::field<int>(par, F.abi.par_height));
        if (gpu >= 0 && spare >= 0 && need > spare) {
            nvdec_why = "NVDEC needs about " + std::to_string((long long) (need >> 20)) + " MiB, GPU " + std::to_string(gpu) +
                        " can spare " + std::to_string((long long) (spare >> 20));
        } else if (gpu >= 0) {
            if (ff::AVBufferRef* dev = nvdec_device(F, gpu, cuda_context, nvdec_why)) {
                ff::AVBufferRef* ref = F.av_buffer_ref(dev);
                std::memcpy((unsigned char*) ctx + F.abi.cc_hw_device_ctx, &ref, sizeof ref);
            }
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

// The sampled frames' conversion: worker threads, each with its own swscale context, take (frame, slot) jobs from a
// short queue (a few frames in flight bound the memory), convert to RGB with the frame's colour matrix and range as
// PyAV does, resize to the canvas and write the slot. The first error stops the rest.
class Converters {
public:
    Converters(const Ffmpeg& f, const ff::AbiRow& a, Video& v, int n) : F(f), A(a), V(v), n_(n) {}
    Converters(const Converters&) = delete;
    Converters& operator=(const Converters&) = delete;
    ~Converters() { finish(); }

    void start() {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const int workers = (int) std::clamp(hw / 4, 1u, 8u);
        cap_ = (size_t) workers * 2;
        for (int t = 0; t < std::min(workers, n_); ++t) threads_.emplace_back([this] { run(); });
    }

    void push(ff::AVFrame* f, size_t slot, int cs, int range) {
        std::unique_lock<std::mutex> l(m_);
        room_.wait(l, [&] { return q_.size() < cap_ || !error_.empty(); });
        if (!error_.empty()) { F.av_frame_free(&f); return; }
        q_.push_back({f, slot, cs, range});
        ready_.notify_one();
    }

    /// Waits for the queued frames; returns the first error, or "".
    std::string finish() {
        {
            std::lock_guard<std::mutex> l(m_);
            done_ = true;
        }
        ready_.notify_all();
        for (auto& t : threads_) t.join();
        threads_.clear();
        std::lock_guard<std::mutex> l(m_);
        for (Job& j : q_) F.av_frame_free(&j.f);
        q_.clear();
        V.ms_convert = convert_ms_;
        return error_;
    }

private:
    struct Job { ff::AVFrame* f; size_t slot; int cs, range; };

    void run() {
        ff::SwsContext* sws = nullptr;
        int sw = 0, sh = 0, sf = -1, scs = -1, sr = -1;
        double ms = 0;
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> l(m_);
                ready_.wait(l, [&] { return !q_.empty() || done_; });
                if (q_.empty()) break;
                j = q_.front();
                q_.pop_front();
                room_.notify_one();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const std::string e = convert(j, sws, sw, sh, sf, scs, sr);
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            F.av_frame_free(&j.f);
            if (!e.empty()) {
                std::lock_guard<std::mutex> l(m_);
                if (error_.empty()) error_ = e;
                room_.notify_all();
            }
        }
        if (sws != nullptr) F.sws_freeContext(sws);
        std::lock_guard<std::mutex> l(m_);
        convert_ms_ = std::max(convert_ms_, ms);
    }

    std::string convert(const Job& j, ff::SwsContext*& sws, int& sw, int& sh, int& sf, int& scs, int& sr) {
        const int w = ff::field<int>(j.f, ff::kFrameWidth), h = ff::field<int>(j.f, ff::kFrameHeight);
        int pf = ff::field<int>(j.f, ff::kFrameFormat);
        uint8_t* data[ff::kNumDataPointers];
        int linesize[ff::kNumDataPointers];
        std::memcpy(data, (const unsigned char*) j.f + ff::kFrameData, sizeof data);
        std::memcpy(linesize, (const unsigned char*) j.f + ff::kFrameLinesize, sizeof linesize);
        // NVDEC's frames come back with their chroma interleaved (NV12, P010); a software decoder's are planar, and
        // swscale converts the two through different routines. The chroma is split into planes first (no value
        // changes), so a frame converts the same whichever decoder made it.
        std::vector<uint8_t> planes;
        if (pf == ff::kPixFmtNv12 || pf == A.pix_fmt_p010) {
            const bool ten = pf != ff::kPixFmtNv12;
            const int cw = (w + 1) / 2, ch = (h + 1) / 2, bytes = ten ? 2 : 1;
            planes.resize((size_t) w * h * bytes + (size_t) 2 * cw * ch * bytes);
            uint8_t* y = planes.data();
            uint8_t* u = y + (size_t) w * h * bytes;
            uint8_t* v = u + (size_t) cw * ch * bytes;
            for (int r = 0; r < h; ++r) {
                const uint8_t* s = data[0] + (size_t) r * linesize[0];
                if (!ten) {
                    std::memcpy(y + (size_t) r * w, s, (size_t) w);
                } else {
                    for (int x = 0; x < w; ++x) {   // P010 keeps the 10 bits at the top of each 16-bit word
                        uint16_t q;
                        std::memcpy(&q, s + 2 * x, 2);
                        q = (uint16_t) (q >> 6);
                        std::memcpy(y + ((size_t) r * w + x) * 2, &q, 2);
                    }
                }
            }
            for (int r = 0; r < ch; ++r) {
                const uint8_t* s = data[1] + (size_t) r * linesize[1];
                for (int x = 0; x < cw; ++x) {
                    if (!ten) {
                        u[(size_t) r * cw + x] = s[2 * x];
                        v[(size_t) r * cw + x] = s[2 * x + 1];
                    } else {
                        uint16_t a, b;
                        std::memcpy(&a, s + 4 * x, 2);
                        std::memcpy(&b, s + 4 * x + 2, 2);
                        a = (uint16_t) (a >> 6);
                        b = (uint16_t) (b >> 6);
                        std::memcpy(u + ((size_t) r * cw + x) * 2, &a, 2);
                        std::memcpy(v + ((size_t) r * cw + x) * 2, &b, 2);
                    }
                }
            }
            data[0] = y; data[1] = u; data[2] = v; data[3] = nullptr;
            linesize[0] = w * bytes; linesize[1] = linesize[2] = cw * bytes; linesize[3] = 0;
            pf = ten ? ff::kPixFmtYuv420p10 : ff::kPixFmtYuv420p;
        }
        if (sws == nullptr || w != sw || h != sh || pf != sf) {
            if (sws != nullptr) F.sws_freeContext(sws);
            sws = F.sws_getContext(w, h, pf, w, h, ff::kPixFmtRgb24, ff::kSwsBilinear, nullptr, nullptr, nullptr);
            sw = w; sh = h; sf = pf; scs = -1; sr = -1;
            if (sws == nullptr) return "FFmpeg cannot convert the video's pixel format";
        }
        // an unspecified matrix is BT.601, an unspecified range the format's own
        if (j.cs != scs || j.range != sr) {
            int *inv = nullptr, *tbl = nullptr, src_range = 0, dst_range = 0, bri = 0, con = 0, sat = 0;
            if (F.sws_getColorspaceDetails(sws, &inv, &src_range, &tbl, &dst_range, &bri, &con, &sat) >= 0) {
                if (j.range == ff::kRangeMpeg) src_range = 0;
                else if (j.range == ff::kRangeJpeg) src_range = 1;
                F.sws_setColorspaceDetails(sws, F.sws_getCoefficients(j.cs), src_range, tbl, dst_range, bri, con, sat);
            }
            scs = j.cs;
            sr = j.range;
        }
        Image img;
        img.w = w;
        img.h = h;
        img.rgb.resize((size_t) w * h * 3);
        uint8_t* dst[4] = {img.rgb.data(), nullptr, nullptr, nullptr};
        const int dst_stride[4] = {w * 3, 0, 0, 0};
        F.sws_scale(sws, data, linesize, 0, h, dst, dst_stride);
        const Image c = video_frame_to_canvas(img, V.canvas);
        std::memcpy(&V.frames[j.slot * c.rgb.size()], c.rgb.data(), c.rgb.size());
        return "";
    }

    const Ffmpeg& F;
    const ff::AbiRow& A;
    Video& V;
    int n_;
    size_t cap_ = 2;
    std::mutex m_;
    std::condition_variable ready_, room_;
    std::deque<Job> q_;
    std::vector<std::thread> threads_;
    std::string error_;
    bool done_ = false;
    double convert_ms_ = 0;
};

}  // namespace

namespace {
bool read_video_once(const std::string& path, const VideoOptions& o, Video& out, std::string& err);
}

bool read_video(const std::string& path, const VideoOptions& o, Video& out, std::string& err) {
    if (read_video_once(path, o, out, err)) return true;
    // only a failure of the decode itself is tried again: a guard or an unreadable file fails the same way twice
    const bool decoding = err.rfind("FFmpeg cannot decode", 0) == 0 || err.rfind("copying a decoded frame", 0) == 0;
    if (o.gpu < 0 || !decoding) return false;
    // NVDEC can run out of device memory after it started (the card's memory is the model's): the file is decoded
    // again on the CPU, and the decoder note says why
    const std::string why = err;
    VideoOptions cpu = o;
    cpu.gpu = -1;
    cpu.cuda_context = nullptr;
    err.clear();
    if (!read_video_once(path, cpu, out, err)) return false;
    out.decoder = "cpu, NVDEC failed: " + why;
    return true;
}

namespace {
bool read_video_once(const std::string& path, const VideoOptions& o, Video& out, std::string& err) {
    const auto t_start = std::chrono::steady_clock::now();
    std::string why;
    const Ffmpeg* F = ffmpeg(why);
    if (F == nullptr) { err = "this process cannot read video: " + why; return false; }
    const ff::AbiRow& a = F->abi;
    Reader rd(*F);
    if (!rd.open(path, err, o.gpu, o.cuda_context, o.gpu_spare)) return false;
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
    // what is held is each sampled frame at its canvas (a frame is resized as it is decoded); a source frame is
    // only decoded one at a time, and FFmpeg allocates it before anything is checked: its own size is capped
    if ((int64_t) pw * ph > o.max_frame_pixels) {
        err = "the video's frames are " + std::to_string(pw) + "x" + std::to_string(ph) + "; the limit is " +
              std::to_string((long long) (o.max_frame_pixels >> 20)) + " megapixels a frame";
        return false;
    }
    // Sampled frames go to converters on worker threads (to RGB with the frame's own colour matrix, then resized
    // to the canvas); a frame NVDEC decoded is copied to the host first, the others never leave the device.
    Converters conv(*F, a, out, (int) n);
    size_t k = 0;
    int64_t i = 0;
    int r;
    while (k < (size_t) n && (r = rd.next(err)) == 1) {
        if (i++ != out.indices[k]) continue;
        const int fw = ff::field<int>(rd.frm, ff::kFrameWidth), fh = ff::field<int>(rd.frm, ff::kFrameHeight);
        if (k == 0 && pw > 0 && (fw != pw || fh != ph)) { err = F->version + ": layout check failed"; break; }
        F->av_frame_apply_cropping(rd.frm, 0);
        const int w = ff::field<int>(rd.frm, ff::kFrameWidth), h = ff::field<int>(rd.frm, ff::kFrameHeight);
        if (w <= 0 || h <= 0 || ff::field<int>(rd.frm, ff::kFrameFormat) < 0) { err = "FFmpeg returned an empty frame"; break; }
        if (k == 0) {
            out.src_w = w;
            out.src_h = h;
            out.decoder = ff::field<int>(rd.frm, ff::kFrameFormat) == a.pix_fmt_cuda ? "nvdec"
                        : o.gpu < 0 ? "cpu"
                        : "cpu, " + (rd.nvdec_why.empty() ? std::string("NVDEC refused the stream") : rd.nvdec_why);
            out.canvas = fit_video((int) n, w, h, o);
            if (out.canvas.w == 0) { err = "the video's frames are " + std::to_string(w) + "x" + std::to_string(h) + ", too narrow"; break; }
            if (n * out.canvas.w * (int64_t) out.canvas.h > o.max_held_pixels) {
                err = "the video's sampled frames would hold " + std::to_string((long long) (n * out.canvas.w * (int64_t) out.canvas.h >> 20)) +
                      " megapixels; the limit is " + std::to_string((long long) (o.max_held_pixels >> 20));
                break;
            }
            out.frames.assign((size_t) n * out.canvas.w * out.canvas.h * 3, 0);
            conv.start();
        }
        // the colour fields stay the decoded frame's: a copy to the host does not carry them
        const int cs = ff::field<int>(rd.frm, a.fr_colorspace), range = ff::field<int>(rd.frm, a.fr_color_range);
        ff::AVFrame* job = nullptr;
        if (ff::field<int>(rd.frm, ff::kFrameFormat) == a.pix_fmt_cuda) {
            const auto t0 = std::chrono::steady_clock::now();
            job = F->av_frame_alloc();
            const int tr = job != nullptr ? F->av_hwframe_transfer_data(job, rd.frm, 0) : -1;
            out.ms_copy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (tr < 0) { if (job) F->av_frame_free(&job); err = "copying a decoded frame from the GPU failed: " + F->error_text(tr); break; }
        } else {
            job = F->av_frame_clone(rd.frm);
            if (job == nullptr) { err = "FFmpeg is out of memory"; break; }
        }
        conv.push(job, k, cs, range);
        ++k;
    }
    const std::string cerr = conv.finish();
    if (err.empty()) err = cerr;
    if (!err.empty()) return false;
    if (k == 0) { err = "no frame of the video could be decoded"; return false; }
    out.indices.resize(k);   // a container that claims more frames than it holds: the tail is gone
    out.frames.resize(k * (size_t) out.canvas.w * out.canvas.h * 3);
    out.ms_total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    return true;
}

}  // namespace

}  // namespace basalt::vision
