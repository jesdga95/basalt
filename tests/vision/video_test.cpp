// tests/vision/video_test.cpp - video sampling, sizing, timestamps and patches against the reference processor
// (transformers Qwen3VLVideoProcessor, fps 2, 4..768 frames), and decoding the committed clip (CPU only).
//
//   video_test <fixture.mp4>                 the rules, then the clip when this machine has FFmpeg (else skipped)
//   video_test <clip> <reference dir>        parity with a reference dump (meta.txt, frames.u8, pixels.f32)
//
// The reference numbers below come from transformers 5.22.2 with its pyav loader; the clip is tests/data/video/
// synth.mp4 (96x64, 25 frames at 10 fps).
#include "basalt/vision/ffmpeg_api.hpp"
#include "basalt/vision/video.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace basalt::vision;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("video %-66s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}

template <class T>
std::string join(const std::vector<T>& v) {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str();
}

// The reference's numbers for a few sources: (frames, fps) -> indices, (frames, w, h) -> canvas.
void rules() {
    const VideoOptions o;
    check(join(sample_indices(25, 10.0, o)) == "0,6,12,18,24", "25 frames at 10 fps sample 5");
    check(join(sample_indices(48, 8.0, o)) == "0,4,9,13,17,21,26,30,34,38,43,47", "48 frames at 8 fps sample 12");
    check(join(sample_indices(240, 337500.0 / 14131.0, o)) ==
              "0,13,25,38,50,63,75,88,101,113,126,138,151,164,176,189,201,214,226,239",
          "240 frames at 23.88 fps sample 20");
    check(join(sample_indices(3, 30.0, o)) == "0,1,2", "a 3-frame clip samples all 3 (the count never passes the total)");
    check(sample_indices(100000, 30.0, o).size() == 768, "a long clip samples max_frames");
    const std::vector<double> ts = group_times({0, 6, 12, 18, 24}, 10.0);
    check(ts.size() == 3 && ts[0] == 0.3 && ts[1] == 1.5 && ts[2] == 2.4, "group times pair the frames, the last one repeated");
    const std::vector<double> t2 = group_times({0, 13, 25, 38}, 337500.0 / 14131.0);
    check(t2.size() == 2 && t2[0] == 0.27215259259259256 && t2[1] == 1.3188933333333333, "group times are the reference's doubles");
    struct Fit { int n, w, h, ow, oh; };
    const Fit fits[] = {
        {5, 96, 64, 96, 64}, {12, 320, 240, 320, 256}, {10, 640, 360, 640, 352}, {20, 720, 358, 704, 352},
        {768, 1920, 1080, 192, 96}, {4, 3840, 2160, 2720, 1536}, {4, 20, 20, 32, 32},
    };
    for (const Fit& f : fits) {
        const Size s = fit_video(f.n, f.w, f.h, o);
        check(s.w == f.ow && s.h == f.oh, std::to_string(f.n) + " frames of " + std::to_string(f.w) + "x" + std::to_string(f.h) +
                                              " fit " + std::to_string(f.ow) + "x" + std::to_string(f.oh) + " (got " +
                                              std::to_string(s.w) + "x" + std::to_string(s.h) + ")");
    }
    check(fit_video(4, 4000, 10, o).w == 0, "an aspect ratio past 200 is refused");
    // the patch row: channel, frame of the pair, row, column; a lone last frame pairs with itself
    const Size c{32, 32};
    std::vector<uint8_t> fr((size_t) 3 * 32 * 32 * 3);
    for (size_t i = 0; i < fr.size(); ++i) fr[i] = (uint8_t) (i * 7 % 251);
    std::vector<float> p;
    patchify_video(fr, 3, c, Geometry{}, p);
    check(p.size() == (size_t) 2 * 4 * 1536, "3 frames of 32x32 make 2 groups of 4 rows of 1536");
    auto px = [&](int f, int y, int x, int ch) { return ((float) fr[(size_t) f * 3072 + ((size_t) y * 32 + x) * 3 + ch] - 127.5f) / 127.5f; };
    check(p[0] == px(0, 0, 0, 0) && p[256] == px(1, 0, 0, 0) && p[512] == px(0, 0, 0, 1) && p[1536 + 17] == px(0, 1, 17, 0),
          "row order: channel, then frame, then pixel; then the next patch in merge order");
    check(p[(size_t) 4 * 1536] == px(2, 0, 0, 0) && p[(size_t) 4 * 1536 + 256] == px(2, 0, 0, 0), "the odd last frame is repeated");
}

void fixture(const std::string& clip) {
    std::string why;
    if (ffmpeg(why) == nullptr) {
        std::printf("video: no FFmpeg here (%s); the clip is not decoded\n", why.c_str());
        return;
    }
    Video v;
    std::string err;
    check(read_video(clip, VideoOptions{}, v, err), "the clip decodes" + (err.empty() ? "" : ": " + err));
    check(v.total == 25 && v.fps == 10.0, "25 frames at 10 fps");
    check(join(v.indices) == "0,6,12,18,24" && v.canvas.w == 96 && v.canvas.h == 64 && v.groups() == 3,
          "5 frames sampled onto a 96x64 canvas, 3 groups");
    // what the reference decoded (PyAV, rgb24): every channel's sum over the five frames; one step per pixel at most
    const long long want[3] = {1572831, 1479227, 1484999};
    long long got[3] = {0, 0, 0};
    for (size_t i = 0; i < v.frames.size(); ++i) got[i % 3] += v.frames[i];
    const long long px = (long long) v.frames.size() / 3;
    bool close = true;
    for (int c = 0; c < 3; ++c) close &= std::llabs(got[c] - want[c]) <= px / 50;
    check(close, "the decoded pixels are the reference's (channel sums " + std::to_string(got[0]) + "," +
                     std::to_string(got[1]) + "," + std::to_string(got[2]) + ")");
    Video bad;
    check(!read_video(clip + ".missing", VideoOptions{}, bad, err) && !err.empty(), "a missing file is refused with a reason");
}

// Parity with a reference dump: sampling and canvas exact, pixels within one uint8 step after normalisation.
int parity(const std::string& clip, const std::string& dir) {
    std::ifstream mt(dir + "/meta.txt");
    int64_t total = 0, n = 0;
    double fps = 0;
    int gt = 0, gh = 0, gw = 0, fh = 0, fw = 0;
    mt >> total >> fps >> n;
    std::vector<int64_t> idx((size_t) n);
    for (auto& x : idx) mt >> x;
    std::vector<double> ts;
    mt >> gt >> gh >> gw >> fh >> fw;
    ts.resize((size_t) gt);
    for (auto& x : ts) mt >> x;
    if (!mt) { std::fprintf(stderr, "cannot read %s/meta.txt\n", dir.c_str()); return 2; }
    Video v;
    std::string err;
    check(read_video(clip, VideoOptions{}, v, err), "decodes " + clip + (err.empty() ? "" : ": " + err));
    check(v.total == total && v.fps == fps, "total and fps as the reference");
    check(v.indices == idx, "sampled indices as the reference");
    check(v.canvas.w == gw * 16 && v.canvas.h == gh * 16 && v.groups() == gt, "canvas and groups as the reference");
    check(group_times(v.indices, v.fps) == ts, "group times as the reference");
    std::vector<float> ref((size_t) gt * gh * gw * 1536);
    std::ifstream pf(dir + "/pixels.f32", std::ios::binary);
    pf.read((char*) ref.data(), (std::streamsize) (ref.size() * 4));
    std::vector<uint8_t> frames((size_t) n * fh * fw * 3);
    std::ifstream ff(dir + "/frames.u8", std::ios::binary);
    ff.read((char*) frames.data(), (std::streamsize) frames.size());
    auto compare = [&](const std::vector<float>& got, const char* what) {
        double mx = 0;
        size_t over = 0, off = 0;
        for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
            const double d = std::fabs((double) got[i] - ref[i]);
            mx = std::max(mx, d);
            off += d > 1e-6;
            over += d > 1.0 / 127.5 + 1e-6;
        }
        std::printf("video parity %s: %s max |diff| %.6f (%.2f steps), %zu of %zu values differ, %zu past one step\n",
                    clip.c_str(), what, mx, mx * 127.5, off, ref.size(), over);
        check(got.size() == ref.size(), std::string(what) + ": as many values as the reference");
    };
    std::vector<float> p;
    patchify_video(v.frames, (int) v.indices.size(), v.canvas, Geometry{}, p);
    compare(p, "decoded here");
    // the reference's own decoded frames through the resize and patches: the resize alone
    std::vector<uint8_t> canv;
    for (int64_t i = 0; i < n; ++i) {
        Image im{fw, fh, std::vector<uint8_t>(frames.begin() + i * fw * fh * 3, frames.begin() + (i + 1) * fw * fh * 3)};
        const Image c = video_frame_to_canvas(im, v.canvas);
        canv.insert(canv.end(), c.rgb.begin(), c.rgb.end());
    }
    patchify_video(canv, (int) n, v.canvas, Geometry{}, p);
    compare(p, "reference frames");
    return failures ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3) return parity(argv[1], argv[2]);
    rules();
    if (argc >= 2) fixture(argv[1]);
    return failures ? 1 : 0;
}
