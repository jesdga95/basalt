// basalt/vision/video.hpp - a video file to the encoder's input: decode, sample, resize, cut into two-frame groups.
//
// The reference processor's rules (Qwen3-VL): frames are sampled at `fps` per second of the source, within
// [min_frames, max_frames], at evenly spaced source indices; every frame is resized (bicubic, stretched) to one canvas
// whose sides are multiples of 32 and whose frames x height x width volume fits the budget; frames are paired in time
// order (the last repeated for an odd count) and each pair becomes one group of 16x16x2 patches. A group is encoded as
// its own two-frame picture and carries the mean time of its two frames. Decoding goes through the FFmpeg loaded at
// run time (ffmpeg_api.hpp), from the first frame on, with no seeking.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "basalt/vision/preprocess.hpp"

namespace basalt::vision {

struct VideoOptions {
    double fps = 2.0;                  ///< frames sampled per second of the source
    int min_frames = 4, max_frames = 768;
    int max_tokens = 8192;             ///< merged tokens over all groups (a volume of max_tokens * 2048 pixels)
    int min_volume = 4096;             ///< the smallest frames x height x width volume
    int group_tokens = 4096;           ///< the most merged tokens one group may have (the encoder's sized picture)
    double max_seconds = 600;          ///< decode guards: the source's length and frames, one decoded frame's
    int64_t max_source_frames = 100000;   ///< pixels, and the pixels of the sampled frames held at their canvas
    int64_t max_frame_pixels = 64ll << 20;
    int64_t max_held_pixels = 256ll << 20;
    int gpu = -1;                      ///< the CUDA device NVDEC decodes on; -1: decode on the CPU
    void* cuda_context = nullptr;      ///< the caller's CUDA context on `gpu` (a CUcontext) for FFmpeg to share
    int64_t gpu_spare = -1;            ///< device bytes NVDEC may take (-1: no limit); a larger stream decodes on the CPU
};

/// The source indices to sample from a video of `total` frames at `src_fps` (no fps: 24).
std::vector<int64_t> sample_indices(int64_t total, double src_fps, const VideoOptions& o);

/// The canvas for `frames` sampled frames of w x h: the reference's video smart_resize (Python round() ties to even,
/// double arithmetic) within [min_volume, max_tokens * 2048], then shrunk until one group fits `group_tokens`. {0, 0}
/// for a frame whose aspect ratio is past 200.
Size fit_video(int frames, int w, int h, const VideoOptions& o);

/// Each group's time in seconds: the mean of its two frames' `index / fps`, the last index repeated for an odd count.
std::vector<double> group_times(const std::vector<int64_t>& indices, double src_fps);

/// `frames` (n frames of the canvas, RGB) as patch rows, group after group: each group's [gx * gy][1536] rows in merge
/// order, each row (channel, frame of the pair, row, column) normalised to [-1, 1]. An odd last frame is paired with
/// itself.
void patchify_video(const std::vector<uint8_t>& frames, int n, Size canvas, const Geometry& g, std::vector<float>& out);

struct Video {
    Size canvas;                       ///< every sampled frame's size after the resize
    int src_w = 0, src_h = 0;          ///< the decoded frames' size
    int64_t total = 0;                 ///< the source's frame count
    double fps = 0;                    ///< the source's frame rate
    std::vector<int64_t> indices;      ///< the sampled source frames
    std::vector<uint8_t> frames;       ///< indices.size() frames of the canvas, RGB
    std::string decoder;               ///< "nvdec" or "cpu" (with why NVDEC did not, when it was asked)
    double ms_total = 0, ms_copy = 0, ms_convert = 0;   ///< all of read_video; GPU-to-host copies; the busiest converter
    int groups() const { return ((int) indices.size() + 1) / 2; }
};

/// Decodes the video file at `path` (UTF-8) and keeps the sampled frames resized to their canvas. False with `err`
/// for a file FFmpeg cannot read, a video past a guard, or when this process has no FFmpeg.
bool read_video(const std::string& path, const VideoOptions& o, Video& out, std::string& err);

/// The sampled frames resized to `canvas` from decoded frames: the step `read_video` takes per frame, for callers that
/// decoded elsewhere.
Image video_frame_to_canvas(const Image& frame, Size canvas);

/// The device bytes NVDEC is expected to take for the video file at `path` (from its frame size; see read_video), or
/// -1 when this process has no FFmpeg or the file has no video stream it can read. For a caller that makes room first.
int64_t video_nvdec_bytes(const std::string& path);

/// An encoder process's video request, `ENCV <video> <output> [fps=F] [max_frames=N] [max_tokens=N]` (paths without
/// spaces): the paths, and the options over `o`'s. False with `err` for a malformed line or a value out of range.
bool parse_encv(const std::string& line, std::string& video, std::string& output, VideoOptions& o, std::string& err);

/// The groups' times for a reply line: each in 17 significant digits (it reads back as the same double), comma
/// separated.
std::string times_text(const std::vector<double>& times);

}  // namespace basalt::vision
