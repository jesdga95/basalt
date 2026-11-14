// basalt/vision/ffmpeg_abi.hpp - the part of FFmpeg's binary interface the video decoder uses, declared by hand.
//
// Nothing of FFmpeg is linked or built: its libraries are loaded at run time (ffmpeg_api.hpp), so this file stands in
// for its headers. The declarations are derived from FFmpeg's public headers (LGPL-2.1-or-later, see
// THIRD_PARTY_LICENSES): opaque structure types, the functions' signatures, the constants used, and the offsets of
// the few structure fields read directly. Fields at the start of a structure have kept their place in every release
// read here; the others move between major versions and come from the row of `kAbiRows` for the loaded release,
// transcribed from the headers of FFmpeg 6.0/6.1, 7.0/7.1 and 8.0 (each pair of the same major has one layout).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace basalt::vision::ff {

struct AVFormatContext;
struct AVStream;
struct AVCodec;
struct AVCodecContext;
struct AVCodecParameters;
struct AVFrame;
struct AVPacket;
struct AVDictionary;
struct AVInputFormat;
struct SwsContext;
struct AVBufferRef;
struct SwsFilter;

struct AVRational {
    int num, den;
};

inline double q2d(AVRational r) { return r.den != 0 ? (double) r.num / (double) r.den : 0.0; }

constexpr int kMediaTypeVideo = 0;          // AVMEDIA_TYPE_VIDEO
constexpr int kPixFmtRgb24 = 2;             // AV_PIX_FMT_RGB24
constexpr int kPixFmtYuv420p = 0;           // AV_PIX_FMT_YUV420P
constexpr int kPixFmtNv12 = 23;             // AV_PIX_FMT_NV12
constexpr int kPixFmtYuv420p10 = 62;        // AV_PIX_FMT_YUV420P10LE
constexpr int kSwsBilinear = 2;             // SWS_BILINEAR
constexpr int kSwsFullChrHInt = 0x2000;     // SWS_FULL_CHR_H_INT
constexpr int kSwsAccurateRnd = 0x40000;    // SWS_ACCURATE_RND
constexpr int kLogQuiet = -8;               // AV_LOG_QUIET
constexpr int kHwDeviceCuda = 2;            // AV_HWDEVICE_TYPE_CUDA
constexpr int kRangeMpeg = 1;               // AVCOL_RANGE_MPEG (limited)
constexpr int kRangeJpeg = 2;               // AVCOL_RANGE_JPEG (full)
constexpr int kNumDataPointers = 8;         // AV_NUM_DATA_POINTERS
constexpr int kErrorEagain = -11;           // AVERROR(EAGAIN) on Linux and Windows alike
constexpr int kErrorEof = -541478725;       // AVERROR_EOF: FFERRTAG('E','O','F',' ')
constexpr int64_t kNoPts = INT64_MIN;       // AV_NOPTS_VALUE
constexpr int kTimeBase = 1000000;          // AV_TIME_BASE

// The functions, by name in the library that exports them.
using avutil_version_fn = unsigned (*)();
using av_frame_alloc_fn = AVFrame* (*)();
using av_frame_free_fn = void (*)(AVFrame**);
using av_frame_unref_fn = void (*)(AVFrame*);
using av_frame_apply_cropping_fn = int (*)(AVFrame*, int flags);
using av_strerror_fn = int (*)(int errnum, char* buf, size_t size);
using av_dict_set_fn = int (*)(AVDictionary** pm, const char* key, const char* value, int flags);
using av_dict_free_fn = void (*)(AVDictionary** m);
using av_log_set_level_fn = void (*)(int level);
using av_frame_clone_fn = AVFrame* (*)(const AVFrame* src);
using av_hwdevice_ctx_create_fn = int (*)(AVBufferRef** device_ctx, int type, const char* device, AVDictionary* opts, int flags);
using av_buffer_ref_fn = AVBufferRef* (*)(const AVBufferRef* buf);
using av_hwdevice_ctx_alloc_fn = AVBufferRef* (*)(int type);
using av_hwdevice_ctx_init_fn = int (*)(AVBufferRef* ref);
using av_buffer_unref_fn = void (*)(AVBufferRef** buf);
using av_hwframe_transfer_data_fn = int (*)(AVFrame* dst, const AVFrame* src, int flags);

using avformat_version_fn = unsigned (*)();
using avformat_open_input_fn = int (*)(AVFormatContext** ps, const char* url, const AVInputFormat* fmt, AVDictionary** options);
using avformat_find_stream_info_fn = int (*)(AVFormatContext* ic, AVDictionary** options);
using av_find_best_stream_fn = int (*)(AVFormatContext* ic, int type, int wanted, int related, const AVCodec** decoder, int flags);
using av_read_frame_fn = int (*)(AVFormatContext* s, AVPacket* pkt);
using avformat_close_input_fn = void (*)(AVFormatContext** s);

using avcodec_version_fn = unsigned (*)();
using avcodec_alloc_context3_fn = AVCodecContext* (*)(const AVCodec* codec);
using avcodec_parameters_to_context_fn = int (*)(AVCodecContext* codec, const AVCodecParameters* par);
using avcodec_open2_fn = int (*)(AVCodecContext* avctx, const AVCodec* codec, AVDictionary** options);
using avcodec_send_packet_fn = int (*)(AVCodecContext* avctx, const AVPacket* avpkt);
using avcodec_receive_frame_fn = int (*)(AVCodecContext* avctx, AVFrame* frame);
using avcodec_free_context_fn = void (*)(AVCodecContext** avctx);
using av_packet_alloc_fn = AVPacket* (*)();
using av_packet_free_fn = void (*)(AVPacket** pkt);
using av_packet_unref_fn = void (*)(AVPacket* pkt);

using swscale_version_fn = unsigned (*)();
using sws_getContext_fn = SwsContext* (*)(int srcW, int srcH, int srcFormat, int dstW, int dstH, int dstFormat, int flags,
                                          SwsFilter* srcFilter, SwsFilter* dstFilter, const double* param);
using sws_scale_fn = int (*)(SwsContext* c, const uint8_t* const srcSlice[], const int srcStride[], int srcSliceY,
                             int srcSliceH, uint8_t* const dst[], const int dstStride[]);
using sws_freeContext_fn = void (*)(SwsContext* swsContext);
using sws_getCoefficients_fn = const int* (*)(int colorspace);
using sws_getColorspaceDetails_fn = int (*)(SwsContext* c, int** inv_table, int* srcRange, int** table, int* dstRange,
                                            int* brightness, int* contrast, int* saturation);
using sws_setColorspaceDetails_fn = int (*)(SwsContext* c, const int inv_table[4], int srcRange, const int table[4],
                                            int dstRange, int brightness, int contrast, int saturation);

/// One FFmpeg release's library majors and the offsets of the fields that move between them.
struct AbiRow {
    const char* release;
    int avformat, avcodec, avutil, swscale;
    size_t fmt_duration;                     // AVFormatContext.duration (int64_t, AV_TIME_BASE units)
    size_t st_codecpar, st_time_base, st_duration, st_nb_frames, st_avg_frame_rate, st_r_frame_rate;   // AVStream
    size_t par_width, par_height;            // AVCodecParameters.width / height (int)
    size_t fr_color_range, fr_colorspace;    // AVFrame.color_range / colorspace (enums, int)
    size_t cc_hw_device_ctx;                 // AVCodecContext.hw_device_ctx (AVBufferRef*)
    int pix_fmt_cuda;                        // AV_PIX_FMT_CUDA
    int pix_fmt_p010;                        // AV_PIX_FMT_P010LE
    size_t hwdev_hwctx;                      // AVHWDeviceContext.hwctx (AVCUDADeviceContext*, cuda_ctx first)
};

/// Newest first: the loader takes the first release whose libraries are all present.
inline constexpr AbiRow kAbiRows[] = {
    {"8", 62, 62, 60, 9, 104, 16, 32, 48, 56, 88, 204, 72, 76, 280, 292, 560, 117, 158, 16},
    {"7", 61, 61, 59, 8, 104, 16, 32, 48, 56, 88, 216, 72, 76, 296, 308, 560, 117, 158, 16},
    {"6", 60, 60, 58, 7, 72, 16, 32, 48, 56, 88, 216, 56, 60, 320, 332, 864, 117, 159, 24},
};

// Fields at the same place in every row.
constexpr size_t kFmtNbStreams = 44;         // AVFormatContext.nb_streams (unsigned)
constexpr size_t kFmtStreams = 48;           // AVFormatContext.streams (AVStream**)
constexpr size_t kFrameData = 0;             // AVFrame.data[8]
constexpr size_t kFrameLinesize = 64;        // AVFrame.linesize[8]
constexpr size_t kFrameWidth = 104;          // AVFrame.width
constexpr size_t kFrameHeight = 108;         // AVFrame.height
constexpr size_t kFrameFormat = 116;         // AVFrame.format
constexpr size_t kPacketStreamIndex = 36;    // AVPacket.stream_index
constexpr size_t kBufferData = 8;            // AVBufferRef.data

/// A field of type T at byte `off` of the structure at `p`.
template <class T>
T field(const void* p, size_t off) {
    T v;
    std::memcpy(&v, (const unsigned char*) p + off, sizeof v);
    return v;
}

}  // namespace basalt::vision::ff
