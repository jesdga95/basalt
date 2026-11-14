// tests/vision/ffmpeg_abi_test.cpp - the hand-declared FFmpeg ABI against FFmpeg's own headers, where the build
// machine has them (compiled only then; nothing is linked). The row for the headers' release must give every
// field's offset and constant as the headers do.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include <cstddef>
#include <cstdio>

#include "basalt/vision/ffmpeg_abi.hpp"

namespace ff = basalt::vision::ff;

int main() {
    const ff::AbiRow* row = nullptr;
    for (const ff::AbiRow& r : ff::kAbiRows)
        if (r.avformat == LIBAVFORMAT_VERSION_MAJOR) row = &r;
    if (row == nullptr) {
        std::printf("ffmpeg_abi: headers of libavformat %d, which the table has no row for\n", LIBAVFORMAT_VERSION_MAJOR);
        return 77;
    }
    int bad = 0;
    auto eq = [&](long long want, long long got, const char* what) {
        if (want != got) { std::printf("ffmpeg_abi: %s is %lld in the headers, %lld in the table\n", what, want, got); ++bad; }
    };
    eq(LIBAVCODEC_VERSION_MAJOR, row->avcodec, "the avcodec major");
    eq(LIBAVUTIL_VERSION_MAJOR, row->avutil, "the avutil major");
    eq(LIBSWSCALE_VERSION_MAJOR, row->swscale, "the swscale major");
    eq(offsetof(AVFormatContext, duration), row->fmt_duration, "AVFormatContext.duration");
    eq(offsetof(AVStream, codecpar), row->st_codecpar, "AVStream.codecpar");
    eq(offsetof(AVStream, time_base), row->st_time_base, "AVStream.time_base");
    eq(offsetof(AVStream, duration), row->st_duration, "AVStream.duration");
    eq(offsetof(AVStream, nb_frames), row->st_nb_frames, "AVStream.nb_frames");
    eq(offsetof(AVStream, avg_frame_rate), row->st_avg_frame_rate, "AVStream.avg_frame_rate");
    eq(offsetof(AVStream, r_frame_rate), row->st_r_frame_rate, "AVStream.r_frame_rate");
    eq(offsetof(AVCodecParameters, width), row->par_width, "AVCodecParameters.width");
    eq(offsetof(AVCodecParameters, height), row->par_height, "AVCodecParameters.height");
    eq(offsetof(AVFrame, color_range), row->fr_color_range, "AVFrame.color_range");
    eq(offsetof(AVFrame, colorspace), row->fr_colorspace, "AVFrame.colorspace");
    eq(offsetof(AVCodecContext, hw_device_ctx), row->cc_hw_device_ctx, "AVCodecContext.hw_device_ctx");
    eq(AV_PIX_FMT_CUDA, row->pix_fmt_cuda, "AV_PIX_FMT_CUDA");
    eq(AV_PIX_FMT_P010LE, row->pix_fmt_p010, "AV_PIX_FMT_P010LE");
    eq(offsetof(AVHWDeviceContext, hwctx), row->hwdev_hwctx, "AVHWDeviceContext.hwctx");
    eq(offsetof(AVBufferRef, data), ff::kBufferData, "AVBufferRef.data");
    eq(AV_PIX_FMT_YUV420P, ff::kPixFmtYuv420p, "AV_PIX_FMT_YUV420P");
    eq(AV_PIX_FMT_NV12, ff::kPixFmtNv12, "AV_PIX_FMT_NV12");
    eq(AV_PIX_FMT_YUV420P10LE, ff::kPixFmtYuv420p10, "AV_PIX_FMT_YUV420P10LE");
    eq(AV_HWDEVICE_TYPE_CUDA, ff::kHwDeviceCuda, "AV_HWDEVICE_TYPE_CUDA");
    eq(AVCOL_RANGE_MPEG, ff::kRangeMpeg, "AVCOL_RANGE_MPEG");
    eq(AVCOL_RANGE_JPEG, ff::kRangeJpeg, "AVCOL_RANGE_JPEG");
    eq(offsetof(AVFormatContext, nb_streams), ff::kFmtNbStreams, "AVFormatContext.nb_streams");
    eq(offsetof(AVFormatContext, streams), ff::kFmtStreams, "AVFormatContext.streams");
    eq(offsetof(AVFrame, data), ff::kFrameData, "AVFrame.data");
    eq(offsetof(AVFrame, linesize), ff::kFrameLinesize, "AVFrame.linesize");
    eq(offsetof(AVFrame, width), ff::kFrameWidth, "AVFrame.width");
    eq(offsetof(AVFrame, height), ff::kFrameHeight, "AVFrame.height");
    eq(offsetof(AVFrame, format), ff::kFrameFormat, "AVFrame.format");
    eq(offsetof(AVPacket, stream_index), ff::kPacketStreamIndex, "AVPacket.stream_index");
    eq(AVMEDIA_TYPE_VIDEO, ff::kMediaTypeVideo, "AVMEDIA_TYPE_VIDEO");
    eq(AV_PIX_FMT_RGB24, ff::kPixFmtRgb24, "AV_PIX_FMT_RGB24");
    eq(SWS_BILINEAR, ff::kSwsBilinear, "SWS_BILINEAR");
    eq(SWS_FULL_CHR_H_INT, ff::kSwsFullChrHInt, "SWS_FULL_CHR_H_INT");
    eq(SWS_ACCURATE_RND, ff::kSwsAccurateRnd, "SWS_ACCURATE_RND");
    eq(AVERROR_EOF, ff::kErrorEof, "AVERROR_EOF");
    eq(AVERROR(EAGAIN), ff::kErrorEagain, "AVERROR(EAGAIN)");
    eq(AV_LOG_QUIET, ff::kLogQuiet, "AV_LOG_QUIET");
    eq(AV_NUM_DATA_POINTERS, ff::kNumDataPointers, "AV_NUM_DATA_POINTERS");
    std::printf("ffmpeg_abi: libavformat %d headers %s the table\n", LIBAVFORMAT_VERSION_MAJOR, bad ? "DIFFER FROM" : "match");
    return bad ? 1 : 0;
}
