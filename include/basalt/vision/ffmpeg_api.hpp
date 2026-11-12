// basalt/vision/ffmpeg_api.hpp - FFmpeg's libraries, found and loaded at run time for video input.
//
// One release's libavformat, libavcodec, libavutil and libswscale are loaded together (never majors from two
// releases), newest release first, where the platform looks for application libraries (DynLib::open_app). Every
// function is looked up before video counts as on, and the four libraries must report the majors of one row of
// `ff::kAbiRows`. A machine without them still runs: video is off, with the reason. The libraries stay loaded for
// the life of the process.
#pragma once

#include <string>
#include <vector>

#include "basalt/vision/ffmpeg_abi.hpp"

namespace basalt::vision {

struct Ffmpeg {
    ff::AbiRow abi{};      ///< the loaded release's row
    std::string version;   ///< "FFmpeg 8 (libavformat 62.3.100)"
    std::string path;      ///< the file libavformat was loaded from

    ff::avutil_version_fn avutil_version = nullptr;
    ff::av_frame_alloc_fn av_frame_alloc = nullptr;
    ff::av_frame_free_fn av_frame_free = nullptr;
    ff::av_frame_unref_fn av_frame_unref = nullptr;
    ff::av_frame_apply_cropping_fn av_frame_apply_cropping = nullptr;
    ff::av_strerror_fn av_strerror = nullptr;
    ff::av_dict_set_fn av_dict_set = nullptr;
    ff::av_dict_free_fn av_dict_free = nullptr;
    ff::av_log_set_level_fn av_log_set_level = nullptr;
    ff::av_frame_clone_fn av_frame_clone = nullptr;
    ff::av_hwdevice_ctx_create_fn av_hwdevice_ctx_create = nullptr;
    ff::av_buffer_ref_fn av_buffer_ref = nullptr;
    ff::av_buffer_unref_fn av_buffer_unref = nullptr;
    ff::av_hwdevice_ctx_alloc_fn av_hwdevice_ctx_alloc = nullptr;
    ff::av_hwdevice_ctx_init_fn av_hwdevice_ctx_init = nullptr;
    ff::av_hwframe_transfer_data_fn av_hwframe_transfer_data = nullptr;

    ff::avformat_version_fn avformat_version = nullptr;
    ff::avformat_open_input_fn avformat_open_input = nullptr;
    ff::avformat_find_stream_info_fn avformat_find_stream_info = nullptr;
    ff::av_find_best_stream_fn av_find_best_stream = nullptr;
    ff::av_read_frame_fn av_read_frame = nullptr;
    ff::avformat_close_input_fn avformat_close_input = nullptr;

    ff::avcodec_version_fn avcodec_version = nullptr;
    ff::avcodec_alloc_context3_fn avcodec_alloc_context3 = nullptr;
    ff::avcodec_parameters_to_context_fn avcodec_parameters_to_context = nullptr;
    ff::avcodec_open2_fn avcodec_open2 = nullptr;
    ff::avcodec_send_packet_fn avcodec_send_packet = nullptr;
    ff::avcodec_receive_frame_fn avcodec_receive_frame = nullptr;
    ff::avcodec_free_context_fn avcodec_free_context = nullptr;
    ff::av_packet_alloc_fn av_packet_alloc = nullptr;
    ff::av_packet_free_fn av_packet_free = nullptr;
    ff::av_packet_unref_fn av_packet_unref = nullptr;

    ff::swscale_version_fn swscale_version = nullptr;
    ff::sws_getContext_fn sws_getContext = nullptr;
    ff::sws_scale_fn sws_scale = nullptr;
    ff::sws_freeContext_fn sws_freeContext = nullptr;
    ff::sws_getCoefficients_fn sws_getCoefficients = nullptr;
    ff::sws_getColorspaceDetails_fn sws_getColorspaceDetails = nullptr;
    ff::sws_setColorspaceDetails_fn sws_setColorspaceDetails = nullptr;

    /// FFmpeg's text for an error code.
    std::string error_text(int code) const;
};

/// Tries each row of `rows` in order and fills `out` from the first whose four libraries load and agree. False
/// with `why` (the names tried, or what the first library found lacks) otherwise.
bool load_ffmpeg(const std::vector<ff::AbiRow>& rows, Ffmpeg& out, std::string& why);

/// The process's FFmpeg, loaded from `ff::kAbiRows` on the first call; null with `why` when video is off.
const Ffmpeg* ffmpeg(std::string& why);

/// The line an encoder process prints at start to say whether it reads video: "VIDEO on <version> (<path>)" or
/// "VIDEO off <reason>". Loads FFmpeg if no call has yet.
std::string video_line();

}  // namespace basalt::vision
