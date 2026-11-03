// src/vision/ffmpeg_api.cpp - see basalt/vision/ffmpeg_api.hpp.
#include "basalt/vision/ffmpeg_api.hpp"

#include <iterator>
#include <mutex>
#include <type_traits>

#include "basalt/platform/dynlib.hpp"

namespace basalt::vision {

namespace {

using platform::DynLib;

std::string version_text(unsigned v) {
    return std::to_string(v >> 16) + "." + std::to_string((v >> 8) & 255) + "." + std::to_string(v & 255);
}

// Looks up every function of `out` in its library; false with `why` naming the first one missing.
bool bind(const DynLib& u, const DynLib& f, const DynLib& c, const DynLib& s, Ffmpeg& out, std::string& why) {
    bool ok = true;
    auto get = [&](const DynLib& lib, const char* name, auto& fn) {
        if (!ok) return;
        fn = lib.sym<std::remove_reference_t<decltype(fn)>>(name);
        if (fn == nullptr) {
            ok = false;
            why = "FFmpeg's libraries lack " + std::string(name);
        }
    };
    get(u, "avutil_version", out.avutil_version);
    get(u, "av_frame_alloc", out.av_frame_alloc);
    get(u, "av_frame_free", out.av_frame_free);
    get(u, "av_frame_unref", out.av_frame_unref);
    get(u, "av_frame_apply_cropping", out.av_frame_apply_cropping);
    get(u, "av_strerror", out.av_strerror);
    get(u, "av_dict_set", out.av_dict_set);
    get(u, "av_dict_free", out.av_dict_free);
    get(u, "av_log_set_level", out.av_log_set_level);
    get(u, "av_frame_clone", out.av_frame_clone);
    get(u, "av_hwdevice_ctx_create", out.av_hwdevice_ctx_create);
    get(u, "av_buffer_ref", out.av_buffer_ref);
    get(u, "av_buffer_unref", out.av_buffer_unref);
    get(u, "av_hwdevice_ctx_alloc", out.av_hwdevice_ctx_alloc);
    get(u, "av_hwdevice_ctx_init", out.av_hwdevice_ctx_init);
    get(u, "av_hwframe_transfer_data", out.av_hwframe_transfer_data);
    get(f, "avformat_version", out.avformat_version);
    get(f, "avformat_open_input", out.avformat_open_input);
    get(f, "avformat_find_stream_info", out.avformat_find_stream_info);
    get(f, "av_find_best_stream", out.av_find_best_stream);
    get(f, "av_read_frame", out.av_read_frame);
    get(f, "avformat_close_input", out.avformat_close_input);
    get(c, "avcodec_version", out.avcodec_version);
    get(c, "avcodec_alloc_context3", out.avcodec_alloc_context3);
    get(c, "avcodec_parameters_to_context", out.avcodec_parameters_to_context);
    get(c, "avcodec_open2", out.avcodec_open2);
    get(c, "avcodec_send_packet", out.avcodec_send_packet);
    get(c, "avcodec_receive_frame", out.avcodec_receive_frame);
    get(c, "avcodec_free_context", out.avcodec_free_context);
    get(c, "av_packet_alloc", out.av_packet_alloc);
    get(c, "av_packet_free", out.av_packet_free);
    get(c, "av_packet_unref", out.av_packet_unref);
    get(s, "swscale_version", out.swscale_version);
    get(s, "sws_getContext", out.sws_getContext);
    get(s, "sws_scale", out.sws_scale);
    get(s, "sws_freeContext", out.sws_freeContext);
    get(s, "sws_getCoefficients", out.sws_getCoefficients);
    get(s, "sws_getColorspaceDetails", out.sws_getColorspaceDetails);
    get(s, "sws_setColorspaceDetails", out.sws_setColorspaceDetails);
    return ok;
}

}  // namespace

std::string Ffmpeg::error_text(int code) const {
    char buf[256] = {0};
    if (av_strerror == nullptr || av_strerror(code, buf, sizeof buf) < 0) return "error " + std::to_string(code);
    return buf;
}

bool load_ffmpeg(const std::vector<ff::AbiRow>& rows, Ffmpeg& out, std::string& why) {
    std::string tried;
    for (const ff::AbiRow& row : rows) {
        const std::string fname = DynLib::versioned_name("avformat", row.avformat);
        DynLib f = DynLib::open_app(fname);
        if (!f) {
            tried += (tried.empty() ? "" : ", ") + fname;
            continue;
        }
        // the rest of the release: libavformat brought its own dependencies, so these find the same files
        DynLib u = DynLib::open_app(DynLib::versioned_name("avutil", row.avutil));
        DynLib c = DynLib::open_app(DynLib::versioned_name("avcodec", row.avcodec));
        DynLib s = DynLib::open_app(DynLib::versioned_name("swscale", row.swscale));
        const char* missing = !u ? "avutil" : !c ? "avcodec" : !s ? "swscale" : nullptr;
        if (missing != nullptr) {
            const int major = !u ? row.avutil : !c ? row.avcodec : row.swscale;
            why = "FFmpeg " + std::string(row.release) + " is incomplete: " + fname + " loaded, " +
                  DynLib::versioned_name(missing, major) + " did not";
            return false;
        }
        Ffmpeg got;
        if (!bind(u, f, c, s, got, why)) return false;
        const unsigned vf = got.avformat_version(), vc = got.avcodec_version(), vu = got.avutil_version(),
                       vs = got.swscale_version();
        if ((int) (vf >> 16) != row.avformat || (int) (vc >> 16) != row.avcodec || (int) (vu >> 16) != row.avutil ||
            (int) (vs >> 16) != row.swscale) {
            why = "FFmpeg's libraries are from different releases (libavformat " + version_text(vf) + ", libavcodec " +
                  version_text(vc) + ", libavutil " + version_text(vu) + ", libswscale " + version_text(vs) + ")";
            return false;
        }
        got.abi = row;
        got.version = "FFmpeg " + std::string(row.release) + " (libavformat " + version_text(vf) + ")";
        got.path = f.path();
        got.av_log_set_level(ff::kLogQuiet);   // failures come back as error codes; the library prints nothing
        out = got;   // a DynLib never closes its library, so the functions stay valid for the life of the process
        return true;
    }
    why = "no FFmpeg libraries found (tried " + tried + ")";
    return false;
}

const Ffmpeg* ffmpeg(std::string& why) {
    static std::once_flag once;
    static Ffmpeg lib;
    static std::string reason;
    static bool ok = false;
    std::call_once(once, [] { ok = load_ffmpeg({std::begin(ff::kAbiRows), std::end(ff::kAbiRows)}, lib, reason); });
    if (!ok) why = reason;
    return ok ? &lib : nullptr;
}

std::string video_line() {
    std::string why;
    const Ffmpeg* f = ffmpeg(why);
    return f != nullptr ? "VIDEO on " + f->version + (f->path.empty() ? "" : " from " + f->path) : "VIDEO off " + why;
}

}  // namespace basalt::vision
