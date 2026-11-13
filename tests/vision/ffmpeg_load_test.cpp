// tests/vision/ffmpeg_load_test.cpp - the run-time FFmpeg loader (CPU only, every platform).
//
// A release nobody ships loads nothing and the reason names the file looked for. Where the machine has FFmpeg the
// process's copy must be one row of the table, with its file named; where it has none, the reason is printed and
// the test still passes (CI runners carry no FFmpeg).
#include "basalt/platform/dynlib.hpp"
#include "basalt/vision/ffmpeg_api.hpp"

#include <cstdio>
#include <string>

using namespace basalt::vision;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("ffmpeg %-62s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    const ff::AbiRow none{"99", 99, 99, 99, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Ffmpeg f;
    std::string why;
    check(!load_ffmpeg({none}, f, why), "a release nobody ships does not load");
    const std::string name = basalt::platform::DynLib::versioned_name("avformat", 99);
    check(why.find(name) != std::string::npos, "and the reason names " + name);

    std::string off;
    const Ffmpeg* sys = ffmpeg(off);
    const std::string line = video_line();
    std::printf("ffmpeg: %s\n", line.c_str());
    if (sys == nullptr) {
        check(line == "VIDEO off " + off, "no FFmpeg here: the line says off with the reason");
        return failures ? 1 : 0;
    }
    check((int) (sys->avformat_version() >> 16) == sys->abi.avformat,
          "the loaded release is a row of the table");
    check(!sys->path.empty(), "and the file it came from is named");
    check(line.rfind("VIDEO on FFmpeg ", 0) == 0, "the line says on with the release");
    check(sys->error_text(ff::kErrorEof).find("End of file") != std::string::npos, "FFmpeg's own text for AVERROR_EOF");
    return failures ? 1 : 0;
}
