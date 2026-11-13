// src/models/deepseek41/file_reads_test.cpp - FileReads lands every blob byte for byte, whatever its alignment (no GPU).
#include "models/deepseek41/file_reads.hpp"

#include "basalt/platform/file.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef BASALT_TEST_TMPDIR
#define BASALT_TEST_TMPDIR "."
#endif

static int g_fail = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                  \
        }                                                              \
    } while (0)

using basalt::models::deepseek41::FileReads;

int main() {
    const std::string path = std::string(BASALT_TEST_TMPDIR) + "/file_reads_test.bin";
    constexpr size_t kBytes = (9u << 20) + 12345;   // not a multiple of the alignment: the last blob ends mid-page
    std::vector<uint8_t> data(kBytes);
    std::mt19937 rng(7);
    for (uint8_t& b : data) b = (uint8_t) rng();
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f && std::fwrite(data.data(), 1, data.size(), f) == data.size());
        if (f) std::fclose(f);
    }
    const int fd = basalt::platform::open_fd_utf8(path);
    CHECK(fd >= 0);
    FileReads reads;
    std::string err;
    CHECK(reads.open(path, fd, err));
    std::printf("file_reads_test: %s\n", reads.unbuffered() ? "unbuffered" : "through the file cache");

    // blobs at aligned and unaligned offsets, shorter and longer than a piece, one ending at the end of the file
    const uint64_t offs[] = {0, 4096, 1, 4095, 123457, (3u << 20) - 7, kBytes - 2500000, kBytes - 777};
    const uint64_t lens[] = {4096, 1u << 20, 2500001, 10, (1u << 20) + 1, 3333333, 2500000, 777};
    for (int round = 0; round < 3; ++round) {
        std::vector<std::vector<uint8_t>> dst;
        std::vector<FileReads::Blob> blobs;
        for (size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); ++i) dst.emplace_back(lens[i], 0);
        for (size_t i = 0; i < dst.size(); ++i) blobs.push_back({offs[i], lens[i], dst[i].data()});
        reads.start(blobs);
        CHECK(reads.busy());
        CHECK(reads.wait(err));
        CHECK(!reads.busy());
        for (size_t i = 0; i < dst.size(); ++i) CHECK(std::memcmp(dst[i].data(), data.data() + offs[i], lens[i]) == 0);
    }
    // a blob past the end of the file fails the batch, and the next batch works again
    std::vector<uint8_t> past(100);
    reads.start({{kBytes - 50, 100, past.data()}});
    CHECK(!reads.wait(err));
    std::vector<uint8_t> again(5000);
    reads.start({{10, 5000, again.data()}});
    CHECK(reads.wait(err));
    CHECK(std::memcmp(again.data(), data.data() + 10, 5000) == 0);
    CHECK(reads.wait(err));   // nothing out: returns at once

    basalt::platform::close_fd(fd);
    std::remove(path.c_str());
    if (g_fail) return 1;
    std::printf("file_reads_test: ok\n");
    return 0;
}
