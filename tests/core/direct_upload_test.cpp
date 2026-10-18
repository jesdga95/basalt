// DirectUploader: ranges at unaligned offsets and lengths, smaller and larger than a chunk, land byte for byte.
#include "basalt/core/direct_upload.hpp"

#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main() {
    const std::string path = "direct_upload_test.bin";
    const size_t size = (size_t) 40 << 20 | 777;
    std::vector<uint8_t> data(size);
    std::mt19937_64 rng(7);
    for (auto& b : data) b = (uint8_t) rng();
    if (FILE* f = basalt::platform::open_file_utf8(path, "wb")) {
        std::fwrite(data.data(), 1, size, f);
        std::fclose(f);
    } else {
        std::fprintf(stderr, "cannot write the test file\n");
        return 1;
    }
    struct Range { uint64_t off, len; };
    const Range ranges[] = {{0, 4096}, {1, 5}, {4095, 2}, {12345, 3 << 20}, {4096, 16 << 20}, {777, (17 << 20) + 13},
                            {(5u << 20) + 3, (35u << 20)}, {size - 100, 100}, {0, size}};
    basalt::core::DirectUploader up;
    int bad = 0;
    for (const Range& r : ranges) {
        void* d = nullptr;
        if (cudaMalloc(&d, r.len) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc failed\n"); return 1; }
        std::string err;
        std::vector<uint8_t> back(r.len);
        const bool ok = up.upload(path, r.off, r.len, d, err) &&
                        cudaMemcpy(back.data(), d, r.len, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        std::memcmp(back.data(), data.data() + r.off, r.len) == 0;
        cudaFree(d);
        if (!ok) { std::fprintf(stderr, "upload off %llu len %llu: %s\n", (unsigned long long) r.off, (unsigned long long) r.len, err.empty() ? "bytes differ" : err.c_str()); ++bad; }
        if (r.len <= basalt::core::DirectUploader::kChunk) {
            const uint8_t* s = up.staged(path, r.off, r.len, err);
            if (s == nullptr || std::memcmp(s, data.data() + r.off, r.len) != 0) {
                std::fprintf(stderr, "staged off %llu len %llu: %s\n", (unsigned long long) r.off, (unsigned long long) r.len, err.c_str());
                ++bad;
            }
        }
    }
    std::string err;
    if (up.upload(path, size - 10, 100, nullptr, err)) { std::fprintf(stderr, "a read past the end succeeded\n"); ++bad; }
    (void) std::remove(path.c_str());
    std::printf("direct_upload_test: %s (%s reads)\n", bad ? "FAIL" : "ok", up.unbuffered() ? "unbuffered" : "buffered");
    return bad ? 1 : 0;
}
