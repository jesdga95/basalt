// tests/vision/preprocess_limit_test.cpp - the decode guard refuses a picture past the pixel budget before decoding.
//
// A PNG's IHDR alone says how many pixels the file claims, so the refusal happens without allocating a pixel buffer.
// The IDAT below is a stored deflate block of the one row a 1x1 image has, which keeps the accepted case a real
// decode rather than a header-only file.
#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::vision;

namespace {

uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xffffffffu;
    while (n-- > 0) {
        c ^= *p++;
        for (int k = 0; k < 8; ++k) {
            const uint32_t low = c & 1u;
            c >>= 1;
            if (low != 0) c ^= 0xedb88320u;
        }
    }
    return ~c;
}

uint32_t adler32(const uint8_t* p, size_t n) {
    uint32_t a = 1, b = 0;
    while (n-- > 0) {
        a = (a + *p++) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

struct Writer {
    std::vector<uint8_t> bytes;

    void be32(uint32_t v) {
        bytes.push_back((uint8_t) (v >> 24));
        bytes.push_back((uint8_t) (v >> 16));
        bytes.push_back((uint8_t) (v >> 8));
        bytes.push_back((uint8_t) v);
    }

    void chunk(const char* type, const std::vector<uint8_t>& data) {
        be32((uint32_t) data.size());
        const size_t start = bytes.size();
        bytes.insert(bytes.end(), type, type + 4);
        bytes.insert(bytes.end(), data.begin(), data.end());
        be32(crc32(bytes.data() + start, data.size() + 4));
    }
};

std::vector<uint8_t> png(int w, int h, const std::vector<uint8_t>& idat) {
    Writer p;
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    p.bytes.assign(sig, sig + 8);
    const std::vector<uint8_t> ihdr = {(uint8_t) (w >> 24), (uint8_t) (w >> 16), (uint8_t) (w >> 8), (uint8_t) w,
                                       (uint8_t) (h >> 24), (uint8_t) (h >> 16), (uint8_t) (h >> 8), (uint8_t) h,
                                       8, 2, 0, 0, 0};
    p.chunk("IHDR", ihdr);
    p.chunk("IDAT", idat);
    p.chunk("IEND", {});
    return p.bytes;
}

/// A 1x1 red RGB PNG: one filter-0 row in a stored deflate block, with its adler32.
std::vector<uint8_t> one_pixel() {
    const uint8_t row[4] = {0x00, 0xff, 0x00, 0x00};
    std::vector<uint8_t> idat = {0x78, 0x01, 0x01, (uint8_t) sizeof row, 0x00, 0xfb, 0xff};
    idat.insert(idat.end(), row, row + sizeof row);
    const uint32_t sum = adler32(row, sizeof row);
    idat.push_back((uint8_t) (sum >> 24));
    idat.push_back((uint8_t) (sum >> 16));
    idat.push_back((uint8_t) (sum >> 8));
    idat.push_back((uint8_t) sum);
    return png(1, 1, idat);
}

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    Image img;
    std::string err;

    // 100 MP is past the 64 MP default; the header is enough to refuse it.
    const std::vector<uint8_t> big = png(10000, 10000, {});
    check(!decode_image(big.data(), big.size(), img, err), "a header past the budget is refused");
    check(err.find("megapixels") != std::string::npos, "the refusal names the pixel limit");
    std::printf("oversized: %s\n", err.c_str());

    // A budget below the picture's pixel count refuses it too, and names that limit.
    const std::vector<uint8_t> small = png(64, 64, {});
    err.clear();
    check(!decode_image(small.data(), small.size(), img, err, 1024), "a smaller budget refuses the image");
    check(err.find("the limit is 1024 pixels") != std::string::npos, "the budget's own limit is named");
    std::printf("budget:    %s\n", err.c_str());

    // A real 1x1 image decodes: the guard does not refuse ordinary pictures.
    const std::vector<uint8_t> one = one_pixel();
    err.clear();
    check(decode_image(one.data(), one.size(), img, err), "a 1x1 image decodes");
    check(img.w == 1 && img.h == 1 && img.rgb.size() == 3 && img.rgb[0] == 255 && img.rgb[1] == 0,
          "the decoded pixels are the image's");
    check(err.empty(), "a decoded image has no error");

    if (failures == 0) std::printf("preprocess limit: ok\n");
    return failures == 0 ? 0 : 1;
}
