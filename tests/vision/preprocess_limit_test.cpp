// tests/vision/preprocess_limit_test.cpp - the decode guard refuses a picture past the pixel budget before decoding.
//
// A PNG's IHDR alone says how many pixels the file claims, so the refusal happens without allocating a pixel buffer.
// The IDAT below is a stored deflate block of the one row a 1x1 image has, which keeps the accepted case a real
// decode rather than a header-only file.
#include "basalt/platform/file.hpp"
#include "basalt/vision/encoder_cpu.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

    // Malformed GGUF projector files must be refused cleanly (no div-by-zero, wraparound OOB, or stack overflow).
    auto write_gguf = [](const std::string& path, const std::vector<uint8_t>& data) {
        std::FILE* f = basalt::platform::open_file_utf8(path, "wb");
        if (!f) return false;
        const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
        std::fclose(f);
        return ok;
    };
    auto append_u32 = [](std::vector<uint8_t>& b, uint32_t v) {
        const size_t p = b.size(); b.resize(p + 4); std::memcpy(b.data() + p, &v, 4);
    };
    auto append_u64 = [](std::vector<uint8_t>& b, uint64_t v) {
        const size_t p = b.size(); b.resize(p + 8); std::memcpy(b.data() + p, &v, 8);
    };
    auto append_f32 = [](std::vector<uint8_t>& b, float v) {
        const size_t p = b.size(); b.resize(p + 4); std::memcpy(b.data() + p, &v, 4);
    };
    auto append_str = [&](std::vector<uint8_t>& b, const std::string& s) {
        append_u64(b, (uint64_t) s.size());
        b.insert(b.end(), s.begin(), s.end());
    };
    auto make_base_gguf = [&](uint64_t n_tensors, uint64_t extra_kv, float std0 = 0.5f) {
        std::vector<uint8_t> b;
        append_u32(b, 0x46554747u);
        append_u32(b, 3u);
        append_u64(b, n_tensors);
        append_u64(b, 11u + extra_kv);
        auto kv_str = [&](const char* k, const char* v) { append_str(b, k); append_u32(b, 8u); append_str(b, v); };
        auto kv_u32 = [&](const char* k, uint32_t v) { append_str(b, k); append_u32(b, 4u); append_u32(b, v); };
        auto kv_f32_3 = [&](const char* k, float a, float c, float d) {
            append_str(b, k); append_u32(b, 9u); append_u32(b, 6u); append_u64(b, 3u);
            append_f32(b, a); append_f32(b, c); append_f32(b, d);
        };
        kv_str("general.architecture", "clip");
        kv_str("clip.projector_type", "qwen3vl_merger");
        kv_u32("clip.vision.image_size", 224);
        kv_u32("clip.vision.patch_size", 16);
        kv_u32("clip.vision.embedding_length", 1152);
        kv_u32("clip.vision.feed_forward_length", 4304);
        kv_u32("clip.vision.block_count", 1);
        kv_u32("clip.vision.attention.head_count", 16);
        kv_u32("clip.vision.projection_dim", 2560);
        kv_f32_3("clip.vision.image_mean", 0.5f, 0.5f, 0.5f);
        kv_f32_3("clip.vision.image_std", std0, 0.5f, 0.5f);
        return b;
    };
    const std::string tmp = basalt::platform::temp_dir() + "/basalt_mmproj_bad_test.gguf";
    {
        // Case 1: general.alignment == 0
        auto b = make_base_gguf(0, 1);
        append_str(b, "general.alignment"); append_u32(b, 4u); append_u32(b, 0u);
        b.resize(b.size() + 64, 0);
        check(write_gguf(tmp, b), "wrote alignment=0 fixture");
        Mmproj mp;
        err.clear();
        check(!mp.open(basalt::core::FileRange(tmp), err) && !err.empty(), "mmproj refuses alignment == 0");
    }
    {
        // Case 2: tensor offset 64-bit wraparound
        auto b = make_base_gguf(1, 0);
        append_str(b, "t0"); append_u32(b, 1u); append_u64(b, 1u); append_u32(b, 0u); append_u64(b, UINT64_MAX - 8);
        b.resize(b.size() + 64, 0);
        check(write_gguf(tmp, b), "wrote wraparound tensor fixture");
        Mmproj mp;
        err.clear();
        check(!mp.open(basalt::core::FileRange(tmp), err) && !err.empty(), "mmproj refuses wrapping tensor offset");
    }
    {
        // Case 3: deeply nested metadata array
        std::vector<uint8_t> b;
        append_u32(b, 0x46554747u); append_u32(b, 3u); append_u64(b, 0u); append_u64(b, 1u);
        append_str(b, "deep");
        for (int d = 0; d < 20; ++d) { append_u32(b, 9u); append_u32(b, 9u); append_u64(b, 1u); }
        append_u32(b, 4u); append_u32(b, 1u);
        check(write_gguf(tmp, b), "wrote nested array fixture");
        Mmproj mp;
        err.clear();
        check(!mp.open(basalt::core::FileRange(tmp), err) && !err.empty(), "mmproj refuses deeply nested arrays");
    }
    {
        // Case 4: zero image_std
        auto b = make_base_gguf(0, 0, 0.0f);
        b.resize(b.size() + 64, 0);
        check(write_gguf(tmp, b), "wrote zero std fixture");
        Mmproj mp;
        err.clear();
        check(!mp.open(basalt::core::FileRange(tmp), err) && !err.empty(), "mmproj refuses zero image_std");
    }
    std::remove(tmp.c_str());

    // CpuEncoder must refuse uninitialized or odd/zero canvas grids cleanly.
    {
        CpuEncoder enc;
        std::vector<float> out;
        check(!enc.encode({}, {0, 0}, out), "uninitialized CpuEncoder refuses encode");
        check(!enc.encode(std::vector<float>(4 * 768, 0.f), {16, 64}, out), "CpuEncoder refuses gx == 1");
        check(!enc.encode(std::vector<float>(4 * 768, 0.f), {64, 16}, out), "CpuEncoder refuses gy == 1");
        check(!enc.encode_padded(img, out), "uninitialized CpuEncoder refuses encode_padded");
    }

    if (failures == 0) std::printf("preprocess limit: ok\n");
    return failures == 0 ? 0 : 1;
}
