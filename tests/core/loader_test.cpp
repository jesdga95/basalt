// The loader: file ranges land byte for byte in host memory and (with a device present) in device memory, on any
// alignment and size, with every conversion, from a path or a descriptor, with items coalesced, counted exactly once,
// copies of bytes the plan itself places run after its reads, its steps between the two, the reads that land on a card
// before those that land in RAM (the start's bar says which), and a failure or a cancel reported.  The host cases need no GPU.
#include "basalt/core/loader.hpp"

#include "basalt/core/load_progress.hpp"

#include "basalt/kernels/f16_bits.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace basalt::core;

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}

struct TestFile {
    std::string path;
    std::vector<uint8_t> bytes;
    explicit TestFile(uint64_t n) : path((std::filesystem::current_path() / "loader_test.bin").string()), bytes(n) {
        std::mt19937_64 rng(7);
        for (uint8_t& b : bytes) b = (uint8_t) rng();
        std::FILE* f = basalt::platform::open_file_utf8(path, "wb");
        check(f != nullptr, "open test file");
        if (f != nullptr) {
            check(std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size(), "write test file");
            std::fclose(f);
        }
    }
    ~TestFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

bool has_device() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

// ranges at every kind of offset and length, into one host buffer each
void host_ranges(const TestFile& f, int threads) {
    struct R { uint64_t off, len; };
    const R ranges[] = {{0, 4096}, {1, 5}, {4095, 2}, {12345, 3 << 20}, {4096, 16 << 20}, {777, (17 << 20) + 13},
                        {(5u << 20) + 3, 35u << 20}, {f.bytes.size() - 100, 100}, {0, f.bytes.size()}};
    LoadPlan plan;
    const int file = plan.file(f.path);
    std::vector<std::vector<uint8_t>> out;
    for (const R& r : ranges) out.emplace_back(r.len + 8, (uint8_t) 0xEE);
    for (size_t i = 0; i < out.size(); ++i) {
        LoadItem it;
        it.file = file;
        it.off = ranges[i].off;
        it.bytes = ranges[i].len;
        it.host = out[i].data();
        plan.add(it);
    }
    Loader loader(threads);
    std::string err;
    check(loader.run(plan, err), "host ranges run");
    for (size_t i = 0; i < out.size(); ++i) {
        check(std::memcmp(out[i].data(), f.bytes.data() + ranges[i].off, ranges[i].len) == 0, "host range bytes");
        check(out[i][ranges[i].len] == 0xEE, "host range wrote past its end");
    }
}

// adjacent items in the file and the destination become one; the count follows
void coalescing(const TestFile& f) {
    LoadPlan plan;
    const int file = plan.file(f.path);
    std::vector<uint8_t> dst(1 << 20);
    for (int i = 0; i < 8; ++i) {
        LoadItem it;
        it.file = file;
        it.off = 4096 + (uint64_t) i * 1000;
        it.bytes = 1000;
        it.host = dst.data() + i * 1000;
        plan.add(it);
    }
    check(plan.items().size() == 1 && plan.items()[0].bytes == 8000, "adjacent items coalesce");
    LoadItem gap;
    gap.file = file;
    gap.off = 20000;
    gap.bytes = 10;
    gap.host = dst.data() + 9000;
    gap.count = 0;
    plan.add(gap);
    check(plan.items().size() == 2 && plan.counted() == 8000, "a gap starts an item; a count of 0 adds nothing");
    Loader loader;
    std::string err;
    check(loader.run(plan, err), "coalesced run");
    check(std::memcmp(dst.data(), f.bytes.data() + 4096, 8000) == 0, "coalesced bytes");
}

// The bar's share per destination comes from the plan: what each item counts, summed where it lands, the main card
// first, the other cards by ordinal, then RAM; a card the plan writes is listed though it counts nothing.  No device
// is touched: the plan is only described.
void shares_by_target() {
    LoadPlan plan;
    const int file = plan.file("unused.bin");
    std::vector<uint8_t> host(1 << 16);
    alignas(16) static uint8_t fake_dev[4][256];   // destinations the plan names, never written
    const auto add = [&](void* dev, int device, uint8_t* h, uint64_t off, uint64_t bytes, uint64_t count, Conv conv) {
        LoadItem it;
        it.file = file;
        it.off = off;
        it.bytes = bytes;
        it.dev = dev;
        it.device = device;
        it.host = h;
        it.conv = conv;
        it.count = count;
        plan.add(it);
    };
    add(nullptr, 0, host.data(), 0, 1000, LoadItem::kOut, Conv::Copy);           // RAM first in the plan
    add(fake_dev[0], 2, nullptr, 4096, 200, LoadItem::kOut, Conv::Copy);         // a second card
    add(fake_dev[1], 1, nullptr, 8192, 400, LoadItem::kOut, Conv::Bf16High);     // the main card: 200 bytes land
    add(fake_dev[2], 3, nullptr, 9000, 64, 0, Conv::Copy);                       // a card that counts nothing
    add(nullptr, 0, host.data() + 2000, 20000, 500, 0, Conv::Copy);              // a RAM image counted elsewhere
    add(fake_dev[3], 1, nullptr, 30000, 100, LoadItem::kOut, Conv::Copy);
    const auto got = plan.counted_by_target(1);
    const std::vector<std::pair<std::string, uint64_t>> want = {{"cuda1", 300}, {"cuda2", 200}, {"cuda3", 0}, {"ram", 1000}};
    check(got == want, "shares: per destination, the main card first, then by ordinal, then ram");
    uint64_t sum = 0;
    for (const auto& [t, b] : got) sum += b;
    check(sum == plan.counted(), "shares: they sum to the plan's count");
    LoadPlan cards_only;
    LoadItem it;
    it.file = cards_only.file("unused.bin");
    it.bytes = 10;
    it.dev = fake_dev[0];
    cards_only.add(it);
    check(cards_only.counted_by_target(0) == std::vector<std::pair<std::string, uint64_t>>{{"cuda0", 10}},
          "shares: no ram share when the plan writes no host memory");
}

void conversions(const TestFile& f) {
    const uint64_t n = 4096 * 5 + 8;   // elements: whole, at an odd file offset
    const uint64_t off = 4099;
    for (Conv c : {Conv::Bf16High, Conv::ToF16, Conv::WidenF16, Conv::Bf16ToF32}) {
        const uint64_t in = n * (c == Conv::Bf16High || c == Conv::ToF16 ? 4 : 2);
        std::vector<uint8_t> out(conv_out_bytes(c, in));
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(f.path);
        it.off = off;
        it.bytes = in;
        it.host = out.data();
        it.conv = c;
        plan.add(it);
        Loader loader;
        std::string err;
        check(loader.run(plan, err), "conversion run");
        const uint8_t* src = f.bytes.data() + off;
        bool ok = true;
        for (uint64_t i = 0; i < n && ok; ++i) {
            switch (c) {
                case Conv::Bf16High: ok = std::memcmp(out.data() + i * 2, src + i * 4 + 2, 2) == 0; break;
                case Conv::ToF16: {
                    float v;
                    std::memcpy(&v, src + i * 4, 4);
                    const uint16_t h = basalt::kernels::f16_from_f32(v);
                    ok = std::memcmp(out.data() + i * 2, &h, 2) == 0;
                    break;
                }
                case Conv::WidenF16: {
                    uint16_t h;
                    std::memcpy(&h, src + i * 2, 2);
                    const float v = basalt::kernels::f32_from_f16(h);
                    ok = std::memcmp(out.data() + i * 4, &v, 4) == 0;
                    break;
                }
                case Conv::Bf16ToF32: {
                    uint16_t h;
                    std::memcpy(&h, src + i * 2, 2);
                    const uint32_t v = (uint32_t) h << 16;
                    ok = std::memcmp(out.data() + i * 4, &v, 4) == 0;
                    break;
                }
                default: break;
            }
        }
        check(ok, "converted elements");
    }
}

// bf16 -> ggml's block formats over several of a conversion's pieces, to host and to device memory, against ggml's reference rules
void bf16_blocks(Conv conv) {
    const bool q8 = conv == Conv::Bf16ToQ8_0;
    const char* tag = q8 ? "q8_0" : "q4_0";
    const int bb = q8 ? 34 : 18;            // bytes a block of 32
    const uint64_t n = (5ull << 20) + 96;   // elements, whole blocks; three 4 MiB pieces and a tail
    const uint64_t off = 4098;
    std::vector<uint16_t> vals(n);
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.f, 0.02f);
    for (uint64_t i = 0; i < n; ++i) {
        const float x = i % 4096 < 32 ? 0.f : g(rng);   // an all-zero block too
        uint32_t u;
        std::memcpy(&u, &x, 4);
        vals[i] = (uint16_t) (u >> 16);
    }
    const std::string path = (std::filesystem::current_path() / (std::string("loader_") + tag + "_test.bin")).string();
    {
        std::vector<uint8_t> pad(off, 0x5a);
        std::FILE* fp = basalt::platform::open_file_utf8(path, "wb");
        check(fp != nullptr, "open block test file");
        if (fp == nullptr) return;
        check(std::fwrite(pad.data(), 1, pad.size(), fp) == pad.size() &&
                  std::fwrite(vals.data(), 2, vals.size(), fp) == vals.size(),
              "write block test file");
        std::fclose(fp);
    }
    std::vector<uint8_t> want(n / 32 * (uint64_t) bb);
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.f, mx = 0.f, x[32];
        for (int i = 0; i < 32; ++i) {
            const uint32_t u = (uint32_t) vals[b * 32 + i] << 16;
            std::memcpy(&x[i], &u, 4);
            if (amax < std::fabs(x[i])) { amax = std::fabs(x[i]); mx = x[i]; }
        }
        uint8_t* o = &want[b * (uint64_t) bb];
        if (q8) {   // ggml's quantize_row_q8_0_ref
            const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
            const uint16_t dh = basalt::kernels::f16_from_f32(d);
            std::memcpy(o, &dh, 2);
            for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) roundf(x[i] * id);
        } else {    // ggml's quantize_row_q4_0_ref
            const float d = mx / -8.f, id = d ? 1.f / d : 0.f;
            const uint16_t dh = basalt::kernels::f16_from_f32(d);
            std::memcpy(o, &dh, 2);
            for (int i = 0; i < 16; ++i) {
                const int x0 = std::min(15, (int) (int8_t) (x[i] * id + 8.5f));
                const int x1 = std::min(15, (int) (int8_t) (x[16 + i] * id + 8.5f));
                o[2 + i] = (uint8_t) (x0 | (x1 << 4));
            }
        }
    }
    check(conv_out_bytes(conv, n * 2) == want.size(), q8 ? "q8_0 output bytes" : "q4_0 output bytes");
    for (const bool to_dev : {false, true}) {
        if (to_dev && !has_device()) continue;
        std::vector<uint8_t> got(want.size());
        void* dev = nullptr;
        if (to_dev && cudaMalloc(&dev, got.size()) != cudaSuccess) {
            check(false, "block device buffer");
            continue;
        }
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(path);
        it.off = off;
        it.bytes = n * 2;
        if (to_dev) it.dev = dev;
        else it.host = got.data();
        it.conv = conv;
        plan.add(it);
        Loader loader;
        std::string err;
        check(loader.run(plan, err), q8 ? "q8_0 run" : "q4_0 run");
        if (to_dev) {
            cudaMemcpy(got.data(), dev, got.size(), cudaMemcpyDeviceToHost);
            cudaFree(dev);
        }
        check(got == want, to_dev ? (q8 ? "q8_0 blocks on the device" : "q4_0 blocks on the device")
                                  : (q8 ? "q8_0 blocks in host memory" : "q4_0 blocks in host memory"));
    }
    {   // a range that is not whole blocks
        std::vector<uint8_t> out(64);
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(path);
        it.bytes = 100;
        it.host = out.data();
        it.conv = conv;
        plan.add(it);
        Loader loader;
        std::string err;
        check(!loader.run(plan, err), q8 ? "a q8_0 range of partial blocks is refused" : "a q4_0 range of partial blocks is refused");
    }
    std::filesystem::remove(path);
}

void errors(const TestFile& f) {
    std::vector<uint8_t> dst(4096);
    {   // a read past the end of the file
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(f.path);
        it.off = f.bytes.size() - 10;
        it.bytes = 100;
        it.host = dst.data();
        plan.add(it);
        Loader loader;
        std::string err;
        check(!loader.run(plan, err) && !err.empty(), "a read past the end fails with a reason");
    }
    {   // a file that is not there
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(f.path + ".missing");
        it.bytes = 10;
        it.host = dst.data();
        plan.add(it);
        Loader loader;
        std::string err;
        check(!loader.run(plan, err), "a missing file fails");
    }
    {   // an item with no destination
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(f.path);
        it.bytes = 10;
        plan.add(it);
        Loader loader;
        std::string err;
        check(!loader.run(plan, err), "an item without a destination fails");
    }
}

// 4 KiB aligned host buffers, offsets and lengths: the read goes straight into the destination
void host_aligned(const TestFile& f) {
    constexpr size_t kLen = 20u << 20;
    uint8_t* dst = (uint8_t*) basalt::platform::alloc_aligned(kLen + 4096, 4096);
    check(dst != nullptr, "aligned allocation");
    if (dst == nullptr) return;
    std::memset(dst, 0xEE, kLen + 4096);
    for (int threads : {1, 4}) {
        LoadPlan plan;
        const int file = plan.file(f.path);
        for (size_t at = 0; at < kLen; at += 4u << 20) {   // several items, each a whole number of blocks
            LoadItem it;
            it.file = file;
            it.off = 8192 + at;
            it.bytes = 4u << 20;
            it.host = dst + at;
            plan.add(it);
        }
        check(plan.items().size() == 1, "aligned host items join into one read");
        Loader loader(threads);
        std::string err;
        check(loader.run(plan, err), "aligned host run");
        check(std::memcmp(dst, f.bytes.data() + 8192, kLen) == 0, "aligned host bytes");
        check(dst[kLen] == 0xEE, "aligned host write past its end");
        std::memset(dst, 0xEE, kLen);
    }
    basalt::platform::free_aligned(dst);
}

// two device items whose addresses touch are two items: one copy cannot span two allocations
void adjacent_device_allocations() {
    LoadPlan plan;
    const int file = plan.file("unused");
    LoadItem a;
    a.file = file;
    a.off = 0;
    a.bytes = 4096;
    a.dev = (void*) 0x100000;
    a.device = 0;
    plan.add(a);
    LoadItem b = a;
    b.off = 4096;
    b.dev = (void*) (0x100000 + 4096);
    plan.add(b);
    check(plan.items().size() == 2, "touching device items stay separate");
    LoadPlan host;
    uint8_t buf[8192];
    LoadItem c;
    c.file = host.file("unused");
    c.bytes = 4096;
    c.host = buf;
    host.add(c);
    c.off = 4096;
    c.host = buf + 4096;
    host.add(c);
    check(host.items().size() == 1, "touching host items join");
    LoadPlan other;
    LoadItem d = a;
    d.file = other.file("second");
    other.add(d);
    d.file = other.file("unused");
    other.add(d);
    plan.append(other);
    check(plan.files().size() == 2 && plan.items().size() == 4 && plan.items()[2].file == 1 && plan.items()[3].file == 0,
          "an appended plan's files are taken in");
    LoadPlan twice;   // a range a device item reads already becomes a copy of its destination; another stays a read
    LoadItem e = a;
    e.file = twice.file("unused");
    e.conv = Conv::Bf16High;
    twice.add(e);
    LoadPlan image;
    uint8_t img[8192];
    LoadItem h;
    h.file = image.file("unused");
    h.off = e.off;
    h.bytes = 4096;
    h.conv = Conv::Bf16High;
    h.host = img;
    image.add(h);
    h.off = 1 << 20;
    h.host = img + 2048;
    image.add(h);
    const uint64_t before = twice.counted();
    twice.append_reusing(image);
    const auto& t = twice.items();
    check(t.size() == 3 && t[2].from == nullptr && t[2].off == (1u << 20) && t[1].from == (const uint8_t*) e.dev &&
              t[1].from_device == 0 && t[1].bytes == 2048 && t[1].conv == Conv::Copy && t[1].host == img,
          "a range read for a device becomes a copy of it");
    check(twice.counted() == before + 2048, "a copy counts nothing, a read its bytes");
    LoadPlan region;   // a device region's items to a host image of it
    uint8_t mirror[8192];
    LoadItem r = a;
    r.file = region.file("unused");
    region.add(r);
    r.dev = (void*) 0x900000;
    region.add(r);
    region.to_host((void*) 0x100000, 8192, mirror);
    check(region.items()[0].host == mirror && region.items()[0].dev == nullptr && region.items()[1].dev == (void*) 0x900000,
          "a region's items land in its host image, the others stay");
}

// copies of bytes the same plan reads run after the reads: a host copy of a host read, listed before it
void memory_sources(const TestFile& f) {
    for (int threads : {1, 8}) {
        constexpr uint64_t kLen = (20u << 20) + 77;
        std::vector<uint8_t> first(kLen), second(kLen);
        LoadPlan plan;
        LoadItem copy;
        copy.from = first.data();
        copy.bytes = kLen;
        copy.host = second.data();
        copy.count = 0;
        plan.add(copy);
        LoadItem read;
        read.file = plan.file(f.path);
        read.off = 333;
        read.bytes = kLen;
        read.host = first.data();
        plan.add(read);
        check(plan.counted() == kLen, "a copy counts what it is given");
        Loader loader(threads);
        std::string err;
        check(loader.run(plan, err) && std::memcmp(second.data(), f.bytes.data() + 333, kLen) == 0,
              "a host copy of a host read of the same plan");
    }
}

// a plan's steps run once its reads have landed and before its copies from memory; an appended plan brings its steps,
// a plan of steps alone runs them, and a step that fails fails the run
void steps(const TestFile& f) {
    for (int threads : {1, 8}) {
        constexpr uint64_t kLen = (20u << 20) + 77;
        std::vector<uint8_t> first(kLen), second(kLen, 0xEE);
        LoadPlan plan;
        LoadItem copy;
        copy.from = first.data();
        copy.bytes = kLen;
        copy.host = second.data();
        copy.count = 0;
        plan.add(copy);
        LoadPlan reads;
        LoadItem read;
        read.file = reads.file(f.path);
        read.off = 333;
        read.bytes = kLen;
        read.host = first.data();
        reads.add(read);
        int ran = 0;
        bool landed = false, before_copy = false;
        reads.after_reads([&](std::string&) {
            ++ran;
            landed = std::memcmp(first.data(), f.bytes.data() + 333, kLen) == 0;
            before_copy = second[0] == 0xEE && second[kLen - 1] == 0xEE;
            return true;
        });
        plan.append(reads);
        check(plan.steps().size() == 1, "an appended plan brings its steps");
        Loader loader(threads);
        std::string err;
        check(loader.run(plan, err), "a run with a step");
        check(ran == 1 && landed && before_copy, "a step runs once, after the reads and before the copies");
        check(std::memcmp(second.data(), f.bytes.data() + 333, kLen) == 0, "the copies after a step");
    }
    {
        LoadPlan only;
        bool ran = false;
        only.after_reads([&](std::string&) { ran = true; return true; });
        Loader loader;
        std::string err;
        check(!only.empty() && loader.run(only, err) && ran, "a plan of steps alone runs them");
    }
    {
        std::vector<uint8_t> dst(4096), out(4096, 0xEE);
        LoadPlan plan;
        LoadItem read;
        read.file = plan.file(f.path);
        read.bytes = dst.size();
        read.host = dst.data();
        plan.add(read);
        LoadItem copy;
        copy.from = dst.data();
        copy.bytes = dst.size();
        copy.host = out.data();
        copy.count = 0;
        plan.add(copy);
        plan.after_reads([](std::string& e) { e = "step refused"; return false; });
        Loader loader;
        std::string err;
        check(!loader.run(plan, err) && err == "step refused" && out[0] == 0xEE, "a failed step fails the run before the copies");
    }
}

void descriptor(const TestFile& f) {
    const int fd = basalt::platform::open_fd_utf8(f.path);
    check(fd >= 0, "open descriptor");
    if (fd < 0) return;
    std::vector<uint8_t> dst(100000);
    LoadPlan plan;
    LoadItem it;
    it.file = plan.file_fd(fd);
    it.off = 5000;
    it.bytes = dst.size();
    it.host = dst.data();
    plan.add(it);
    Loader loader;
    std::string err;
    check(loader.run(plan, err) && std::memcmp(dst.data(), f.bytes.data() + 5000, dst.size()) == 0, "a descriptor source");
    basalt::platform::close_fd(fd);
}

void device(const TestFile& f) {
    if (!has_device()) {
        std::printf("loader_test: no CUDA device, the device cases are not run\n");
        return;
    }
    {   // more pieces than workers: the staging slots are reused and each copy is waited for before its slot is read into
        for (int threads : {1, 2}) {
            constexpr uint64_t kItems = 24, kLen = (5u << 20) + 123;
            void* d = nullptr;
            check(cudaMalloc(&d, kItems * kLen) == cudaSuccess, "cudaMalloc");
            LoadPlan plan;
            const int file = plan.file(f.path);
            for (uint64_t i = 0; i < kItems; ++i) {
                LoadItem it;
                it.file = file;
                it.off = 999 + i * 1000;
                it.bytes = kLen;
                it.dev = (uint8_t*) d + i * kLen;
                plan.add(it);
            }
            LoadItem big;   // one item of several pieces for the single worker
            big.file = file;
            big.off = 4096;
            big.bytes = 33u << 20;
            void* bigd = nullptr;
            check(cudaMalloc(&bigd, big.bytes) == cudaSuccess, "cudaMalloc");
            big.dev = bigd;
            plan.add(big);
            Loader loader(threads);
            std::string err;
            check(loader.run(plan, err), "slot reuse run");
            std::vector<uint8_t> back(kLen), bigback(big.bytes);
            bool ok = true;
            for (uint64_t i = 0; i < kItems && ok; ++i) {
                ok = cudaMemcpy(back.data(), (uint8_t*) d + i * kLen, kLen, cudaMemcpyDeviceToHost) == cudaSuccess &&
                     std::memcmp(back.data(), f.bytes.data() + 999 + i * 1000, kLen) == 0;
            }
            check(ok, "slot reuse bytes");
            check(cudaMemcpy(bigback.data(), bigd, big.bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                      std::memcmp(bigback.data(), f.bytes.data() + 4096, big.bytes) == 0,
                  "slot reuse bytes of a long item");
            cudaFree(d);
            cudaFree(bigd);
        }
    }
    {   // blobs whose sizes and offsets are not multiples of the block size, into slots of one arena, with staging
        // that sits beside an odd-sized pinned allocation
        void* odd = nullptr;
        check(cudaHostAlloc(&odd, 1000, cudaHostAllocPortable) == cudaSuccess, "pinned allocation");
        constexpr uint64_t kBlob = 1382401, kSlot = (kBlob + 255) / 256 * 256, kCount = 6, kBase = (2u << 20) + 777;
        void* arena = nullptr;
        check(cudaMalloc(&arena, kSlot * kCount) == cudaSuccess, "cudaMalloc");
        LoadPlan plan;
        const int file = plan.file(f.path);
        const uint64_t order[kCount] = {3, 1, 4, 0, 5, 2};
        for (uint64_t i = 0; i < kCount; ++i) {
            LoadItem it;
            it.file = file;
            it.off = kBase + order[i] * kBlob;
            it.bytes = kBlob;
            it.dev = (uint8_t*) arena + i * kSlot;
            plan.add(it);
        }
        Loader loader(3);
        std::string err;
        check(loader.run(plan, err), "unaligned blob run");
        std::vector<uint8_t> back(kBlob);
        bool ok = true;
        for (uint64_t i = 0; i < kCount && ok; ++i)
            ok = cudaMemcpy(back.data(), (uint8_t*) arena + i * kSlot, kBlob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                 std::memcmp(back.data(), f.bytes.data() + kBase + order[i] * kBlob, kBlob) == 0;
        check(ok, "unaligned blob bytes");
        cudaFree(arena);
        cudaFreeHost(odd);
    }
    struct R { uint64_t off, len; };
    const R ranges[] = {{0, 4096}, {1, 5}, {12345, 3 << 20}, {777, (17 << 20) + 13}, {(5u << 20) + 3, 35u << 20},
                        {f.bytes.size() - 100, 100}};
    for (const R& r : ranges) {
        void* d = nullptr;
        check(cudaMalloc(&d, r.len) == cudaSuccess, "cudaMalloc");
        LoadPlan plan;
        LoadItem it;
        it.file = plan.file(f.path);
        it.off = r.off;
        it.bytes = r.len;
        it.dev = d;
        plan.add(it);
        Loader loader;
        std::string err;
        std::vector<uint8_t> back(r.len);
        check(loader.run(plan, err), "device range run");
        check(cudaMemcpy(back.data(), d, r.len, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  std::memcmp(back.data(), f.bytes.data() + r.off, r.len) == 0,
              "device range bytes");
        cudaFree(d);
    }
    {   // a host source to the device, and a converted range
        std::vector<uint8_t> src(1 << 20);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t) (i * 31);
        void* d = nullptr;
        void* w = nullptr;
        check(cudaMalloc(&d, src.size()) == cudaSuccess && cudaMalloc(&w, 8 << 20) == cudaSuccess, "cudaMalloc");
        LoadPlan plan;
        LoadItem a;
        a.from = src.data();
        a.bytes = src.size();
        a.dev = d;
        a.count = 0;
        plan.add(a);
        LoadItem b;
        b.file = plan.file(f.path);
        b.off = 8192;
        b.bytes = 4 << 20;
        b.dev = w;
        b.conv = Conv::WidenF16;
        plan.add(b);
        Loader loader;
        std::string err;
        check(loader.run(plan, err), "host source and conversion run");
        std::vector<uint8_t> back(src.size()), wide(8 << 20);
        cudaMemcpy(back.data(), d, back.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(wide.data(), w, wide.size(), cudaMemcpyDeviceToHost);
        check(back == src, "host source bytes");
        bool ok = true;
        for (uint64_t i = 0; i < (4u << 20) / 2 && ok; ++i) {
            uint16_t h;
            std::memcpy(&h, f.bytes.data() + 8192 + i * 2, 2);
            const float v = basalt::kernels::f32_from_f16(h);
            ok = std::memcmp(wide.data() + i * 4, &v, 4) == 0;
        }
        check(ok, "device widened elements");
        cudaFree(d);
        cudaFree(w);
    }
    {   // a device read copied back to host memory and to a second allocation, in the same plan
        constexpr uint64_t kLen = (9u << 20) + 5;
        void* d = nullptr;
        void* d2 = nullptr;
        uint8_t* h = nullptr;
        check(cudaMalloc(&d, kLen) == cudaSuccess && cudaMalloc(&d2, kLen) == cudaSuccess &&
                  cudaHostAlloc((void**) &h, kLen, cudaHostAllocPortable) == cudaSuccess, "allocations");
        LoadPlan plan;
        LoadItem back;
        back.from = (const uint8_t*) d;
        back.from_device = 0;
        back.bytes = kLen;
        back.host = h;
        back.count = 0;
        plan.add(back);
        LoadItem twin = back;
        twin.host = nullptr;
        twin.dev = d2;
        twin.device = 0;
        plan.add(twin);
        LoadItem read;
        read.file = plan.file(f.path);
        read.off = 4097;
        read.bytes = kLen;
        read.dev = d;
        plan.add(read);
        Loader loader;
        std::string err;
        check(loader.run(plan, err), "device copies run");
        std::vector<uint8_t> twin_back(kLen);
        cudaMemcpy(twin_back.data(), d2, kLen, cudaMemcpyDeviceToHost);
        check(std::memcmp(h, f.bytes.data() + 4097, kLen) == 0, "a device read copied to host memory");
        check(std::memcmp(twin_back.data(), f.bytes.data() + 4097, kLen) == 0, "a device read copied to another allocation");
        cudaFree(d);
        cudaFree(d2);
        cudaFreeHost(h);
    }
}

// A start's pass reads the card's share first though the plan queued its RAM item first: one worker, so the order is the
// order the bytes land in, read back from the bar's lines (`into cuda0`, `weights`, `into ram`).  Last: it announces a
// start's total, which every later placement would count against.
void cards_first(const TestFile& f) {
    if (!has_device()) return;
    constexpr uint64_t kLen = 16u << 20;   // one staging piece each: every placement is a line of the bar
    void* d = nullptr;
    uint8_t* h = (uint8_t*) basalt::platform::alloc_aligned(kLen, 4096);
    check(cudaMalloc(&d, kLen) == cudaSuccess && h != nullptr, "cards first: buffers");
    LoadPlan plan;
    const int file = plan.file(f.path);
    LoadItem ram;
    ram.file = file;
    ram.off = 8192;
    ram.bytes = kLen;
    ram.host = h;
    plan.add(ram);
    LoadItem card = ram;
    card.off = 4096 * 3 + 5;
    card.host = nullptr;
    card.dev = d;
    plan.add(card);
    const std::string path = (std::filesystem::current_path() / "loader_test_bar.txt").string();
    std::fflush(stderr);
    check(std::freopen(path.c_str(), "w", stderr) != nullptr, "cards first: capture the bar");
    Loader loader(1);
    std::string err;
    const bool ok = place_start(loader, plan, err);
    std::fflush(stderr);
    const std::string after = path + ".after";   // let go of the file before reading it (Windows)
    check(std::freopen(after.c_str(), "w", stderr) != nullptr, "cards first: release the bar");
    check(ok, "cards first: run");
    std::vector<uint8_t> back(kLen);
    check(cudaMemcpy(back.data(), d, kLen, cudaMemcpyDeviceToHost) == cudaSuccess &&
              std::memcmp(back.data(), f.bytes.data() + card.off, kLen) == 0 && std::memcmp(h, f.bytes.data() + ram.off, kLen) == 0,
          "cards first: bytes");
    std::ifstream in(path);
    std::vector<std::string> seq;   // "cuda0", "ram" and the weights counts, in order
    std::vector<std::string> shares;   // the share lines, as printed
    for (std::string l; std::getline(in, l);) {
        unsigned long long a = 0, b = 0;
        char into[32] = {};
        if (l.rfind("basalt load: share ", 0) == 0) shares.push_back(l.substr(19));
        else if (l == "basalt load: stages cuda0 ram") seq.push_back("stages");
        else if (std::sscanf(l.c_str(), "basalt load: into %31s", into) == 1) seq.push_back(into);
        else if (std::sscanf(l.c_str(), "basalt load: weights %llu %llu", &a, &b) == 2) seq.push_back(std::to_string(a));
    }
    in.close();
    const std::vector<std::string> want = {"0", "stages", "cuda0", std::to_string(kLen), "ram", std::to_string(2 * kLen)};
    check(seq == want, "cards first: the card's bytes land first, then the bar reads ram");
    const std::string k = std::to_string(kLen);
    check(shares == std::vector<std::string>{"cuda0 0 " + k, "ram 0 " + k, "cuda0 " + k + " " + k, "ram " + k + " " + k},
          "cards first: each share announced at 0, then once at its own total");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(after, ec);
    cudaFree(d);
    basalt::platform::free_aligned(h);
}

}  // namespace

int main() {
    TestFile f((size_t) 40 << 20 | 777);
    host_ranges(f, 1);
    host_ranges(f, 8);
    coalescing(f);
    shares_by_target();
    conversions(f);
    bf16_blocks(Conv::Bf16ToQ8_0);
    bf16_blocks(Conv::Bf16ToQ4_0);
    errors(f);
    host_aligned(f);
    adjacent_device_allocations();
    memory_sources(f);
    steps(f);
    descriptor(f);
    device(f);
    cards_first(f);
    std::printf("loader_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
