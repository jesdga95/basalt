// The loader: file ranges land byte for byte in host memory and (with a device present) in device memory, on any
// alignment and size, with every conversion, from a path or a descriptor, with items coalesced, counted exactly once,
// copies of bytes the plan itself places run after its reads, and a failure or a cancel reported.  The host cases need no GPU.
#include "basalt/core/loader.hpp"

#include "basalt/kernels/f16_bits.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
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
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }
    ~TestFile() { std::filesystem::remove(path); }
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
    for (const R& r : ranges) out.emplace_back(r.len + 8, 0xEE);
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

void descriptor(const TestFile& f) {
    const int fd = basalt::platform::open_fd_utf8(f.path);
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

}  // namespace

int main() {
    TestFile f((size_t) 40 << 20 | 777);
    host_ranges(f, 1);
    host_ranges(f, 8);
    coalescing(f);
    conversions(f);
    errors(f);
    host_aligned(f);
    adjacent_device_allocations();
    memory_sources(f);
    descriptor(f);
    device(f);
    std::printf("loader_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
