// tests/platform/mapped_file_test.cpp - platform::MappedFile, file_size and pread_all on one scratch file. CPU only,
// no model. The same assertions hold on every platform: a whole-file view, a section at an offset, the refusals, and
// the view's hints (which may do nothing, but must not fault).
#include "basalt/platform/file.hpp"
#include "basalt/platform/mapped_file.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("mapped_file %-56s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}

uint8_t byte_at(uint64_t i) { return (uint8_t) ((i * 2654435761ull) >> 13); }

}  // namespace

int main() {
    using basalt::platform::MapOptions;
    using basalt::platform::MappedFile;

    const std::string path = basalt::platform::temp_dir() + "/basalt_mapped_file_test.bin";
    // an odd size, past two granularity units, so a section can start inside the file and end short of it
    const uint64_t total = 3 * 65536 + 1234;
    {
        std::FILE* f = basalt::platform::open_file_utf8(path, "wb");
        if (!f) { std::printf("mapped_file: cannot create %s\n", path.c_str()); return 1; }
        std::vector<uint8_t> buf(total);
        for (uint64_t i = 0; i < total; ++i) buf[i] = byte_at(i);
        std::fwrite(buf.data(), 1, buf.size(), f);
        std::fclose(f);
    }

    uint64_t sz = 0;
    check(basalt::platform::file_size(path, sz) && sz == total, "file_size of a regular file");
    check(!basalt::platform::file_size(path + ".missing", sz), "file_size of a missing file fails");
    check(!basalt::platform::file_size(basalt::platform::temp_dir(), sz), "file_size of a directory fails");

    std::string err;
    {
        auto m = MappedFile::open_ro(path, &err);
        check(m.has_value() && m->size() == total && m->file_size() == total, "whole file: size");
        bool same = m.has_value();
        for (uint64_t i = 0; same && i < total; ++i) same = m->data()[i] == byte_at(i);
        check(same, "whole file: every byte");

        MappedFile moved = std::move(*m);
        check(!*m && moved.data() != nullptr && moved.size() == total, "move transfers the view");
        moved.prefetch(0, total);
        moved.prefetch(total - 10, 100);
        const uint64_t released = moved.release(0, total);
        check(released <= total, "release reports at most the range");
        const uint64_t rel_huge = moved.release(0, UINT64_MAX);
        check(rel_huge <= total, "release clips huge length to the view");
        const uint64_t rel_wrap = moved.release(total - 10, UINT64_MAX);
        check(rel_wrap <= 10, "release clips wrapping off + len to the view");
        check(moved.data()[total - 1] == byte_at(total - 1), "bytes are still readable after release");
        moved.reset();
        check(!moved && moved.data() == nullptr && moved.size() == 0, "reset empties the view");
        moved.reset();
    }
    {
        // Unaligned offsets (4 KiB and arbitrary byte offset) work on both 4 KiB and 64 KiB page systems.
        for (uint64_t off : {4096ull, 123ull}) {
            MapOptions o;
            o.offset = off;
            o.bytes = 70000;
            auto m = MappedFile::open_ro(path, &err, o);
            check(m.has_value() && m->size() == 70000 && m->file_size() == total,
                  "unaligned offset " + std::to_string(off) + ": size");
            bool same = m.has_value();
            for (uint64_t i = 0; same && i < 70000; ++i) same = m->data()[i] == byte_at(off + i);
            check(same, "unaligned offset " + std::to_string(off) + ": every byte");
            if (m) {
                m->prefetch(0, 70000);
                check(m->release(0, 70000) <= 70000, "unaligned offset " + std::to_string(off) + ": release");
                check(m->data()[69999] == byte_at(off + 69999), "unaligned offset " + std::to_string(off) + ": readable after release");
            }
        }
    }
    {
        err.clear();
        MapOptions o;
        o.offset = total;
        o.bytes = 0;
        check(!MappedFile::open_ro(path, &err, o).has_value() && !err.empty(), "empty trailing range is refused");
    }
    {
        MapOptions o;
        o.offset = 65536;
        o.bytes = 70000;
        o.shared = true;
        auto m = MappedFile::open_ro(path, &err, o);
        check(m.has_value() && m->size() == 70000 && m->file_size() == total, "section: size");
        bool same = m.has_value();
        for (uint64_t i = 0; same && i < 70000; ++i) same = m->data()[i] == byte_at(65536 + i);
        check(same, "section: bytes are the file's from the offset");
    }
    {
        MapOptions o;
        o.offset = 65536;
        auto m = MappedFile::open_ro(path, &err, o);
        check(m.has_value() && m->size() == total - 65536, "offset with no length runs to the end");
    }
    {
        err.clear();
        MapOptions o;
        o.offset = total - 10;
        o.bytes = 11;
        check(!MappedFile::open_ro(path, &err, o).has_value() && !err.empty(), "a section past the end is refused");
        err.clear();
        check(!MappedFile::open_ro(path + ".missing", &err).has_value() && !err.empty(), "a missing file is refused");
    }
    {
        const int fd = basalt::platform::open_fd_utf8(path);
        std::vector<uint8_t> got(total);
        check(fd >= 0 && basalt::platform::pread_all(fd, got.data(), total, 0), "pread_all reads the whole file");
        bool same = true;
        for (uint64_t i = 0; same && i < total; ++i) same = got[i] == byte_at(i);
        check(same, "pread_all: every byte");
        std::vector<uint8_t> past(8);
        check(!basalt::platform::pread_all(fd, past.data(), 8, total - 4), "pread_all past the end fails");
        basalt::platform::close_fd(fd);
    }
    std::remove(path.c_str());

    return failures ? 1 : 0;
}
