// tests/platform/file_write_test.cpp - the file seam's writes and directory calls: create, write at offsets, flush,
// read back, list with sizes and write times, replace by rename, touch, remove, nested directories and free space, on a
// directory with a non-ASCII name (the bytes must reach the OS as UTF-8 on both platforms). CPU only.
#include "basalt/platform/direct_file.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(bool ok, const char* what) {
    std::printf("file_write %-48s -> %s\n", what, ok ? "ok" : "FAILED");
    failures |= !ok;
}

const basalt::platform::DirFile* find(const std::vector<basalt::platform::DirFile>& files, const std::string& name) {
    for (const auto& f : files)
        if (f.name == name) return &f;
    return nullptr;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    const std::string dir = temp_dir() + "/basalt-file-write-" + std::to_string((long long) process_id()) +
                            "/\xc3\xa9-\xe6\x97\xa5";   // é-日 under a fresh parent
    check(make_dirs(dir), "nested directories are made");
    check(make_dirs(dir), "making them again is fine");

    const std::string a = dir + "/a.bin", b = dir + "/b.bin";
    std::vector<uint8_t> data(3 * 4096);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) (i * 7 + 1);
    int fd = open_write_utf8(a);
    check(fd >= 0, "a file is created for writing");
    check(pwrite_all(fd, data.data() + 4096, 8192, 4096) && pwrite_all(fd, data.data(), 4096, 0),
          "writes land at their offsets, out of order");
    check(sync_fd(fd), "the writes are flushed to the device");
    close_fd(fd);
    uint64_t size = 0;
    check(file_size(a, size) && size == data.size(), "the file has the bytes written");
    std::vector<uint8_t> back(data.size());
    fd = open_fd_utf8(a);
    check(fd >= 0 && pread_all(fd, back.data(), back.size(), 0) && back == data, "they read back the same");
    close_fd(fd);

    fd = open_write_utf8(b);
    check(fd >= 0 && pwrite_all(fd, data.data(), 100, 0), "a second file");
    close_fd(fd);
    std::vector<DirFile> files;
    check(list_files(dir, files) && files.size() == 2 && find(files, "a.bin") && find(files, "b.bin") &&
              find(files, "a.bin")->bytes == data.size() && find(files, "b.bin")->bytes == 100,
          "the listing names both with their sizes");
    const int64_t now = file_time_now();
    check(find(files, "a.bin") && now - find(files, "a.bin")->mtime >= -2 && now - find(files, "a.bin")->mtime < 600,
          "a write time is on the listing's clock");

    check(rename_replace(b, a), "a rename replaces the file there");
    check(list_files(dir, files) && files.size() == 1 && find(files, "a.bin") && find(files, "a.bin")->bytes == 100,
          "the replaced file has the moved bytes");
    check(touch_file(a), "a file is touched");
    check(disk_free_bytes(dir) > 0, "the volume reports free space");
    check(remove_file(a) && !file_size(a, size), "a file is removed");
    check(list_files(dir, files) && files.empty(), "the directory is empty again");

    const int direct = open_write_unbuffered(dir + "/c.bin");
    if (direct >= 0) {   // where the platform offers unbuffered writes: aligned buffer, offset and count
        void* buf = DirectFile::alloc_aligned(8192);
        if (buf != nullptr) {
            std::memset(buf, 0x5a, 8192);
            check(pwrite_all(direct, buf, 8192, 4096) && sync_fd(direct), "unbuffered aligned writes");
            DirectFile::free_aligned(buf);
        }
        close_fd(direct);
        remove_file(dir + "/c.bin");
    }
    remove_file(dir);
    remove_file(dir.substr(0, dir.rfind('/')));
    std::printf("file_write: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
