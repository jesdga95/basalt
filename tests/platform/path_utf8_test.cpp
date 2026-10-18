// tests/platform/path_utf8_test.cpp - a path the engine carries as UTF-8 reaches the OS as those same bytes, and the
// bytes the OS names come back as UTF-8. CPU only, no model: one temp directory and a few files.
//
// What is being proven is the seam, not the file system: `fopen` and `_open` read a path in the machine's ANSI code
// page on Windows, so an accent or a CJK name survives at one call site and is replaced by "?" at another. Every
// open here goes through `basalt::platform`, which converts to UTF-16 on Windows and hands the bytes to the OS on
// Linux, so the same assertion holds on both.
//
// The second case (`--name`) is about the other half: the path a user typed. This program runs itself as a child
// with the fixture name as one argument, and the child checks that the bytes it was given are the bytes the program
// was compiled with. Windows decodes argv with the process's ANSI code page, which only becomes UTF-8 when the image
// carries the application manifest (src/platform/basalt.exe.manifest), so this case fails in a build without it.
// The fixture directory itself is made through `std::filesystem`, whose narrow strings the C++ library also reads
// with the process code page: without the manifest that directory and the one the seam asks for are two different
// places, so the cases above fail too, which is the point of the test.
#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
// The fixture's directory name: an accented Latin letter, Japanese, and Cyrillic, spelled as UTF-8 bytes. A profile
// name with an accent and a pack directory with a CJK name are the two shapes a real pack path takes.
const char* const kName = "\xc3\xa9-\xe6\x97\xa5\xe6\x9c\xac-\xd0\x96";   // é-日本-Ж

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("path_utf8 %-56s -> %s%s\n", what.c_str(), ok ? "ok" : "FAILED", ok ? "" : "   <-- WRONG");
    failures |= !ok;
}

/// The name with its non-ASCII bytes spelled out, so a mangled path is visible in a log that stays ASCII.
std::string esc(const std::string& s) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    for (const unsigned char c : s) {
        if (c >= 0x20 && c < 0x7f) out += (char) c;
        else { out += "\\x"; out += kHex[c >> 4]; out += kHex[c & 0xf]; }
    }
    return out;
}

/// The path as one spelled with `/` everywhere: the OS answers with its own separator, and the bytes of the name are
/// what is under test here, not how the directory part is spelled.
std::string slashes(const std::string& s) {
    std::string out = s;
    for (char& c : out) if (c == '\\') c = '/';
    return out;
}

/// The last component, with either separator.
std::string tail(const std::string& s) {
    const size_t at = s.find_last_of("/\\");
    return at == std::string::npos ? s : s.substr(at + 1);
}

std::vector<uint8_t> pattern(size_t n) {
    std::vector<uint8_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) (i * 31u + (i >> 9) * 7u);
    return p;
}

bool write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::FILE* f = basalt::platform::open_file_utf8(path, "wb");
    if (f == nullptr) return false;
    const size_t got = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return got == bytes.size();
}

/// Reads the whole file back through the stream seam.
bool read_file(const std::string& path, const std::vector<uint8_t>& want) {
    std::FILE* f = basalt::platform::open_file_utf8(path, "rb");
    if (f == nullptr) return false;
    std::vector<uint8_t> got(want.size());
    const size_t n = std::fread(got.data(), 1, got.size(), f);
    std::fclose(f);
    return n == want.size() && got == want;
}

/// Reads at two offsets through the descriptor seam, the way an expert or PLE section is read.
bool read_positional(const std::string& path, const std::vector<uint8_t>& want) {
    const int fd = basalt::platform::open_fd_utf8(path);
    if (fd < 0) return false;
    bool ok = true;
    std::vector<uint8_t> buf(64);
    for (const uint64_t off : {(uint64_t) 0, (uint64_t) (want.size() - buf.size()), (uint64_t) (want.size() / 2)}) {
        const int64_t got = basalt::platform::pread(fd, buf.data(), buf.size(), off);
        if (got != (int64_t) buf.size()) { ok = false; break; }
        if (std::memcmp(buf.data(), want.data() + off, buf.size()) != 0) { ok = false; break; }
    }
    // a 64-bit seek through the stream seam, past where a 32-bit one would wrap
    std::FILE* f = basalt::platform::open_file_utf8(path, "rb");
    if (f == nullptr) { ok = false; } else {
        const long long at = (long long) (want.size() - 8);
        if (basalt::platform::fseek64(f, at, SEEK_SET) != 0) ok = false;
        uint64_t tail8 = 0;
        if (std::fread(&tail8, 8, 1, f) != 1) ok = false;
        uint64_t want_tail;
        std::memcpy(&want_tail, want.data() + want.size() - 8, 8);
        if (tail8 != want_tail) ok = false;
        if (basalt::platform::file_fd(f) < 0) ok = false;
        std::fclose(f);
    }
    basalt::platform::close_fd(fd);
    return ok;
}

/// The directory the fixtures live in: the system temp directory plus the non-ASCII name.
std::string fixture_dir() {
    return basalt::platform::temp_dir() + "/" + kName;
}

int seam_cases() {
    const std::string dir = fixture_dir();
    const std::string file = dir + "/mod\xc3\xa8le-\xe6\x97\xa5.basalt";   // modèle-日.basalt
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        check(false, "create " + esc(dir));
        return failures;
    }
    const std::vector<uint8_t> bytes = pattern(4096);
    check(write_file(file, bytes), "write through a UTF-8 path");
    check(read_file(file, bytes), "read it back through the stream seam");
    check(read_positional(file, bytes), "read it back through pread and fseek64");

    // the absolute form of the path, and that form opened again: the name the OS answers with has to be the same
    // bytes going the other way, or a pack path made absolute stops naming the pack
    std::string abs;
    check(basalt::platform::absolute_path(file, abs), "absolute_path on a UTF-8 path");
    check(slashes(abs) == slashes(dir) + "/" + tail(file), "absolute_path leaves the name alone");
    check(read_file(abs, bytes), "the absolute path opens");
    std::printf("path_utf8 %-56s -> %s\n", "absolute form", esc(abs).c_str());

    // the working directory, and a path relative to it: `_chdir` is an ANSI call, `_wchdir` is not
    std::string back;
    const bool have_cwd = basalt::platform::absolute_path(".", back);
    check(have_cwd, "absolute_path of the current directory");
    if (have_cwd && basalt::platform::set_working_dir(dir) == 0) {
        std::string here;
        check(basalt::platform::absolute_path(".", here) && slashes(here) == slashes(dir),
              "set_working_dir took the UTF-8 path");
        const std::string base = tail(file);
        check(read_file(base, bytes), "a path relative to it opens");
        check(basalt::platform::set_working_dir(back) == 0, "back to the directory we came from");
    } else {
        check(false, "set_working_dir took the UTF-8 path");
    }

    // this program's own path, named by the OS, opened through the seam: the OS-to-engine direction
    const std::string self = basalt::platform::module_path();
    check(!self.empty() && basalt::platform::open_file_utf8(self, "rb") != nullptr, "module_path names a readable file");
    std::printf("path_utf8 %-56s -> %s\n", "this program", esc(self).c_str());

    std::filesystem::remove_all(dir, ec);
    check(!ec, "the fixture directory is removed");
    return failures;
}

/// A path longer than the 260 the legacy API refuses. `longPathAware` in the manifest is what lets the OS open it;
/// a CRT that caps its own path handling still refuses, and that is reported rather than assumed.
int long_path_case() {
    const std::string base = fixture_dir() + "/deep";
    std::string dir = base;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (int i = 0; i < 8; ++i) {
        dir += "/\xc3\xa9\xe6\x97\xa5\xd0\x96-dir-" + std::string(20, (char) ('a' + i));
        std::filesystem::create_directories(dir, ec);
    }
    const std::string file = dir + "/\xc3\xa9-\xe6\x97\xa5.basalt";
    const std::vector<uint8_t> bytes = pattern(1024);
    const bool wrote = write_file(file, bytes);
    std::printf("path_utf8 %-56s %zu chars\n", "path length", file.size());
    check(wrote && read_file(file, bytes), "a path past 260 chars opens");
    std::filesystem::remove_all(base, ec);
    return failures;
}

/// The child mode: the fixture name arrived as one argument. It has to be the bytes the program was compiled with
/// (argv decoded as UTF-8), and it has to open.
int argv_case(const std::string& name) {
    std::printf("path_utf8 %-56s -> %s\n", "child argument bytes", esc(name).c_str());
    check(name == kName, "argv carried the UTF-8 bytes of the name");
    const std::string dir = basalt::platform::temp_dir() + "/" + name;
    const std::string file = dir + "/pack.basalt";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        check(false, "create the directory the argument names");
        return failures;
    }
    const std::vector<uint8_t> bytes = pattern(512);
    check(write_file(file, bytes), "write through the path built from argv");
    check(read_positional(file, bytes), "read it back through the seam");
    std::filesystem::remove_all(dir, ec);
    return failures;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--name") == 0) return argv_case(argv[2]) ? 1 : 0;
    seam_cases();
    long_path_case();
    // The argument case runs as a child of this program, so the name reaches it the way a pack path typed by a user
    // reaches a tool: as one argument, decoded by the OS on the way in.
    const std::string self = basalt::platform::module_path();
    if (self.empty()) {
        check(false, "module_path to re-run this program");
    } else {
        const int code = basalt::platform::spawn_and_wait({self, "--name", kName});
        std::printf("path_utf8 %-56s %d\n", "child exit code", code);
        check(code == 0, "the child saw the UTF-8 name in argv and opened it");
    }
    std::printf("path_utf8_test: %s\n", failures ? "FAILED" : "OK");
    return failures;
}
