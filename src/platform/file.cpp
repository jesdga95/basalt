// src/platform/file.cpp - see include/basalt/platform/file.hpp.
//
// One implementation per platform behind one spelling per call. The Linux half is the plain POSIX call the engine
// made there before; the Windows half is the wide-call form of it, plus what reading a CRT descriptor needs there.
#include "basalt/platform/file.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <direct.h>   // _wchdir, _wfullpath
#include <fcntl.h>    // _O_RDONLY, _O_BINARY
#include <io.h>       // _wopen, _waccess, _close, _fileno, _get_osfhandle
#include <sys/stat.h>  // _S_IREAD, _S_IWRITE
#else
#include <fcntl.h>    // O_RDONLY, posix_fadvise
#include <sys/stat.h>  // ::stat
#include <unistd.h>   // ::open, ::close, ::pread, ::access, ::chdir, ::readlink, fileno
#endif

namespace basalt::platform {

#if defined(_WIN32)
// --------------------------------------------------------------------------------------------- Windows

std::wstring to_wide(const std::string& utf8) {
    // -1 for the length so the terminator travels with the string; the result keeps its own instead. No
    // MB_ERR_INVALID_CHARS: bytes that are not valid UTF-8 become the replacement character, so the open that
    // follows them fails on the name rather than the conversion refusing it.
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (n <= 0) return std::wstring();   // an empty string is the only one that converts to nothing
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &w[0], n);
    w.pop_back();
    return w;
}

namespace {
/// The way back, for a path the OS names (its own module path, a resolved path, the temp directory). An empty
/// result means the name could not be read or is not representable.
std::string to_utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    s.pop_back();
    return s;
}

/// This thread's handle for reading `fd`, valid until the read that asked for it is done. A handle opened without
/// FILE_FLAG_OVERLAPPED serializes the I/O issued through it, so threads reading through one shared descriptor would
/// take turns: each read opens the same file again and gets a file object of its own. The handle is not kept - a
/// caller that closed its descriptor has to have let go of the file, or nothing can replace it. Returns the shared
/// descriptor's handle when the file cannot be reopened (it still reads correctly, only in turn).
HANDLE read_handle(HANDLE cur) {
    if (cur == INVALID_HANDLE_VALUE) return INVALID_HANDLE_VALUE;
    wchar_t path[4096];
    const DWORD cap = (DWORD) (sizeof path / sizeof path[0]);
    const DWORD len = GetFinalPathNameByHandleW(cur, path, cap, VOLUME_NAME_DOS);
    if (len == 0 || len >= cap) return cur;
    const HANDLE own = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    return own == INVALID_HANDLE_VALUE ? cur : own;
}
}  // namespace

std::FILE* open_file_utf8(const std::string& path, const std::string& mode) {
    return _wfopen(to_wide(path).c_str(), to_wide(mode).c_str());
}

int open_fd_utf8(const std::string& path) {
    return _wopen(to_wide(path).c_str(), _O_RDONLY | _O_BINARY);
}

void close_fd(int fd) {
    if (fd >= 0) _close(fd);
}

bool absolute_path(const std::string& path, std::string& out) {
    const std::wstring w = to_wide(path);
    if (w.empty()) return false;
    wchar_t real[4096];
    if (_wfullpath(real, w.c_str(), sizeof real / sizeof real[0] - 1) == nullptr) return false;
    out = to_utf8(real);
    return !out.empty();
}

int set_working_dir(const std::string& path) {
    return _wchdir(to_wide(path).c_str());
}

bool path_executable(const std::string& path) {
    // The mode asks only that the file exist: there is no executable bit to test here.
    return _waccess(to_wide(path).c_str(), 0) == 0;
}

std::string module_path() {
    wchar_t exe[4096];
    const DWORD cap = (DWORD) (sizeof exe / sizeof exe[0]) - 1;
    const DWORD n = GetModuleFileNameW(nullptr, exe, cap);
    if (n == 0 || n >= cap) return std::string();
    exe[n] = 0;
    return to_utf8(exe);
}

std::string temp_dir() {
    wchar_t dir[MAX_PATH];
    const DWORD cap = (DWORD) (sizeof dir / sizeof dir[0]);
    const DWORD n = GetTempPathW(cap, dir);
    if (n == 0 || n >= cap) return ".";
    std::string s = to_utf8(dir);
    // The OS names the directory with its own trailing separator; the engine joins its own. A bare drive root
    // keeps its separator: without it "C:" names a directory relative to the current one on that drive.
    while (s.size() > 3 && (s.back() == '\\' || s.back() == '/')) s.pop_back();
    return s.empty() ? std::string(".") : s;
}

int64_t pread(int fd, void* dst, size_t n, uint64_t off) {
    if (fd < 0) return -1;   // asking the CRT for a negative descriptor's handle stops the process here
    const HANDLE shared = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    const HANDLE h = read_handle(shared);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const DWORD want = (DWORD) (n < (size_t) (1ull << 30) ? n : (size_t) (1ull << 30));
    OVERLAPPED ov{};
    ov.Offset = (DWORD) (off & 0xFFFFFFFFull);
    ov.OffsetHigh = (DWORD) (off >> 32);
    DWORD got = 0;
    if (!ReadFile(h, dst, want, &got, &ov)) {
        // A read that ran into the end of the file reports the bytes it did get as an error, so `got` stands.
        // ERROR_IO_PENDING means the handle is an overlapped one: this thread has no other read outstanding on it,
        // so waiting for that request is the whole of what a synchronous read would have done.
        const DWORD e = GetLastError();
        if (e == ERROR_IO_PENDING) {
            got = 0;
            if (!GetOverlappedResult(h, &ov, &got, TRUE) && GetLastError() != ERROR_HANDLE_EOF) {
                if (h != shared) CloseHandle(h);
                return -1;
            }
        } else if (e != ERROR_HANDLE_EOF) {
            if (h != shared) CloseHandle(h);
            return -1;
        }
    }
    if (h != shared) CloseHandle(h);   // the read is done: the file is no longer this call's to hold open
    return (int64_t) got;
}

bool file_size(const std::string& path, uint64_t& out) {
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExW(to_wide(path).c_str(), GetFileExInfoStandard, &d)) return false;
    if (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) return false;   // stat's S_ISREG, in the form Windows gives
    ULARGE_INTEGER sz;
    sz.HighPart = d.nFileSizeHigh;
    sz.LowPart = d.nFileSizeLow;
    out = sz.QuadPart;
    return true;
}

int fseek64(std::FILE* f, int64_t off, int whence) {
    return _fseeki64(f, (long long) off, whence);
}

int file_fd(std::FILE* f) {
    return _fileno(f);
}

void advise_dontneed(int fd, uint64_t off, uint64_t bytes) {
    // Nothing to ask: the cache manager takes clean pages back on pressure by itself, and a handle takes no range
    // hint. Bytes read through a descriptor are cached unless the caller opted out of that cache.
    (void) fd; (void) off; (void) bytes;
}

int open_unbuffered(int fd) {
    // No unbuffered descriptor on this platform. A handle that bypasses the cache is opened by path with
    // FILE_FLAG_NO_BUFFERING, and every read through it has to line up with the volume's sector size; what this call
    // is given is a descriptor, and a name is not recoverable from it. The caller that gets -1 reads the same bytes
    // through its mapping, which is the file cache doing the copy the unbuffered read would have saved.
    (void) fd;
    return -1;
}

int open_unbuffered(const std::string& path) {
    // The same answer for the path form: those ranges are read through the file cache here, so a caller that gets -1
    // opens the file ordinarily and the bytes arrive by the cached route.
    (void) path;
    return -1;
}

int open_write_utf8(const std::string& path) {
    return _wopen(to_wide(path).c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
}

int open_write_unbuffered(const std::string& path) {
    // As for reads: a caller that gets -1 writes through the file cache.
    (void) path;
    return -1;
}

bool pwrite_all(int fd, const void* src, uint64_t n, uint64_t off) {
    if (fd < 0) return false;
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) return false;
    const uint8_t* p = (const uint8_t*) src;
    while (n > 0) {
        const DWORD want = (DWORD) (n < (1ull << 30) ? n : (1ull << 30));
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD put = 0;
        if (!WriteFile(h, p, want, &put, &ov)) {
            if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult(h, &ov, &put, TRUE)) return false;
        }
        if (put == 0) return false;
        p += put;
        off += put;
        n -= put;
    }
    return true;
}

bool sync_fd(int fd) {
    if (fd < 0) return false;
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    return h != INVALID_HANDLE_VALUE && FlushFileBuffers(h) != 0;
}

namespace {
std::filesystem::path fs_path(const std::string& utf8) { return std::filesystem::path(to_wide(utf8)); }
std::string fs_name(const std::filesystem::path& p) { return to_utf8(p.filename().wstring().c_str()); }
}  // namespace

#else
// ----------------------------------------------------------------------------------------------- POSIX

std::FILE* open_file_utf8(const std::string& path, const std::string& mode) {
    return std::fopen(path.c_str(), mode.c_str());
}

int open_fd_utf8(const std::string& path) {
    return ::open(path.c_str(), O_RDONLY);
}

void close_fd(int fd) {
    if (fd >= 0) ::close(fd);
}

bool absolute_path(const std::string& path, std::string& out) {
    char real[4096];
    if (::realpath(path.c_str(), real) == nullptr) return false;
    out = real;
    return true;
}

int set_working_dir(const std::string& path) {
    return ::chdir(path.c_str());
}

bool path_executable(const std::string& path) {
    return ::access(path.c_str(), X_OK) == 0;
}

std::string module_path() {
    char exe[4096];
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return std::string();
    exe[n] = 0;
    return std::string(exe);
}

std::string temp_dir() {
    return "/tmp";
}

int64_t pread(int fd, void* dst, size_t n, uint64_t off) {
    return (int64_t) ::pread(fd, dst, n, (off_t) off);
}

bool file_size(const std::string& path, uint64_t& out) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) return false;
    out = (uint64_t) st.st_size;
    return true;
}

int fseek64(std::FILE* f, int64_t off, int whence) {
    return fseeko(f, (off_t) off, whence);
}

int file_fd(std::FILE* f) {
    return fileno(f);
}

void advise_dontneed(int fd, uint64_t off, uint64_t bytes) {
    // A hint, and its own failure is not the caller's problem: the pages it declines to release are reclaimed
    // under pressure. A `bytes` of 0 is the rest of the file, spelled the same way here.
    (void) posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED);
}

int open_unbuffered(int fd) {
    if (fd < 0) return -1;
    // The descriptor's own path, reopened: /proc/self/fd names the file a descriptor stands for, so this is a second
    // handle on it with the unbuffered flag, and the first keeps its mapping and its cached reads.
    const std::string self = "/proc/self/fd/" + std::to_string(fd);
    return ::open(self.c_str(), O_RDONLY | O_DIRECT);
}

int open_unbuffered(const std::string& path) {
    // The same flag taken from the path directly. A file system that refuses it answers -1, and the caller reads the
    // bytes cached, which is the copy the flag exists to avoid.
    return ::open(path.c_str(), O_RDONLY | O_DIRECT);
}

int open_write_utf8(const std::string& path) {
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
}

int open_write_unbuffered(const std::string& path) {
    // A file system that refuses the flag answers -1, and the caller writes through the cache.
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0644);
}

bool pwrite_all(int fd, const void* src, uint64_t n, uint64_t off) {
    const uint8_t* p = (const uint8_t*) src;
    while (n > 0) {
        const ssize_t put = ::pwrite(fd, p, (size_t) (n < (1ull << 30) ? n : (1ull << 30)), (off_t) off);
        if (put <= 0) {
            if (put < 0 && errno == EINTR) continue;
            return false;
        }
        p += put;
        off += (uint64_t) put;
        n -= (uint64_t) put;
    }
    return true;
}

bool sync_fd(int fd) {
    return ::fdatasync(fd) == 0;
}

namespace {
std::filesystem::path fs_path(const std::string& utf8) { return std::filesystem::path(utf8); }
std::string fs_name(const std::filesystem::path& p) { return p.filename().string(); }
}  // namespace

#endif

bool make_dirs(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(fs_path(path), ec);
    return std::filesystem::is_directory(fs_path(path), ec);
}

bool list_files(const std::string& dir, std::vector<DirFile>& out) {
    out.clear();
    std::error_code ec;
    std::filesystem::directory_iterator it(fs_path(dir), ec), end;
    if (ec) return false;
    for (; it != end; it.increment(ec)) {
        if (ec) return false;
        std::error_code fe;
        if (!it->is_regular_file(fe) || fe) continue;
        DirFile f;
        f.name = fs_name(it->path());
        f.bytes = (uint64_t) it->file_size(fe);
        if (fe) continue;
        const auto t = it->last_write_time(fe);
        if (fe) continue;
        f.mtime = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
        out.push_back(std::move(f));
    }
    return true;
}

bool rename_replace(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::rename(fs_path(from), fs_path(to), ec);
    return !ec;
}

bool remove_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(fs_path(path), ec);
    return !std::filesystem::exists(fs_path(path), ec);
}

bool touch_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::last_write_time(fs_path(path), std::filesystem::file_time_type::clock::now(), ec);
    return !ec;
}

int64_t file_time_now() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::filesystem::file_time_type::clock::now().time_since_epoch()).count();
}

uint64_t disk_free_bytes(const std::string& path) {
    std::error_code ec;
    const auto s = std::filesystem::space(fs_path(path), ec);
    return ec ? 0 : (uint64_t) s.available;
}

bool pread_all(int fd, void* dst, uint64_t n, uint64_t off) {
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const int64_t got = pread(fd, p, (size_t) (n < (1ull << 30) ? n : (1ull << 30)), off);
        if (got <= 0) {
#if !defined(_WIN32)
            // A signal can interrupt a read before it has transferred anything; that one is restarted.
            if (got < 0 && errno == EINTR) continue;
#endif
            return false;
        }
        p += got;
        off += (uint64_t) got;
        n -= (uint64_t) got;
    }
    return true;
}

}  // namespace basalt::platform
