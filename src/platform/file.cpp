// src/platform/file.cpp - see include/basalt/platform/file.hpp.
//
// One implementation per platform behind one spelling per call. The Linux half is the plain POSIX call the engine
// made there before; the Windows half is the wide-call form of it, plus what reading a CRT descriptor needs there.
#include "basalt/platform/file.hpp"

#include <cerrno>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <direct.h>   // _wchdir, _wfullpath
#include <fcntl.h>    // _O_RDONLY, _O_BINARY
#include <io.h>       // _wopen, _waccess, _close, _fileno, _get_osfhandle
#else
#include <fcntl.h>    // O_RDONLY, posix_fadvise
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

/// This thread's handle for reading `fd`. A handle opened without FILE_FLAG_OVERLAPPED serializes the I/O issued
/// through it, so threads reading through one shared descriptor take turns rather than reading in parallel: each
/// thread opens the same file again and gets a file object of its own. Returns the shared descriptor's handle when
/// the file cannot be reopened (it still reads correctly, only in turn).
HANDLE read_handle(int fd) {
    // What the handle was opened for has to be checked, not only the descriptor: a closed descriptor's number comes
    // back as the next open's number, and the handle value it carried can come back with it. The volume and the
    // file index inside it say which file this is.
    struct PerThread {
        ~PerThread() { release(); }
        void release() {
            if (own != INVALID_HANDLE_VALUE) CloseHandle(own);   // the shared descriptor belongs to the CRT
            own = INVALID_HANDLE_VALUE;
        }
        bool holds(const BY_HANDLE_FILE_INFORMATION& i) const {
            return own != INVALID_HANDLE_VALUE && i.dwVolumeSerialNumber == vol &&
                   i.nFileIndexHigh == hi && i.nFileIndexLow == lo;
        }
        HANDLE own = INVALID_HANDLE_VALUE;   // opened here, so closing it is ours to do
        DWORD vol = 0, hi = 0, lo = 0;       // the file `own` was opened for
    };
    thread_local PerThread pt;
    const HANDLE cur = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    BY_HANDLE_FILE_INFORMATION info{};
    if (cur != INVALID_HANDLE_VALUE && GetFileInformationByHandle(cur, &info)) {
        if (pt.holds(info)) return pt.own;
        pt.release();
        wchar_t path[4096];
        const DWORD cap = (DWORD) (sizeof path / sizeof path[0]);
        const DWORD len = GetFinalPathNameByHandleW(cur, path, cap, VOLUME_NAME_DOS);
        if (len != 0 && len < cap) {
            pt.own = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (pt.own != INVALID_HANDLE_VALUE) {
                pt.vol = info.dwVolumeSerialNumber;
                pt.hi = info.nFileIndexHigh;
                pt.lo = info.nFileIndexLow;
                return pt.own;
            }
        }
    }
    return cur;
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
    const HANDLE h = read_handle(fd);
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
            if (!GetOverlappedResult(h, &ov, &got, TRUE) && GetLastError() != ERROR_HANDLE_EOF) return -1;
        } else if (e != ERROR_HANDLE_EOF) {
            return -1;
        }
    }
    return (int64_t) got;
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

#endif

}  // namespace basalt::platform
