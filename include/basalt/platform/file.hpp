// include/basalt/platform/file.hpp - the file calls the engine makes, so that Win32 stays out of the engine.
//
// A path is UTF-8 everywhere in the engine. On Windows the plain entry points (`fopen`, `_open`, `_fullpath`,
// `_chdir`, `_access`, `GetModuleFileNameA`) read a path in the machine's ANSI code page, so one path works at one
// call site and fails at another; every path taken here is converted to UTF-16 and handed to the wide function. On
// Linux the same names are the plain POSIX calls and no conversion happens.
//
// The read and seek helpers keep the shape of the calls they stand in for: `pread` reads at most one chunk, so a
// short count is normal and the caller loops; `fseek64` and `file_fd` answer the question "how do I spell a 64-bit
// seek / a stream's descriptor here", which the expert files need because they are larger than 4 GiB.
//
// `to_wide` is declared on Windows only. Nothing outside this header has any use for a std::wstring, and a portable
// stub would only tell a call site that the wide form means something on Linux, where it does not.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::platform {

#if defined(_WIN32)
/// UTF-8 to UTF-16, the conversion every path open here goes through. An empty string converts to an empty one.
/// Bytes that are not valid UTF-8 become the replacement character rather than an error, so a name that came from
/// somewhere broken fails at the open that follows rather than here.
std::wstring to_wide(const std::string& utf8);
#endif

/// Opens `path` as `mode` spells it ("rb", "wb", "ab"). nullptr on failure. `mode` is passed through as given: text
/// or binary stays the caller's choice, because the serve protocol and the dump files depend on which one they get.
std::FILE* open_file_utf8(const std::string& path, const std::string& mode);

/// Opens `path` read-only and binary, for positional reads. A descriptor; -1 on failure, with errno set on Linux.
/// The descriptor's handle is one the OS opened, so a read through it needs no second handle: see `pread`.
int open_fd_utf8(const std::string& path);

/// Closes a descriptor, from `open_fd_utf8` or any other open.
void close_fd(int fd);

/// The absolute form of `path`, in UTF-8. False when the OS will not answer. The two answers differ as the two
/// calls they wrap do: Linux resolves against the directory tree and so refuses a path that does not exist,
/// Windows spells the path out without looking at the tree, and both refuse one too long for their buffer.
bool absolute_path(const std::string& path, std::string& out);

/// Makes `path` the process's working directory. 0 on success, -1 on failure with errno set.
int set_working_dir(const std::string& path);

/// Whether `path` names a file this process may run. Windows has no executable bit, so there it asks only that the
/// file exist, which is what the call site that uses it means.
bool path_executable(const std::string& path);

/// This executable's own path, as the OS knows it; empty when the OS does not say.
std::string module_path();

/// The directory for scratch files, with no trailing separator: the system's temp directory on Windows, /tmp on
/// Linux. Never empty; where the OS cannot name one it says ".".
std::string temp_dir();

/// Reads up to `n` bytes at the absolute offset `off` into `dst`, without moving any file position. Returns the
/// bytes read (0 at the end of the file), or -1 on failure - Linux sets errno as pread(2) does, Windows reports it
/// by the return value alone.
/// A count short of `n` is not an error and every caller loops over it. Windows reads at most 1 GiB in one call,
/// because its read call takes a 32-bit byte count, and issues the read through a handle of its own for the
/// duration of it: a handle opened without overlapped I/O serializes the reads issued through it, so threads sharing
/// one descriptor would take turns, and a handle the call kept would keep the file open after its descriptor was
/// closed. When the file cannot be reopened the shared descriptor is used, which still reads correctly, only in
/// turn.
/// A descriptor this process has already closed is not a value to hand here: finding a descriptor's handle is what
/// Windows does first, and the CRT stops the process for a number it knows is closed. `fd` < 0 says -1 instead.
/// `dst` has to hold everything `n` asks for: on Windows that is up to the 1 GiB the cap gives, so a caller asking
/// for gigabytes is committing to a buffer of gigabytes.
int64_t pread(int fd, void* dst, size_t n, uint64_t off);

/// Reads exactly `n` bytes at `off` by looping over `pread`'s short counts, a read a signal interrupted before it
/// transferred anything being restarted. False when the file ends first or a read fails.
bool pread_all(int fd, void* dst, uint64_t n, uint64_t off);

/// The size of the regular file at `path`, in bytes. False for anything else - a missing file, a directory.
bool file_size(const std::string& path, uint64_t& out);

/// The 64-bit seek for a stream: an expert file is larger than 4 GiB, so a 32-bit seek would wrap. Returns what the
/// platform's seek returns: 0 on success, nonzero on failure.
int fseek64(std::FILE* f, int64_t off, int whence = SEEK_SET);

/// The descriptor a std::FILE was opened on (read-ahead advice is asked of the file, not the stream).
int file_fd(std::FILE* f);

/// Asks the OS to release what it is holding for [off, off + bytes) of an open descriptor; a `bytes` of 0 is the
/// rest of the file, which is what the one caller that uses it asks for. A hint, never an error. Windows does
/// nothing: its cache manager takes no range hint for a handle, and a mapped view's pages are clean, so the
/// manager takes them back by itself.
void advise_dontneed(int fd, uint64_t off, uint64_t bytes);

/// A descriptor on the same open file as `fd`, whose reads bypass the OS cache: the caller's buffer gets the bytes
/// as the drive gives them, so a large fill neither copies them through the cache nor evicts anything to do it.
/// The reads through it have to be aligned - buffer address, file offset and count to the device's block size.
/// -1 where the platform gives no such descriptor, which is the caller's signal to read the same bytes some other
/// way. `close_fd` lets go of it on both platforms.
int open_unbuffered(int fd);

/// The same for a path: the descriptor `open_unbuffered(fd)` would give for an already-open file, opened from the
/// path in the first place. -1 where the platform gives no such descriptor, and the caller opens the file ordinarily
/// instead.
int open_unbuffered(const std::string& path);

/// Creates or truncates `path` for writing, binary. A descriptor for `pwrite_all`; -1 on failure.
int open_write_utf8(const std::string& path);

/// The same with writes that bypass the OS cache: buffer address, offset and count aligned to 4096. -1 where the
/// platform gives no such descriptor, and the caller writes through `open_write_utf8` instead.
int open_write_unbuffered(const std::string& path);

/// Writes exactly `n` bytes at `off`, looping over short writes. False when a write fails.
bool pwrite_all(int fd, const void* src, uint64_t n, uint64_t off);

/// Waits until what was written through `fd` is on the device. False when the OS reports a failure.
bool sync_fd(int fd);

/// Creates `path` and its missing parents. True when the directory exists afterwards.
bool make_dirs(const std::string& path);

/// A regular file in a directory: its name (no directory part), size and last write time (seconds on the file
/// system's clock, `file_time_now` the same clock's present: only differences of two times mean something).
struct DirFile {
    std::string name;
    uint64_t bytes = 0;
    int64_t mtime = 0;
};
/// The regular files directly in `dir`. False when the directory cannot be read.
bool list_files(const std::string& dir, std::vector<DirFile>& out);

/// Moves `from` to `to`, replacing a file already there, in one step on the same volume.
bool rename_replace(const std::string& from, const std::string& to);

/// Deletes a file. True when it is gone afterwards.
bool remove_file(const std::string& path);

/// Sets a file's last write time to now. False when the file cannot be touched.
bool touch_file(const std::string& path);

/// The present on the clock `list_files` gives write times on, in seconds.
int64_t file_time_now();

/// Bytes this process may still write on the volume that holds `path`; 0 when the OS does not say.
uint64_t disk_free_bytes(const std::string& path);

}  // namespace basalt::platform
