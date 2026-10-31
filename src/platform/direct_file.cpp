// src/platform/direct_file.cpp - see include/basalt/platform/direct_file.hpp.
#include "basalt/platform/direct_file.hpp"

#include "basalt/platform/file.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace basalt::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    // A fresh VirtualAlloc block starts at an allocation granularity boundary, so it satisfies the unbuffered
    // contract without the cache manager ever having held these pages.
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

namespace {
/// A queued read: `submit` only queues it; the pool's threads issue it .  One thread
/// issuing every read would limit the n-gram table's prompt reads to the kernel time per read, while an NVMe
/// drive serves several times that at depth.
struct Pending {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

bool valid_workers(int workers, std::string& err) {
    if (workers >= 0 && workers <= 256) return true;
    err = "DirectFile: workers must be 0 (default) or 1..256";
    return false;
}
}  // namespace

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED is the first member so that a completion packet's OVERLAPPED* converts
/// back to the request it belongs to.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};

/// The completion key of a posted packet that carries no request. A `wake` is turned into a WAKE_TAG completion
/// for the caller; the nudge of an immediate completion only says the internal buffer has something in it, and
/// never reaches the caller, so `wake()` keeps its meaning. The key of a real read is the one the file was
/// associated with (0) and it always carries an OVERLAPPED.
constexpr ULONG_PTR WAKE_KEY = 1;
constexpr ULONG_PTR NUDGE_KEY = 2;
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::mutex mu;                      // guards everything below
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // completions ready to hand out: a read that hit end of file or failed
                                        // without a port packet, a wake, and any that did not fit the last wait
    std::deque<Pending> queue;          // submitted, not yet issued
    size_t inflight = 0;                // reads whose completion packet has not been retrieved yet
    std::condition_variable cv;
    bool stop = false;
    std::vector<std::thread> pool;

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }

    void issue(const Pending& p) {
        Req* r;
        {
            std::lock_guard<std::mutex> lk(mu);
            r = take();
            // Counted before the read starts: a read that finishes at once can have its packet retrieved by a
            // waiter before this thread would get the chance to record it, and a count that goes past zero leaves
            // close() waiting for a packet that has already been taken.
            ++inflight;
        }
        std::memset(&r->ov, 0, sizeof r->ov);
        r->ov.Offset = (DWORD) (p.offset & 0xFFFFFFFFull);
        r->ov.OffsetHigh = (DWORD) (p.offset >> 32);
        r->tag = p.tag;
        if (!ReadFile(file, p.buffer, p.length, nullptr, &r->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                std::lock_guard<std::mutex> lk(mu);
                free_reqs.push_back(r);
                --inflight;   // a read the kernel never took produces no packet
                // at or past end of file: a zero-byte completion; anything else: a failed one
                immediate.push_back(Completion{p.tag, 0, e == ERROR_HANDLE_EOF});
                PostQueuedCompletionStatus(port, 0, NUDGE_KEY, nullptr);   // tell a waiter the buffer has work
                return;
            }
        }
    }

    /// Retrieve the packet of every read still in flight and hand its request back to the free list. A packet is
    /// queued only once the kernel has stopped writing the OVERLAPPED and the caller's buffer, so after this no
    /// late completion can land in memory the caller has released. Cancelling keeps a teardown from waiting out
    /// reads nobody wants any more; a cancelled read still produces a packet.
    void drain() {
        if (file != INVALID_HANDLE_VALUE) CancelIoEx(file, nullptr);
        if (port == nullptr) return;
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu);
                if (inflight == 0) return;
            }
            DWORD bytes = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* ov = nullptr;
            const BOOL ok = GetQueuedCompletionStatus(port, &bytes, &key, &ov, INFINITE);
            // A cancelled/failed read still supplies its OVERLAPPED and must be reaped before buffers are freed.
            if (ov == nullptr) {
                if (!ok) return;
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            free_reqs.push_back((Req*) ov);
            --inflight;
        }
    }

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            issue(p);
            lk.lock();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err, int workers, bool mapped_elsewhere) {
    close();
    if (!valid_workers(workers, err)) return false;
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM. It is left out for
    // a file mapped elsewhere: while a file has a mapped view or cached reads, NTFS completes its unbuffered reads one
    // at a time, and its cached reads still run side by side.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    const DWORD buffering = mapped_elsewhere ? 0 : FILE_FLAG_NO_BUFFERING;
    impl_->file = CreateFileW(to_wide_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              buffering | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    return finish_open(path, err, workers);
}

bool DirectFile::finish_open(const std::string& path, std::string& err, int workers) {
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    const int n = workers == 0 ? 16 : workers;
    try {
        impl_->pool.reserve(n);
        for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { impl_->worker(); });
    } catch (const std::exception& e) {
        close();
        err = "DirectFile: cannot start issuing threads: " + std::string(e.what());
        return false;
    }
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();   // after this no thread is left that could issue another read
    impl_->drain();       // ... and every read already in flight has finished writing
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->immediate.clear();
    impl_->queue.clear();
    impl_->inflight = 0;   // the port is gone, so no packet is left to retrieve
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv.notify_one();
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    if (max <= 0) return 0;
    using clock = std::chrono::steady_clock;
    const bool timed = timeout_ms >= 0;
    const auto deadline = clock::now() + std::chrono::milliseconds(timed ? timeout_ms : 0);
    int n = 0;
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            while (n < max && !impl_->immediate.empty()) {
                out[n++] = impl_->immediate.front();
                impl_->immediate.pop_front();
            }
        }
        if (n > 0 || !is_open()) return n;   // hand out whatever is ready rather than waiting to fill `max`
        DWORD t = INFINITE;
        if (timed) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clock::now()).count();
            t = (DWORD) (left > 0 ? left : 0);
        }
        // Packets are taken in batches of up to 64 whatever `max` is: a completion that does not fit the caller's
        // array goes into the buffer for the next call, so `out` never holds more than `max` and one call leaves
        // at most one batch behind in the buffer.
        OVERLAPPED_ENTRY entries[64];
        ULONG got = 0;
        if (!GetQueuedCompletionStatusEx(impl_->port, entries, 64, &got, t, FALSE)) return n;
        std::lock_guard<std::mutex> lk(impl_->mu);
        for (ULONG i = 0; i < got; ++i) {
            if (entries[i].lpOverlapped == nullptr) {
                // The nudge of an immediate completion is already represented by the buffered completion, so it
                // is swallowed here; only a real wake is turned into a completion of its own.
                if (entries[i].lpCompletionKey == WAKE_KEY) {
                    Completion w{WAKE_TAG, 0, true};
                    if (n < max) out[n++] = w;
                    else impl_->immediate.push_back(w);
                }
                continue;
            }
            Req* r = (Req*) entries[i].lpOverlapped;
            const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
            Completion c;
            c.tag = r->tag;
            c.bytes = entries[i].dwNumberOfBytesTransferred;
            // STATUS_END_OF_FILE is a legal short read at the table's last page.
            c.ok = status == 0 || status == 0xC0000011u;
            if (n < max) out[n++] = c;
            else impl_->immediate.push_back(c);
            impl_->free_reqs.push_back(r);
            --impl_->inflight;
        }
        // A batch of nothing but nudges hands the caller nothing: keep waiting until the deadline instead of
        // reporting a completion-less wait that had time left.
        if (n > 0) return n;
    }
}

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, WAKE_KEY, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// A pool of threads, each doing blocking O_DIRECT preads, so reads run in parallel (the thread
// count is the queue depth).
struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Pending> queue;
    std::deque<Completion> done;
    bool stop = false;
    std::vector<std::thread> pool;

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            const ssize_t got = pread(fd, p.buffer, p.length, (off_t) p.offset);
            lk.lock();
            // only a waiter that found the list empty is blocked, so only the first completion needs to wake it
            const bool was_empty = done.empty();
            done.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
            if (was_empty) cv_done.notify_all();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err, int workers, bool mapped_elsewhere) {
    close();
    if (!valid_workers(workers, err)) return false;
    (void) mapped_elsewhere;   // a mapping of the file does not hold up O_DIRECT reads of it here
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    return finish_open(path, err, workers);
}

bool DirectFile::finish_open(const std::string& path, std::string& err, int workers) {
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
    // one blocking read per thread: a drive serves random pages at its full rate only with a deep queue
    const int n = workers == 0 ? 256 : workers;
    try {
        impl_->pool.reserve(n);
        for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { impl_->worker(); });
    } catch (const std::exception& e) {
        close();
        err = "DirectFile: cannot start issuing threads: " + std::string(e.what());
        return false;
    }
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv_work.notify_one();
    return true;
}

void DirectFile::wake() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->done.push_back(Completion{WAKE_TAG, 0, true});
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    if (max <= 0 || !is_open()) return 0;
    std::unique_lock<std::mutex> lk(impl_->mu);
    if (impl_->done.empty() && timeout_ms != 0) {
        auto ready = [&] { return !impl_->done.empty(); };
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}
#endif

}  // namespace basalt::platform
