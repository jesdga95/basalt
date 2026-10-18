#include "basalt/core/direct_upload.hpp"

#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>

namespace basalt::core {

namespace {
constexpr uint64_t kAlign = 4096;
}

struct DirectUploader::Impl {
    std::string path;
    int fd = -1;
    bool direct = false;
    uint8_t* buf[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
    cudaStream_t stream = nullptr;
    bool ready = false;
    bool pending[2] = {false, false};
    int next = 0;

    bool init(std::string& err) {
        if (ready) return true;
        for (int i = 0; i < 2; ++i)
            if (cudaHostAlloc((void**) &buf[i], kChunk + 2 * kAlign, cudaHostAllocPortable) != cudaSuccess ||
                cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming) != cudaSuccess) {
                (void) cudaGetLastError();
                err = "cannot allocate the staging buffers for the file reads";
                return false;
            }
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            err = "cannot create the stream for the file reads";
            return false;
        }
        ready = true;
        return true;
    }

    bool open_path(const std::string& p, std::string& err) {
        if (fd >= 0 && p == path) return true;
        if (fd >= 0) basalt::platform::close_fd(fd);
        fd = basalt::platform::open_unbuffered(p);
        direct = fd >= 0;
        if (fd < 0) fd = basalt::platform::open_fd_utf8(p);
        if (fd < 0) {
            path.clear();
            err = "cannot open " + p;
            return false;
        }
        path = p;
        return true;
    }

    // `n` bytes at `off` into `b`; returns where they start inside it
    const uint8_t* read_into(uint8_t* b, uint64_t off, uint64_t n, std::string& err) {
        const uint64_t a0 = direct ? off - off % kAlign : off;
        const uint64_t a1 = direct ? (off + n + kAlign - 1) / kAlign * kAlign : off + n;
        uint64_t got = 0;
        while (got < off + n - a0) {
            const int64_t r = basalt::platform::pread(fd, b + got, (size_t) (a1 - a0 - got), a0 + got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) {
                err = "short read from " + path;
                return nullptr;
            }
            got += (uint64_t) r;
        }
        return b + (off - a0);
    }

    void drain() {
        for (int i = 0; i < 2; ++i)
            if (pending[i]) {
                cudaEventSynchronize(done[i]);
                pending[i] = false;
            }
    }
};

DirectUploader::DirectUploader() : impl_(new Impl) {}

DirectUploader::~DirectUploader() {
    Impl& m = *impl_;
    if (m.ready) cudaStreamSynchronize(m.stream);
    for (int i = 0; i < 2; ++i) {
        if (m.buf[i]) cudaFreeHost(m.buf[i]);
        if (m.done[i]) cudaEventDestroy(m.done[i]);
    }
    if (m.stream) cudaStreamDestroy(m.stream);
    if (m.fd >= 0) basalt::platform::close_fd(m.fd);
    delete impl_;
}

bool DirectUploader::unbuffered() const { return impl_->direct; }

const uint8_t* DirectUploader::staged(const std::string& path, uint64_t file_off, uint64_t bytes, std::string& err) {
    Impl& m = *impl_;
    if (bytes > kChunk) { err = "staged read larger than the staging buffer"; return nullptr; }
    if (!m.init(err) || !m.open_path(path, err)) return nullptr;
    m.drain();
    return m.read_into(m.buf[0], file_off, bytes, err);
}

bool DirectUploader::upload(const std::string& path, uint64_t file_off, uint64_t bytes, void* dst, std::string& err) {
    Impl& m = *impl_;
    if (!m.init(err) || !m.open_path(path, err)) return false;
    for (uint64_t done = 0; done < bytes;) {
        const int b = m.next;
        m.next ^= 1;
        if (m.pending[b]) {   // this buffer's previous copy has to finish before it is read into again
            cudaEventSynchronize(m.done[b]);
            m.pending[b] = false;
        }
        const uint64_t n = std::min<uint64_t>(kChunk, bytes - done);
        const uint8_t* src = m.read_into(m.buf[b], file_off + done, n, err);
        if (src == nullptr) { m.drain(); return false; }
        const cudaError_t e = cudaMemcpyAsync((uint8_t*) dst + done, src, (size_t) n, cudaMemcpyHostToDevice, m.stream);
        if (e != cudaSuccess) {
            err = std::string("upload: ") + cudaGetErrorString(e);
            m.drain();
            return false;
        }
        cudaEventRecord(m.done[b], m.stream);
        m.pending[b] = true;
        done += n;
    }
    const cudaError_t e = cudaStreamSynchronize(m.stream);
    m.pending[0] = m.pending[1] = false;
    if (e != cudaSuccess) { err = std::string("upload: ") + cudaGetErrorString(e); return false; }
    return true;
}

}  // namespace basalt::core
