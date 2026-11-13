// src/models/deepseek41/file_reads.hpp - expert blobs read from the pack into RAM places, off the caller's thread.
//
// A lookup that routes to experts only the file holds reads each one whole into a place of the RAM tier. The reads go
// around the OS file cache (platform::DirectFile) in aligned pieces staged through a few buffers, so the CPU never
// faults the file in a page at a time and the file cache does not compete with the RAM tier for memory. A file system
// that refuses unbuffered reads is read through the file cache instead.
#pragma once

#include "basalt/platform/direct_file.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {

class FileReads {
public:
    /// `bytes` of the file at `off` go to `dst`.
    struct Blob {
        uint64_t off = 0, bytes = 0;
        uint8_t* dst = nullptr;
    };

    FileReads() = default;
    ~FileReads();
    FileReads(const FileReads&) = delete;
    FileReads& operator=(const FileReads&) = delete;

    /// `path` is the file, `fd` a descriptor of it that reads through the file cache.
    bool open(const std::string& path, int fd, std::string& err);
    bool is_open() const { return thread_.joinable(); }
    /// Whether the reads go around the file cache.
    bool unbuffered() const { return direct_.is_open(); }

    /// Starts reading `blobs` (the previous batch was waited for). Their destinations stay untouched by the caller until `wait`.
    void start(const std::vector<Blob>& blobs);
    bool busy() const { return started_; }
    /// The batch has landed; false (with `err`) when a blob could not be read.
    bool wait(std::string& err);

private:
    struct Piece {
        uint64_t at = 0;          ///< aligned file offset of the read
        uint32_t length = 0;      ///< aligned length of the read
        uint32_t skip = 0;        ///< bytes of the read before the wanted ones
        uint32_t copy = 0;        ///< wanted bytes
        uint8_t* dst = nullptr;
    };
    void run();
    bool read_direct(const std::vector<Blob>& blobs);
    bool read_buffered(const std::vector<Blob>& blobs);

    platform::DirectFile direct_;
    int fd_ = -1;
    uint8_t* stage_ = nullptr;    ///< kDepth aligned buffers of one piece each
    std::vector<Piece> pieces_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Blob> batch_;
    bool ready_ = false, done_ = false, ok_ = true, quit_ = false;
    bool started_ = false;        ///< caller side: a batch is out
};

}  // namespace basalt::models::deepseek41
