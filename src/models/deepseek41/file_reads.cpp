// src/models/deepseek41/file_reads.cpp - expert blobs read from the pack into RAM places, off the caller's thread.
#include "models/deepseek41/file_reads.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <cstring>

namespace basalt::models::deepseek41 {
namespace {

constexpr uint32_t kPiece = 1u << 20;    // one unbuffered read
constexpr int kDepth = 16;               // pieces in flight: the readers stay busy while landed ones are copied out
constexpr int kReaders = 8;              // threads issuing the reads
constexpr uint64_t kAlign = platform::DirectFile::alignment();
constexpr size_t kStage = kPiece + kAlign;   // a piece and the alignment its ends round out to

}  // namespace

FileReads::~FileReads() {
    if (thread_.joinable()) {
        std::string e;
        wait(e);
        {
            std::lock_guard<std::mutex> g(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    direct_.close();
    platform::free_aligned(stage_);
}

bool FileReads::open(const std::string& path, int fd, std::string& err) {
    fd_ = fd;
    std::string why;
    if (direct_.open(path, why, kReaders, /*mapped_elsewhere=*/true)) {
        stage_ = (uint8_t*) platform::alloc_aligned(kStage * kDepth, kAlign);
        if (stage_ == nullptr) {
            err = "deepseek41: cannot allocate the file read buffers";
            return false;
        }
    } else {
        basalt::log::warn("basalt deepseek41", "file experts read through the file cache: %s", why.c_str());
    }
    try {
        thread_ = std::thread([this] { run(); });
    } catch (const std::exception& e) {
        err = std::string("deepseek41: cannot start the file read thread: ") + e.what();
        return false;
    }
    return true;
}

void FileReads::start(const std::vector<Blob>& blobs) {
    {
        std::lock_guard<std::mutex> g(mu_);
        batch_ = blobs;
        ready_ = true;
        done_ = false;
    }
    cv_.notify_all();
    started_ = true;
}

bool FileReads::wait(std::string& err) {
    if (!started_) return true;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return done_; });
    started_ = false;
    if (ok_) return true;
    ok_ = true;
    err = "deepseek41: cannot read an expert from the pack";
    return false;
}

void FileReads::run() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return quit_ || ready_; });
        if (quit_) return;
        ready_ = false;
        lk.unlock();
        const bool ok = direct_.is_open() ? read_direct(batch_) : read_buffered(batch_);
        lk.lock();
        ok_ = ok;
        done_ = true;
        cv_.notify_all();
    }
}

bool FileReads::read_buffered(const std::vector<Blob>& blobs) {
    bool ok = true;
    for (const Blob& b : blobs) ok = platform::pread_all(fd_, b.dst, b.bytes, b.off) && ok;
    return ok;
}

// Each blob in aligned pieces; a landed piece's wanted bytes are copied to their place and its buffer takes the next piece.
// A piece the unbuffered read fails goes through the file cache.
bool FileReads::read_direct(const std::vector<Blob>& blobs) {
    pieces_.clear();
    for (const Blob& b : blobs) {
        const uint64_t end = b.off + b.bytes;
        for (uint64_t at = b.off / kAlign * kAlign; at < end; at += kPiece) {
            const uint64_t lo = std::max(at, b.off), hi = std::min(at + kPiece, end);
            const uint64_t stop = (hi + kAlign - 1) / kAlign * kAlign;
            pieces_.push_back({at, (uint32_t) (stop - at), (uint32_t) (lo - at), (uint32_t) (hi - lo), b.dst + (lo - b.off)});
        }
    }
    bool ok = true;
    const auto by_cache = [&](const Piece& p) { ok = platform::pread_all(fd_, p.dst, p.copy, p.at + p.skip) && ok; };
    size_t next = 0;
    int in_flight = 0;
    int piece_of[kDepth];
    std::string err;
    const auto submit = [&](int buf) {
        while (next < pieces_.size()) {
            const size_t i = next++;
            if (direct_.submit(pieces_[i].at, stage_ + (size_t) buf * kStage, pieces_[i].length, (uint64_t) buf, err)) {
                piece_of[buf] = (int) i;
                ++in_flight;
                return;
            }
            by_cache(pieces_[i]);
        }
    };
    for (int buf = 0; buf < kDepth; ++buf) submit(buf);
    platform::Completion done[kDepth];
    while (in_flight > 0) {
        const int n = direct_.wait(done, kDepth, -1);
        for (int c = 0; c < n; ++c) {
            if (done[c].tag == platform::DirectFile::WAKE_TAG) continue;
            const int buf = (int) done[c].tag;
            const Piece& p = pieces_[(size_t) piece_of[buf]];
            --in_flight;
            if (done[c].ok && done[c].bytes >= p.skip + p.copy) std::memcpy(p.dst, stage_ + (size_t) buf * kStage + p.skip, p.copy);
            else by_cache(p);
            submit(buf);
        }
    }
    return ok;
}

}  // namespace basalt::models::deepseek41
