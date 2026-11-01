// include/basalt/ngram/ple_ram_quant.hpp - background packing into resident PLE RAM.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::ngram {

inline constexpr uint32_t kPleQuantChunkRows = 262144;

struct LockedPleBuffer {
    uint8_t* data = nullptr;
    void* mapping = nullptr;
    uint64_t mapping_bytes = 0;
    uint64_t locked_bytes = 0;

    LockedPleBuffer();
    ~LockedPleBuffer();
    LockedPleBuffer(const LockedPleBuffer&) = delete;
    LockedPleBuffer& operator=(const LockedPleBuffer&) = delete;

    // Ready rows take only an acquire load; unfinished rows are packed on demand.
    bool ensure_row(uint64_t row, std::string& err);
    // Wait for conversion and source/worker teardown. Reports background failures.
    bool wait(std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<LockedPleBuffer> open_ple_q8_0(
        const std::string&, uint64_t, uint64_t, std::string&, int, uint32_t);
};

// Allocates and locks the entire final table before returning; conversion continues in the background.
// data has a fixed address and 170-byte row stride, but callers must ensure_row before reading it.
// chunk_rows is for isolated tests, not user tuning. Destruction cancels/drains and joins all workers.
std::unique_ptr<LockedPleBuffer> open_ple_q8_0(const std::string& path, uint64_t offset, uint64_t rows,
                                            std::string& err, int workers = 1,
                                            uint32_t chunk_rows = kPleQuantChunkRows);

}  // namespace basalt::ngram
