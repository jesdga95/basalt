// src/ngram/ple_ram_quant.cpp - bounded streaming and demand packing into locked Q8_0 RAM.
#include "basalt/ngram/ple_ram_quant.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/ngram.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/direct_file.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::ngram {
namespace {
constexpr uint64_t kRegionAlignment = 2ull << 20;
constexpr uint32_t kRegionRows = 128;
constexpr uint32_t kScratchValues = kRegionRows * kernels::PLE_HEAD_DIM;
constexpr uint32_t kSourceStride = kernels::PLE_ROW_BYTES_BF16;
constexpr uint32_t kOutputStride = kernels::PLE_ROW_BYTES_Q8_0;
enum State : uint8_t { Empty, Loading, Ready, Failed };

struct Stage {
    uint8_t* data = nullptr;
    uint32_t prefix = 0;
    uint32_t rows = 0;
    uint32_t required = 0;
    ~Stage() { platform::DirectFile::free_aligned(data); }
};

uint64_t stage_size(uint64_t rows) {
    const uint64_t alignment = platform::DirectFile::alignment();
    return (rows * kSourceStride + 2 * (alignment - 1)) & ~(alignment - 1);
}
}  // namespace

struct LockedPleBuffer::Impl {
    uint8_t* output = nullptr;
    uint64_t offset = 0, rows = 0;
    uint32_t chunk = 0;
    int workers = 1;
    ggml_from_float_t quantize = nullptr;
    std::unique_ptr<std::atomic<uint8_t>[]> states;
    std::atomic<bool> stop{false}, failed{false};
    std::mutex state_mu;
    std::condition_variable changed;
    std::string failure;
    bool finished = false;
    std::thread background;
    std::unique_ptr<core::LoadProgress> progress;

    // Demand I/O never queues behind a large background read. Closing it takes the same mutex as a miss.
    std::mutex demand_mu;
    Stage demand_stage;
    std::unique_ptr<float[]> demand_scratch;
    platform::DirectFile demand_file;
    std::mutex background_io_mu;  // serializes wake with source close, not with read waits
    Stage stages[2];
    platform::DirectFile background_file;

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(state_mu);
            stop.store(true, std::memory_order_release);
        }
        {
            std::lock_guard<std::mutex> lock(background_io_mu);
            if (background_file.is_open()) background_file.wake();
        }
        changed.notify_all();
        if (background.joinable()) background.join();
        // Failed initialization can leave handles open without a background thread.
        background_file.close();
        demand_file.close();
    }

    void fail(const std::string& err) {
        {
            std::lock_guard<std::mutex> lock(state_mu);
            if (!failed.load(std::memory_order_relaxed)) {
                failure = err;
                failed.store(true, std::memory_order_release);
                basalt::log::error("basalt", "PLE CPU BF16->Q8_0: %s", failure.c_str());
            }
        }
        changed.notify_all();
    }

    void publish(uint64_t region, State state) {
        // Predicate changes share the wait mutex: no notify can fall between checking and sleeping.
        {
            std::lock_guard<std::mutex> lock(state_mu);
            states[region].store(state, std::memory_order_release);
        }
        changed.notify_all();
    }

    void convert(uint64_t row, uint32_t count, const uint8_t* source, float* scratch) {
        const int values = (int) (count * kernels::PLE_HEAD_DIM);
        basalt::dequantize_bf16(source, scratch, values);
        quantize(scratch, output + row * kOutputStride, values);
        publish(row / kRegionRows, Ready);
    }

    bool submit(platform::DirectFile& file, Stage& stage, uint64_t row, uint32_t count,
                uint64_t tag, std::string& err) {
        const uint64_t alignment = platform::DirectFile::alignment();
        const uint64_t start = offset + row * kSourceStride;
        stage.prefix = (uint32_t) (start % alignment);
        stage.rows = count;
        stage.required = stage.prefix + count * kSourceStride;
        const uint32_t length = (uint32_t) (((uint64_t) stage.required + alignment - 1) & ~(alignment - 1));
        return file.submit(start - stage.prefix, stage.data, length, tag, err);
    }

    bool ensure(uint64_t row, std::string& err) {
        if (row >= rows) { err = "PLE Q8_0 row is out of range"; return false; }
        const uint64_t region = row / kRegionRows;
        if (states[region].load(std::memory_order_acquire) == Ready) return true;
        try {
            std::unique_lock<std::mutex> demand_lock(demand_mu);
            if (states[region].load(std::memory_order_acquire) == Ready) return true;
            if (!failed.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire)) {
                uint8_t expected = Empty;
                if (states[region].compare_exchange_strong(expected, Loading, std::memory_order_acq_rel)) {
                    const uint64_t first = region * kRegionRows;
                    const uint32_t count = (uint32_t) std::min<uint64_t>(kRegionRows, rows - first);
                    err.clear();
                    platform::Completion completion;
                    if (!submit(demand_file, demand_stage, first, count, 0, err) ||
                        demand_file.wait(&completion, 1, -1) != 1 || completion.tag != 0 ||
                        !completion.ok || completion.bytes < demand_stage.required) {
                        if (err.empty()) err = "PLE Q8_0 demand read failed or was short at row " + std::to_string(first);
                        fail(err);
                        publish(region, Failed);
                        return false;
                    }
                    convert(first, count, demand_stage.data + demand_stage.prefix, demand_scratch.get());
                    return true;
                }
            }
            // A background worker already owns this small region; it never claims before its I/O finishes.
            demand_lock.unlock();
            std::unique_lock<std::mutex> lock(state_mu);
            changed.wait(lock, [&] {
                return states[region].load(std::memory_order_acquire) != Loading ||
                       failed.load(std::memory_order_acquire) || stop.load(std::memory_order_acquire);
            });
            if (states[region].load(std::memory_order_acquire) == Ready) return true;
            err = failure.empty() ? "PLE Q8_0 conversion stopped" : failure;
            return false;
        } catch (const std::exception& e) {
            fail(std::string("PLE Q8_0 demand conversion failed: ") + e.what());
        } catch (...) {
            fail("PLE Q8_0 demand conversion failed with an unknown exception");
        }
        std::lock_guard<std::mutex> lock(state_mu);
        err = failure;
        return false;
    }

    // A single persistent team processes disjoint regions of each ready stage, never individual rows.
    struct Pool {
        Impl& owner;
        std::vector<std::thread> threads;
        std::unique_ptr<float[]> scratch;
        std::mutex mu;
        std::condition_variable work, done;
        bool stopping = false;
        uint64_t generation = 0, first = 0;
        uint32_t count = 0;
        int pending = 0;
        const uint8_t* source = nullptr;

        explicit Pool(Impl& impl) : owner(impl), scratch(new float[(size_t) impl.workers * kScratchValues]) {}
        ~Pool() {
            { std::lock_guard<std::mutex> lock(mu); stopping = true; }
            work.notify_all();
            for (auto& thread : threads) thread.join();
        }
        void convert(int id) {
            try {
                for (uint64_t at = (uint64_t) id * kRegionRows; at < count;
                     at += (uint64_t) owner.workers * kRegionRows) {
                    if (owner.stop.load(std::memory_order_acquire) || owner.failed.load(std::memory_order_acquire)) return;
                    const uint64_t row = first + at;
                    uint8_t expected = Empty;
                    if (owner.states[row / kRegionRows].compare_exchange_strong(expected, Loading, std::memory_order_acq_rel))
                        owner.convert(row, (uint32_t) std::min<uint64_t>(kRegionRows, count - at),
                                      source + at * kSourceStride, scratch.get() + (size_t) id * kScratchValues);
                }
            } catch (const std::exception& e) {
                owner.fail(std::string("PLE Q8_0 background conversion failed: ") + e.what());
            } catch (...) {
                owner.fail("PLE Q8_0 background conversion failed with an unknown exception");
            }
        }
        void start() {
            threads.reserve((size_t) owner.workers - 1);
            for (int id = 1; id < owner.workers; ++id) threads.emplace_back([this, id] {
                uint64_t seen = 0;
                std::unique_lock<std::mutex> lock(mu);
                for (;;) {
                    work.wait(lock, [&] { return stopping || generation != seen; });
                    if (stopping) return;
                    seen = generation;
                    lock.unlock();
                    convert(id);
                    lock.lock();
                    if (--pending == 0) done.notify_one();
                }
            });
        }
        void run(uint64_t row, uint32_t rows, const uint8_t* data) {
            {
                std::lock_guard<std::mutex> lock(mu);
                first = row; count = rows; source = data;
                pending = owner.workers - 1;
                ++generation;
            }
            work.notify_all();
            convert(0);
            std::unique_lock<std::mutex> lock(mu);
            done.wait(lock, [&] { return pending == 0; });
        }
    };

    void run() {
        try {
            Pool pool(*this);
            pool.start();
            std::string err;
            int slot = 0;
            for (uint64_t row = 0; row < rows;) {
                if (stop.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)) break;
                if (row == 0 && !submit(background_file, stages[slot], 0,
                                       (uint32_t) std::min<uint64_t>(chunk, rows), slot, err)) {
                    fail(err);
                    break;
                }
                platform::Completion completion;
                const int count = background_file.wait(&completion, 1, -1);
                if (stop.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)) break;
                Stage& stage = stages[slot];
                if (count != 1 || completion.tag != (uint64_t) slot || !completion.ok || completion.bytes < stage.required) {
                    fail("PLE Q8_0 background read failed or was short at row " + std::to_string(row));
                    break;
                }
                const uint64_t next = row + stage.rows;
                if (next < rows && !submit(background_file, stages[slot ^ 1], next,
                        (uint32_t) std::min<uint64_t>(chunk, rows - next), slot ^ 1, err)) {
                    fail(err);
                    break;
                }
                pool.run(row, stage.rows, stage.data + stage.prefix);
                if (next < rows && !stop.load(std::memory_order_acquire) && !failed.load(std::memory_order_acquire))
                    progress->add((uint64_t) stage.rows * kOutputStride);
                row = next;
                slot ^= 1;
            }
        } catch (const std::exception& e) {
            fail(std::string("PLE Q8_0 background failed: ") + e.what());
        } catch (...) {
            fail("PLE Q8_0 background failed with an unknown exception");
        }
        // No conversion worker remains. Drain/cancel I/O before releasing either stage, including on failure.
        {
            std::lock_guard<std::mutex> lock(background_io_mu);
            background_file.close();
        }
        for (auto& stage : stages) {
            platform::DirectFile::free_aligned(stage.data);
            stage.data = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(demand_mu);
            demand_file.close();
            platform::DirectFile::free_aligned(demand_stage.data);
            demand_stage.data = nullptr;
            demand_scratch.reset();
        }
        if (!stop.load(std::memory_order_acquire) && !failed.load(std::memory_order_acquire)) progress->finish();
        {
            std::lock_guard<std::mutex> lock(state_mu);
            finished = true;
        }
        changed.notify_all();
    }
};

LockedPleBuffer::LockedPleBuffer() = default;
LockedPleBuffer::~LockedPleBuffer() {
    impl_.reset();
    if (locked_bytes != 0) platform::unlock_resident(mapping, locked_bytes);
    platform::free_region(mapping, mapping_bytes);
}
bool LockedPleBuffer::ensure_row(uint64_t row, std::string& err) {
    if (!impl_) { err = "PLE Q8_0 table is not open"; return false; }
    return impl_->ensure(row, err);
}
bool LockedPleBuffer::wait(std::string& err) {
    if (!impl_) { err = "PLE Q8_0 table is not open"; return false; }
    std::unique_lock<std::mutex> lock(impl_->state_mu);
    impl_->changed.wait(lock, [&] { return impl_->finished; });
    if (impl_->failed.load(std::memory_order_acquire)) { err = impl_->failure; return false; }
    return true;
}

std::unique_ptr<LockedPleBuffer> open_ple_q8_0(const std::string& path, uint64_t offset, uint64_t rows,
                                            std::string& err, int workers, uint32_t chunk_rows) {
    try {
        if (workers < 1 || rows == 0 || chunk_rows == 0) {
            err = "PLE CPU quantization requires workers >= 1, rows > 0 and chunk_rows > 0";
            return nullptr;
        }
        const uint64_t max = std::numeric_limits<uint64_t>::max();
        if (rows > (max - offset) / kSourceStride ||
            rows > (std::numeric_limits<size_t>::max() - kRegionAlignment) / kOutputStride) {
            err = "PLE CPU quantization source or destination extent overflows";
            return nullptr;
        }
        // Chunk boundaries are region boundaries; no region ever has two partial producers.
        const uint64_t rounded_chunk = ((uint64_t) chunk_rows + kRegionRows - 1) / kRegionRows * kRegionRows;
        const uint64_t chunk = std::min<uint64_t>(rows, rounded_chunk);
        const uint64_t stage_bytes = stage_size(chunk);
        if (stage_bytes > std::numeric_limits<uint32_t>::max() || stage_bytes > std::numeric_limits<size_t>::max()) {
            err = "PLE CPU quantization chunk exceeds the direct-read size limit";
            return nullptr;
        }
        const uint64_t regions = (rows + kRegionRows - 1) / kRegionRows;
        workers = (int) std::min<uint64_t>(workers, (chunk + kRegionRows - 1) / kRegionRows);
        const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        if (traits == nullptr || traits->from_float_ref == nullptr) {
            err = "PLE CPU quantization has no Q8_0 reference quantizer";
            return nullptr;
        }
        auto output = std::make_unique<LockedPleBuffer>();
        output->impl_ = std::make_unique<LockedPleBuffer::Impl>();
        auto& impl = *output->impl_;
        if (!impl.background_file.open(path, err) || !impl.demand_file.open(path, err)) return nullptr;
        const uint64_t source_bytes = rows * kSourceStride;
        if (offset > impl.background_file.size() || source_bytes > impl.background_file.size() - offset ||
            offset > impl.demand_file.size() || source_bytes > impl.demand_file.size() - offset) {
            err = "PLE CPU quantization source range exceeds the file";
            return nullptr;
        }
        const uint64_t output_bytes = rows * kOutputStride;
        const uint64_t allocation_bytes = (output_bytes + kRegionAlignment - 1) & ~(kRegionAlignment - 1);
        uint64_t reserve = 0;
        const uint64_t additions[] = {2 * stage_bytes, stage_size(kRegionRows),
            ((uint64_t) workers + 1) * kScratchValues * sizeof(float),
            regions * sizeof(std::atomic<uint8_t>), platform::ram_margin(),
            platform::kLateHostBytes, platform::kRequestHostBytes};
        for (uint64_t bytes : additions) {
            if (bytes > max - reserve) { err = "PLE CPU quantization memory budget overflows"; return nullptr; }
            reserve += bytes;
        }
        platform::RegionBacking backing = platform::RegionBacking::NormalPages;
        std::string note;
        impl.progress = std::make_unique<core::LoadProgress>("ngram", output_bytes);
        output->data = (uint8_t*) platform::alloc_region_budgeted(allocation_bytes, reserve, backing, note,
                                                                output->mapping, output->mapping_bytes);
        if (output->data == nullptr) { err = "PLE CPU Q8_0 allocation failed: " + note; return nullptr; }
        if (backing != platform::RegionBacking::LargePages) {
            const auto lock = platform::lock_resident(output->mapping, output->mapping_bytes,
                [&](uint64_t done) { output->locked_bytes = done; });
            output->locked_bytes = lock.locked_bytes;
            if (!lock.ok || lock.locked_bytes < output->mapping_bytes) {
                err = "PLE CPU Q8_0 requires a complete RAM lock: " + lock.note;
                return nullptr;
            }
        }
        impl.output = output->data; impl.offset = offset; impl.rows = rows;
        impl.chunk = (uint32_t) chunk; impl.workers = workers;
        impl.quantize = traits->from_float_ref;
        impl.states.reset(new std::atomic<uint8_t>[(size_t) regions]);
        for (uint64_t i = 0; i < regions; ++i) impl.states[i].store(Empty, std::memory_order_relaxed);
        impl.demand_scratch.reset(new float[kScratchValues]);
        impl.demand_stage.data = (uint8_t*) platform::DirectFile::alloc_aligned((size_t) stage_size(kRegionRows));
        for (auto& stage : impl.stages) stage.data = (uint8_t*) platform::DirectFile::alloc_aligned((size_t) stage_bytes);
        if (!impl.demand_stage.data || !impl.stages[0].data || !impl.stages[1].data) {
            err = "PLE CPU quantization cannot allocate direct-read staging";
            return nullptr;
        }
        basalt::log::info("basalt", "PLE CPU BF16->Q8_0: %llu B, %d background workers, %s; unfinished rows pack on demand",
                          (unsigned long long) output_bytes, workers,
                          backing == platform::RegionBacking::LargePages ? "large pages" : "locked normal pages");
        impl.background = std::thread([&impl] { impl.run(); });
        return output;
    } catch (const std::exception& e) {
        err = std::string("PLE CPU quantization failed: ") + e.what();
    } catch (...) {
        err = "PLE CPU quantization failed with an unknown exception";
    }
    return nullptr;
}

}  // namespace basalt::ngram
