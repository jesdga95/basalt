// include/basalt/core/loader.hpp - the one reader of the model file: file ranges to host memory and to device memory.
//
// A model decides what goes where (its allocations, tiers, homes and spares) and describes it as a LoadPlan: items that
// each say "these bytes of the file land there", with an optional elementwise conversion on the way.  The Loader moves
// the bytes: unbuffered reads on a pool of threads, straight into host memory where the destination allows it and
// through pinned staging with asynchronous copies on per-device streams where it is device memory, so the drive, the
// bus and the conversion overlap.  Every byte placed adds to the one progress counter (load_progress.hpp).
//
// An item may instead copy bytes the same plan places (a second home of an expert, a stage's image of weights already on
// a card): the loader runs those after every file read of the plan is complete, so one run is one pass over the file.
// What must happen to the read bytes before those copies (a RAM tier registered with the driver) is a step of the plan,
// run between the two.
//
// A start places its whole plan in one run (place_start); its staging is sized to that plan and freed with it, and its
// copy streams are made before the start sizes anything (prepare).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {

/// What the loader does to a range between the file and its destination.  The elements are little-endian.
enum class Conv : uint8_t {
    Copy,        ///< the bytes as they are
    Bf16High,    ///< f32 -> its high 16 bits (the bf16 encoding)
    ToF16,       ///< f32 -> fp16, round to nearest even
    WidenF16,    ///< fp16 -> f32, exact
    Bf16ToF32,   ///< bf16 -> f32, exact
    Bf16ToQ8_0,  ///< bf16 -> ggml's block_q8_0 (34 bytes per 32 elements), rounded as quantize_bf16_q8_0
};

/// The bytes a converted range of `in` file bytes takes at its destination.
uint64_t conv_out_bytes(Conv conv, uint64_t in);

/// `n` bf16 values (a multiple of 32) -> ggml's block_q8_0, rounded as ggml's reference quantizer does: per 32 values
/// d = amax / 127 stored as fp16, q = round(x / d).
void quantize_bf16_q8_0(const uint16_t* bf16, uint64_t n, uint8_t* blocks);

struct LoadItem {
    static constexpr uint64_t kOut = ~0ull;   ///< `count`: what the item adds to the bar is its destination bytes

    int file = 0;                    ///< LoadPlan::file()
    uint64_t off = 0;                ///< the range in the file ...
    uint64_t bytes = 0;              ///< ... of this many bytes
    const uint8_t* from = nullptr;   ///< a source in memory instead of the file: bytes this plan's file reads place
    int from_device = -1;            ///< `from` is memory of this device (-1: host memory); a device source is copied as is
    uint8_t* host = nullptr;         ///< the destination in host memory, or
    void* dev = nullptr;             ///< in memory of device `device`
    int device = 0;
    Conv conv = Conv::Copy;
    uint64_t count = kOut;           ///< bytes the bar counts for the item (0: counted elsewhere)
};

/// An ordered list of items over a set of files.
class LoadPlan {
public:
    /// A file of the plan: by path, or by a descriptor the caller keeps open (the loader reads around the file cache
    /// through a descriptor of its own on the same file).
    struct File {
        std::string path;
        int fd = -1;
    };
    /// The index of `path` in this plan, added when new.
    int file(const std::string& path);
    /// The index of the open descriptor `fd`, added when new.
    int file_fd(int fd);
    /// Appends an item; a host item adjacent to the last in the file and in memory, with the same conversion, extends it.
    void add(LoadItem item);
    /// Appends every item of `other`, its files taken into this plan's.
    void append(const LoadPlan& other);
    /// Appends every item of `other` as append() does, except that one whose file range and conversion a device item of
    /// this plan already reads becomes a copy of that item's destination (made after the reads, counted once): the
    /// file is read once for both.
    void append_reusing(const LoadPlan& other);
    /// Every device item whose destination lies in [dev_base, dev_base + bytes) lands at the same offset of `host`
    /// instead: a host image of that device region, laid out as the region would be.
    void to_host(const void* dev_base, uint64_t bytes, uint8_t* host);
    /// Runs `step` once every file read of the plan has landed, before its copies from memory; a false fails the run.
    void after_reads(std::function<bool(std::string&)> step);
    const std::vector<File>& files() const { return files_; }
    const std::vector<LoadItem>& items() const { return items_; }
    const std::vector<std::function<bool(std::string&)>>& steps() const { return steps_; }
    bool empty() const { return items_.empty() && steps_.empty(); }
    /// The bytes the bar counts for the whole plan.
    uint64_t counted() const;

private:
    std::vector<File> files_;
    std::vector<LoadItem> items_;
    std::vector<std::function<bool(std::string&)>> steps_;
};

class Loader {
public:
    static constexpr uint64_t kChunk = 16ull << 20;   ///< what a thread reads at once

    explicit Loader(int threads = 8);
    ~Loader();
    /// Makes its copy streams on `devices` now, which it keeps until it is destroyed: a start prepares its loader before
    /// it sizes anything from the free memory, so what the streams take on a card is not the caches'.  -1: the calling
    /// thread's current device.
    bool prepare(const std::vector<int>& devices, std::string& err);
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

    /// Executes `plan`: its file reads, its steps, then its copies from memory; complete when it returns true.  False with `err` on
    /// the first failure (the rest of the plan is not read).  The calling thread's current device is as it was.
    bool run(const LoadPlan& plan, std::string& err);
    /// Frees the staging buffers (pinned when the plan wrote a card); the next run allocates them again.
    void release_staging();
    /// Whether every byte read so far came around the file cache (a file that would not open that way, or a read that
    /// fell back, says no).
    bool unbuffered() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// The one pass of a start: announces the bytes `plan` counts as the start's total, then places them with `loader`, whose
/// staging is freed after it (a start that serves keeps its loader, not its pinned staging).
bool place_start(Loader& loader, const LoadPlan& plan, std::string& err);

}  // namespace basalt::core
