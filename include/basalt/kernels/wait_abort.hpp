// include/basalt/kernels/wait_abort.hpp - the sticky word of the bounded cross-block device waits.
//
// A wait that passes its bound (a hand-off that never comes: a block that got no SM, a commit that never ran) writes
// the word and falls through instead of trapping, and every bounded wait that has waited kWaitPollNs falls through
// once the word is set, so the launch and the graph after it end with garbage results and the context stays usable.
// The host reads the word after each window, fails that window and runs no further one.
#pragma once

#include <string>

namespace basalt::kernels {

/// who timed out (the code's bits 56..62)
constexpr unsigned kWaitMidChain = 1, kWaitCommitFinisher = 2, kWaitHostRelease = 3;
/// a bounded wait reads the word only once it has waited this long
constexpr unsigned long long kWaitPollNs = 1000000ull;

/// Allocate the word (mapped host memory, shared portable across cards when multi-GPU with per-device fallback) and
/// its alias on the current device; outside a stream capture, once per device.  Returns false on failure after logging.
bool wait_abort_try_init();
/// Non-throwing wrapper over wait_abort_try_init() for existing callers.
void wait_abort_init();
/// The word's alias on the current device (initialises on demand if needed; nullptr if allocation failed).
unsigned long long* wait_abort_device_word();
/// 0 while every bounded wait has finished in time; else a timed-out wait's code.
unsigned long long wait_abort_code();
/// "<who>: kind K, line L, block B" for a code.
std::string wait_abort_text(unsigned long long code);
/// Set the word (unless a wait already did): every bounded wait falls through once it has waited kWaitPollNs.
void wait_abort_release();
/// Tests only: clear the word.
void wait_abort_clear();

}  // namespace basalt::kernels

#if defined(__CUDACC__)
namespace basalt::kernels {

__device__ __forceinline__ unsigned long long wait_abort_code_of(unsigned who, unsigned kind, unsigned line, unsigned block) {
    return (1ull << 63) | ((unsigned long long) (who & 0x7fu) << 56) | ((unsigned long long) (kind & 0xffu) << 48) |
           ((unsigned long long) (line & 0xffffu) << 32) | (unsigned long long) block;
}
// A wait that has run `elapsed` ns (>= kWaitPollNs): true when it gives up, i.e. the word is set, or `elapsed` has
// passed `bound` (the word is then set to `code`).
static __device__ __noinline__ bool wait_give_up(unsigned long long* word, unsigned long long elapsed,
                                                 unsigned long long bound, unsigned long long code) {
    if (word == nullptr) return elapsed > bound;
    volatile unsigned long long* w = word;
    if (*w != 0) return true;
    if (elapsed <= bound) return false;
    *w = code;
    __threadfence_system();
    return true;
}

}  // namespace basalt::kernels
#endif
