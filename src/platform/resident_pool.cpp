#include "basalt/platform/memory.hpp"
#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <sys/mman.h>
#endif

namespace basalt::platform {

void* alloc_resident_pool(uint64_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    (void) madvise(p, bytes, MADV_HUGEPAGE);
    return p;
#endif
}

LockResult commit_resident_pool(void* p, uint64_t bytes) {
#if defined(_WIN32)
    if (!VirtualAlloc(p, (SIZE_T) bytes, MEM_COMMIT, PAGE_READWRITE))
        return {false, 0, "host pool recommit failed"};
#endif
    return lock_resident(p, bytes);
}

bool discard_resident_pool(void* p, uint64_t bytes) {
    if (!p || !bytes) return false;
#if defined(_WIN32)
    unlock_resident(p, bytes);
    return VirtualFree(p, (SIZE_T) bytes, MEM_DECOMMIT) != 0;
#else
    if (munlock(p, bytes) != 0) return false;
    return madvise(p, bytes, MADV_DONTNEED) == 0;
#endif
}

}  // namespace basalt::platform
