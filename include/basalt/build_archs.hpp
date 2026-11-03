// include/basalt/build_archs.hpp - the compute capabilities this build holds device code for.
//
// CMake passes CMAKE_CUDA_ARCHITECTURES as BASALT_CUDA_ARCHS, its entries joined by commas ("120,89", "89",
// "120a-real,89-real"). A card of another compute capability would run its kernels from the PTX of a lower
// architecture, where the code that card's paths need is compiled out (a trap), so the engine refuses it at start.
#pragma once

#ifndef BASALT_CUDA_ARCHS
#error "BASALT_CUDA_ARCHS is defined by CMake from CMAKE_CUDA_ARCHITECTURES"
#endif

namespace basalt {

/// Whether the comma-separated architecture list `list` ("120,89"; a suffix such as "a" or "-real" after the number is
/// ignored) holds compute capability major.minor: 120 is 12.0, 89 is 8.9.
inline bool arch_list_has(const char* list, int major, int minor) {
    const int want = major * 10 + minor;
    for (const char* s = list; *s;) {
        int v = 0;
        bool digits = false;
        for (; *s >= '0' && *s <= '9'; ++s) {
            v = v * 10 + (*s - '0');
            digits = true;
        }
        if (digits && v == want) return true;
        while (*s && *s != ',') ++s;
        if (*s == ',') ++s;
    }
    return false;
}

/// This build's architecture list, as CMake gave it ("120,89").
inline const char* build_archs() { return BASALT_CUDA_ARCHS; }

/// Whether this build holds device code for compute capability major.minor.
inline bool build_has_arch(int major, int minor) { return arch_list_has(BASALT_CUDA_ARCHS, major, minor); }

}  // namespace basalt
