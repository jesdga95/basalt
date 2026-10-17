/* The two scalar F16C conversions for MSVC.
 *
 * MSVC's headers never declare the pair, and with __F16C__ defined the compiler claims the names as its own:
 * a definition compiled under that macro does not build as written. This file is therefore compiled without
 * __F16C__ (see the CMake wiring), the callers carry prototypes and #pragma function so the calls reach these
 * definitions, and the bodies use the vector intrinsics, which are the same instruction with the same result.
 * The rounding mode is a constant in the intrinsic, so the four cases are written out.
 *
 * Only the AVX2 and AVX-512 dot products reach these, and they run behind the run-time F16C probe.
 */
#if defined(_MSC_VER)
#include <immintrin.h>

unsigned short _cvtss_sh(float a, const int rounding) {
    const __m128 v = _mm_set_ss(a);
    switch (rounding & 3) {
        case 0:  return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 0));
        case 1:  return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 1));
        case 2:  return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 2));
        default: return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 3));
    }
}

float _cvtsh_ss(unsigned short a) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) a)));
}
#endif
