# BASALT_ARM64: whether the target is Arm64, whose CPU kernels are NEON with the dot product (x86-64 builds the AVX2 /
# AVX-512 ones).  Read right after project() by the root project and by the standalone `tools/vision` configuration,
# since the compute capability a build defaults to follows it: 12.1 (the integrated GPU) on Arm64, 12.0 elsewhere.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$" OR CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
  set(BASALT_ARM64 ON)
  message(STATUS "basalt: CPU kernels NEON (Arm64)")
else()
  set(BASALT_ARM64 OFF)
endif()
