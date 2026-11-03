# cmake/target_arch.cmake - the two things a build is for: a CPU architecture and one GPU compute capability.
# Included right after project() by the root project and by the standalone tools/vision configuration.
#
#   BASALT_ARM64    ON when the target is Arm64: the CPU kernels are NEON. OFF on x86-64 (AVX2 / AVX-512).
#   BASALT_CUDA_CC  120 (discrete Blackwell) or 121 (the integrated Blackwell of the N1X and GB10 systems-on-chip).
#                   It follows CMAKE_CUDA_ARCHITECTURES, which defaults to 121 on Arm64 and 120 elsewhere. The two are
#                   independent: -DCMAKE_CUDA_ARCHITECTURES=120 on Arm64 builds for a discrete card in an Arm machine.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$" OR "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" MATCHES "^ARM64")
  set(BASALT_ARM64 ON)
else()
  set(BASALT_ARM64 OFF)
endif()

# _basalt_cuda_arch_default is set before project() when the caller named no architecture; compiler detection has
# since filled in the default of nvcc itself, which is replaced here.
if(_basalt_cuda_arch_default)
  if(BASALT_ARM64)
    set(CMAKE_CUDA_ARCHITECTURES 121 CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()

set(BASALT_CUDA_CC "")
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "[-a].*$" "" _base "${_a}")     # 120-real / 121a are still 120 / 121
  if(NOT _base MATCHES "^(120|121|89)$")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}': basalt builds for one compute capability, 12.0 or 12.1, plus sm_89 (Ada) beside it (the Ada port; its card gate is basalt::build_has_arch)")
  endif()
  if(NOT _base STREQUAL "89" AND BASALT_CUDA_CC AND NOT BASALT_CUDA_CC STREQUAL "89" AND NOT _base STREQUAL BASALT_CUDA_CC)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}': basalt builds for one Blackwell capability, 12.0 or 12.1 (plus sm_89 beside it for the Ada port)")
  endif()
  # sm_89 (the Ada port) is the expert tier beside a Blackwell arch, never the reference capability
  if(NOT _base STREQUAL "89" OR NOT BASALT_CUDA_CC)
    set(BASALT_CUDA_CC ${_base})
  endif()
endforeach()
if(BASALT_CUDA_CC STREQUAL "89")
  # an Ada-only build: the engine's capability gate is build_archs.hpp, and the message texts and the gpu_select_test
  # rigs are Blackwell-based, so the reference capability stays 12.0 (kCudaCC is only read for those)
  set(BASALT_CUDA_CC 120)
endif()
# the architecture of the targets that use the FP4 / block-scaled tensor-core instructions
set(BASALT_CUDA_ARCH_A "${BASALT_CUDA_CC}a-real")
add_compile_definitions(BASALT_CUDA_CC=${BASALT_CUDA_CC})
if(BASALT_ARM64)
  include(${CMAKE_CURRENT_LIST_DIR}/arm64.cmake)
  basalt_arm64_global_flags()
  message(STATUS "basalt: Arm64, NEON CPU kernels; CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "basalt: x86-64, AVX2 / AVX-512 CPU kernels; CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
endif()
