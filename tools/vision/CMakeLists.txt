# tools/vision - basalt-vision, the image encoder for the multimodal path, on Basalt's own kernels (no ggml or
# llama.cpp).  Built with the root project, or on its own:
#
#   cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-vision --target basalt-vision
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  project(basalt_vision CXX CUDA)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CUDA_STANDARD 20)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  option(BUILD_TESTING "Build the parity programs" OFF)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(VISION_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

# host side: preprocessing, projector file, CPU encoder
add_library(vision_core STATIC
  ${VISION_ROOT}/src/vision/preprocess.cpp
  ${VISION_ROOT}/src/vision/mmproj.cpp
  ${VISION_ROOT}/src/vision/encoder_cpu.cpp
  ${VISION_ROOT}/src/vision/gemm_cpu.cpp
  ${VISION_ROOT}/src/vision/gemm_cpu_avx512.cpp)
target_include_directories(vision_core PUBLIC ${VISION_ROOT}/include PRIVATE ${VISION_ROOT}/src/vision)
if(MSVC)
  # /arch: selects the baseline; the intrinsics used below are available to a function that uses them either way.
  set_source_files_properties(${VISION_ROOT}/src/vision/gemm_cpu.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  # the AVX-512 BF16 file is only called after a run-time CPU check
  set_source_files_properties(${VISION_ROOT}/src/vision/gemm_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties(${VISION_ROOT}/src/vision/gemm_cpu.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  # the AVX-512 BF16 file is only called after a run-time CPU check
  set_source_files_properties(${VISION_ROOT}/src/vision/gemm_cpu_avx512.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512bf16;-mfma")
endif()
target_link_libraries(vision_core PUBLIC platform Threads::Threads)

# device side
add_library(vision_cuda STATIC
  ${VISION_ROOT}/src/vision/encoder_cuda.cu
  ${VISION_ROOT}/src/vision/vision_kernels.cu)
target_include_directories(vision_cuda PUBLIC ${VISION_ROOT}/include ${VISION_ROOT}/src/vision)
target_link_libraries(vision_cuda PUBLIC vision_core CUDA::cudart CUDA::cublas)

add_executable(basalt-vision vision.cpp)
target_compile_definitions(basalt-vision PRIVATE BASALT_VISION_CUDA)
target_link_libraries(basalt-vision PRIVATE vision_cuda)
set_target_properties(basalt-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${VISION_ROOT}/bin)
# The image and projector paths arrive on the command line, so this program needs the same manifest as the engine:
# UTF-8 as the process code page and long paths. Built here directly rather than through the root project's helper,
# because this project also configures on its own.
if(MSVC)
  target_link_options(basalt-vision PRIVATE "/MANIFEST:EMBED"
                      "/MANIFESTINPUT:${VISION_ROOT}/src/platform/basalt.exe.manifest")
endif()

if(BUILD_TESTING)
  add_executable(vision_cpu_parity ${VISION_ROOT}/tests/vision/cpu_parity.cpp)
  target_link_libraries(vision_cpu_parity PRIVATE vision_core)
  add_executable(vision_cuda_parity ${VISION_ROOT}/tests/vision/cuda_parity.cu)
  target_link_libraries(vision_cuda_parity PRIVATE vision_cuda)
  # the parity program is outside src/vision now and needs the private kernel header
  target_include_directories(vision_cuda_parity PRIVATE ${VISION_ROOT}/src/vision)
endif()
