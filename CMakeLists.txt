# basalt: a local inference engine for one NVIDIA GPU.
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
endif()
project(basalt VERSION 0.0.1 LANGUAGES C CXX CUDA)
add_compile_definitions(BASALT_VERSION="${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BUILD_TESTING "Build the tests and parity programs" OFF)
option(WERROR "Treat compiler warnings as errors" OFF)
if(BUILD_TESTING)
  enable_testing()
endif()

# Every test is registered through one of these so it carries a label: `cpu` tests run with no GPU or driver present,
# `gpu` tests need a device. `ctest -L cpu` is what the hosted CI runs.
function(_basalt_labelled_test label)
  add_test(${ARGN})
  cmake_parse_arguments(_t "" "NAME" "" ${ARGN})
  set_property(TEST ${_t_NAME} APPEND PROPERTY LABELS ${label})
endfunction()
function(basalt_cpu_test)
  _basalt_labelled_test(cpu ${ARGN})
endfunction()
function(basalt_gpu_test)
  _basalt_labelled_test(gpu ${ARGN})
endfunction()

# the CUDA language gives the .cu compile rules but not the imported targets, so `CUDA::cudart` is brought in
# explicitly for the host-side programs that call the runtime themselves
find_package(CUDAToolkit REQUIRED)
# One architecture only: the kernels use its instructions and nothing else is tested.
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "[-a].*$" "" _base "${_a}")     # 120-real / 120a are still 120
  if(NOT _base STREQUAL "120")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}': basalt builds for compute capability 12.0 only")
  endif()
endforeach()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION}: basalt needs CUDA 13.0 or newer")
endif()
message(STATUS "basalt: CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
set(_gpu_runtime_target CUDA::cudart)
set(_gpu_blas_target CUDA::cublas)
set(_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# MSVC reports __cplusplus as 199711L unless /Zc:__cplusplus is given. cuda.h applies alignas(TENSOR_MAP_ALIGN) to
# CUtensorMap only when __cplusplus says C++11 or later, so without the option the type keeps the 8-byte alignment
# of its single array member and the TMA descriptors inside a __grid_constant__ kernel-argument struct fault on a
# misaligned address. The option has to be global: the alignment is decided in the translation unit that reads the
# header, which is not necessarily one that links the target carrying a compile option.
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/Zc:__cplusplus>)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
  # nvcc takes host-compiler options through -Xcompiler, and this reaches every CUDA target.
  string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:__cplusplus")
endif()
# CCCL in CUDA 13.2 and later rejects cl.exe's traditional preprocessor (C1189); nvcc's default Windows host is cl.exe.
if(WIN32)
  file(TO_CMAKE_PATH "${CMAKE_CUDA_HOST_COMPILER}" _cuda_host)
  get_filename_component(_cuda_host "${_cuda_host}" NAME)
  string(TOLOWER "${_cuda_host}" _cuda_host)
  if(_cuda_host STREQUAL "" OR _cuda_host MATCHES "^cl(\\.exe)?$")
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:preprocessor")
  endif()
endif()

# Warnings live on an INTERFACE target the project's own C++ code links, NOT in add_compile_options: applied
# globally they would also hit the extracted ggml C and CUDA sources, whose warnings are not this project's to fix.
# The flags are guarded by COMPILE_LANGUAGE:CXX so nvcc never receives them as inputs, and by the compiler id:
# MSVC reads -Wall/-Wextra as /Wall and the invalid /Wextra, so it takes its own spellings. The sources are UTF-8,
# which MSVC only assumes when told (the Windows SDK's headers are not), hence /utf-8 for both languages.
add_library(warnings INTERFACE)
target_compile_options(warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>)
if(WERROR)
  target_compile_options(warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# The application manifest, for every program that names a file path in its command line. Windows decodes argv and
# the CRT's narrow opens in the machine's ANSI code page unless the image says otherwise, so the same pack path
# opens at one call site and fails at another. The manifest says the process code page is UTF-8 and that long paths
# are in use; `link.exe` merges the file into the RT_MANIFEST resource it already writes, so there is no mt.exe step
# and no external .manifest beside the exe. Nothing changes on Linux, where a path is bytes.
set(_basalt_manifest ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/basalt.exe.manifest)
function(basalt_embed_manifest target)
  if(MSVC)
    target_link_options(${target} PRIVATE "/MANIFEST:EMBED" "/MANIFESTINPUT:${_basalt_manifest}")
    # a manifest edit has to relink the program, and the link command is the only thing that reads the file
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${_basalt_manifest})
  endif()
endfunction()

# The vector-extension flags per file, in each compiler's spelling. MSVC has no per-feature ISA switches: /arch:
# turns on the whole rung (and a TU built with it may use those instructions anywhere in itself, exactly as the
# GNU flags do), so the run-time probe still decides which of these objects runs.
#
# /arch:AVX512 predefines __AVX__, __AVX2__ and __AVX512F/BW/VL/DQ__ but leaves __F16C__, __AVX512VNNI__ and
# __AVX512VBMI__ undefined, while accepting the intrinsics of all three. The sources branch on those macros (the
# ggml dot products pick their VNNI form by __AVX512VNNI__), so they are defined to match what the GNU flags give:
# without it a file compiles down its no-VNNI path and runs slower than the same build on Linux. __BMI2__ is left
# alone because -mavx2 and -mavx512* do not enable BMI2 either, so that branch is out on both platforms.
if(MSVC)
  set(_isa_avx512_flags "/arch:AVX512;-D__F16C__=1;-D__AVX512VNNI__=1;-D__AVX512VBMI__=1")
  set(_isa_avx2_flags "/arch:AVX2;-D__F16C__=1")
  set(_isa_avxvnni_flags "/arch:AVX2;-D__F16C__=1")
  set(_isa_avx1_flags "/arch:AVX")
else()
  set(_isa_avx512_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set(_isa_avx2_flags "-mavx2;-mfma;-mf16c")
  set(_isa_avxvnni_flags "-mavx2;-mfma;-mf16c;-mavxvnni")
  set(_isa_avx1_flags "-mavx")
endif()

# ---- artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(artifact INTERFACE)
target_include_directories(artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(artifact INTERFACE platform warnings)

add_executable(basalt-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(basalt-gguf PRIVATE artifact)
basalt_embed_manifest(basalt-gguf)

add_executable(basalt-dequant src/artifact/dequant.cpp)
target_link_libraries(basalt-dequant PRIVATE artifact)
basalt_embed_manifest(basalt-dequant)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(basalt-dequant PRIVATE DEQUANT_SELFTEST)
if(BUILD_TESTING)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test tests/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE artifact)
  basalt_embed_manifest(gguf_reader_test)
  basalt_cpu_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE artifact)
  basalt_embed_manifest(gguf_split_test)
  basalt_cpu_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the logit_bias protocol key: what parses and what is refused (host only)
  add_executable(logit_bias_test tests/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  basalt_cpu_test(NAME logit_bias_test COMMAND logit_bias_test)
endif()

# ---- plan ----------------------------------------------------------
# Also header-only: the planner holds no state, so its decision is arithmetic you can check by hand.
add_library(plan INTERFACE)
target_include_directories(plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(plan INTERFACE warnings)

add_executable(basalt-plan src/plan/plan_main.cpp)
target_link_libraries(basalt-plan PRIVATE plan)

# ---- platform (host OS boundary) ------------------------------------
# The file, memory and process calls the engine makes, so the Win32 and POSIX spellings live in one place.
# Host-only: no CUDA runtime, which is why the header-only `artifact` target, the vision library, the bench and the
# standalone test programs can all link it.
add_library(platform STATIC src/platform/file.cpp src/platform/memory.cpp src/platform/direct_file.cpp src/platform/process.cpp
  src/platform/mapped_file.cpp src/platform/cpu.cpp src/platform/system.cpp src/platform/dynlib.cpp)
target_include_directories(platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# dlopen (DynLib) lives in libdl before glibc 2.34
target_link_libraries(platform PUBLIC warnings ${CMAKE_DL_LIBS})
if(WIN32)
  target_link_libraries(platform PUBLIC advapi32)   # the processor name in the registry
endif()

# ---- core (GPU runtime) ---------------------------------------------
add_library(core STATIC src/core/device.cu src/core/pinned.cu src/core/graph.cpp src/core/weights.cpp src/core/direct_upload.cpp
                             src/core/layout.cpp src/core/model_file.cpp)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
# cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
# linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
target_link_libraries(core PUBLIC plan artifact platform warnings ${_gpu_runtime_target})

add_executable(basalt-device src/core/device_main.cpp)
target_link_libraries(basalt-device PRIVATE core)

add_executable(basalt-load src/core/load_main.cpp)
target_link_libraries(basalt-load PRIVATE core kernels)

add_executable(basalt-concurrent src/core/concurrent_main.cpp)
target_include_directories(basalt-concurrent PRIVATE ${_gpu_include_directories})
target_link_libraries(basalt-concurrent PRIVATE core kernels ${_gpu_runtime_target})

add_executable(basalt-overlap src/core/overlap_main.cpp)
# a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
target_include_directories(basalt-overlap PRIVATE ${_gpu_include_directories})
target_link_libraries(basalt-overlap PRIVATE core kernels ${_gpu_runtime_target})

# ---- kernels ----
# Naive first, each with a parity test against a reference that was itself validated against ggml.  The
# kernels link `artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
# reference is the whole point - a kernel checked against another copy of itself proves nothing.
add_library(kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                        src/kernels/cuda/s_gemv.cu
                        src/kernels/cuda/router_top10.cu
                        src/kernels/cuda/quantize_act.cu
                        src/kernels/cuda/rope.cu
                        src/kernels/cuda/sampler.cu
                        src/kernels/cuda/s2_gemv_q8.cu
  src/kernels/cuda/s2_expert_grouped.cu
                        src/kernels/cuda/shared_expert.cu
                        src/kernels/cuda/gr.cu
                        src/kernels/cuda/gdn.cu
                        src/kernels/cuda/qsa.cu
                        src/kernels/cuda/elementwise.cu
                        src/kernels/cuda/bf16_gemv.cu
                        src/kernels/cuda/native_mmvq.cu
                        src/kernels/cuda/native_bf16.cu
                        src/kernels/cuda/native_gr_norm.cu
                        src/kernels/cuda/native_moe.cu
                        src/kernels/cuda/native_gdn.cu
                        src/kernels/cuda/native_gr_postops.cu
                        src/kernels/cuda/native_router.cu
                        src/kernels/cuda/native_gdn_preprocess.cu
                        src/kernels/cuda/native_qsa.cu
                        src/kernels/cuda/native_rope.cu
                        src/kernels/cuda/native_ple_postops.cu
                        src/kernels/cuda/native_qsa_indexer.cu
                        src/kernels/cuda/native_qsa_score.cu
                        src/kernels/cuda/ple.cu
                        src/kernels/cuda/kv_q8.cu
                        src/kernels/cuda/kv_stream.cu
                        src/kernels/cuda/kv_q4.cu
                        src/kernels/cuda/fused_gr.cu
                        src/kernels/cuda/cvec.cu
                        src/kernels/cuda/qsa_decode_attn.cu
                        src/kernels/cuda/qsa_dec_attn.cu
                        src/kernels/cuda/qsa_dec_fused.cu
                        src/kernels/cuda/q5k_head.cu
                        src/kernels/cuda/q6k_head.cu
                        src/kernels/cuda/q80_head.cu
                        src/kernels/cuda/qsa_prompt_attn.cu
                        src/kernels/cuda/fused_gdn.cu
                        src/kernels/cuda/dequant_bf16.cu
                        src/kernels/cuda/qsa_select.cu
                        src/kernels/cuda/verify_kernels.cu
                        src/kernels/cuda/dec_gdn_rec.cu
                        src/kernels/cuda/dense_tc.cu
                        src/kernels/cuda/mid_chain.cu
                        src/kernels/cuda/iq_kernels.cu
                        src/kernels/ngram.cpp
                        src/ngram/ple_reader.cpp)
target_link_libraries(kernels PUBLIC core artifact)
target_sources(kernels PRIVATE src/kernels/cuda/hc_fused.cu src/kernels/cuda/hc8_gr.cu)
# the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
if(BUILD_TESTING)
  add_executable(qsa_prompt_attn_parity tests/kernels/qsa_prompt_attn_parity.cpp)
  target_include_directories(qsa_prompt_attn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(qsa_prompt_attn_parity PRIVATE kernels CUDA::cudart)
  basalt_gpu_test(NAME qsa_prompt_attn_parity COMMAND qsa_prompt_attn_parity 32768 2048 1)
  set_tests_properties(qsa_prompt_attn_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
# the top-k's active-block bound against
# the capacity rule, selected ids bitwise
if(BUILD_TESTING)
  add_executable(qsa_topk_active_parity tests/kernels/qsa_topk_active_parity.cpp)
  target_include_directories(qsa_topk_active_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(qsa_topk_active_parity PRIVATE kernels CUDA::cudart)
  basalt_gpu_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
  set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()
# Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/native_gr_norm.cu
    src/kernels/cuda/native_moe.cu
    src/kernels/cuda/native_gdn.cu
    src/kernels/cuda/native_gr_postops.cu
    src/kernels/cuda/native_router.cu
    src/kernels/cuda/native_gdn_preprocess.cu
    src/kernels/cuda/native_qsa.cu
    src/kernels/cuda/native_rope.cu
    src/kernels/cuda/native_ple_postops.cu
    src/kernels/cuda/native_qsa_indexer.cu
    src/kernels/cuda/native_qsa_score.cu
    src/kernels/cuda/qsa_dec_fused.cu
    src/kernels/cuda/q5k_head.cu
    src/kernels/cuda/q6k_head.cu
    src/kernels/cuda/q80_head.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")

# ---- engine -----------------------------------------------------
# The COMPOSED layer lives here and not in `core`, because it calls kernels and `kernels`
# already links `core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
# library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
# about arithmetic, and this one knows the ORDER the arithmetic happens in.
add_library(engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
  src/core/remote_expert_opt.cu
  src/core/expert_cache.cpp src/core/vmm.cpp src/core/kv_arena.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
  src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp src/core/staged_copy.cpp)
target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(engine PUBLIC core kernels kernels_cpu ${_gpu_runtime_target})
if(BUILD_TESTING)
  add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
  target_link_libraries(file_expert_source_test PRIVATE engine)
  basalt_embed_manifest(file_expert_source_test)
  basalt_cpu_test(NAME file_expert_source_test COMMAND file_expert_source_test)
  set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
endif()
# file ranges to the GPU through unbuffered reads
if(BUILD_TESTING)
  add_executable(direct_upload_test tests/core/direct_upload_test.cpp)
  target_link_libraries(direct_upload_test PRIVATE core)
  basalt_embed_manifest(direct_upload_test)
  basalt_gpu_test(NAME direct_upload_test COMMAND direct_upload_test)
  set_tests_properties(direct_upload_test PROPERTIES TIMEOUT 60)
endif()
# the learned expert profile (--expert-profile-save): its order and file format (CPU only)
if(BUILD_TESTING)
  add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
  target_link_libraries(expert_profile_save_test PRIVATE engine)
  basalt_embed_manifest(expert_profile_save_test)
  basalt_cpu_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
  set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
endif()
if(BUILD_TESTING)
  # the shape check over a synthetic weight table: the canonical shapes pass, one wrong dimension or form fails
  add_executable(layout_check_test tests/core/layout_check_test.cpp)
  target_link_libraries(layout_check_test PRIVATE core)
  basalt_cpu_test(NAME layout_check_test COMMAND layout_check_test)
  # the expert layout text: real models' layouts and the refusals
  add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
  target_link_libraries(expert_layout_test PRIVATE engine)
  basalt_cpu_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
  set_tests_properties(expert_layout_test PROPERTIES TIMEOUT 60)
  # the conversation cache: retention bookkeeping (header-only), the memory budget, the snapshot's validation (host
  # only: CUDA is linked but never initialized) and its transfer-failure paths, and a snapshot round trip on a device
  add_executable(conversation_cache_test tests/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  basalt_cpu_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  # conversation_memory.cpp asks the platform layer for the free RAM, so the probe comes along; it needs the system
  # allocator and the OS memory query, and nothing from the GPU runtime.
  add_executable(conversation_memory_test tests/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  target_link_libraries(conversation_memory_test PRIVATE platform)
  basalt_cpu_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(conversation_validation_test tests/core/conversation_validation_test.cpp)
  target_link_libraries(conversation_validation_test PRIVATE engine)
  basalt_cpu_test(NAME conversation_validation_test COMMAND conversation_validation_test)
  set_tests_properties(conversation_validation_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
  # The transfer-failure paths are reached by GNU ld's --wrap, which symbolizes the three runtime calls. The Windows
  # linker has no equivalent, so that one program is not built here; the same source builds the ordinary snapshot
  # test above, which does port.
  if(NOT WIN32)
    add_executable(conversation_transfer_test tests/core/conversation_validation_test.cpp)
    target_compile_definitions(conversation_transfer_test PRIVATE CONVERSATION_TEST_TRANSFERS)
    target_link_options(conversation_transfer_test PRIVATE
      -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaDeviceSynchronize -Wl,--wrap=cudaGetLastError)
    target_link_libraries(conversation_transfer_test PRIVATE engine)
    basalt_cpu_test(NAME conversation_transfer_test COMMAND conversation_transfer_test)
    set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
  endif()
  add_executable(conversation_snapshot_test tests/core/conversation_snapshot_test.cpp)
  target_link_libraries(conversation_snapshot_test PRIVATE engine)
  basalt_gpu_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
  set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # the prompt-time layer split's state moves (a layer range into a prompt session, back, into a checkpoint)
  add_executable(prompt_split_state_test tests/core/prompt_split_state_test.cpp)
  target_link_libraries(prompt_split_state_test PRIVATE engine)
  basalt_gpu_test(NAME prompt_split_state_test COMMAND prompt_split_state_test)
  set_tests_properties(prompt_split_state_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # the .basalt reader: table, meta JSON, the GGUF view of the source tensors, the layout text, the hc8 table
  add_executable(model_file_test tests/core/model_file_test.cpp)
  target_link_libraries(model_file_test PRIVATE engine)
  basalt_embed_manifest(model_file_test)
  basalt_cpu_test(NAME model_file_test COMMAND model_file_test)
  set_tests_properties(model_file_test PROPERTIES TIMEOUT 60)
  # a PLE key held unquantized in the dense index stays in the arena; a quantized key row loads as before
  add_executable(native_dense_ple_key_test tests/core/native_dense_ple_key_test.cpp)
  target_link_libraries(native_dense_ple_key_test PRIVATE engine)
  basalt_embed_manifest(native_dense_ple_key_test)
  basalt_gpu_test(NAME native_dense_ple_key_test COMMAND native_dense_ple_key_test)
  set_tests_properties(native_dense_ple_key_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  # per-layer admission on a sized cache starts each layer in its own range
  add_executable(expert_cache_per_layer_test tests/core/expert_cache_per_layer_test.cpp)
  target_link_libraries(expert_cache_per_layer_test PRIVATE engine)
  basalt_gpu_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
  set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
  target_link_libraries(expert_cache_segmented_test PRIVATE engine)
  basalt_gpu_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
  set_tests_properties(expert_cache_segmented_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  add_executable(file_expert_homes_test tests/core/file_expert_homes_test.cpp)
  target_link_libraries(file_expert_homes_test PRIVATE engine)
  basalt_embed_manifest(file_expert_homes_test)
  basalt_gpu_test(NAME file_expert_homes_test COMMAND file_expert_homes_test)
  set_tests_properties(file_expert_homes_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  add_executable(kv_arena_test tests/core/kv_arena_test.cpp)
  target_link_libraries(kv_arena_test PRIVATE engine)
  basalt_gpu_test(NAME kv_arena_test COMMAND kv_arena_test)
  set_tests_properties(kv_arena_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()

# `basalt` (generate.cpp): the engine program that composes the components into a token.  The prompt arrives as
# ids, so a tokenizer is never on one side of a logit comparison.
# The prompt path: batched prompt processing (tensor-core GEMMs through cuBLAS).
add_library(prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/gdn_chunked.cu src/prefill/prefill.cpp)
target_include_directories(prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_gpu_include_directories})
target_link_libraries(prefill PUBLIC engine ${_gpu_blas_target} ${_gpu_runtime_target})

add_executable(basalt src/program/generate.cpp src/program/probe_rates.cpp src/program/serve/loans.cpp
                       src/program/measure/gpu_modes.cpp src/program/measure/dump_set.cpp)
set_target_properties(basalt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_include_directories(basalt PRIVATE ${_gpu_include_directories})
target_link_libraries(basalt PRIVATE engine engine_session grammar tokenizer prefill spec ${_gpu_runtime_target})
basalt_embed_manifest(basalt)
# `basalt serve` runs the Python server from the source tree
target_compile_definitions(basalt PRIVATE BASALT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# The standard benchmark: a host-only program that drives the engine over its stdin/stdout protocol.
# The commit stamp is refreshed on every build (not only at configure time), by a target that rewrites the header
# only when the commit changed.
set(_git_stamp ${CMAKE_BINARY_DIR}/stamp/basalt_git.h)
add_custom_target(git_stamp
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${_git_stamp} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_stamp.cmake
  BYPRODUCTS ${_git_stamp}
  COMMENT "Refreshing the commit stamp")
execute_process(COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${_git_stamp}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_stamp.cmake)
# the engine driven as a child process and the report logic, shared by basalt-bench and basalt-probe
add_library(engine_session STATIC src/program/engine_session.cpp src/program/load_summary.cpp src/program/probe_report.cpp src/program/probe_derive.cpp src/program/auto_plan.cpp src/program/community_row.cpp src/program/host_rates.cpp
            src/program/gpu_select.cpp src/program/multi_gpu_plan.cpp src/program/image_loan.cpp)
target_include_directories(engine_session PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(engine_session PUBLIC platform Threads::Threads)
# the GBNF grammar engine and the JSON-schema and tool-call grammar builders (CPU only, no engine dependency)
add_library(grammar STATIC src/program/grammar/grammar.cpp src/program/grammar/schema.cpp src/program/grammar/json_schema.cpp
            src/program/grammar/schema_convert.cpp src/program/grammar/guard.cpp)
target_include_directories(grammar PUBLIC include)
target_link_libraries(grammar PRIVATE warnings)

# the C++ byte-level BPE tokenizer the engine runs on request text (CPU only)
add_library(tokenizer STATIC src/program/tokenizer/bpe.cpp src/program/tokenizer/pretok_qwen35.cpp
            src/program/tokenizer/unicode_flags.cpp)
target_include_directories(tokenizer PUBLIC include PRIVATE src/program/tokenizer)
target_link_libraries(tokenizer PRIVATE warnings)
add_executable(tokenizer_test tests/program/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE tokenizer warnings)
basalt_embed_manifest(tokenizer_test)
add_executable(tokenizer_bench tests/program/tokenizer_bench.cpp)
target_link_libraries(tokenizer_bench PRIVATE tokenizer warnings)
basalt_embed_manifest(tokenizer_bench)
basalt_cpu_test(NAME tokenizer_test COMMAND tokenizer_test)
set_tests_properties(tokenizer_test PROPERTIES TIMEOUT 60)
add_executable(basalt-bench src/program/bench_main.cpp)
set_target_properties(basalt-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# the image encoder sidecar, bin/basalt-vision (its own project, built with this one)
add_subdirectory(tools/vision)
target_link_libraries(basalt PRIVATE vision_cuda)
if(BUILD_TESTING)
  # the host half of the vision library needs no GPU: the decode guard is checked here
  add_executable(vision_preprocess_test tests/vision/preprocess_limit_test.cpp)
  target_link_libraries(vision_preprocess_test PRIVATE vision_core)
  basalt_cpu_test(NAME vision_preprocess_test COMMAND vision_preprocess_test)
endif()
add_dependencies(basalt-bench basalt basalt-device git_stamp)
target_include_directories(basalt-bench PRIVATE ${CMAKE_BINARY_DIR}/stamp)
target_link_libraries(basalt-bench PRIVATE engine_session platform warnings ${CMAKE_DL_LIBS})
basalt_embed_manifest(basalt-bench)
target_compile_definitions(basalt-bench PRIVATE
  BASALT_BENCH_ENGINE="$<TARGET_FILE:basalt>"
  BASALT_BENCH_DEVICE="$<TARGET_FILE:basalt-device>"
  BASALT_BENCH_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/data/bench"
  BASALT_BENCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  BASALT_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  BASALT_BENCH_CUDA="${CMAKE_CUDA_COMPILER_VERSION}")

# basalt-probe: starts the engine, measures its rates and cold decode speed, prints a report
add_executable(basalt-probe src/program/probe_main.cpp)
set_target_properties(basalt-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
add_dependencies(basalt-probe basalt)
target_link_libraries(basalt-probe PRIVATE engine_session platform warnings ${CMAKE_DL_LIBS})
basalt_embed_manifest(basalt-probe)
target_compile_definitions(basalt-probe PRIVATE
  BASALT_BENCH_ENGINE="$<TARGET_FILE:basalt>"
  BASALT_BENCH_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/data/bench")

# closing the engine's stdin mid-request stops it (runs the real engine on a model pack:
# -DBASALT_STDIN_STOP_MODEL=<pack>, with python3)
if(BASALT_STDIN_STOP_MODEL AND Python3_Interpreter_FOUND)
  add_test(NAME stdin_stop_test
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/stdin_stop_test.py
                   $<TARGET_FILE:basalt> ${BASALT_STDIN_STOP_MODEL} ${CMAKE_CURRENT_SOURCE_DIR}/data/bench/conv.ids
                   ${BASALT_STDIN_STOP_ARGS})
  set_property(TEST stdin_stop_test APPEND PROPERTY LABELS gpu)
  set_tests_properties(stdin_stop_test PROPERTIES TIMEOUT 600)
endif()

# the engine enforces a request grammar on what it samples (runs the real engine on a model pack:
# -DBASALT_GRAMMAR_MODEL=<pack>, with python3)
if(BASALT_GRAMMAR_MODEL AND Python3_Interpreter_FOUND)
  add_test(NAME grammar_engine_test
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/grammar_engine_test.py
                   $<TARGET_FILE:basalt> ${BASALT_GRAMMAR_MODEL} ${BASALT_GRAMMAR_ARGS})
  set_property(TEST grammar_engine_test APPEND PROPERTY LABELS gpu)
  set_tests_properties(grammar_engine_test PROPERTIES TIMEOUT 900)
endif()

# grammar requests under --batch: each slot held to its own grammar (starts the server on a model pack, same
# -DBASALT_GRAMMAR_MODEL, with python3; -DBASALT_GRAMMAR_BATCH_ARGS=<server args>)
if(BASALT_GRAMMAR_MODEL AND Python3_Interpreter_FOUND)
  add_test(NAME grammar_batch_test
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/grammar_batch_test.py
                   $<TARGET_FILE:basalt> ${BASALT_GRAMMAR_MODEL} ${BASALT_GRAMMAR_BATCH_ARGS})
  set_property(TEST grammar_batch_test APPEND PROPERTY LABELS gpu)
  set_tests_properties(grammar_batch_test PROPERTIES TIMEOUT 1500)
endif()

add_executable(dequant_s2_parity tests/kernels/dequant_s2_parity.cpp)
# The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
# and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
# target is just a C++ target and gets neither.
target_include_directories(dequant_s2_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dequant_s2_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

add_executable(s2_gemv_parity tests/kernels/s2_gemv_parity.cpp)
target_include_directories(s2_gemv_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(s2_gemv_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

add_executable(iq_parity tests/kernels/iq_parity.cpp)
target_include_directories(iq_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(iq_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_embed_manifest(iq_parity)
# the i-quant fixtures are generated, not shipped.  tools/iq_fixture.py needs python3 + numpy
# and the repository's vendored gguf-py (no pip installs); it writes deterministic fixtures into the
# BUILD directory (the source tree is not touched).  The CTest fixture makes generation-to-test order
# explicit even under parallel execution; a fixture-generation failure fails iq_parity with it.
find_package(Python3 COMPONENTS Interpreter QUIET)

# appending a community row to docs/community/benchmarks.json: formatting cases, and a byte comparison with
# tools/check_benchmarks.py --fix (python3)
add_executable(community_row_test tests/program/community_row_test.cpp)
target_link_libraries(community_row_test PRIVATE engine_session platform warnings)
basalt_embed_manifest(community_row_test)
basalt_cpu_test(NAME community_row_test COMMAND community_row_test)
set_tests_properties(community_row_test PROPERTIES TIMEOUT 60)
if(Python3_Interpreter_FOUND)
  basalt_cpu_test(NAME community_row_fix_test
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/community_row_fix_test.py
                          $<TARGET_FILE:community_row_test> ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(community_row_fix_test PROPERTIES TIMEOUT 60)
endif()

# the C++ tokenizer's ids against tools/tokenizer.py over a corpus
# (-DBASALT_TOKENIZER_DIR=<a pack's extracted tokenizer/ directory>, with python3 and the regex package)
if(BASALT_TOKENIZER_DIR AND Python3_Interpreter_FOUND)
  basalt_cpu_test(NAME tokenizer_parity
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/tokenizer_parity.py
                          $<TARGET_FILE:tokenizer_test> ${BASALT_TOKENIZER_DIR} ${BASALT_TOKENIZER_BIG})
  set_tests_properties(tokenizer_parity PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
endif()
# the generator needs an interpreter WITH numpy: probe the configured one, a .venv, and the
# system python3; -DIQ_FIXTURE_PYTHON=... overrides the search outright.
set(_iq_python "")
set(_iq_candidates "")
if(IQ_FIXTURE_PYTHON)
  list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
else()
  if(Python3_Interpreter_FOUND)
    list(APPEND _iq_candidates ${Python3_EXECUTABLE})
  endif()
  list(APPEND _iq_candidates "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
                              /usr/bin/python3
                              python3)
endif()
foreach(_cand IN LISTS _iq_candidates)
  if(_cand STREQUAL "")
    continue()
  endif()
  execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
  if(_iq_probe EQUAL 0)
    set(_iq_python "${_cand}")
    break()
  endif()
endforeach()
if(NOT _iq_python STREQUAL "")
  basalt_cpu_test(NAME iq_parity_fixtures
           COMMAND ${_iq_python} ${CMAKE_CURRENT_SOURCE_DIR}/tools/iq_fixture.py
                   --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
  # exit 3: numpy or the vendored gguf-py is missing (a trimmed checkout) - a skip, not a failure
  set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
  basalt_gpu_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
  set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  # the converter's own tests (the checkpoint repack, the pack sections, the requantizers): plain unittest, host only
  basalt_cpu_test(NAME convert_tests COMMAND ${_iq_python} -m unittest discover -s tools/convert -p "test_*.py"
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(convert_tests PROPERTIES TIMEOUT 300)
else()
  message(STATUS "basalt tests: no python3 with numpy was found (the configured interpreter and .venv "
                 "were probed) - the iq_parity fixtures and its test and the converter tests are not registered; "
                 "they need python3 with numpy and gguf-py")
endif()
# native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
# bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
add_executable(mmvq_multi_parity tests/kernels/mmvq_multi_parity.cpp)
target_include_directories(mmvq_multi_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(mmvq_multi_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)
# the decode GDN block against the per-stage verify kernels, bitwise (synthetic, GPU, no model; --bench)
add_executable(dec_gdn_parity tests/kernels/dec_gdn_parity.cpp)
target_include_directories(dec_gdn_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dec_gdn_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME dec_gdn_parity COMMAND dec_gdn_parity)
set_tests_properties(dec_gdn_parity PROPERTIES SKIP_RETURN_CODE 77)
# a GDN layer split by key-head group across two cards against the whole layer and a CPU reference (synthetic, GPU,
# no model; --bench)
add_executable(gdn_head_split_parity tests/kernels/gdn_head_split_parity.cpp)
target_include_directories(gdn_head_split_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(gdn_head_split_parity PRIVATE kernels artifact ${_gpu_runtime_target})
basalt_gpu_test(NAME gdn_head_split_parity COMMAND gdn_head_split_parity)
set_tests_properties(gdn_head_split_parity PROPERTIES SKIP_RETURN_CODE 77)
# the peer table update by kernel and the counter zeroing against the span copy and memset they replace, bitwise
# (GPU, synthetic, no model; --bench times each behind a burst of expert copies on another stream)
add_executable(dm_table_put_parity tests/kernels/dm_table_put_parity.cpp)
target_include_directories(dm_table_put_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dm_table_put_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME dm_table_put_parity COMMAND dm_table_put_parity)
set_tests_properties(dm_table_put_parity PROPERTIES SKIP_RETURN_CODE 77)
# the overlapped windows kernels: the list copy of a stage snapshot against a CPU copy, and the flag raised behind a
# stream writes (GPU, synthetic, no model; --bench)
add_executable(pair_snapshot_parity tests/kernels/pair_snapshot_parity.cpp)
target_include_directories(pair_snapshot_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(pair_snapshot_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME pair_snapshot_parity COMMAND pair_snapshot_parity)
set_tests_properties(pair_snapshot_parity PROPERTIES SKIP_RETURN_CODE 77)
# dense_tc (DEC_DTC) against native_mmvq: its columns' T contract bitwise, its error vs the dp4a kernels
add_executable(dense_tc_parity tests/kernels/dense_tc_parity.cpp)
target_include_directories(dense_tc_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dense_tc_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME dense_tc_parity COMMAND dense_tc_parity)
set_tests_properties(dense_tc_parity PROPERTIES SKIP_RETURN_CODE 77)
# the DEC_COMMIT launches against the commit graph's per-layer calls, bitwise (synthetic, GPU; --bench)
add_executable(dec_commit_parity tests/kernels/dec_commit_parity.cpp)
target_include_directories(dec_commit_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dec_commit_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME dec_commit_parity COMMAND dec_commit_parity)
basalt_gpu_test(NAME dec_commit_finisher COMMAND dec_commit_parity --finisher)
basalt_gpu_test(NAME dec_commit_lazy COMMAND dec_commit_parity --lazy)
# the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
# (GPU, synthetic, no model; --bench adds old/new timings)
add_executable(iq_multi_parity tests/kernels/iq_multi_parity.cpp)
target_include_directories(iq_multi_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(iq_multi_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME iq_multi_parity COMMAND iq_multi_parity)
# the prompt path's chunked DeltaNet recurrence against the token-serial kernel before it and a double-precision
# CPU reference, output and state (GPU, synthetic, no model; --bench adds timings)
add_executable(gdn_chunk_parity tests/prefill/gdn_chunk_parity.cu)
target_link_libraries(gdn_chunk_parity PRIVATE prefill ${_gpu_runtime_target})
target_include_directories(gdn_chunk_parity PRIVATE ${CMAKE_SOURCE_DIR}/include)
basalt_gpu_test(NAME gdn_chunk_parity COMMAND gdn_chunk_parity)
# the hyper-connection reads and prompt-path products on matrices held only as 8-bit copies
# against the bf16 ones (GPU, synthetic, no model; --bench adds timings)
add_executable(hc8_parity tests/kernels/hc8_parity.cpp)
target_include_directories(hc8_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(hc8_parity PRIVATE prefill kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME hc8_parity COMMAND hc8_parity)
# native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the launches before them (GPU, synthetic,
# no model; --bench adds v1/new timings of a window's 48 calls in a graph)
add_executable(native_grouped_parity tests/kernels/native_grouped_parity.cpp)
target_include_directories(native_grouped_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(native_grouped_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME native_grouped_parity COMMAND native_grouped_parity)
# dec_moe_glue (plan built inside, gate/up pair kernel, SwiGLU, down) bitwise vs native_expert_grouped, every covered pair
basalt_gpu_test(NAME native_grouped_parity_glue COMMAND native_grouped_parity --glue)
set_tests_properties(native_grouped_parity_glue PROPERTIES TIMEOUT 300)
add_executable(shared_expert_parity tests/kernels/shared_expert_parity.cpp)
target_include_directories(shared_expert_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(shared_expert_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

# the MoE half of a decode layer: the engine's launch sequence against the persistent experts kernel (GPU, synthetic)
add_executable(pk_moe_bench tests/kernels/pk_moe_bench.cpp)
target_include_directories(pk_moe_bench PRIVATE ${_gpu_include_directories})
target_link_libraries(pk_moe_bench PRIVATE kernels ${_gpu_runtime_target})

add_executable(hc_fused_parity tests/kernels/hc_fused_parity.cpp)
target_include_directories(hc_fused_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(hc_fused_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME hc_fused_parity COMMAND hc_fused_parity 20 1 8 --check-only --stress 200)
add_executable(mid_chain_parity tests/kernels/mid_chain_parity.cpp)
target_include_directories(mid_chain_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(mid_chain_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME mid_chain_parity COMMAND mid_chain_parity 20)
set_tests_properties(mid_chain_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
add_executable(gr_parity tests/kernels/gr_parity.cpp)
target_include_directories(gr_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(gr_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME gr_parity COMMAND gr_parity --selftest)

add_executable(gdn_parity tests/kernels/gdn_parity.cpp)
target_include_directories(gdn_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(gdn_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME gdn_parity COMMAND gdn_parity --selftest)

add_executable(s2_gemv_q8_parity tests/kernels/s2_gemv_q8_parity.cpp)
target_include_directories(s2_gemv_q8_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(s2_gemv_q8_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

add_executable(sampler_parity tests/kernels/sampler_parity.cpp)
target_include_directories(sampler_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(sampler_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME sampler_parity COMMAND sampler_parity --selftest)
# the decode QSA top-k and the greedy argmax on thread-block clusters vs the one-CTA kernels, bitwise,
# captured in a graph too (GPU, synthetic, no model; skips without cluster support); --bench: per-call times
add_executable(decode_cluster_parity tests/kernels/decode_cluster_parity.cpp)
target_include_directories(decode_cluster_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(decode_cluster_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME decode_cluster_parity COMMAND decode_cluster_parity --selftest)

add_executable(rope_parity tests/kernels/rope_parity.cpp)
target_include_directories(rope_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(rope_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME rope_parity COMMAND rope_parity --selftest)

add_executable(quantize_act_parity tests/kernels/quantize_act_parity.cpp)
target_include_directories(quantize_act_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(quantize_act_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

# the verify window's routing in one multi-column BF16 projection + one top-k, bitwise the per-token calls
add_executable(route_window_parity tests/kernels/route_window_parity.cpp)
target_include_directories(route_window_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(route_window_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME route_window_parity COMMAND route_window_parity)

add_executable(router_top10_parity tests/kernels/router_top10_parity.cpp)
target_include_directories(router_top10_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(router_top10_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)

add_executable(s_gemv_parity tests/kernels/s_gemv_parity.cpp)
target_include_directories(s_gemv_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(s_gemv_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

add_executable(elementwise_parity tests/kernels/elementwise_parity.cpp)
target_include_directories(elementwise_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(elementwise_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

add_executable(bf16_gemv_parity tests/kernels/bf16_gemv_parity.cpp)
target_include_directories(bf16_gemv_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(bf16_gemv_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

add_executable(s_gemv_q8k_parity tests/kernels/s_gemv_q8k_parity.cpp)
target_include_directories(s_gemv_q8k_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(s_gemv_q8k_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

add_executable(qsa_parity tests/kernels/qsa_parity.cpp)
target_include_directories(qsa_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(qsa_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME qsa_parity COMMAND qsa_parity --selftest)

# the QSA top-k the engine dispatches against the original kernel: identical ids on both sides of the register
# kernel's reach, with tied and with equal scores (synthetic, no model)
add_executable(qsa_scores_parity tests/kernels/qsa_scores_parity.cpp)
target_include_directories(qsa_scores_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(qsa_scores_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME qsa_scores_parity COMMAND qsa_scores_parity 20)
add_executable(dequant_f16_parity tests/kernels/dequant_f16_parity.cpp)
target_include_directories(dequant_f16_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dequant_f16_parity PRIVATE kernels artifact ${_gpu_runtime_target})
add_executable(qsa_topk_parity tests/kernels/qsa_topk_parity.cpp)
target_include_directories(qsa_topk_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(qsa_topk_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME qsa_topk_parity COMMAND qsa_topk_parity --selftest)

# ---- the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
# `ngram.cpp` is host-only but lives in `kernels` because `ple.cu` shares its geometry constants, and
# splitting them across two libraries would only add a layer.
#
# The dense section (extracted to a file), the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
# artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
# INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
# re-learning.
add_executable(ple_parity tests/kernels/ple_parity.cpp)
target_include_directories(ple_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(ple_parity PRIVATE kernels ${_gpu_runtime_target})
set(PLE_FIXTURE_DIR "" CACHE PATH "Directory containing the ple_in.bin and ple_out.bin parity fixtures")
if(PLE_FIXTURE_DIR)
  basalt_gpu_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${PLE_FIXTURE_DIR}/ple_in.bin" --out "${PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
add_executable(kv_q8_parity tests/kernels/kv_q8_parity.cpp)
target_include_directories(kv_q8_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(kv_q8_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME kv_q8_parity COMMAND kv_q8_parity)

# Decode attention for a few queries vs qsa_decode_attn_batch + the gate, bitwise (GPU, synthetic, no model).
add_executable(qsa_dec_attn_parity tests/kernels/qsa_dec_attn_parity.cpp)
target_include_directories(qsa_dec_attn_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(qsa_dec_attn_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME qsa_dec_attn_parity COMMAND qsa_dec_attn_parity 20)
add_executable(qsa_dec_fused_parity tests/kernels/qsa_dec_fused_parity.cpp)
target_include_directories(qsa_dec_fused_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(qsa_dec_fused_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME qsa_dec_fused_parity COMMAND qsa_dec_fused_parity 20)
add_executable(qsa_seg_parity tests/kernels/qsa_seg_parity.cpp)
target_include_directories(qsa_seg_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(qsa_seg_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME qsa_seg_parity COMMAND qsa_seg_parity 20)
add_executable(ple_win_parity tests/kernels/ple_win_parity.cpp)
target_include_directories(ple_win_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(ple_win_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME ple_win_parity COMMAND ple_win_parity 20)
add_executable(q5k_head_parity tests/kernels/q5k_head_parity.cpp)
target_include_directories(q5k_head_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(q5k_head_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME q5k_head_parity COMMAND q5k_head_parity 20000 5)
add_executable(q80_head_parity tests/kernels/q80_head_parity.cpp)
target_include_directories(q80_head_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(q80_head_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME q80_head_parity COMMAND q80_head_parity 20000 5)
add_executable(q6k_head_parity tests/kernels/q6k_head_parity.cpp)
target_include_directories(q6k_head_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(q6k_head_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME q6k_head_parity COMMAND q6k_head_parity 20000 5)
add_executable(q6k_vhead_parity tests/kernels/q6k_vhead_parity.cpp)
target_include_directories(q6k_vhead_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(q6k_vhead_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME q6k_vhead_parity COMMAND q6k_vhead_parity 20000 5)

# KV streaming: streamed vs resident attention, bitwise (GPU, synthetic, no model).
add_executable(kv_stream_parity tests/kernels/kv_stream_parity.cpp)
target_include_directories(kv_stream_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(kv_stream_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME kv_stream_parity COMMAND kv_stream_parity)

# the hybrid K8V4 KV (INT8 K + rotated Q4_0 V) vs host references (GPU, synthetic, no model).
add_executable(kv_hybrid_parity tests/kernels/kv_hybrid_parity.cpp)
target_include_directories(kv_hybrid_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(kv_hybrid_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME kv_hybrid_parity COMMAND kv_hybrid_parity)
# Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
add_executable(kv_q4_parity tests/kernels/kv_q4_parity.cpp)
target_include_directories(kv_q4_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(kv_q4_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME kv_q4_parity COMMAND kv_q4_parity)
add_executable(cvec_parity tests/kernels/cvec_parity.cpp)
target_include_directories(cvec_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(cvec_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME cvec_parity COMMAND cvec_parity)

# The rewritten per-hit and grouped expert kernels vs the previous ones, bitwise (GPU, synthetic, no model).
# `--bench` adds previous/new timings on engine-like shapes.
add_executable(s2_expert_grouped_parity tests/kernels/s2_expert_grouped_parity.cpp)
target_include_directories(s2_expert_grouped_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(s2_expert_grouped_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME s2_expert_grouped_parity COMMAND s2_expert_grouped_parity --selftest)

# working-set lock of a host region (CPU only).
add_executable(platform_memory_test tests/platform/memory_test.cpp)
target_link_libraries(platform_memory_test PRIVATE core ${_gpu_runtime_target})
basalt_embed_manifest(platform_memory_test)
basalt_cpu_test(NAME platform_memory_test COMMAND platform_memory_test)

# A path carried as UTF-8 reaching the OS intact: the seam in both directions, and a path past 260 chars
# (longPathAware). The program also runs itself as a child with a non-ASCII name as one argument, which is how a
# pack path typed by a user arrives; on Windows argv is decoded with the process code page, so that case only
# passes because this executable carries the application manifest.
add_executable(path_utf8_test tests/platform/path_utf8_test.cpp)
target_link_libraries(path_utf8_test PRIVATE platform)
basalt_embed_manifest(path_utf8_test)
basalt_cpu_test(NAME path_utf8_test COMMAND path_utf8_test)
set_tests_properties(path_utf8_test PROPERTIES TIMEOUT 60)

# platform::MappedFile, file_size and pread_all on a scratch file (CPU only).
add_executable(mapped_file_test tests/platform/mapped_file_test.cpp)
target_link_libraries(mapped_file_test PRIVATE platform)
basalt_embed_manifest(mapped_file_test)
basalt_cpu_test(NAME mapped_file_test COMMAND mapped_file_test)
set_tests_properties(mapped_file_test PROPERTIES TIMEOUT 60)
if(NOT WIN32)
  add_executable(mapped_file_large_test tests/platform/mapped_file_large_test.cpp)
  target_link_libraries(mapped_file_large_test PRIVATE platform)
  basalt_cpu_test(NAME mapped_file_large_test COMMAND mapped_file_large_test)
  set_tests_properties(mapped_file_large_test PROPERTIES TIMEOUT 60)
endif()

# the platform seams below the engine: thread placement and topology, the compiler helpers, large host regions, the
# piped child, the system questions and the dynamic loader (CPU only, both platforms).
foreach(_t cpu_test compiler_test region_test process_pipe_test system_test)
  add_executable(platform_${_t} tests/platform/${_t}.cpp)
  target_link_libraries(platform_${_t} PRIVATE platform Threads::Threads)
  target_include_directories(platform_${_t} PRIVATE include)
  basalt_embed_manifest(platform_${_t})
  basalt_cpu_test(NAME platform_${_t} COMMAND platform_${_t})
  set_tests_properties(platform_${_t} PROPERTIES TIMEOUT 120)
endforeach()

# the engine's load lines condensed for basalt-bench: lines, throttling, terminal mode (CPU only)
add_executable(load_summary_test tests/program/load_summary_test.cpp)
target_link_libraries(load_summary_test PRIVATE engine_session warnings)
basalt_embed_manifest(load_summary_test)
basalt_cpu_test(NAME load_summary_test COMMAND load_summary_test)
set_tests_properties(load_summary_test PROPERTIES TIMEOUT 60)

# the engine as a child process: an error line in a healthy start does not fail it, an exit before READY does (CPU only)
add_executable(engine_session_test tests/program/engine_session_test.cpp)
target_link_libraries(engine_session_test PRIVATE engine_session warnings)
basalt_embed_manifest(engine_session_test)
basalt_cpu_test(NAME engine_session_test COMMAND engine_session_test)
set_tests_properties(engine_session_test PROPERTIES TIMEOUT 60)

# the engine log helper: the line each level renders (CPU only)
add_executable(log_test tests/core/log_test.cpp)
target_include_directories(log_test PRIVATE include)
target_link_libraries(log_test PRIVATE warnings)
basalt_embed_manifest(log_test)
basalt_cpu_test(NAME log_test COMMAND log_test)
set_tests_properties(log_test PROPERTIES TIMEOUT 60)

# basalt-probe's report: rates block and layout (CPU only)
add_executable(probe_report_test tests/program/probe_report_test.cpp)
target_link_libraries(probe_report_test PRIVATE engine_session warnings)
basalt_embed_manifest(probe_report_test)
basalt_cpu_test(NAME probe_report_test COMMAND probe_report_test)
set_tests_properties(probe_report_test PROPERTIES TIMEOUT 60)

# the gpu setting: parsing, the card choice and the roles, over fake card tables (CPU only)
add_executable(gpu_select_test tests/program/gpu_select_test.cpp)
target_link_libraries(gpu_select_test PRIVATE engine_session warnings)
basalt_embed_manifest(gpu_select_test)
basalt_cpu_test(NAME gpu_select_test COMMAND gpu_select_test)
set_tests_properties(gpu_select_test PROPERTIES TIMEOUT 60)
add_executable(multi_gpu_plan_test tests/program/multi_gpu_plan_test.cpp)
target_link_libraries(multi_gpu_plan_test PRIVATE engine_session warnings)
basalt_embed_manifest(multi_gpu_plan_test)
basalt_cpu_test(NAME multi_gpu_plan_test COMMAND multi_gpu_plan_test)
set_tests_properties(multi_gpu_plan_test PROPERTIES TIMEOUT 60)

# basalt-probe's suggested values: the worker pick, the balance share and the suggestion lines (CPU only)
add_executable(probe_derive_test tests/program/probe_derive_test.cpp)
target_link_libraries(probe_derive_test PRIVATE engine_session warnings)
basalt_embed_manifest(probe_derive_test)
basalt_cpu_test(NAME probe_derive_test COMMAND probe_derive_test)
set_tests_properties(probe_derive_test PROPERTIES TIMEOUT 60)

# the start-up picks for settings left out: the rules and the printed line (CPU only)
add_executable(image_loan_test tests/program/image_loan_test.cpp)
target_link_libraries(image_loan_test PRIVATE engine_session warnings)
basalt_embed_manifest(image_loan_test)
basalt_cpu_test(NAME image_loan_test COMMAND image_loan_test)
set_tests_properties(image_loan_test PROPERTIES TIMEOUT 60)
add_executable(auto_plan_test tests/program/auto_plan_test.cpp)
target_link_libraries(auto_plan_test PRIVATE engine_session warnings)
basalt_embed_manifest(auto_plan_test)
basalt_cpu_test(NAME auto_plan_test COMMAND auto_plan_test)
set_tests_properties(auto_plan_test PROPERTIES TIMEOUT 60)

# the grammar engine: parse, accept/reject, UTF-8 across tokens, clone, lazy triggers and the vocabulary mask (CPU only)
add_executable(grammar_test tests/program/grammar_test.cpp)
target_link_libraries(grammar_test PRIVATE grammar warnings)
basalt_embed_manifest(grammar_test)
basalt_cpu_test(NAME grammar_test COMMAND grammar_test)
set_tests_properties(grammar_test PROPERTIES TIMEOUT 120)

# the pack vocabulary reader and the per-request window check (CPU only)
add_executable(grammar_guard_test tests/program/grammar_guard_test.cpp)
target_link_libraries(grammar_guard_test PRIVATE grammar warnings)
basalt_embed_manifest(grammar_guard_test)
basalt_cpu_test(NAME grammar_guard_test COMMAND grammar_guard_test)
set_tests_properties(grammar_guard_test PROPERTIES TIMEOUT 120)

# schema-to-grammar and the tool-call grammar (CPU only)
add_executable(grammar_schema_test tests/program/grammar_schema_test.cpp)
target_link_libraries(grammar_schema_test PRIVATE grammar warnings)
basalt_embed_manifest(grammar_schema_test)
basalt_cpu_test(NAME grammar_schema_test COMMAND grammar_schema_test)
set_tests_properties(grammar_schema_test PROPERTIES TIMEOUT 120)

# llama.cpp's own grammar tests (parser, accept/reject, state, schema-to-grammar) run against the grammar library (CPU only)
foreach(_t grammar_llama_parser_test grammar_llama_state_test grammar_llama_accept_test grammar_llama_schema_test)
  add_executable(${_t} tests/program/${_t}.cpp)
  target_link_libraries(${_t} PRIVATE grammar warnings)
  basalt_embed_manifest(${_t})
  basalt_cpu_test(NAME ${_t} COMMAND ${_t})
  set_tests_properties(${_t} PROPERTIES TIMEOUT 300)
endforeach()
target_include_directories(grammar_llama_schema_test PRIVATE src/program/grammar)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(pinned_shared_test tests/core/pinned_shared_test.cpp)
  target_link_libraries(pinned_shared_test PRIVATE core ${_gpu_runtime_target})
  basalt_cpu_test(NAME pinned_shared_test COMMAND pinned_shared_test)
endif()

# the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
add_executable(ple_reader_test tests/ngram/ple_reader_test.cpp)
target_include_directories(ple_reader_test PRIVATE ${_gpu_include_directories})
target_link_libraries(ple_reader_test PRIVATE kernels core ${_gpu_runtime_target})
basalt_embed_manifest(ple_reader_test)
basalt_gpu_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

# The GPU test runs ON THE TARGET PC. `--selftest` allocates
# and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
# which is correct: an engine that targets one GPU should not report a pass for having no GPU.
basalt_gpu_test(NAME cuda_device_selftest COMMAND basalt-device --selftest)

# ---- kernels_cpu ----
#
# A SEPARATE LIBRARY from `kernels`, and the AVX-512 flags are set on the SOURCE rather than on the target
# so that they apply only to the TU that needs them.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI);
#   * a TU compiled WITH AVX-512 may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests.
# speculation components that need no GPU.
add_library(spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(spec PUBLIC include)
if(BUILD_TESTING)
  add_executable(suffix_drafter_test tests/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE spec)
  basalt_cpu_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test tests/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE spec)
  basalt_cpu_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test tests/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE spec)
  basalt_cpu_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()
if(BUILD_TESTING)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test tests/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  basalt_cpu_test(NAME conv_cache_test COMMAND conv_cache_test)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test tests/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  basalt_cpu_test(NAME bf16_bits_test COMMAND bf16_bits_test)
endif()

add_library(kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/cpu_quota.cpp src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(kernels_cpu PUBLIC include)
# only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
set_source_files_properties(src/kernels/cpu/expert.cpp
                            PROPERTIES COMPILE_OPTIONS "${_isa_avx512_flags}")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
find_package(Threads REQUIRED)
target_link_libraries(kernels_cpu PUBLIC Threads::Threads platform)

# ---- ggml, extracted ----
#
# The parts of ggml (MIT, llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d) the engine uses, in
# src/kernels/cpu/ggml and src/prefill/mmq: the reference quantizers and the format table, the CPU row dot
# products, and the prompt path's MMQ kernels.  Nothing is fetched at build time.
add_library(ggml_quants STATIC src/kernels/cpu/ggml/ggml_quants.c)
target_include_directories(ggml_quants PUBLIC include PRIVATE src/kernels/cpu/ggml)
set_target_properties(ggml_quants PROPERTIES C_STANDARD 11 C_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
# The CPU dot products are built for AVX2 and for AVX-512, each with its own name for the table lookup;
# ggml_cpu_dispatch.cpp picks one at run time (cpu_avx512_ok / cpu_avx2_ok).
set(_ggml_dots_avx2_flags ${_isa_avx2_flags})
set(_ggml_dots_avx512_flags ${_isa_avx512_flags})
foreach(_isa avx2 avx512)
  add_library(ggml_dots_${_isa} OBJECT src/kernels/cpu/ggml/ggml_cpu_dots.c)
  target_include_directories(ggml_dots_${_isa} PRIVATE include src/kernels/cpu/ggml)
  target_compile_options(ggml_dots_${_isa} PRIVATE ${_ggml_dots_${_isa}_flags})
  target_compile_definitions(ggml_dots_${_isa} PRIVATE ggml_get_type_traits_cpu=ggml_get_type_traits_cpu_${_isa})
  set_target_properties(ggml_dots_${_isa} PROPERTIES C_STANDARD 11 C_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
endforeach()
# part of kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
# one-pass static linking has no cycle to resolve)
target_sources(kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                         src/kernels/cpu/nvfp4_avx512.cpp
                                         src/kernels/cpu/nvfp4_avx2.cpp
                                         src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/iq_avx2_vnni.cpp
                                         src/kernels/cpu/kq_avx2.cpp
                                         src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/ggml/ggml_cpu_dispatch.cpp
                                         $<TARGET_OBJECTS:ggml_dots_avx2> $<TARGET_OBJECTS:ggml_dots_avx512>)
# the i-quant rows also use BMI2 (pdep/pext for the sign masks; cpu_avx512_ok requires it). MSVC compiles the
# intrinsics without a switch.
if(MSVC)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx512_flags}")
else()
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx512_flags};-mbmi2")
endif()
set_source_files_properties(src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                            "${_isa_avx512_flags}")
set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
# the same rows with the AVX-VNNI instructions, called only behind cpu_avxvnni_ok()
set_source_files_properties(src/kernels/cpu/iq_avx2_vnni.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avxvnni_flags}")
set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
# the router lookahead's AVX1 dot (called behind cpu_avx1_ok)
set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx1_flags}")
# MSVC gives the scalar F16C conversions no working code: its headers declare them nowhere, and with
# __F16C__ defined the compiler claims the names as its own, so a definition compiled under that macro does not
# build as written. The file below supplies the pair from the vector intrinsics, which are the same instruction,
# and it must be compiled without __F16C__ for its definitions to survive; the callers carry prototypes and
# #pragma function so the calls reach it. The file is empty for every other compiler, and it is not added to the
# library at all there.
if(MSVC)
  target_sources(kernels_cpu PRIVATE src/kernels/cpu/ggml/f16c_scalar_msvc.c)
  set_source_files_properties(src/kernels/cpu/ggml/f16c_scalar_msvc.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()
target_link_libraries(kernels_cpu PUBLIC ggml_quants)
  # the router lookahead's dot on each rung this CPU has (AVX2, the older CPUs' AVX1) against a double reference
add_executable(router_dot_parity tests/kernels/cpu/router_dot_parity.cpp)
target_link_libraries(router_dot_parity PRIVATE kernels_cpu)
if(BUILD_TESTING)
  basalt_gpu_test(NAME router_dot_parity COMMAND router_dot_parity)
endif()
# MMQ (ggml-cuda) for the prompt path's experts: the kernel, one instance per format the packs use, and the
# activation quantizer; mmq_host.cu holds its few host symbols
set(_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 q5_1 q4_k q5_k q6_k nvfp4)
# the Q2_0 pack's prompt experts on the engine's own int8 kernels (moe_fused.cu)
add_library(mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
            src/prefill/mmq/mmq_host.cu src/prefill/mmq/quantize.cu)
# q8_0: the drafter's matrices
foreach(_t IN LISTS _mmq_types)
  target_sources(mmq PRIVATE src/prefill/mmq/instances/mmq_${_t}.cu)
endforeach()
target_include_directories(mmq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
# MSVC's COFF object holds 65536 sections; the instantiated MMQ formats exceed that in one TU without /bigobj.
if(MSVC)
  target_compile_options(mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()
target_link_libraries(mmq PUBLIC ggml_quants ${_gpu_runtime_target})
target_link_libraries(prefill PUBLIC mmq)
# the FP4 block-scaled tensor-core experts (need the a-variant of the architecture)
add_library(fp4 STATIC src/prefill/fp4/moe_fp4.cu)
set_target_properties(fp4 PROPERTIES CUDA_ARCHITECTURES 120a-real)
target_include_directories(fp4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fp4 PUBLIC mmq ${_gpu_runtime_target})
target_link_libraries(prefill PUBLIC fp4)
if(BUILD_TESTING)
  # the fused int8 experts against a double-precision reference and the MMQ path (exit 77 without int8 tensor cores)
  add_executable(prefill_fused_moe_test tests/cuda/prefill_fused_moe_test.cpp)
  target_link_libraries(prefill_fused_moe_test PRIVATE mmq ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_fused_moe_test COMMAND prefill_fused_moe_test)
  set_tests_properties(prefill_fused_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # the same for the native packs' formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL)
  add_executable(prefill_fused_iq_test tests/cuda/prefill_fused_iq_test.cpp)
  target_link_libraries(prefill_fused_iq_test PRIVATE mmq ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_fused_iq_test COMMAND prefill_fused_iq_test)
  set_tests_properties(prefill_fused_iq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # a small chunk's experts on the CPU pool against MMQ and a double-precision reference
  add_executable(prefill_cpu_share_test tests/cuda/prefill_cpu_share_test.cpp)
  target_link_libraries(prefill_cpu_share_test PRIVATE mmq kernels_cpu ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_cpu_share_test COMMAND prefill_cpu_share_test)
  set_tests_properties(prefill_cpu_share_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # the prompt path's routed-entry histogram against a host count
  add_executable(prefill_route_count_test tests/cuda/prefill_route_count_test.cpp)
  target_link_libraries(prefill_route_count_test PRIVATE prefill ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_route_count_test COMMAND prefill_route_count_test)
  set_tests_properties(prefill_route_count_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # the prompt path's buffer layout against its sizing (no device memory), once per expert layout
  add_executable(prefill_layout_test tests/cuda/prefill_layout_test.cpp)
  target_link_libraries(prefill_layout_test PRIVATE prefill ${_gpu_runtime_target})
  foreach(_mode nvfp4 mixed kquant q2)
    basalt_gpu_test(NAME prefill_layout_${_mode} COMMAND prefill_layout_test ${_mode})
    set_tests_properties(prefill_layout_${_mode} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  endforeach()
  # MMQ for the K-quant experts against the FP16 dequant + cuBLAS path and a double-precision reference (--bench: timings)
  add_executable(prefill_mmq_kq_test tests/cuda/prefill_mmq_kq_test.cpp)
  target_link_libraries(prefill_mmq_kq_test PRIVATE prefill mmq kernels ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_mmq_kq_test COMMAND prefill_mmq_kq_test)
  set_tests_properties(prefill_mmq_kq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()
if(BUILD_TESTING)
  # NVFP4 experts through MMQ (tile loader, per-16-scale dot, scale tail in SwiGLU) against CPU references, with a
  # microbench line
  add_executable(prefill_mmq_nvfp4_test tests/cuda/prefill_mmq_nvfp4_test.cpp)
  target_link_libraries(prefill_mmq_nvfp4_test PRIVATE mmq ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_mmq_nvfp4_test COMMAND prefill_mmq_nvfp4_test --no-bench)
  set_tests_properties(prefill_mmq_nvfp4_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
if(BUILD_TESTING)
  # NVFP4 block decode against the E2M1/UE4M3 spec (exhaustive) and ggml's row dequantizer; with an experts.bin,
  # a real NVFP4 expert's blocks and scale tail
  add_executable(nvfp4_format_test tests/core/nvfp4_format_test.cpp)
  target_link_libraries(nvfp4_format_test PRIVATE artifact ggml_quants)
  basalt_cpu_test(NAME nvfp4_format_test COMMAND nvfp4_format_test)
  set_tests_properties(nvfp4_format_test PROPERTIES TIMEOUT 60)
endif()
if(BUILD_TESTING)
  # the hc up product fused with the mix against cuBLAS + gr_mix_r (bitwise) and a double-precision reference
  add_executable(prefill_hc_up_mix_test tests/cuda/prefill_hc_up_mix_test.cpp)
  target_link_libraries(prefill_hc_up_mix_test PRIVATE prefill kernels ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_hc_up_mix_test COMMAND prefill_hc_up_mix_test --no-bench)
  set_tests_properties(prefill_hc_up_mix_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(prefill_fp4_test tests/cuda/prefill_fp4_test.cpp)
  target_link_libraries(prefill_fp4_test PRIVATE fp4 mmq ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_fp4_test COMMAND prefill_fp4_test --no-bench)
  set_tests_properties(prefill_fp4_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
add_executable(native_expert_parity tests/kernels/native_expert_parity.cpp)
target_include_directories(native_expert_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(native_expert_parity PRIVATE kernels kernels_cpu artifact ${_gpu_runtime_target})
add_executable(native_expert_bench tests/kernels/native_expert_bench.cpp)
target_include_directories(native_expert_bench PRIVATE ${_gpu_include_directories})
target_link_libraries(native_expert_bench PRIVATE kernels kernels_cpu artifact ${_gpu_runtime_target})
# DEC_MOE's kernels on real GGUF experts, bitwise vs the path they replace (--bench: timings)
add_executable(dec_moe_parity tests/kernels/dec_moe_parity.cpp)
target_include_directories(dec_moe_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dec_moe_parity PRIVATE kernels artifact ${_gpu_runtime_target})
basalt_gpu_test(NAME dec_moe_parity_nvfp4 COMMAND dec_moe_parity --nvfp4)   # synthetic experts: no GGUF needed
basalt_gpu_test(NAME dec_moe_parity_kq COMMAND dec_moe_parity --kq)   # synthetic K-quant experts at the real geometry (Q5_K and Q6_K gate/up, Q8_0 down)
if(BUILD_TESTING)
  # native expert pairs on synthetic ggml-quantized weights at 2560/640: GPU and CPU experts against the float
  # reference, GPU dequantizers against to_float.
  foreach(_pair q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q6_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0)
    string(REPLACE "/" "_" _name "${_pair}")
    basalt_gpu_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
    set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
  endforeach()
  basalt_cpu_test(NAME native_expert_parity_cpu_q6_K_q8_0 COMMAND native_expert_parity --cpu-synthetic q6_K/q8_0)
  basalt_gpu_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
  # the AVX-512 Q8_K activation quantizer bit for bit against ggml's reference
  basalt_cpu_test(NAME native_expert_parity_q8k_quant COMMAND native_expert_parity --q8k-quant)
  # the BF16 token embedding gathered and dequantized bit for bit
  basalt_gpu_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
  # the Q6_K token embedding dequantized against ggml's reference
  basalt_gpu_test(NAME native_expert_parity_q6k_embd COMMAND native_expert_parity --q6k-embd)
  # the Q8_0 token embedding with its hot rows in VRAM, bit for bit against the mapped-only gather and ggml's reference
  basalt_gpu_test(NAME native_expert_parity_q8_embd_hot COMMAND native_expert_parity --q8-embd-hot)
  # NVFP4 on the CPU tier: AVX2/AVX-512 dots vs plain C, the expert with its tail scales vs a double reference
  basalt_gpu_test(NAME native_expert_parity_nvfp4 COMMAND native_expert_parity --synthetic nvfp4/nvfp4 nvfp4/nvfp4)
  set_tests_properties(native_expert_parity_nvfp4 PROPERTIES TIMEOUT 120)
endif()
# real-weight checks: test programs, not part of a release
if(BUILD_TESTING)
  add_executable(ple_q5_parity tests/kernels/ple_q5_parity.cpp)
  target_link_libraries(ple_q5_parity PRIVATE kernels artifact ggml_quants ${_gpu_runtime_target})
  add_executable(q5_projection_parity tests/kernels/q5_projection_parity.cpp)
  target_include_directories(q5_projection_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(q5_projection_parity PRIVATE kernels artifact ggml_quants ${_gpu_runtime_target})
endif()

if(BUILD_TESTING)
  # Both programs test a Linux-only surface: the cgroup quota files and the Linux affinity backend's headers.
  # Neither exists on Windows, so neither is built or run there.
  if(NOT WIN32)
    add_executable(cpu_quota_test tests/kernels/cpu/cpu_quota_test.cpp)
    target_link_libraries(cpu_quota_test PRIVATE kernels_cpu)
    basalt_cpu_test(NAME cpu_quota_test COMMAND cpu_quota_test)
    add_executable(pool_affinity_test tests/platform/affinity_linux_test.cpp)
    target_include_directories(pool_affinity_test PRIVATE src/platform)
    target_link_libraries(pool_affinity_test PRIVATE kernels_cpu)
    basalt_cpu_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  endif()
  # one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the full model; run it by hand)
  add_executable(dequant_bf16_test tests/kernels/dequant_bf16_test.cpp)
  target_link_libraries(dequant_bf16_test PRIVATE kernels artifact ${_gpu_runtime_target})

  # the CPU expert kernels on each instruction set side by side (timing; the parity checks are native_expert_parity)
  add_executable(avx_bench tests/kernels/cpu/avx_bench.cpp)
  target_link_libraries(avx_bench PRIVATE kernels_cpu)

  add_executable(expert_multi_test tests/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE kernels_cpu)
  basalt_cpu_test(NAME expert_multi_test COMMAND expert_multi_test)
  # needs the AVX-512 VNNI and VBMI instruction sets, which not every host has
  set_property(TEST expert_multi_test APPEND PROPERTY LABELS avx512)

  add_executable(nvfp4_cpu_rows_test tests/kernels/cpu/nvfp4_cpu_rows_test.cpp)
  target_link_libraries(nvfp4_cpu_rows_test PRIVATE kernels_cpu)
  basalt_cpu_test(NAME nvfp4_cpu_rows_test COMMAND nvfp4_cpu_rows_test)
  set_tests_properties(nvfp4_cpu_rows_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "cpu;avx512")

  add_executable(expert_parity tests/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE kernels_cpu platform)
  # Both tests run on a synthetic expert fixture by default; --file experts.bin (the pack's experts section) checks real blobs.
  basalt_cpu_test(NAME expert_parity COMMAND expert_parity --selftest)

  add_executable(pool_test tests/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME pool_test COMMAND pool_test --selftest)
  # tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress tests/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE kernels_cpu)
  basalt_cpu_test(NAME pool_stress COMMAND pool_stress 20)
  # the multi-token native path against a serial reference, bitwise, at several worker counts
  add_executable(pool_native_test tests/kernels/cpu/pool_native_test.cpp)
  target_link_libraries(pool_native_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME pool_native_test COMMAND pool_native_test)
  # the AVX2 i-quant rows: the madd build against the AVX-VNNI build (bitwise) and a double reference
  add_executable(iq_avx2_test tests/kernels/cpu/iq_avx2_test.cpp)
  target_link_libraries(iq_avx2_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME iq_avx2_test COMMAND iq_avx2_test)
  # the AVX-512 i-quant rows: frozen output hashes, token-count independence and a double reference
  add_executable(iq_avx512_test tests/kernels/cpu/iq_avx512_test.cpp)
  target_link_libraries(iq_avx512_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME iq_avx512_test COMMAND iq_avx512_test)
  set_property(TEST iq_avx512_test APPEND PROPERTY LABELS avx512)

  # DirectFile has a Linux backend and a Windows backend, and this test covers those two; macOS is not a platform
  # the engine builds for.
  if((UNIX AND NOT APPLE) OR WIN32)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(direct_file_async_test PRIVATE BASALT_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(direct_file_async_test PRIVATE platform Threads::Threads)
    basalt_embed_manifest(direct_file_async_test)
    basalt_cpu_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
endif()
