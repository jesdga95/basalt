# cmake/arm64.cmake - what an Arm64 build changes in the targets the root CMakeLists.txt defines for x86-64.
#
# The root file lists the AVX2 / AVX-512 kernels where it always has; on Arm64 the functions below, called right after
# each target is complete, take those sources out and put the NEON ones in. Keeping the edits here leaves the root file
# the x86 build's text plus one call per target. Included by cmake/target_arch.cmake users after project().

# -ffp-contract=off on GCC and Clang: Arm64 always has FMA, so the default (fast) would fuse a*b+c wherever the
# optimizer likes, and a kernel's scalar reference would no longer round like the x86 build's (which is compiled
# without FMA outside its AVX files) nor like MSVC's Arm64 output. The kernels ask for the FMAs they want (vfmaq).
function(basalt_arm64_global_flags)
  if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
  endif()
endfunction()

# The NEON kernels use the Armv8.2 dot product (sdot), which GCC and Clang gate behind -march; MSVC has no switch.
if(MSVC)
  set(BASALT_NEON_FLAGS "")
else()
  set(BASALT_NEON_FLAGS "-march=armv8.2-a+dotprod")
endif()

# Removes from `target` every source whose path matches `regex` (generator expressions included).
function(basalt_drop_sources target regex)
  get_target_property(_src ${target} SOURCES)
  list(FILTER _src EXCLUDE REGEX "${regex}")
  set_property(TARGET ${target} PROPERTY SOURCES ${_src})
endfunction()

# kernels_cpu: the x86 kernels and the two x86 builds of the ggml dot products out; the NEON kernels, the NEON build of
# the dot products, and the x86 kernels' names (probes that say no) in.
function(basalt_arm64_kernels_cpu)
  basalt_drop_sources(kernels_cpu "(_avx[0-9]*(_vnni)?\\.cpp$|/expert\\.cpp$|f16c_scalar_msvc\\.c$|TARGET_OBJECTS:ggml_dots_avx)")
  set_target_properties(ggml_dots_avx2 ggml_dots_avx512 PROPERTIES EXCLUDE_FROM_ALL ON)
  add_library(ggml_dots_neon OBJECT src/kernels/cpu/ggml/ggml_cpu_dots_neon.c)
  target_include_directories(ggml_dots_neon PRIVATE include src/kernels/cpu/ggml)
  target_compile_options(ggml_dots_neon PRIVATE ${BASALT_NEON_FLAGS})
  target_compile_definitions(ggml_dots_neon PRIVATE ggml_get_type_traits_cpu=ggml_get_type_traits_cpu_neon
                             ggml_vec_dot_mxfp8_q8_0_scalar=ggml_vec_dot_mxfp8_q8_0_scalar_neon
                             ggml_mxfp8_q8_0_prep=ggml_mxfp8_q8_0_prep_neon ggml_mxfp8_q8_0_rows=ggml_mxfp8_q8_0_rows_neon)
  set_target_properties(ggml_dots_neon PROPERTIES C_STANDARD 11 C_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
  target_sources(kernels_cpu PRIVATE src/kernels/cpu/q2_neon.cpp src/kernels/cpu/x86_absent.cpp
                                     $<TARGET_OBJECTS:ggml_dots_neon>)
  set_source_files_properties(src/kernels/cpu/q2_neon.cpp PROPERTIES COMPILE_OPTIONS "${BASALT_NEON_FLAGS}")
endfunction()

# Tests and benches of the x86 kernels alone: not built, and their ctest entries disabled.
function(basalt_arm64_disable)
  foreach(_t IN LISTS ARGN)
    if(TARGET ${_t})
      set_target_properties(${_t} PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES DISABLED ON)
    endif()
  endforeach()
endfunction()
