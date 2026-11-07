# DeepSeek-V4.1 kernels: the library and its parity tests. Included from the top-level CMakeLists.txt.
add_library(deepseek41_kernels STATIC
  src/models/deepseek41/kernels/router.cu
  src/models/deepseek41/kernels/mhc.cu
  src/models/deepseek41/kernels/norm.cu
  src/models/deepseek41/kernels/engram.cu
  src/models/deepseek41/kernels/attn_index.cu
  src/models/deepseek41/kernels/engram_hash.cpp)
target_include_directories(deepseek41_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek41_kernels PUBLIC ${_gpu_runtime_target})

foreach(_t router_parity norm_parity hc_parity engram_parity attn_index_parity prefill_index_parity)
  if(BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/models/deepseek41/kernels/${_t}.cpp)
    add_executable(ds41_${_t} src/models/deepseek41/kernels/${_t}.cpp)
    target_include_directories(ds41_${_t} PRIVATE ${_gpu_include_directories})
    target_link_libraries(ds41_${_t} PRIVATE deepseek41_kernels kernels artifact ${_gpu_runtime_target})
    basalt_gpu_test(NAME ds41_${_t} COMMAND ds41_${_t})
  endif()
endforeach()
if(BUILD_TESTING)
  add_executable(ds41_cache_rows_parity src/models/deepseek41/kernels/cache_rows_parity.cu)
  target_include_directories(ds41_cache_rows_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds41_cache_rows_parity PRIVATE ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds41_cache_rows_parity COMMAND ds41_cache_rows_parity)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/src/models/deepseek41/kernels/attn.cmake)
