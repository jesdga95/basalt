# DeepSeek-V4.1 attention, compute side: the query path, sparse MQA with the sink, the grouped output.
add_library(deepseek41_attn STATIC
  src/models/deepseek41/kernels/attn_mqa.cu
  src/models/deepseek41/kernels/q_path.cu
  src/models/deepseek41/kernels/attn_out.cu
  src/models/deepseek41/kernels/attn_out_grouped.cu
  src/models/deepseek41/kernels/prefill_attn.cu)
target_include_directories(deepseek41_attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek41_attn PUBLIC kernels ${_gpu_runtime_target})

if(BUILD_TESTING)
  add_executable(ds41_attn_mqa_parity src/models/deepseek41/kernels/attn_mqa_parity.cpp)
  target_include_directories(ds41_attn_mqa_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(ds41_attn_mqa_parity PRIVATE deepseek41_attn ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds41_attn_mqa_parity COMMAND ds41_attn_mqa_parity)
  add_executable(ds41_attn_out_parity src/models/deepseek41/kernels/attn_out_parity.cpp)
  target_include_directories(ds41_attn_out_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(ds41_attn_out_parity PRIVATE deepseek41_attn ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds41_attn_out_parity COMMAND ds41_attn_out_parity)
  add_executable(ds41_prefill_attn_parity src/models/deepseek41/kernels/prefill_attn_parity.cpp)
  target_include_directories(ds41_prefill_attn_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(ds41_prefill_attn_parity PRIVATE deepseek41_attn ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds41_prefill_attn_parity COMMAND ds41_prefill_attn_parity)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/models/deepseek41/kernels/attn_fixture_parity.cpp)
    # needs the model pack and the llama.cpp fixtures: run by hand, not registered in ctest
    add_executable(ds41_attn_fixture_parity src/models/deepseek41/kernels/attn_fixture_parity.cpp)
    target_include_directories(ds41_attn_fixture_parity PRIVATE ${_gpu_include_directories})
    target_link_libraries(ds41_attn_fixture_parity PRIVATE deepseek41_attn ${_gpu_runtime_target})
  endif()
endif()
