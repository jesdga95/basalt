# DeepSeek-V4 kernels and their parity programs. Included only when the deepseek4 model is selected.
add_library(deepseek4_kernels STATIC
  src/models/deepseek4/compress.cu
  src/models/deepseek4/indexer_compress.cu
  src/models/deepseek4/indexer_project.cu
  src/models/deepseek4/indexer_select.cu)
target_include_directories(deepseek4_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek4_kernels PUBLIC kernels ${_gpu_runtime_target})

add_library(deepseek4_vision_plan STATIC src/models/deepseek4/vision/plan.cpp)
target_include_directories(deepseek4_vision_plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek4_vision_plan PUBLIC vision_core warnings)

if(BUILD_TESTING)
  foreach(_name compress indexer_compress indexer_project indexer_select)
    add_executable(ds4_${_name}_parity src/models/deepseek4/${_name}_parity.cpp)
    target_include_directories(ds4_${_name}_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_gpu_include_directories})
    target_link_libraries(ds4_${_name}_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
    basalt_gpu_test(NAME ds4_${_name}_parity COMMAND ds4_${_name}_parity)
  endforeach()
  add_executable(ds4_vision_plan_test src/models/deepseek4/vision/plan_test.cpp)
  target_link_libraries(ds4_vision_plan_test PRIVATE deepseek4_vision_plan warnings)
  basalt_cpu_test(NAME ds4_vision_plan_test COMMAND ds4_vision_plan_test)
endif()
