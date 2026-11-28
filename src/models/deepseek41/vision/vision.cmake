# DeepSeek-V4.1 vision encoder: the library and its parity tool. Included from the top-level CMakeLists.txt.
add_library(deepseek41_vision STATIC
  src/models/deepseek41/vision/plan.cpp
  src/models/deepseek41/vision/encoder.cu)
target_include_directories(deepseek41_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek41_vision PUBLIC vision_core CUDA::cudart CUDA::cublas)
if(BUILD_TESTING)
  add_executable(ds41_vision_plan_test
    src/models/deepseek41/vision/plan_test.cpp
    src/models/deepseek41/vision/plan.cpp)
  target_include_directories(ds41_vision_plan_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds41_vision_plan_test PRIVATE vision_core warnings)
  basalt_cpu_test(NAME ds41_vision_plan_test COMMAND ds41_vision_plan_test)
  add_executable(ds41_vision_parity src/models/deepseek41/vision/parity.cpp)
  target_link_libraries(ds41_vision_parity PRIVATE deepseek41_vision)
  # the encoder on a card short of room: refusals leave no CUDA error, and the retry encodes as a fresh encoder (GPU)
  add_executable(ds41_vision_reserve_test src/models/deepseek41/vision/reserve_test.cpp)
  target_link_libraries(ds41_vision_reserve_test PRIVATE deepseek41_vision warnings)
  basalt_gpu_test(NAME ds41_vision_reserve_test COMMAND ds41_vision_reserve_test)
  set_tests_properties(ds41_vision_reserve_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
