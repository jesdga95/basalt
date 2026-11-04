# DeepSeek-V4.1 vision encoder: the library and its parity tool. Included from the top-level CMakeLists.txt.
add_library(deepseek41_vision STATIC
  src/models/deepseek41/vision/plan.cpp
  src/models/deepseek41/vision/encoder.cu)
target_include_directories(deepseek41_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek41_vision PUBLIC vision_core CUDA::cudart CUDA::cublas)
if(BUILD_TESTING)
  add_executable(ds41_vision_parity src/models/deepseek41/vision/parity.cpp)
  target_link_libraries(ds41_vision_parity PRIVATE deepseek41_vision)
endif()
