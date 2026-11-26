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
target_link_libraries(deepseek4_vision_plan PUBLIC vision_core vision_transformer warnings)

add_library(deepseek4_pack STATIC src/models/deepseek4/pack.cpp)
target_include_directories(deepseek4_pack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(deepseek4_pack PUBLIC core)

if(BUILD_TESTING)
  add_executable(ds4_meta_test src/models/deepseek4/meta_test.cpp)
  target_include_directories(ds4_meta_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ds4_meta_test PRIVATE deepseek4_pack warnings)
  basalt_cpu_test(NAME ds4_meta_test COMMAND ds4_meta_test)

  foreach(_name compress indexer_compress indexer_project indexer_select)
    add_executable(ds4_${_name}_parity src/models/deepseek4/${_name}_parity.cpp)
    target_include_directories(ds4_${_name}_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_gpu_include_directories})
    target_link_libraries(ds4_${_name}_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
    basalt_gpu_test(NAME ds4_${_name}_parity COMMAND ds4_${_name}_parity)
  endforeach()
  add_executable(ds4_mhc_parity src/models/deepseek4/mhc_parity.cpp)
  target_include_directories(ds4_mhc_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                     ${_gpu_include_directories})
  target_link_libraries(ds4_mhc_parity PRIVATE kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_mhc_parity COMMAND ds4_mhc_parity)

  add_executable(ds4_vision_plan_test src/models/deepseek4/vision/plan_test.cpp)
  target_link_libraries(ds4_vision_plan_test PRIVATE deepseek4_vision_plan warnings)
  basalt_cpu_test(NAME ds4_vision_plan_test COMMAND ds4_vision_plan_test)
  add_executable(ds4_pack_test src/models/deepseek4/pack_test.cpp)
  target_include_directories(ds4_pack_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ds4_pack_test PRIVATE deepseek4_pack vision_core warnings)
  if(BASALT_DEEPSEEK4_PACK)
    basalt_cpu_test(NAME ds4_pack_test COMMAND ds4_pack_test "${BASALT_DEEPSEEK4_PACK}")
  else()
    basalt_cpu_test(NAME ds4_pack_test COMMAND ds4_pack_test)
    set_tests_properties(ds4_pack_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  add_executable(ds4_vision_tower_test src/models/deepseek4/vision/tower_test.cpp)
  target_include_directories(ds4_vision_tower_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ds4_vision_tower_test PRIVATE deepseek4_pack deepseek4_vision_plan vision_core core warnings)
  if(BASALT_DEEPSEEK4_PACK)
    if(BASALT_DEEPSEEK4_VISION_REFERENCE)
      basalt_gpu_test(NAME ds4_vision_tower_test COMMAND ds4_vision_tower_test
        "${BASALT_DEEPSEEK4_PACK}" "${BASALT_DEEPSEEK4_VISION_REFERENCE}")
    else()
      basalt_gpu_test(NAME ds4_vision_tower_test COMMAND ds4_vision_tower_test "${BASALT_DEEPSEEK4_PACK}")
    endif()
    set_tests_properties(ds4_vision_tower_test PROPERTIES TIMEOUT 600)
  else()
    basalt_gpu_test(NAME ds4_vision_tower_test COMMAND ds4_vision_tower_test)
    set_tests_properties(ds4_vision_tower_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
