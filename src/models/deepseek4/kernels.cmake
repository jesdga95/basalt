# DeepSeek-V4 kernels and their parity programs. Included only when the deepseek4 model is selected.
add_library(deepseek4_kernels STATIC
  src/models/deepseek4/attention.cpp
  src/models/deepseek4/rope.cpp
  src/models/deepseek4/head.cpp
  src/models/deepseek4/layer.cpp
  src/models/deepseek4/shared_expert.cu
  src/models/deepseek4/routed_expert.cu
  src/models/deepseek4/attention.cu
  src/models/deepseek4/compress.cu
  src/models/deepseek4/indexer_compress.cu
  src/models/deepseek4/indexer_project.cu
  src/models/deepseek4/indexer_select.cu)
target_include_directories(deepseek4_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek4_kernels PUBLIC kernels ${_gpu_runtime_target})

add_library(deepseek4_vision_plan STATIC src/models/deepseek4/vision/plan.cpp)
target_include_directories(deepseek4_vision_plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deepseek4_vision_plan PUBLIC vision_core vision_transformer warnings)

add_library(deepseek4_pack STATIC src/models/deepseek4/pack.cpp src/models/deepseek4/expert_tier_spec.cpp)
target_include_directories(deepseek4_pack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(deepseek4_pack PUBLIC core)

add_library(deepseek4_model_weights STATIC src/models/deepseek4/model_weights.cpp)
target_include_directories(deepseek4_model_weights PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_gpu_include_directories})
target_link_libraries(deepseek4_model_weights PUBLIC deepseek4_pack kernels core ${_gpu_runtime_target})

if(BUILD_TESTING)
  add_executable(ds4_rope_test src/models/deepseek4/rope_test.cpp src/models/deepseek4/rope.cpp)
  target_include_directories(ds4_rope_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds4_rope_test PRIVATE warnings)
  basalt_cpu_test(NAME ds4_rope_test COMMAND ds4_rope_test)

  add_executable(ds4_context_policy_test src/models/deepseek4/context_policy_test.cpp)
  target_include_directories(ds4_context_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds4_context_policy_test PRIVATE warnings)
  basalt_cpu_test(NAME ds4_context_policy_test COMMAND ds4_context_policy_test)

  add_executable(ds4_router_test src/models/deepseek4/router_test.cpp)
  target_include_directories(ds4_router_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds4_router_test PRIVATE warnings)
  basalt_cpu_test(NAME ds4_router_test COMMAND ds4_router_test)

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

  add_executable(ds4_attention_kv_parity src/models/deepseek4/attention_kv_parity.cpp)
  target_include_directories(ds4_attention_kv_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                              ${_gpu_include_directories})
  target_link_libraries(ds4_attention_kv_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_attention_kv_parity COMMAND ds4_attention_kv_parity)
  basalt_gpu_test(NAME ds4_attention_kv_bench COMMAND ds4_attention_kv_parity --bench)

  add_executable(ds4_attention_project_parity src/models/deepseek4/attention_project_parity.cpp)
  target_include_directories(ds4_attention_project_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                                   ${_gpu_include_directories})
  target_link_libraries(ds4_attention_project_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_attention_project_parity COMMAND ds4_attention_project_parity)
  basalt_gpu_test(NAME ds4_attention_project_bench COMMAND ds4_attention_project_parity --bench)

  add_executable(ds4_attention_step_parity src/models/deepseek4/attention_step_parity.cpp)
  target_include_directories(ds4_attention_step_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                               ${_gpu_include_directories})
  target_link_libraries(ds4_attention_step_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_attention_step_parity COMMAND ds4_attention_step_parity)
  basalt_gpu_test(NAME ds4_attention_step_bench COMMAND ds4_attention_step_parity --bench)
  set_tests_properties(ds4_attention_step_parity PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(ds4_attention_step_bench PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_logits_parity src/models/deepseek4/logits_parity.cpp)
  target_include_directories(ds4_logits_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                        ${_gpu_include_directories})
  target_link_libraries(ds4_logits_parity PRIVATE deepseek4_kernels deepseek4_pack core ${_gpu_runtime_target})
  if(BASALT_DEEPSEEK4_PACK)
    basalt_gpu_test(NAME ds4_logits_parity COMMAND ds4_logits_parity "${BASALT_DEEPSEEK4_PACK}")
    basalt_gpu_test(NAME ds4_logits_bench COMMAND ds4_logits_parity "${BASALT_DEEPSEEK4_PACK}" --bench)
  else()
    basalt_gpu_test(NAME ds4_logits_parity COMMAND ds4_logits_parity)
    basalt_gpu_test(NAME ds4_logits_bench COMMAND ds4_logits_parity --bench)
  endif()
  set_tests_properties(ds4_logits_parity ds4_logits_bench PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_ffn_route_parity src/models/deepseek4/ffn_route_parity.cpp)
  target_include_directories(ds4_ffn_route_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                            ${_gpu_include_directories})
  target_link_libraries(ds4_ffn_route_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_ffn_route_parity COMMAND ds4_ffn_route_parity)
  basalt_gpu_test(NAME ds4_ffn_layer_bench COMMAND ds4_ffn_route_parity --bench)
  set_tests_properties(ds4_ffn_route_parity PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_shared_expert_parity src/models/deepseek4/shared_expert_parity.cpp)
  target_include_directories(ds4_shared_expert_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                              ${_gpu_include_directories})
  target_link_libraries(ds4_shared_expert_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_shared_expert_parity COMMAND ds4_shared_expert_parity)
  basalt_gpu_test(NAME ds4_shared_expert_bench COMMAND ds4_shared_expert_parity --bench)
  set_tests_properties(ds4_shared_expert_parity PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_routed_expert_parity src/models/deepseek4/routed_expert_parity.cpp)
  target_include_directories(ds4_routed_expert_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                              ${_gpu_include_directories})
  target_link_libraries(ds4_routed_expert_parity PRIVATE deepseek4_kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_routed_expert_parity COMMAND ds4_routed_expert_parity)
  basalt_gpu_test(NAME ds4_routed_expert_bench COMMAND ds4_routed_expert_parity --bench)
  set_tests_properties(ds4_routed_expert_parity PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_vision_plan_test src/models/deepseek4/vision/plan_test.cpp)
  target_link_libraries(ds4_vision_plan_test PRIVATE deepseek4_vision_plan warnings)
  basalt_cpu_test(NAME ds4_vision_plan_test COMMAND ds4_vision_plan_test)
  # the encoder on a card short of room: refusals leave no CUDA error, and the retry encodes as a fresh encoder (GPU)
  # the drafter's expert kernels against a double reference, the staging commit's slots, the kept-expert order (GPU)
  # the packed cache rows' codec against the CPU reference packing, decoding and rounding (GPU)
  add_executable(ds4_cache_rows_parity src/models/deepseek4/cache_rows_parity.cu)
  target_include_directories(ds4_cache_rows_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${_gpu_include_directories})
  target_link_libraries(ds4_cache_rows_parity PRIVATE core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_cache_rows_parity COMMAND ds4_cache_rows_parity)
  set_tests_properties(ds4_cache_rows_parity PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_dspark_parity src/models/deepseek4/dspark_parity.cpp)
  target_include_directories(ds4_dspark_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${_gpu_include_directories})
  target_link_libraries(ds4_dspark_parity PRIVATE model_deepseek4 ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds4_dspark_parity COMMAND ds4_dspark_parity)
  set_tests_properties(ds4_dspark_parity PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ds4_vision_reserve_test src/models/deepseek4/vision/reserve_test.cpp)
  target_link_libraries(ds4_vision_reserve_test PRIVATE deepseek4_vision_plan warnings)
  basalt_gpu_test(NAME ds4_vision_reserve_test COMMAND ds4_vision_reserve_test)
  set_tests_properties(ds4_vision_reserve_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(ds4_pack_test src/models/deepseek4/pack_test.cpp)
  target_include_directories(ds4_pack_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ds4_pack_test PRIVATE deepseek4_pack vision_core warnings)
  if(BASALT_DEEPSEEK4_PACK)
    basalt_cpu_test(NAME ds4_pack_test COMMAND ds4_pack_test "${BASALT_DEEPSEEK4_PACK}")
  else()
    basalt_cpu_test(NAME ds4_pack_test COMMAND ds4_pack_test)
    set_tests_properties(ds4_pack_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  add_executable(ds4_model_weights_test src/models/deepseek4/model_weights_test.cpp)
  target_include_directories(ds4_model_weights_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${_gpu_include_directories})
  target_link_libraries(ds4_model_weights_test PRIVATE deepseek4_model_weights ${_gpu_runtime_target})
  if(BASALT_DEEPSEEK4_PACK)
    basalt_gpu_test(NAME ds4_model_weights_test COMMAND ds4_model_weights_test "${BASALT_DEEPSEEK4_PACK}")
  else()
    basalt_gpu_test(NAME ds4_model_weights_test COMMAND ds4_model_weights_test)
  endif()
  set_tests_properties(ds4_model_weights_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

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
