# src/models/ernie55/ernie55.cmake - ERNIE-5.5-Flash model library, CUDA kernels, and unit tests.
# Included from CMakeLists.txt under `if(BASALT_MODEL_ERNIE55)`.

add_library(ernie55_kernels STATIC
    src/models/ernie55/kernels/ops.cu
    src/models/ernie55/kernels/hca.cu
    src/models/ernie55/kernels/mla_dsa.cu
    src/models/ernie55/kernels/dspark.cu
    src/models/ernie55/kernels/moe_dec.cu
    src/models/ernie55/kernels/q8x.cu
)
target_include_directories(ernie55_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ernie55_kernels PUBLIC ${_gpu_runtime_target})

add_library(model_ernie55 STATIC
    src/models/ernie55/adapter.cpp
    src/models/ernie55/pack.cpp
    src/models/ernie55/dspark.cpp
    src/models/ernie55/model.cpp
    src/models/ernie55/session.cpp
    src/models/ernie55/decode_moe.cpp
    src/models/ernie55/tokenizer/pretok_ernie55.cpp
)
target_include_directories(model_ernie55
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src/program/tokenizer
)
target_link_libraries(model_ernie55 PUBLIC
    model
    tokenizer
    engine_session
    ernie55_kernels
    expert_tier
    tiers
    kernels
    kernels_cpu
    mmq
    core
    CUDA::cublas
    ${_gpu_runtime_target}
)

if(BUILD_TESTING)
    add_executable(ernie55_pretok_test
        src/models/ernie55/pretok_test.cpp
        src/models/ernie55/tokenizer/pretok_ernie55.cpp
    )
    target_include_directories(ernie55_pretok_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/program/tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(ernie55_pretok_test PRIVATE tokenizer)
    basalt_cpu_test(NAME ernie55_pretok_test COMMAND ernie55_pretok_test)

    add_executable(ernie55_pack_test
        src/models/ernie55/pack_test.cpp
        src/models/ernie55/pack.cpp
    )
    target_include_directories(ernie55_pack_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(ernie55_pack_test PRIVATE model_file platform)
    basalt_cpu_test(NAME ernie55_pack_test COMMAND ernie55_pack_test)

    add_executable(ernie55_kernels_parity
        src/models/ernie55/kernels/kernels_parity.cpp
    )
    target_include_directories(ernie55_kernels_parity PRIVATE ${_gpu_include_directories})
    target_link_libraries(ernie55_kernels_parity PRIVATE ernie55_kernels kernels CUDA::cublas ${_gpu_runtime_target})
    basalt_gpu_test(NAME ernie55_kernels_parity COMMAND ernie55_kernels_parity)

    add_executable(ernie55_dspark_parity
        src/models/ernie55/kernels/dspark_parity.cpp
    )
    target_include_directories(ernie55_dspark_parity PRIVATE ${_gpu_include_directories})
    target_link_libraries(ernie55_dspark_parity PRIVATE ernie55_kernels ${_gpu_runtime_target})
    basalt_gpu_test(NAME ernie55_dspark_parity COMMAND ernie55_dspark_parity)
endif()
