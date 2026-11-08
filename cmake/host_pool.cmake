# Shared by the engine and the standalone vision executable.
if(NOT TARGET host_memory)
  get_filename_component(_host_pool_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
  add_library(host_memory STATIC ${_host_pool_root}/src/core/host_pool.cpp)
  target_include_directories(host_memory PUBLIC ${_host_pool_root}/include)
  target_link_libraries(host_memory PUBLIC platform warnings CUDA::cudart)
endif()
