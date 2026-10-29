# The `warnings` and `platform` targets in one file, read by the root project and by the standalone
# `tools/vision` configuration alike, so the platform source list and the warning flags are written once.
# Paths hang off this file's own directory, so either project can include it wherever it is configured.

get_filename_component(BASALT_CMAKE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Warnings live on an INTERFACE target the project's own C++ code links, NOT in add_compile_options: applied
# globally they would also hit the extracted ggml C and CUDA sources, whose warnings are not this project's to fix.
# The flags are guarded by COMPILE_LANGUAGE:CXX so nvcc never receives them as inputs, and by the compiler id:
# MSVC reads -Wall/-Wextra as /Wall and the invalid /Wextra, so it takes its own spellings. The sources are UTF-8,
# which MSVC only assumes when told (the Windows SDK's headers are not), hence /utf-8 for both languages.
add_library(warnings INTERFACE)
target_compile_options(warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>)
if(WERROR)
  target_compile_options(warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# The file, memory and process calls the engine makes, so the Win32 and POSIX spellings live in one place.
# Host-only: no CUDA runtime, which is why the header-only `artifact` target, the vision library, the bench and the
# standalone test programs can all link it.
add_library(platform STATIC
  ${BASALT_CMAKE_ROOT}/src/platform/file.cpp ${BASALT_CMAKE_ROOT}/src/platform/memory.cpp
  ${BASALT_CMAKE_ROOT}/src/platform/direct_file.cpp ${BASALT_CMAKE_ROOT}/src/platform/process.cpp
  ${BASALT_CMAKE_ROOT}/src/platform/mapped_file.cpp ${BASALT_CMAKE_ROOT}/src/platform/cpu.cpp
  ${BASALT_CMAKE_ROOT}/src/platform/system.cpp ${BASALT_CMAKE_ROOT}/src/platform/dynlib.cpp)
target_include_directories(platform PUBLIC ${BASALT_CMAKE_ROOT}/include)
# dlopen (DynLib) lives in libdl before glibc 2.34
target_link_libraries(platform PUBLIC warnings ${CMAKE_DL_LIBS})
if(WIN32)
  target_link_libraries(platform PUBLIC advapi32)   # the processor name in the registry
endif()
