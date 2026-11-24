# The models a build contains. Adding a model is one word in BASALT_BUILTIN_MODELS plus its own library under
# src/models/<arch> (named model_<arch>, in the namespace basalt::models::<arch>, with make_adapter()), gated by
# `if(BASALT_MODEL_<ARCH>)`. A model left out is not compiled, not linked and not registered.

# Every model that ships: the architecture ids, as `general.architecture` names them.
set(BASALT_BUILTIN_MODELS qwen4exp deepseek41 qwen35 hy4flash inkling2)

# "all" (the default) follows BASALT_BUILTIN_MODELS, so a build folder picks up a model added later. The cache keeps the
# word, not the list: a list written as the default would stay in the folder and leave every newer model out.
set(BASALT_MODELS "all" CACHE STRING
    "Models to build: all, or a semicolon list from: ${BASALT_BUILTIN_MODELS}")
# A folder configured before "all" holds the list that was then the default (0.4.x): it meant every model.
if(BASALT_MODELS STREQUAL "qwen4exp;deepseek41;qwen35")
  message(STATUS "basalt: BASALT_MODELS held 0.4's default list (every model then); it is now 'all'")
  set(BASALT_MODELS "all" CACHE STRING "Models to build: all, or a semicolon list from: ${BASALT_BUILTIN_MODELS}" FORCE)
endif()
# The resolved list, as a normal variable over the cache entry for the rest of the configure.
if(BASALT_MODELS STREQUAL "all")
  set(BASALT_MODELS ${BASALT_BUILTIN_MODELS})
endif()

list(REMOVE_DUPLICATES BASALT_MODELS)
string(REPLACE ";" ", " _basalt_valid "${BASALT_BUILTIN_MODELS}")
if(NOT BASALT_MODELS)
  message(FATAL_ERROR "BASALT_MODELS is empty: say all or name at least one of ${_basalt_valid}")
endif()
foreach(_m IN LISTS BASALT_MODELS)
  if(NOT _m IN_LIST BASALT_BUILTIN_MODELS)
    message(FATAL_ERROR "BASALT_MODELS names '${_m}', which is not a Basalt model. Valid: all, ${_basalt_valid}")
  endif()
endforeach()

foreach(_m IN LISTS BASALT_BUILTIN_MODELS)
  string(TOUPPER ${_m} _up)
  if(_m IN_LIST BASALT_MODELS)
    set(BASALT_MODEL_${_up} ON)
  else()
    set(BASALT_MODEL_${_up} OFF)
  endif()
endforeach()
message(STATUS "basalt: models ${BASALT_MODELS}")

# The registration list the composition root (src/models/builtin.cpp) expands, one entry per built model.
set(_basalt_registrations "")
foreach(_m IN LISTS BASALT_MODELS)
  string(APPEND _basalt_registrations "BASALT_BUILT_MODEL(${_m})\n")
endforeach()
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generated/builtin_models.inc CONTENT "${_basalt_registrations}" @ONLY)

# The registry names the models in its error for a pack whose model this build left out.
string(REPLACE ";" "," _basalt_known "${BASALT_BUILTIN_MODELS}")
string(REPLACE ";" "," _basalt_built "${BASALT_MODELS}")
