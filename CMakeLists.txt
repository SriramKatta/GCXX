cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(
  GPUCXX
  VERSION 0.0
  LANGUAGES CXX
)

# ---- Include guards ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (e.g. build) and run CMake from there."
  )
endif()

# Determine whether GPUCXX is the top-level project or included into another
# project via add_subdirectory()
set(GPUCXX_TOPLEVEL_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  set(GPUCXX_TOPLEVEL_PROJECT ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(GPUCXX_CUDA_MODE "to compile for CUDA programming language" OFF)
option(GPUCXX_HIP_MODE "to compile for HIP programming language" OFF)
option(GPUCXX_BUILD_EXAMPLES "Enables building of examples" OFF)
option(GPUCXX_BUILD_BENCHMARKS "Enables building of benchmarks" OFF)
option(GPUCXX_BUILD_TESTS "Enables building of tests" OFF)

# basically XOR(GPUCXX_CUDA_MODE, GPUCXX_HIP_MODE); since only one mode can be
# active at a time
if((GPUCXX_CUDA_MODE AND GPUCXX_HIP_MODE) OR (NOT GPUCXX_CUDA_MODE
                                              AND NOT GPUCXX_HIP_MODE)
)
  message(
    FATAL_ERROR
      "Exactly one GPU back-end must be enabled: "
      "set either GPUCXX_CUDA_MODE=ON or GPUCXX_HIP_MODE=ON (but not both)."
  )
endif()

if(GPUCXX_CUDA_MODE)
  set(GPUCXX_GPU_LANG
      "CUDA"
      CACHE STRING "GPU programming language to use"
  )
endif()

if(GPUCXX_HIP_MODE)
  set(GPUCXX_GPU_LANG
      "HIP"
      CACHE STRING "GPU programming language to use"
  )
  set(ROCM_ROOT
      "/opt/rocm"
      CACHE PATH "Root directory of the ROCm installation"
  )
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
endif()

message(STATUS "GPU back-end selected: ${GPUCXX_GPU_LANG}")

enable_language(${GPUCXX_GPU_LANG})
set(CMAKE_${GPUCXX_GPU_LANG}_STANDARD 17)
set(CMAKE_${GPUCXX_GPU_LANG}_EXTENSIONS OFF)
set(CMAKE_${GPUCXX_GPU_LANG}_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)

if(GPUCXX_TOPLEVEL_PROJECT)
  include(cmake/format.cmake)
  include(cmake/tools.cmake)
endif()

add_subdirectory(lib)

if(GPUCXX_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GPUCXX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(GPUCXX_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(testing)
endif()
