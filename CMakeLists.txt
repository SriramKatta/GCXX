cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(
  GCXX
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

# Determine whether GCXX is the top-level project or included into another
# project via add_subdirectory()
set(GCXX_TOPLEVEL_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  set(GCXX_TOPLEVEL_PROJECT ON)
  include(cmake/clangd_compileinfo.cmake)
endif()

option(GCXX_CUDA_MODE "to compile for CUDA programming language" OFF)
option(GCXX_HIP_MODE "to compile for HIP programming language" OFF)
option(GCXX_BUILD_EXAMPLES "Enables building of examples" OFF)
option(GCXX_BUILD_BENCHMARKS "Enables building of benchmarks" OFF)
option(GCXX_BUILD_TESTS "Enables building of tests" OFF)

# basically XOR(GCXX_CUDA_MODE, GCXX_HIP_MODE); since only one mode can be
# active at a time
if((GCXX_CUDA_MODE AND GCXX_HIP_MODE) OR (NOT GCXX_CUDA_MODE AND NOT
                                                                 GCXX_HIP_MODE)
)
  message(
    FATAL_ERROR
      "Exactly one GPU back-end must be enabled: "
      "set either GCXX_CUDA_MODE=${GCXX_CUDA_MODE} or GCXX_HIP_MODE=${GCXX_HIP_MODE} (but not both)."
  )
endif()

if(GCXX_CUDA_MODE)
  set(GCXX_GPU_LANG
      "CUDA"
      CACHE STRING "GPU programming language to use"
  )
endif()

if(GCXX_HIP_MODE)
  set(GCXX_GPU_LANG
      "HIP"
      CACHE STRING "GPU programming language to use"
  )
  set(ROCM_ROOT
      "/opt/rocm"
      CACHE PATH "Root directory of the ROCm installation"
  )
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
endif()

message(STATUS "GPU back-end selected: ${GCXX_GPU_LANG}")

enable_language(${GCXX_GPU_LANG})
set(CMAKE_${GCXX_GPU_LANG}_STANDARD 17)
set(CMAKE_${GCXX_GPU_LANG}_EXTENSIONS OFF)
set(CMAKE_${GCXX_GPU_LANG}_STANDARD_REQUIRED ON)

if(GCXX_CUDA_MODE)
  find_package(CUDAToolkit REQUIRED)
elseif(GCXX_HIP_MODE)
  find_package(hip REQUIRED)
endif()

include(cmake/CPM.cmake)

if(GCXX_TOPLEVEL_PROJECT)
  include(cmake/format.cmake)
  include(cmake/tools.cmake)
endif()

add_subdirectory(lib)

if(GCXX_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GCXX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(GCXX_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(testing)
endif()
