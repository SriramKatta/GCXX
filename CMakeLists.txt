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

# Determine whether GPUCXX is the top-level project or included into another project via
# add_subdirectory()
set(GPUCXX_TOPLEVEL_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  set(GPUCXX_TOPLEVEL_PROJECT ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(GPUCXX_CUDA_MODE "Enables building of examples" ON)
option(GPUCXX_HIP_MODE "Enables building of examples" OFF)
option(GPUCXX_ENABLE_EXAMPLES "Enables building of examples" ON)
option(GPUCXX_ENABLE_BENCHMARKS "Enables building of benchmarks" OFF)
option(GPUCXX_ENABLE_TESTS "Enables building of tests" OFF)

# basically XOR(GPUCXX_CUDA_MODE, GPUCXX_HIP_MODE); since only one mode can be active at a time
if((GPUCXX_CUDA_MODE AND GPUCXX_HIP_MODE) OR (NOT GPUCXX_CUDA_MODE AND NOT GPUCXX_HIP_MODE))
  message(FATAL_ERROR "Exactly one GPU back-end must be enabled: "
                      "set either GPUCXX_CUDA_MODE=ON or GPUCXX_HIP_MODE=ON (but not both)."
  )
endif()

if(GPUCXX_CUDA_MODE)
  set(BUILD_GPU_LANGUAGE "CUDA")
endif()

if(GPUCXX_HIP_MODE)
  set(BUILD_GPU_LANGUAGE "HIP")
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation" )
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
endif()

message(STATUS "GPU back-end selected: ${BUILD_GPU_LANGUAGE}")

enable_language(${BUILD_GPU_LANGUAGE})
set(CMAKE_${BUILD_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${BUILD_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${BUILD_GPU_LANGUAGE}_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)
include(cmake/fmt.cmake)

if(GPUCXX_TOPLEVEL_PROJECT)
  include(cmake/format.cmake)
  include(cmake/tools.cmake)
endif()

add_subdirectory(lib)


if(GPUCXX_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GPUCXX_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(GPUCXX_ENABLE_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(testing)
endif()
