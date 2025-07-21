cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

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
endif()

option(GPUCXX_BUILD_EXAMPLES "Enables building of examples" ON)
option(GPUCXX_CUDA_MODE "Enables building of examples" ON)
option(GPUCXX_HIP_MODE "Enables building of examples" OFF)

# basically XOR(GPUCXX_CUDA_MODE, GPUCXX_HIP_MODE); since only one mode can be active at a time
if((GPUCXX_CUDA_MODE AND GPUCXX_HIP_MODE) OR (NOT GPUCXX_CUDA_MODE AND NOT GPUCXX_HIP_MODE))
  message(FATAL_ERROR "Exactly one GPU back-end must be enabled: "
                      "set either GPUCXX_CUDA_MODE=ON or GPUCXX_HIP_MODE=ON (but not both)."
  )
endif()

if(GPUCXX_CUDA_MODE)
  message(STATUS "GPU back-end selected: CUDA")
  set(BUILD_GPU_LANGUAGE "CUDA")
endif()

if(GPUCXX_HIP_MODE)
  message(STATUS "GPU back-end selected: HIP")
  set(BUILD_GPU_LANGUAGE "HIP")
endif()

enable_language(${BUILD_GPU_LANGUAGE})

include(cmake/CPM.cmake)
include(cmake/fmt.cmake)

if(GPUCXX_TOPLEVEL_PROJECT)
  include(cmake/format.cmake)
  include(cmake/tools.cmake)
endif()

add_subdirectory(lib)

if(GPUCXX_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
