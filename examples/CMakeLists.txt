project(GCXX_Examples)
add_custom_target(examples)

# All example executables go to <runtime-output>/examples, keeping them
# grouped under the single runtime output folder defined by the presets.
if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/examples")
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/examples)
endif()

include(cmake/fmt.cmake)
include(cmake/argparse.cmake)
include(cmake/gcxx_add_example.cmake)

add_subdirectory(by_api_module)
add_subdirectory(vector_add)
add_subdirectory(vector_add_overlap)
# add_subdirectory(multi_gpu_testspace)
add_subdirectory(Reduction)
add_subdirectory(testspace)
