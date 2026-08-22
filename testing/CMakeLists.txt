include(cmake/GTest.cmake)

include(GoogleTest)

# Umbrella target: `cmake --build . --target tests` builds every test
# executable.
add_custom_target(tests)

include(cmake/gcxx_add_test.cmake)

# All test executables go to <runtime-output>/tests, keeping them grouped
# under the single runtime output folder defined by the presets.
if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/tests")
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
endif()

include_directories(common)
add_subdirectory(event_testing)
add_subdirectory(span_testing)
add_subdirectory(vector_types_testing)
add_subdirectory(memory_testing)
add_subdirectory(mempool_testing)
add_subdirectory(device_testing)
add_subdirectory(macros_testing)
add_subdirectory(blas_testing)
add_subdirectory(sparse_testing)
