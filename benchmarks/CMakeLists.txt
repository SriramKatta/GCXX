include(cmake/google_bench.cmake)
include(cmake/gcxx_add_benchmark.cmake)

# Umbrella target: `cmake --build . --target benchmarks` builds every
# benchmark executable.
add_custom_target(benchmarks)

# All benchmark executables go to <runtime-output>/benchmarks, keeping them
# grouped under the single runtime output folder defined by the presets.
if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/benchmarks")
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/benchmarks)
endif()

add_subdirectory(gemm_raw_vs_wrapped)
add_subdirectory(kernel_params_direct_vs_builder)
