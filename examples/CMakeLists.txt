project(GCXX_Examples)
add_custom_target(examples)

include(cmake/fmt.cmake)
include(cmake/argparse.cmake)
include(cmake/gcxx_add_example.cmake)

# When building in CPU-only mode, only add the mdspan example to avoid
# pulling in GPU/runtime-dependent examples and toolchain requirements.
if(GCXX_CPU_MODE)
  add_subdirectory(by_api_module/mdspan_htu)
else()
  add_subdirectory(by_api_module)
  add_subdirectory(vector_add)
  add_subdirectory(vector_add_overlap)
  # add_subdirectory(multi_gpu_testspace)
  add_subdirectory(Reduction)
  add_subdirectory(testspace)
endif()
