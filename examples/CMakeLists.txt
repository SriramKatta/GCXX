project(GCXX_Examples)
add_custom_target(examples)

include(cmake/fmt.cmake)
include(cmake/argparse.cmake)
include(cmake/gcxx_add_example.cmake)

add_subdirectory(by_api_module)
add_subdirectory(vector_add)
add_subdirectory(vector_add_overlap)
# add_subdirectory(multi_gpu_testspace)
add_subdirectory(Reduction)
add_subdirectory(testspace)
