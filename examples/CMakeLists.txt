project(GCXX_Examples)
add_custom_target(examples)

set(CMAKE_CUDA_ARCHITECTURES 80 86 # 87 89 90
                             # 100 120
)

include(cmake/fmt.cmake)
include(cmake/argparse.cmake)

add_subdirectory(by_api_module)
add_subdirectory(vector_add)
add_subdirectory(vector_add_overlap)
add_subdirectory(multi_gpu_testspace)
add_subdirectory(Reduction)
