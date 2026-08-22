project(GCXX_Examples)
add_custom_target(examples)

include(cmake/gcxx_add_example.cmake)
include(cmake/fmt.cmake)
# include(cmake/argparse.cmake)  # uncomment when a sample uses argparse

add_subdirectory(0_Introduction)
# add_subdirectory(1_Utilities)
# add_subdirectory(2_Concepts_and_Techniques)
# add_subdirectory(3_CUDA_Features)
# add_subdirectory(4_CUDA_Libraries)
# add_subdirectory(5_Domain_Specific)
# add_subdirectory(6_Performance)
