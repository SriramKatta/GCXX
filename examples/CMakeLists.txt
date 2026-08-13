project(GCXX_Examples)
add_custom_target(examples)

<<<<<<< HEAD
# All example executables go to <runtime-output>/examples, keeping them
# grouped under the single runtime output folder defined by the presets.
if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/examples")
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/examples)
endif()

include(cmake/fmt.cmake)
include(cmake/argparse.cmake)
=======
>>>>>>> f6989c9 (Amending to new examples)
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
