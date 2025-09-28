set_source_files_properties(kernels.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG})

set(EXE_NAME vector_add)

add_library(vector_add_lib kernels.cpp)

set_target_properties(
  vector_add_lib PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(vector_add_lib PUBLIC gcxx::gcxx fmt::fmt argparse::argparse)

add_executable(${EXE_NAME} main.cpp)
target_link_libraries(${EXE_NAME} PRIVATE vector_add_lib)
# set_target_properties(
#   ${EXE_NAME} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON
# )
set_target_properties(${EXE_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)