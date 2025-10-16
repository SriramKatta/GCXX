set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG})

# helps in intellisense
set_source_files_properties(main.hpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG})

set(EXE_NAME vector_add_overlap)
set(LIB_NAME ${EXE_NAME}_lib)

add_library(${LIB_NAME} INTERFACE)

set_target_properties(
  ${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(
  ${LIB_NAME} INTERFACE gcxx::gcxx fmt::fmt argparse::argparse
)

add_executable(${EXE_NAME} main.cpp)
target_link_libraries(${EXE_NAME} PRIVATE ${LIB_NAME})

set_target_properties(${EXE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
