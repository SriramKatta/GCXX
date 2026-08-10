set(LIB_INTERNAL_NAME gpu_cxx)

include(cmake/mdspan.cmake)

add_library(${LIB_INTERNAL_NAME} INTERFACE)
add_library(gcxx::gcxx ALIAS ${LIB_INTERNAL_NAME})
target_include_directories(
  ${LIB_INTERNAL_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE mdspan_headers)

set_target_properties(
  ${LIB_INTERNAL_NAME} PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON
)

set_property(
  TARGET ${LIB_INTERNAL_NAME} PROPERTY "${GCXX_GPU_LANG}_STANDARD" 17
)

target_compile_features(${LIB_INTERNAL_NAME} INTERFACE cxx_std_17)

target_compile_definitions(
  ${LIB_INTERNAL_NAME} INTERFACE "CMAKE_GCXX_${GCXX_GPU_LANG}_MODE"
)

if(GCXX_WITH_EXCEPTIONS)
  target_compile_definitions(
    ${LIB_INTERNAL_NAME} INTERFACE "GCXX_WITH_EXCEPTIONS"
  )
endif()

if(GCXX_DISABLE_RUNTIME_CHECKS)
  target_compile_definitions(
    ${LIB_INTERNAL_NAME} INTERFACE "GCXX_DISABLE_RUNTIME_CHECKS"
  )
endif()

if(GCXX_CUDA_MODE)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE CUDA::cudart)
  set_target_properties(
    ${LIB_INTERNAL_NAME} PROPERTIES CMAKE_CUDA_SEPRABLE_COMPILATION ON
  )
elseif(GCXX_HIP_MODE)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE hip::host)
endif()

# BLAS backend (cuBLAS/hipBLAS).Consumers that use BlasHandle opt in by linking gcxx::blas.
# And including the appropriate header file
add_library(gcxx_blas INTERFACE)
add_library(gcxx::blas ALIAS gcxx_blas)
target_link_libraries(gcxx_blas INTERFACE gcxx::gcxx)
if(GCXX_CUDA_MODE)
  target_link_libraries(gcxx_blas INTERFACE CUDA::cublas)
elseif(GCXX_HIP_MODE)
  find_package(hipblas REQUIRED)
  target_link_libraries(gcxx_blas INTERFACE hip::hipblas)
endif()

# Register all library headers as clang-tidy targets.
# gcxx_tidy_add_target is a no-op when GCXX_ENABLE_CLANG_TIDY=OFF.
file(
  GLOB_RECURSE
  _gcxx_headers
  "${CMAKE_CURRENT_SOURCE_DIR}/gcxx/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/gcxx/*.hxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/gcxx/*.hh"
  "${CMAKE_CURRENT_SOURCE_DIR}/gcxx/*.cuh"
)
gcxx_tidy_add_target(
  TARGET
  gcxx
  SOURCES
  ${_gcxx_headers}
)
