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

if(GCXX_ENABLE_RUNTIME_CHECKS)
  target_compile_definitions(
    ${LIB_INTERNAL_NAME} INTERFACE "GCXX_ENABLE_RUNTIME_CHECKS"
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
  target_link_libraries(gcxx_blas INTERFACE roc::hipblas)
endif()

# SPARSE backend (cuSPARSE/hipSPARSE, Generic API only). Consumers that use
# SparseHandle or the Sp*/Dn* descriptors opt in by linking gcxx::sparse.
add_library(gcxx_sparse INTERFACE)
add_library(gcxx::sparse ALIAS gcxx_sparse)
target_link_libraries(gcxx_sparse INTERFACE gcxx::gcxx)
if(GCXX_CUDA_MODE)
  target_link_libraries(gcxx_sparse INTERFACE CUDA::cusparse)
elseif(GCXX_HIP_MODE)
  find_package(hipsparse REQUIRED)
  target_link_libraries(gcxx_sparse INTERFACE roc::hipsparse)
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

# Skip backend headers for the *inactive* backend: they include that backend's
# SDK unconditionally (cuda_runtime.h / hip_runtime.h), which is not present
# when tidy runs inside a single-backend container.
set(_gcxx_tidy_exclude "")
if(GCXX_CUDA_MODE)
  set(_gcxx_tidy_exclude "backend/hip_[^/]*\\.hpp$")
elseif(GCXX_HIP_MODE)
  set(_gcxx_tidy_exclude "backend/cuda_[^/]*\\.hpp$")
endif()
if(_gcxx_tidy_exclude)
  list(
    FILTER
    _gcxx_headers
    EXCLUDE
    REGEX
    "${_gcxx_tidy_exclude}"
  )
endif()

gcxx_tidy_add_target(
  TARGET
  gcxx
  SOURCES
  ${_gcxx_headers}
)
