set(SRC_BASE_NAMES
    test_blas_handle
    test_blas_transpose
    test_blas_gemm
    test_blas_gemv
    test_blas_pointer_mode
)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
  # The cublas/hipblas backend is intentionally NOT linked into the core gcxx
  target_link_libraries(${SRC_BASE_NAME} PRIVATE gcxx::blas)
endforeach()
