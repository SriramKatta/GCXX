set(SRC_BASE_NAMES
    test_blas_handle
    test_blas_transposed
    test_blas_gemm
    test_blas_gemv
    test_blas_pointer_mode
    test_blas_l1
    test_blas_geam_dgmm
    test_blas_gemm_batched
)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
  # The cublas/hipblas backend is intentionally NOT linked into the core gcxx
  target_link_libraries(${SRC_BASE_NAME} PRIVATE gcxx::blas)
endforeach()
