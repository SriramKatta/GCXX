set(SRC_BASE_NAMES test_sparse_handle test_sparse_descr)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
  # The cusparse/hipsparse backend is intentionally NOT linked into the core gcxx
  target_link_libraries(${SRC_BASE_NAME} PRIVATE gcxx::sparse)
endforeach()
