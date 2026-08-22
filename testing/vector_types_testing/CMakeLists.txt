set(SRC_BASE_NAMES test_vector_types test_vector_ops)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  gcxx_add_test(${SRC_BASE_NAME})
endforeach()
