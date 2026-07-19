set(SRC_BASE_NAMES
    test_memset
    test_buffer_size
    test_buffer_constructors
    test_buffer_accessors
    test_buffer_constructors_range
    test_copy
)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
endforeach()
