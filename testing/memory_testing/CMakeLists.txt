set(SRC_BASE_NAMES
    test_memset
    test_buffer_size
    test_buffer_constructors
    test_buffer_accessors
    test_buffer_constructors_range
    test_buffer_properties
    test_any_resource
    test_heterogeneous_iterator
    test_stride_iterator
    test_properties
    test_copy
)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
endforeach()

# device_mock_resource models the resource interface but is only exercised via
# static property checks in test_buffer_properties, so its allocate/deallocate
# are never called — silence nvcc #177-D ("declared but never referenced") there.
if(TARGET test_buffer_properties)
  target_compile_options(
    test_buffer_properties
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>
  )
endif()
