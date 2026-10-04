set(SRC_BASE_NAMES
    test_mempool
    test_device_memory_pool
    test_managed_memory_pool
    test_pinned_memory_pool
    test_pool_memory_kinds
)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  gcxx_add_test(${SRC_BASE_NAME})
endforeach()
