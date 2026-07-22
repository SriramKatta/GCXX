set(SRC_BASE_NAMES test_mempool test_mempool_view
                    test_device_memory_pool test_managed_memory_pool
                    test_pinned_memory_pool)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
endforeach()
