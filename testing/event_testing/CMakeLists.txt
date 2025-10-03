set(SRC_BASE_NAMES test_event_base test_event_wrap test_event)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
endforeach()
