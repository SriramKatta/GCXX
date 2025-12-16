set(SRC_BASE_NAMES test_event_view test_event)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  add_gcxx_test(${SRC_BASE_NAME})
endforeach()
