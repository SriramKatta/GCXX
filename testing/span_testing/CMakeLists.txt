set(SRC_BASE_NAMES test_span_deduction_guides test_span_constructors)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  gcxx_add_test(${SRC_BASE_NAME})
endforeach()
