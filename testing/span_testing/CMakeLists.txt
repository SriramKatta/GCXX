set(SRC_BASE_NAMES test_span_deduction_guides test_span_constructors
                   test_span_init_list
)

foreach(SRC_BASE_NAME IN LISTS SRC_BASE_NAMES)
  gcxx_add_test(${SRC_BASE_NAME})
endforeach()
