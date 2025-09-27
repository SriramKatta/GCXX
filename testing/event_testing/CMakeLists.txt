

if (BUILD_TESTING)
    add_executable(test_event_base event_base_test.cpp)
    target_link_libraries(test_event_base PRIVATE gcxx::gcxx)
    target_link_libraries(test_event_base PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(test_event_base)

    add_executable(test_event_ref event_ref_test.cpp)
    target_link_libraries(test_event_ref PRIVATE gcxx::gcxx)
    target_link_libraries(test_event_ref PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(test_event_ref)


endif()


