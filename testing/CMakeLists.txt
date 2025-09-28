include(cmake/GTest.cmake)

function(add_gcxx_test SRC_BASE_NAME)
  if(BUILD_TESTING)
    add_executable(${SRC_BASE_NAME} ${SRC_BASE_NAME}.cpp)
    target_link_libraries(${SRC_BASE_NAME} PRIVATE gcxx::gcxx)
    target_link_libraries(${SRC_BASE_NAME} PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(${SRC_BASE_NAME})
  endif()
endfunction()

add_subdirectory(event_testing)
