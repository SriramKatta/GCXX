include(cmake/GTest.cmake)

include(GoogleTest)

function(add_gcxx_test SRC_BASE_NAME)
  if(BUILD_TESTING)
    add_executable(${SRC_BASE_NAME} ${SRC_BASE_NAME}.cpp)
    set_source_files_properties(
      ${SRC_BASE_NAME}.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG}
    )
    target_link_libraries(${SRC_BASE_NAME} PRIVATE gcxx::gcxx)
    target_link_libraries(${SRC_BASE_NAME} PRIVATE GTest::gtest_main)
    gtest_discover_tests(${SRC_BASE_NAME})
  endif()
endfunction()

include_directories(common)
add_subdirectory(event_testing)
