# gcxx_add_test
# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for adding a single GTest executable.
#
# Usage:
#   gcxx_add_test(<src-base-name>)
#
# Expects "<src-base-name>.cpp" next to the calling CMakeLists.txt and:
#   • tags the source with LANGUAGE ${GCXX_GPU_LANG}
#   • links gcxx::gcxx and GTest::gtest_main PRIVATE
#   • registers the executable with CTest via gtest_discover_tests()
#   • attaches it to the umbrella `tests` target
# ─────────────────────────────────────────────────────────────────────────────
function(gcxx_add_test SRC_BASE_NAME)
  gcxx_tidy_add_target(
    TARGET
    ${SRC_BASE_NAME}
    SOURCES
    "${SRC_BASE_NAME}.cpp"
  )

  if(BUILD_TESTING)
    add_executable(${SRC_BASE_NAME} ${SRC_BASE_NAME}.cpp)
    set_source_files_properties(
      ${SRC_BASE_NAME}.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG}
    )
    target_link_libraries(${SRC_BASE_NAME} PRIVATE gcxx::gcxx)
    target_link_libraries(${SRC_BASE_NAME} PRIVATE GTest::gtest_main)
    add_dependencies(tests ${SRC_BASE_NAME})
    gtest_discover_tests(${SRC_BASE_NAME})
  endif()
endfunction()
