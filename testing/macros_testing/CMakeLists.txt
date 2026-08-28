# The concept DSL in template_helper_macros.hpp has two implementations selected by
# GCXX_HAS_CONCEPTS(): a C++20 requires(...) path and a pre-C++20 SFINAE-emulation
# path. GCXX defaults to C++17, so build the same (host-only) test source under
# both standards to exercise each implementation. The source has no device code,
# so it just needs the CUDA language (for the __host__/__device__ decorators in
# the gcxx headers) at two different dialects.

# The single source is registered with clang-tidy once (it is shared by both
# targets below; registering twice would collide on the generated tidy target).
gcxx_tidy_add_target(
  TARGET
  test_concept_macros
  SOURCES
  "test_concept_macros.cpp"
)

function(add_gcxx_concept_macro_test TARGET_NAME CUDA_STD)
  if(BUILD_TESTING)
    add_executable(${TARGET_NAME} test_concept_macros.cpp)
    set_source_files_properties(
      test_concept_macros.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG}
    )
    set_target_properties(
      ${TARGET_NAME} PROPERTIES CUDA_STANDARD ${CUDA_STD} CUDA_STANDARD_REQUIRED
                                                          ON
    )

    target_link_libraries(${TARGET_NAME} PRIVATE gcxx::gcxx GTest::gtest_main)
    gtest_discover_tests(${TARGET_NAME})
  endif()
endfunction()

add_gcxx_concept_macro_test(test_concept_macros_cpp17 17
)# SFINAE-emulation branch
add_gcxx_concept_macro_test(test_concept_macros_cpp20 20) # requires(...) branch
