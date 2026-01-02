include(cmake/GTest.cmake)

include(GoogleTest)

if(GCXX_CUDA_MODE)
    set(
        CMAKE_CUDA_ARCHITECTURES 
        80 # for A100
        86 # for A40
        # 87 89 90 100 120 # since i have access only to these archs
    )
elseif(GCXX_HIP_MODE)
    set(
        CMAKE_HIP_ARCHITECTURES 
        gfx908 # for MI100
        gfx90a # for MI210A
        gfx1030 # for RX 6900 XT 
        gfx940 # for MI300X & MI300A
    )
    # link for all the arch list
    # https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/wiki/AMD-GPU-Arches-Lists-Map 
endif()

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
add_subdirectory(vector_types_testing)
