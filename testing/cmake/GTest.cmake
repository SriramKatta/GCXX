CPMAddPackage(
  NAME GTEST
  GITHUB_REPOSITORY "google/googletest"
  GIT_TAG "main"
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

if(GTEST_ADDED)
  # Treat gtest/gmock headers as system headers and silence their own warnings.
  foreach(_t gtest gtest_main gmock gmock_main)
    if(TARGET ${_t})
      get_target_property(_inc ${_t} INTERFACE_INCLUDE_DIRECTORIES)
      if(_inc)
        set_target_properties(${_t} PROPERTIES
          INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
      endif()
      target_compile_options(${_t} PRIVATE -w)
    endif()
  endforeach()
endif()
