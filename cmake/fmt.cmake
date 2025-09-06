CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.2.0
)

if(fmt_ADDED)
  set_target_properties(fmt PROPERTIES CXX_CLANG_TIDY "")
  add_library(fmt_system INTERFACE)
  target_link_libraries(fmt_system INTERFACE fmt)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GCC")
    target_compile_options(fmt_system INTERFACE -diag-suppress=128,2417)
  endif()
endif()
