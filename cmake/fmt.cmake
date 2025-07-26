CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.2.0
)

if(fmt_ADDED)
  target_compile_options(fmt INTERFACE -diag-suppress=128,2417)
endif()
