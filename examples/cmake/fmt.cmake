CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.2.0
)

if(fmt_ADDED)
  # Treat fmt's headers as system headers (so they don't warn in our code)
  # and silence warnings from compiling fmt's own sources.
  get_target_property(FMT_INCLUDES fmt INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(fmt PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${FMT_INCLUDES}")
  target_compile_options(fmt PRIVATE -w)
endif()
