# gcxx_add_example
# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for adding a single-executable GPU example.
#
# Usage:
#   gcxx_add_example(
#     NAME   <exe-name>
#     [SOURCES  <src> ...]          # defaults to main.cpp
#     [HEADERS  <hdr> ...]          # extra files tagged with GPU language
#                                   # (improves IDE / intellisense support)
#     [LINKS    <target> ...]       # additional link targets beyond gcxx::gcxx
#     [SEPARABLE_COMPILATION]       # set CUDA_SEPARABLE_COMPILATION ON
#   )
#
# Every example automatically:
#   • tags all SOURCES (and HEADERS) with LANGUAGE ${GCXX_GPU_LANG}
#   • links gcxx::gcxx PRIVATE
# ─────────────────────────────────────────────────────────────────────────────
function(gcxx_add_example)
  cmake_parse_arguments(
    ARG                       # prefix
    "SEPARABLE_COMPILATION"   # options  (boolean flags)
    "NAME"                    # one-value keywords
    "SOURCES;HEADERS;LINKS"   # multi-value keywords
    ${ARGN}
  )

  if(NOT ARG_NAME)
    message(FATAL_ERROR "gcxx_add_example: NAME is required")
  endif()

  # Default source file
  if(NOT ARG_SOURCES)
    set(ARG_SOURCES main.cpp)
  endif()

  # Tag every source file with the GPU language (CUDA / HIP)
  foreach(_src IN LISTS ARG_SOURCES)
    set_source_files_properties(${_src} PROPERTIES LANGUAGE ${GCXX_GPU_LANG})
  endforeach()

  # Tag header files so IDEs pick up the correct language for intellisense
  foreach(_hdr IN LISTS ARG_HEADERS)
    set_source_files_properties(${_hdr} PROPERTIES LANGUAGE ${GCXX_GPU_LANG})
  endforeach()

  add_executable(${ARG_NAME} ${ARG_SOURCES})

  target_link_libraries(${ARG_NAME} PRIVATE gcxx::gcxx ${ARG_LINKS})

  if(ARG_SEPARABLE_COMPILATION)
    set_target_properties(${ARG_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
endfunction()
