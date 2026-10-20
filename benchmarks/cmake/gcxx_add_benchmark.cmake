# gcxx_add_benchmark
# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for adding a single-executable micro-benchmark.
#
# Usage:
#   gcxx_add_benchmark(
#     NAME   <exe-name>
#     [SOURCES  <src> ...]          # defaults to main.cpp
#     [HEADERS  <hdr> ...]          # extra files tagged with GPU language
#                                   # (improves IDE / intellisense support)
#     [LINKS    <target> ...]       # additional link targets beyond
#                                   # gcxx::gcxx + benchmark::benchmark
#     [SEPARABLE_COMPILATION]       # set CUDA_SEPARABLE_COMPILATION ON
#   )
#
# Every benchmark automatically:
#   • tags all SOURCES (and HEADERS) with LANGUAGE ${GCXX_GPU_LANG}
#   • links gcxx::gcxx and benchmark::benchmark PRIVATE
#   • names the executable "bench_<NAME>"
# ─────────────────────────────────────────────────────────────────────────────
function(gcxx_add_benchmark)
  # cmake-format: off
  cmake_parse_arguments(
    ARG                       # prefix
    "SEPARABLE_COMPILATION"   # options  (boolean flags)
    "NAME"                    # one-value keywords
    "SOURCES;HEADERS;LINKS"   # multi-value keywords
    ${ARGN}
  )
  # cmake-format: on

  if(NOT ARG_NAME)
    message(FATAL_ERROR "gcxx_add_benchmark: NAME is required")
  endif()

  set(_bench_target bench_${ARG_NAME})

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

  gcxx_tidy_add_target(
    TARGET
    ${_bench_target}
    SOURCES
    ${ARG_SOURCES}
  )

  add_executable(${_bench_target} ${ARG_SOURCES})

  # Attach to the umbrella target so `--target benchmarks` builds every
  # benchmark.
  add_dependencies(benchmarks ${_bench_target})

  target_link_libraries(
    ${_bench_target} PRIVATE gcxx::gcxx benchmark::benchmark ${ARG_LINKS}
  )

  if(ARG_SEPARABLE_COMPILATION)
    set_target_properties(
      ${_bench_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
  endif()
endfunction()
