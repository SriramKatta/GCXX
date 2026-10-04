CPMAddPackage(
  URI "gh:google/benchmark@1.5.2" OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)

if(benchmark_ADDED)
  set_target_properties(benchmark PROPERTIES CXX_STANDARD ${GCXX_CXX_STANDARD})
endif()
