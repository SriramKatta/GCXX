CPMAddPackage(
  URI "gh:google/benchmark@1.10.0" 
  OPTIONS 
    "BENCHMARK_ENABLE_TESTING Off"
    "BENCHMARK_ENABLE_WERROR Off"
)

if(benchmark_ADDED)
  set_target_properties(benchmark PROPERTIES CXX_STANDARD ${GCXX_CXX_STANDARD})
endif()
