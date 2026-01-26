CPMAddPackage(
  NAME mdspan
  GITHUB_REPOSITORY kokkos/mdspan
  GIT_TAG single-header
)

add_library(mdspan_headers INTERFACE)

target_include_directories(mdspan_headers SYSTEM INTERFACE ${mdspan_SOURCE_DIR})

target_compile_definitions(
  mdspan_headers
  INTERFACE MDSPAN_IMPL_STANDARD_NAMESPACE=gcxx
            $<$<COMPILE_LANGUAGE:CUDA>:MDSPAN_IMPL_HAS_CUDA=1>
            $<$<COMPILE_LANGUAGE:HIP>:MDSPAN_IMPL_HAS_HIP=1>
)
