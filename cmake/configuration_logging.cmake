# ---- Verbose configuration logging ----
if(GCXX_CUDA_MODE)
  set(GPU_BACKEND_VERSION "${CUDAToolkit_VERSION}")
elseif(GCXX_HIP_MODE)
  set(GPU_BACKEND_VERSION "${hip_VERSION}")
endif()

message(STATUS "\n=== GCXX Configuration Summary ===")
message(STATUS "GPU Backend:")
message(STATUS "  GCXX_GPU_LANG                    = ${GCXX_GPU_LANG}")
message(STATUS "  GCXX_CUDA_MODE                   = ${GCXX_CUDA_MODE}")
message(STATUS "  GCXX_HIP_MODE                    = ${GCXX_HIP_MODE}")
if(GCXX_HIP_MODE)
  message(STATUS "  ROCM_ROOT                        = ${ROCM_ROOT}")
endif()
message(
  STATUS
    "  ${GCXX_GPU_LANG} Version                     = ${GPU_BACKEND_VERSION}"
)
message(STATUS "")
message(STATUS "Compiler Standards:")
message(STATUS "  GCXX_CXX_STANDARD                = ${GCXX_CXX_STANDARD}")
message(STATUS "  CMAKE_CXX_STANDARD               = ${CMAKE_CXX_STANDARD}")
message(
  STATUS
    "  CMAKE_${GCXX_GPU_LANG}_STANDARD              = ${CMAKE_${GCXX_GPU_LANG}_STANDARD}"
)
message(STATUS "  CMAKE_CXX_EXTENSIONS             = ${CMAKE_CXX_EXTENSIONS}")
message(
  STATUS
    "  CMAKE_${GCXX_GPU_LANG}_EXTENSIONS            = ${CMAKE_${GCXX_GPU_LANG}_EXTENSIONS}"
)
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  GCXX_MULTI_GPU                   = ${GCXX_MULTI_GPU}")
message(STATUS "  GCXX_WITH_EXCEPTIONS             = ${GCXX_WITH_EXCEPTIONS}")
message(STATUS "  GCXX_BUILD_EXAMPLES              = ${GCXX_BUILD_EXAMPLES}")
message(STATUS "  GCXX_BUILD_BENCHMARKS            = ${GCXX_BUILD_BENCHMARKS}")
message(STATUS "  GCXX_BUILD_TESTS                 = ${GCXX_BUILD_TESTS}")
message(
  STATUS "  GCXX_DISABLE_RUNTIME_CHECKS      = ${GCXX_DISABLE_RUNTIME_CHECKS}"
)
message(STATUS "")
message(STATUS "Project Structure:")
message(STATUS "  GCXX_TOPLEVEL_PROJECT            = ${GCXX_TOPLEVEL_PROJECT}")
message(STATUS "  CMAKE_SOURCE_DIR                 = ${CMAKE_SOURCE_DIR}")
message(STATUS "  CMAKE_BINARY_DIR                 = ${CMAKE_BINARY_DIR}")
string(
  REPEAT
  "="
  35
  GCXX_EQ_LINE
)
message(STATUS "${GCXX_EQ_LINE}\n")
