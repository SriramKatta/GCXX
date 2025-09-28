set(LIB_INTERNAL_NAME gpu_cxx)

add_library(${LIB_INTERNAL_NAME} INTERFACE)
add_library(gcxx::gcxx ALIAS ${LIB_INTERNAL_NAME})
target_include_directories(
  ${LIB_INTERNAL_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
  ${LIB_INTERNAL_NAME} PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE ON
)

set_property(
  TARGET ${LIB_INTERNAL_NAME} PROPERTY "${GCXX_GPU_LANG}_STANDARD" 17
)

target_compile_features(${LIB_INTERNAL_NAME} INTERFACE cxx_std_17)
target_compile_definitions(
  ${LIB_INTERNAL_NAME} INTERFACE "GCXX_${GCXX_GPU_LANG}_MODE"
)
set_target_properties(
  ${LIB_INTERNAL_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(GCXX_CUDA_MODE)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE CUDA::cudart)
  set_target_properties(
    ${LIB_INTERNAL_NAME} PROPERTIES CMAKE_CUDA_SEPRABLE_COMPILATION ON
  )
elseif(GCXX_HIP_MODE)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE hip::host)
endif()
