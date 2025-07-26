set(LIB_INTERNAL_NAME gpu_cxx)

add_library(${LIB_INTERNAL_NAME} INTERFACE)
add_library(gpucxx::gpucxx ALIAS ${LIB_INTERNAL_NAME})
target_include_directories(${LIB_INTERNAL_NAME} INTERFACE include)
target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE fmt::fmt)

set_property(
  TARGET ${LIB_INTERNAL_NAME} PROPERTY "${BUILD_GPU_LANGUAGE}_STANDARD" 17 
  PROPERTIES LANGUAGE ${BUILD_GPU_LANGUAGE}
)
target_compile_definitions(${LIB_INTERNAL_NAME} INTERFACE "GPUCXX_${BUILD_GPU_LANGUAGE}_MODE")
set_target_properties(${LIB_INTERNAL_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(GPUCXX_CUDA_MODE)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE CUDA::cudart)
elseif(GPUCXX_HIP_MODE)
  #test
endif()
