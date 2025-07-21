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

if(GPUCXX_CUDA_MODE)
  set(CMAKE_CUDA_ARCHITECTURES 80)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE CUDA::cudart)
elseif(GPUCXX_HIP_MODE)
  target_compile_definitions(${LIB_INTERNAL_NAME} INTERFACE __HIP_PLATFORM_AMD__)
  target_link_libraries(${LIB_INTERNAL_NAME} INTERFACE hip::device)
  target_include_directories(${LIB_INTERNAL_NAME} INTERFACE ${HIP_INCLUDE_DIRS})
endif()
