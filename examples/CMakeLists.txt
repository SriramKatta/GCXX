project(GPUCXX_EXamples)
add_custom_target(examples)

  set(CMAKE_CUDA_ARCHITECTURES
      80
      86
      87
      89
      90
      100
      120
  )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(by_api_module)
add_subdirectory(vector_add)
