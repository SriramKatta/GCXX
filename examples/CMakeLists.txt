project(GCXX_Examples)
add_custom_target(examples)


if(GCXX_CUDA_MODE)
    set(
        CMAKE_CUDA_ARCHITECTURES 
        80 # for A100
        86 # for A40
        90 # for GH200
        )
        # link for cuda compute capability list
        # https://developer.nvidia.com/cuda/gpus
elseif(GCXX_HIP_MODE)
    set(
        CMAKE_HIP_ARCHITECTURES 
        gfx908 # for MI100
        gfx90a # for MI210A
        gfx1030 # for RX 6900 XT 
        gfx942 # for MI300X & MI300A
    )
    # link for all hip arch list
    # https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
endif()


include(cmake/fmt.cmake)
include(cmake/argparse.cmake)

add_subdirectory(by_api_module)
add_subdirectory(vector_add)
add_subdirectory(vector_add_overlap)
# add_subdirectory(multi_gpu_testspace)
add_subdirectory(Reduction)
