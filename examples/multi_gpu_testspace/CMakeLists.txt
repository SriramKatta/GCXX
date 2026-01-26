set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG})

set(EXE_NAME multi_gpu_testspace)

include(../cmake/mpicxx.cmake)

add_executable(${EXE_NAME} main.cpp)

set(NCCL_HOME
    "/apps/SPACK/0.22.2/opt/linux-almalinux8-zen/gcc-8.4.1/nvhpc-25.5-5tjelrgdrvojmmo2k3qbvnusezsijacy/Linux_x86_64/25.5/comm_libs/nccl"
)

if(DEFINED NCCL_HOME)
  find_path(
    NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${NCCL_HOME}/include REQUIRED
  )
  find_library(
    NCCL_LIB
    NAMES nccl
    PATHS ${NCCL_HOME}/lib REQUIRED
  )
  target_include_directories(${EXE_NAME} PRIVATE "${NCCL_INCLUDE_DIR}")
  target_link_libraries(${EXE_NAME} PRIVATE "${NCCL_LIB}")
else()
  message(FATAL_ERROR "Please set -DNCCL_HOME to compile NCCL code")
endif()

target_link_libraries(
  ${EXE_NAME}
  PRIVATE gcxx::gcxx
          fmt::fmt
          cudart
          nvidia-ml
          MPI::MPI_CXX
)
