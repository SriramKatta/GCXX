# OpenMP is the default concurrency path (the #else branch of main.cpp); pthreads
# is available behind -DUSE_PTHREADS and needs Threads.
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

gcxx_add_example(
  NAME UnifiedMemoryStreams
  LINKS fmt::fmt gcxx::blas Threads::Threads
)

# gcxx_add_example compiles main.cpp as the GPU language (CUDA/HIP), and
# CMAKE_CXX_COMPILER is the system g++ -- so OpenMP::OpenMP_CXX would detect
# GNU/libgomp while the GPU compiler (nvcc/clang++) emits libomp (__kmpc_*)
# calls. That mismatch breaks the link. Instead, drive OpenMP through the actual
# GPU compiler for both compile and link: nvcc needs -Xcompiler wrapping,
# clang++ takes -fopenmp directly. Without this the loop runs serially.
if(GCXX_CUDA_MODE)
  target_compile_options(UnifiedMemoryStreams PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
else()
  target_compile_options(UnifiedMemoryStreams PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>)
endif()
# The link step runs under the host compiler (g++ for the CUDA target, clang++
# for the HIP target), so a plain -fopenmp pulls the matching host OpenMP runtime
# (libgomp / libomp respectively).
target_link_options(UnifiedMemoryStreams PRIVATE -fopenmp)
