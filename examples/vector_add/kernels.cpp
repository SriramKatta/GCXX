#include "main.hpp"

#include <cstddef>
#include <gcxx/api.hpp>

__global__ void kernel_scalar(size_t N, double* a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < N; i += stride) {
    a[i] = a[i] + 1.0;
  }
}

__global__ void kernel_2vec(size_t N, double* a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < N / 2; i += stride) {
    double2* a2 = reinterpret_cast<double2*>(a) + i;
    a2->x       = a2->x + 1.0;
    a2->y       = a2->y + 1.0;
  }
  if (N % 2 != 0 && start == 0) {
    a[N - 1] += 1.0;
  }
}

__global__ void kernel_4vec(size_t N, double* a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < N / 4; i += stride) {
    double4* a4 = reinterpret_cast<double4*>(a) + i;
    a4->x       = a4->x + 1.0;
    a4->y       = a4->y + 1.0;
    a4->z       = a4->z + 1.0;
    a4->w       = a4->w + 1.0;
  }
  // in only one thread, process final elements (if there are any)
  int remainder = N % 4;
  if (start == N / 4 && remainder != 0) {
    while (remainder) {
      a[N - remainder + start] += 1.0;
    }
  }
}

void launch_scalar_kernel(const Args& arg, const gcxx::Stream& str,
                          std::size_t N, double* ptr) {
  kernel_scalar<<<arg.blocks, arg.threads, 0, str.get()>>>(N, ptr);
}

void launch_vec2_kernel(const Args& arg, const gcxx::Stream& str, std::size_t N,
                        double* ptr) {
  kernel_2vec<<<arg.blocks, arg.threads, 0, str.get()>>>(N, ptr);
}

void launch_vec4_kernel(const Args& arg, const gcxx::Stream& str, std::size_t N,
                        double* ptr) {
  kernel_4vec<<<arg.blocks, arg.threads, 0, str.get()>>>(N, ptr);
}