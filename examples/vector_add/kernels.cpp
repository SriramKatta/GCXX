#include "main.hpp"

#include <cstddef>
#include <gcxx/api.hpp>

__global__ void kernel_scalar(gcxx::span<double> a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size(); i += stride) {
    a[i] += 1.0;
  }
}

__global__ void kernel_2vec(gcxx::span<double> a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 2; i += stride) {
    double2* a2 = reinterpret_cast<double2*>(a.data()) + i;
    a2->x += 1.0;
    a2->y += 1.0;
  }
  if (a.size() % 2 != 0 && start == 0) {
    a.back() += 1.0;
  }
}

__global__ void kernel_4vec(gcxx::span<double> a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 4; i += stride) {
    double4* a4 = reinterpret_cast<double4*>(a.data()) + i;
    a4->x += 1.0;
    a4->y += 1.0;
    a4->z += 1.0;
    a4->w += 1.0;
  }
  // 0 thread, process final elements (if there are any)
  int remainder = a.size() % 4;
  if (start == 0)
    for (auto& i : a.last(remainder)) {
      i += 1.0;
    }
}

void launch_scalar_kernel(const Args& arg, const gcxx::Stream& str,
                          gcxx::span<double>& ptr) {
  kernel_scalar<<<arg.blocks, arg.threads, 0, str.get()>>>(ptr);
}

void launch_vec2_kernel(const Args& arg, const gcxx::Stream& str,
                        gcxx::span<double>& ptr) {
  kernel_2vec<<<arg.blocks, arg.threads, 0, str.get()>>>(ptr);
}

void launch_vec4_kernel(const Args& arg, const gcxx::Stream& str,
                        gcxx::span<double>& ptr) {
  kernel_4vec<<<arg.blocks, arg.threads, 0, str.get()>>>(ptr);
}