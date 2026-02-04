#pragma once


#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gcxx/api.hpp>

struct Args {
  size_t N{};
  size_t rep{};
  size_t blocks{};
  size_t threads{};
};

inline Args parse_args(int argc, char** argv) {
  argparse::ArgumentParser program("vector_add");

  program.add_argument("-N", "--num-entries")
    .help("Number of elements")
    .default_value<size_t>(32'000'000)
    .scan<'i', std::size_t>();

  program.add_argument("-R", "--reps")
    .help("Number of kernel repetitions")
    .default_value<size_t>(10)
    .scan<'i', std::size_t>();

  program.add_argument("-B", "--blocks")
    .help("Number of blocks")
    .default_value<size_t>(3456)
    .scan<'i', std::size_t>();

  program.add_argument("-T", "--threads")
    .help("Threads per block")
    .default_value<size_t>(256)
    .scan<'i', std::size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    fmt::print(stderr, "{}\n", err.what());
    fmt::print(stderr, "{}\n", program.help().str());
    std::exit(1);
  }

  return {program.get<size_t>("N"), program.get<size_t>("reps"),
          program.get<size_t>("blocks"), program.get<size_t>("threads")};
}

// prefer using vec2 types in case of double to improve
// bandwidth also test with vec4_32a and vec4_16a to get
// an idea of if they improve the performance
template <typename VT>
GCXX_FD VT thread_partial_sum(const gcxx::span<VT> a) {
  VT sum{};
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 4; i += stride) {
    // using vectorized loads to improve the bandwidth
    auto* a4 = gcxx::cast_as_vec4_ptr(a.data()) + i;
    sum += (a4->x + a4->y + a4->z + a4->w);
  }

  // 0 thread, process final elements (if there are any)
  int remainder = a.size() % 4;
  if (start == 0 && remainder != 0) {
    for (auto& i : a.last(remainder)) {
      sum += i;
    }
  }

  return sum;
}

// need to improve since huge thread divergence and should use the warp shuffles
// to utilize the registers in place of shared memory since they would be even
// quicker access
template <typename VT>
GCXX_FD void in_block_reduction(VT* smem, size_t N) {
  const auto tid = threadIdx.x;
  for (size_t i = N / 2; i > 0; i >>= 1) {
    __syncthreads();
    if (tid < i) {
      smem[tid] += smem[tid + i];
    }
  }
}

// okay for now but not possible in terms of old cuda with no atomic support for
// doubles
template <typename VT>
GCXX_FDC void inter_block_reduction(VT* smem, VT* res) {
  if (threadIdx.x == 0) {
    atomicAdd(res, smem[0]);
  }
}

template <typename VT>
__global__ void kernel_reduction(const gcxx::span<VT> a, VT* result) {
  VT* sdata      = gcxx::dynamicSharedMemory<VT>();
  const auto tid = threadIdx.x;
  sdata[tid]     = thread_partial_sum(a);

  in_block_reduction(sdata, blockDim.x);

  inter_block_reduction(sdata, result);
}

template <typename VT>
VT launch_reduction_kernel(const Args& arg, const gcxx::Stream& str,
                           gcxx::span<VT>& ptr) {
  auto res_raii = gcxx::memory::make_device_unique_ptr<VT>(1, str);
  gcxx::memory::Memset(res_raii, 0, 1, str);
  VT* res = res_raii.get();

  gcxx::launch::Kernel(str, arg.blocks, arg.threads, arg.threads * sizeof(VT),
                       kernel_reduction<VT>, ptr, res);

  auto res_host = gcxx::memory::make_host_pinned_unique_ptr<VT>(1);
  gcxx::memory::Copy(res_host, res_raii, 1, str);
  return *res_host;
}