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

  program.add_argument("-B", "--blocksmult")
    .help("Number of blocks multiple of SMS")
    .default_value<size_t>(32)
    .scan<'i', std::size_t>();

  program.add_argument("-T", "--threadsmult")
    .help("Threads per block multiple of warpsize")
    .default_value<size_t>(8)
    .scan<'i', std::size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    fmt::print(stderr, "{}\n", err.what());
    fmt::print(stderr, "{}\n", program.help().str());
    std::exit(1);
  }

  auto devhand = gcxx::Device::get();
  int SMcount =
    devhand.getAttribute(gcxx::flags::deviceAttribute::MultiProcessorCount);
  int warpSize = devhand.getAttribute(gcxx::flags::deviceAttribute::WarpSize);

  return {program.get<size_t>("N"), program.get<size_t>("reps"),
          SMcount * program.get<size_t>("blocksmult"),
          warpSize * program.get<size_t>("threadsmult")};
}

template <typename VT>
__global__ void kernel_scalar(const gcxx::span<VT> a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size(); i += stride) {
    a[i] += 1.0;
  }
}

template <typename VT>
__global__ void kernel_2vec(const gcxx::span<VT> a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 2; i += stride) {
    auto* a2 = reinterpret_cast<gcxx::vec2_t<VT>*>(a.data()) + i;
    a2->x += 1.0;
    a2->y += 1.0;
  }
  if (a.size() % 2 != 0 && start == 0) {
    a.back() += 1.0;
  }
}

template <typename VT>
__global__ void kernel_4vec(const gcxx::span<VT> a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 4; i += stride) {
    auto* a4 = reinterpret_cast<gcxx::vec4_t<VT>*>(a.data()) + i;
    a4->x += 1.0;
    a4->y += 1.0;
    a4->z += 1.0;
    a4->w += 1.0;
  }
  // 0 thread, process final elements (if there are any)
  int remainder = a.size() % 4;
  if (start == a.size() / 4)
    for (auto& i : a.last(remainder)) {
      i += 1.0;
    }
}

template <typename VT>
void launch_scalar_kernel(const Args& arg, const gcxx::Stream& str,
                          gcxx::span<VT> ptr) {
  gcxx::launch::Kernel(str, arg.blocks, arg.threads, 0, kernel_scalar<VT>, ptr);
}

template <typename VT>
void launch_vec2_kernel(const Args& arg, const gcxx::Stream& str,
                        gcxx::span<VT> ptr) {
  gcxx::launch::Kernel(str, arg.blocks, arg.threads, 0, kernel_2vec<VT>, ptr);
}

template <typename VT>
void launch_vec4_kernel(const Args& arg, const gcxx::Stream& str,
                        gcxx::span<VT> ptr) {
  gcxx::launch::Kernel(str, arg.blocks, arg.threads, 0, kernel_4vec<VT>, ptr);
}