#pragma once


#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gcxx/api.hpp>

struct Args {
  size_t N{};
  size_t rep{};
  size_t blocks{};
  size_t threads{};
  size_t numstreams{};
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


  program.add_argument("-S", "--streams")
    .help("Number of streams")
    .default_value<size_t>(4)
    .scan<'i', std::size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    fmt::print(stderr, "{}\n", err.what());
    fmt::print(stderr, "{}\n", program.help().str());
    std::exit(1);
  }

  return {program.get<size_t>("N"), program.get<size_t>("reps"),
          program.get<size_t>("blocks"), program.get<size_t>("threads"),
          program.get<size_t>("streams")};
}

template <typename VT>
__global__ void kernel_scalar(const gcxx::span<VT> a) {
  const auto incval = static_cast<VT>(1);
  int start         = threadIdx.x + blockDim.x * blockIdx.x;
  int stride        = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size(); i += stride) {
    a[i] += incval;
  }
}

template <typename VT>
__global__ void kernel_2vec(const gcxx::span<VT> a) {
  const auto incval = static_cast<VT>(1);
  int start         = threadIdx.x + blockDim.x * blockIdx.x;
  int stride        = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 2; i += stride) {
    auto& a2 = gcxx::cast_as_vec2_ptr(a.data())[i];
    a2       = a2 + incval;
  }
  if (a.size() % 2 != 0 && start == 0) {
    a.back() += incval;
  }
}

template <typename VT>
__global__ void kernel_4vec(const gcxx::span<VT> a) {
  const auto incval = static_cast<VT>(1);
  int start         = threadIdx.x + blockDim.x * blockIdx.x;
  int stride        = blockDim.x * gridDim.x;
  for (size_t i = start; i < a.size() / 4; i += stride) {
    auto& a4 = gcxx::cast_as_vec4_ptr(a.data())[i];
    a4       = a4 + incval;
  }
  // 0 thread, process final elements (if there are any)
  int remainder = a.size() % 4;
  if (start == a.size() / 4)
    for (auto& i : a.last(remainder)) {
      i += incval;
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