// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <utility>

#include <benchmark/benchmark.h>

#include <gcxx/runtime/graph/params/graph_kernel_node_params.hpp>

namespace {

  __global__ void bench_kernel_noargs() {}

  __global__ void bench_kernel_args(float* out, const float* in, int count,
                                    float scale) {}

  constexpr unsigned int kGridX     = 128;
  constexpr unsigned int kBlockX    = 256;
  constexpr unsigned int kSharedMem = 1024;
  constexpr std::size_t kMaxArgs    = 8;

  using RawParams = gcxx::KernelNodeParamsView::deviceKernelNodeParams_t;

  using ArgStorage = std::tuple<float, int, double, unsigned long long, float,
                                short, unsigned int, long long>;

  auto make_arg_storage() -> ArgStorage {
    return {1.0F, 42, 3.14159, 7ULL, -2.5F, -1, 99U, -99LL};
  }

  template <std::size_t N, typename Tuple, std::size_t... Is>
  auto take_head_impl(Tuple& storage, std::index_sequence<Is...>)
    -> decltype(std::forward_as_tuple(std::get<Is>(storage)...)) {
    return std::forward_as_tuple(std::get<Is>(storage)...);
  }

  template <std::size_t N, typename Tuple>
  auto take_head(Tuple& storage)
    -> decltype(take_head_impl<N>(storage, std::make_index_sequence<N>{})) {
    return take_head_impl<N>(storage, std::make_index_sequence<N>{});
  }

  template <typename ParamsT>
  auto sink_params(const ParamsT& params) -> void {
    auto raw = params.getRawParams();
    benchmark::DoNotOptimize(raw);
    benchmark::ClobberMemory();
  }

  // Floor: hand-fill the pointer array and aggregate-initialize the raw
  // driver struct exactly as the library's make_raw_params() does.
  template <std::size_t N>
  void BM_RawDriverStruct(benchmark::State& state) {
    static ArgStorage storage = make_arg_storage();
    auto args                 = take_head<N>(storage);
    for (auto _ : state) {
      auto ptrs = std::apply(
        [](auto&... arg) {
          return std::array<void*, sizeof...(arg)>{
            static_cast<void*>(std::addressof(arg))...};
        },
        args);
      // Field assignment, not positional aggregate init: the field order of
      // hipKernelNodeParams differs from cudaKernelNodeParams.
      RawParams raw{};
      raw.func           = reinterpret_cast<void*>(bench_kernel_args);
      raw.gridDim        = dim3{kGridX};
      raw.blockDim       = dim3{kBlockX};
      raw.sharedMemBytes = kSharedMem;
      raw.kernelParams   = ptrs.data();
      raw.extra          = nullptr;
      benchmark::DoNotOptimize(raw);
      benchmark::ClobberMemory();
    }
  }

  // Direct construction: one variadic ctor call; arg pointers captured
  // internally by KernelNodeParams::make_arg_array().
  template <std::size_t N>
  void BM_DirectCtor(benchmark::State& state) {
    static ArgStorage storage = make_arg_storage();
    auto args                 = take_head<N>(storage);
    for (auto _ : state) {
      auto params = std::apply(
        [](auto&... arg) {
          return gcxx::KernelNodeParams<sizeof...(arg)>(
            reinterpret_cast<void*>(bench_kernel_args), dim3{kGridX},
            dim3{kBlockX}, kSharedMem, arg...);
        },
        args);
      sink_params(params);
    }
  }

  // Direct construction where the caller supplies the explicit pointer array
  // (the second KernelNodeParams ctor overload).
  template <std::size_t N>
  void BM_DirectPtrArray(benchmark::State& state) {
    static ArgStorage storage = make_arg_storage();
    auto args                 = take_head<N>(storage);
    for (auto _ : state) {
      auto params = std::apply(
        [](auto&... arg) {
          std::array<void*, sizeof...(arg)> ptrs{
            static_cast<void*>(std::addressof(arg))...};
          return gcxx::KernelNodeParams<sizeof...(arg)>(
            reinterpret_cast<void*>(bench_kernel_args), dim3{kGridX},
            dim3{kBlockX}, kSharedMem, ptrs);
        },
        args);
      sink_params(params);
    }
  }

  // Full type-state builder chain: setKernel/setGridDim/setBlockDim/
  // setSharedMemBytes/setArgs/build(). Each setter returns a new builder state.
  template <std::size_t N>
  void BM_BuilderFull(benchmark::State& state) {
    static ArgStorage storage = make_arg_storage();
    auto args                 = take_head<N>(storage);
    for (auto _ : state) {
      auto params = std::apply(
        [](auto&... arg) {
          return gcxx::KernelParamsBuilder()
            .setKernel(bench_kernel_args)
            .setGridDim(kGridX)
            .setBlockDim(kBlockX)
            .setSharedMemBytes(kSharedMem)
            .setArgs(arg...)
            .build();
        },
        args);
      sink_params(params);
    }
  }

  void BM_DirectNoArgs(benchmark::State& state) {
    for (auto _ : state) {
      gcxx::KernelNodeParams<0> params(
        reinterpret_cast<void*>(bench_kernel_noargs), dim3{kGridX},
        dim3{kBlockX}, 0U);
      sink_params(params);
    }
  }

  void BM_BuilderNoArgs(benchmark::State& state) {
    for (auto _ : state) {
      auto params = gcxx::KernelParamsBuilder()
                      .setKernel(bench_kernel_noargs)
                      .setGridDim(kGridX)
                      .setBlockDim(kBlockX)
                      .build();
      sink_params(params);
    }
  }

  template <typename Tuple, std::size_t... Is>
  auto direct_from_head(Tuple& args, std::index_sequence<Is...>)
    -> gcxx::KernelNodeParams<sizeof...(Is)> {
    return gcxx::KernelNodeParams<sizeof...(Is)>(
      reinterpret_cast<void*>(bench_kernel_args), dim3{kGridX}, dim3{kBlockX},
      kSharedMem, std::get<Is>(args)...);
  }

  template <typename Tuple, std::size_t... Is>
  auto built_from_head(Tuple& args, std::index_sequence<Is...>)
    -> gcxx::KernelNodeParams<sizeof...(Is)> {
    return gcxx::KernelParamsBuilder()
      .setKernel(bench_kernel_args)
      .setGridDim(kGridX)
      .setBlockDim(kBlockX)
      .setSharedMemBytes(kSharedMem)
      .setArgs(std::get<Is>(args)...)
      .build();
  }

  auto verify_equivalence() -> bool {
    static ArgStorage storage = make_arg_storage();
    constexpr std::size_t n   = 4;
    auto args                 = take_head<n>(storage);

    auto direct   = direct_from_head(args, std::make_index_sequence<n>{});
    auto built    = built_from_head(args, std::make_index_sequence<n>{});
    const auto& d = direct.getRawParams();
    const auto& b = built.getRawParams();

    if (d.func != b.func || d.gridDim.x != b.gridDim.x ||
        d.gridDim.y != b.gridDim.y || d.gridDim.z != b.gridDim.z ||
        d.blockDim.x != b.blockDim.x || d.blockDim.y != b.blockDim.y ||
        d.blockDim.z != b.blockDim.z || d.sharedMemBytes != b.sharedMemBytes ||
        d.extra != b.extra) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (d.kernelParams[i] != b.kernelParams[i]) {
        return false;
      }
    }
    return true;
  }

}  // namespace

BENCHMARK_TEMPLATE(BM_RawDriverStruct, 2);
BENCHMARK_TEMPLATE(BM_RawDriverStruct, 4);
BENCHMARK_TEMPLATE(BM_RawDriverStruct, kMaxArgs);
BENCHMARK_TEMPLATE(BM_DirectCtor, 2);
BENCHMARK_TEMPLATE(BM_DirectCtor, 4);
BENCHMARK_TEMPLATE(BM_DirectCtor, kMaxArgs);
BENCHMARK_TEMPLATE(BM_DirectPtrArray, 2);
BENCHMARK_TEMPLATE(BM_DirectPtrArray, 4);
BENCHMARK_TEMPLATE(BM_DirectPtrArray, kMaxArgs);
BENCHMARK_TEMPLATE(BM_BuilderFull, 2);
BENCHMARK_TEMPLATE(BM_BuilderFull, 4);
BENCHMARK_TEMPLATE(BM_BuilderFull, kMaxArgs);
BENCHMARK(BM_DirectNoArgs);
BENCHMARK(BM_BuilderNoArgs);

auto main(int argc, char** argv) -> int {
  if (!verify_equivalence()) {
    std::fputs("FATAL: builder output does not match direct construction\n",
               stderr);
    return EXIT_FAILURE;
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return EXIT_FAILURE;
  }
  benchmark::RunSpecifiedBenchmarks();
  return EXIT_SUCCESS;
}
