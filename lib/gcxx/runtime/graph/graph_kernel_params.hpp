#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_KERNEL_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_KERNEL_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceKernelNodeParams_t = GCXX_RUNTIME_BACKEND(KernelNodeParams);

template <std::size_t N>
class KernelNodeParams {
 private:
  deviceKernelNodeParams_t params_{};
  std::array<void*, N> kernelargs_{};


 public:
  template <typename... Args>
  KernelNodeParams(void* func, dim3 grid, dim3 block, unsigned int shmem,
                   Args&... args) {
    static_assert(sizeof...(args) == N, "Arg count mismatch!");

    std::size_t i = 0;
    ((kernelargs_[i++] = static_cast<void*>(&args)), ...);

    params_.func           = func;
    params_.gridDim        = grid;
    params_.blockDim       = block;
    params_.sharedMemBytes = shmem;
    params_.kernelParams   = kernelargs_.data();  // Points to sibling member
    params_.extra          = nullptr;
  }

  // Constructor from pre-built array of void* pointers (used by builder)
  KernelNodeParams(void* func, dim3 grid, dim3 block, unsigned int shmem,
                   std::array<void*, N> arg_ptrs)
      : kernelargs_(arg_ptrs) {
    params_.func           = func;
    params_.gridDim        = grid;
    params_.blockDim       = block;
    params_.sharedMemBytes = shmem;
    params_.kernelParams   = kernelargs_.data();
    params_.extra          = nullptr;
  }

  auto getRawParams() const -> const deviceKernelNodeParams_t& {
    return params_;
  }

  // Disable move/copy to ensure params_.kernelParams never points to an old
  // stack location
  KernelNodeParams(const KernelNodeParams&) = delete;
  KernelNodeParams(KernelNodeParams&&)      = delete;
};

GCXX_NAMESPACE_DETAILS_BEGIN

template <typename... Args>
class KernelArgPack {
  static_assert((std::is_trivially_copyable_v<Args> && ...),
                "All kernel arguments must be trivially copyable");

 public:
  std::tuple<Args&...> args;
  std::array<void*, sizeof...(Args)> ptrs{};

  KernelArgPack(Args&... a) : args(a...) {
    std::size_t i = 0;
    std::apply([&](auto&... unpacked) { ((ptrs[i++] = &unpacked), ...); },
               args);
  }
};

class KernelParamsBuilder {
 private:
  void* kernel_{nullptr};
  dim3 grid_{1, 1, 1};
  dim3 block_{1, 1, 1};
  unsigned shmem_{0};
  std::vector<void*> arg_ptrs_;

 public:
  static KernelParamsBuilder create() { return {}; }

  template <typename Kernel>
  KernelParamsBuilder& setKernel(Kernel k) {
    static_assert(details_::is_void_function_pointer<Kernel>::value,
                  "Passed value must be a function pointer and function should "
                  "return void");
    kernel_ = reinterpret_cast<void*>(k);
    return *this;
  }

  KernelParamsBuilder& setGridDim(dim3 g) {
    grid_ = g;
    return *this;
  }

  KernelParamsBuilder& setGridDim(unsigned int x = 1, unsigned int y = 1,
                                  unsigned int z = 1) {
    grid_ = {x, y, z};
    return *this;
  }

  KernelParamsBuilder& setBlockDim(dim3 b) {
    block_ = b;
    return *this;
  }

  KernelParamsBuilder& setBlockDim(unsigned int x = 1, unsigned int y = 1,
                                   unsigned int z = 1) {
    block_ = {x, y, z};
    return *this;
  }

  KernelParamsBuilder& setSharedMem(unsigned s) {
    shmem_ = s;
    return *this;
  }

  template <typename... Args>
  KernelParamsBuilder& setArgs(Args&... args) {
    static_assert((std::is_trivially_copyable_v<Args> && ...),
                  "All kernel args must be trivially copyable");

    arg_ptrs_.clear();
    arg_ptrs_.reserve(sizeof...(Args));
    (arg_ptrs_.push_back(&args), ...);

    return *this;
  }

  template <std::size_t N>
  gcxx::KernelNodeParams<N> build() {
    // static_assert(N > 0, "Kernel must have at least one argument");
    assert(arg_ptrs_.size() == N);
    assert(kernel_ != nullptr);

    std::array<void*, N> final_args{};
    std::copy_n(arg_ptrs_.begin(), N, final_args.begin());

    return KernelNodeParams<N>(kernel_, grid_, block_, shmem_, final_args);
  }
};

GCXX_NAMESPACE_DETAILS_END

details_::KernelParamsBuilder KernelParamsBuilder() {
  return details_::KernelParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END


#endif