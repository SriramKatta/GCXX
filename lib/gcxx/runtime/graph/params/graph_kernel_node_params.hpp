#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_KERNEL_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_KERNEL_NODE_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/details/type_traits.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceKernelNodeParams_t = GCXX_RUNTIME_BACKEND(KernelNodeParams);

// KernelNodeParamsView is a non-owning view. It assumes that
// params_.kernelParams (and any memory it points to) remains valid for the
// lifetime of the view. No mutation is allowed through this interface.
class KernelNodeParamsView {
 protected:
  deviceKernelNodeParams_t params_{}; // NOLINT

 public:
  GCXX_FHC auto getRawParams() const -> const deviceKernelNodeParams_t& {
    return params_;
  }

  GCXX_FHC auto getFunc() const -> void* { return params_.func; }

  GCXX_FHC auto getGridDim() const -> dim3 { return params_.gridDim; }

  GCXX_FHC auto getBlockDim() const -> dim3 { return params_.blockDim; }

  GCXX_FHC auto getSharedMemBytes() const -> unsigned int {
    return params_.sharedMemBytes;
  }

  GCXX_FHC auto getKernelParams() const -> void* const* {
    return params_.kernelParams;
  }

  GCXX_FHC auto getExtraArgs() const -> void* { return params_.extra; }
};

template <std::size_t NumParams>
class KernelNodeParams : public KernelNodeParamsView {
 private:
  std::array<void*, NumParams> kernelargs_{};

 public:
  template <typename... Args>
  GCXX_FHC KernelNodeParams(void* func, dim3 grid, dim3 block,
                            unsigned int shmem, Args&... args) {
    static_assert(sizeof...(args) == NumParams, "Arg count mismatch!");

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
  GCXX_FHC KernelNodeParams(void* func, dim3 grid, dim3 block,
                            unsigned int shmem,
                            std::array<void*, NumParams> arg_ptrs)
      : kernelargs_(arg_ptrs) {
    params_.func           = func;
    params_.gridDim        = grid;
    params_.blockDim       = block;
    params_.sharedMemBytes = shmem;
    params_.kernelParams   = kernelargs_.data();
    params_.extra          = nullptr;
  }

  // Disable move/copy to ensure params_.kernelParams never points to an old
  // stack location
  KernelNodeParams(const KernelNodeParams&) = delete;
  KernelNodeParams(KernelNodeParams&&)      = delete;

  KernelNodeParams operator=(const KernelNodeParams&) = delete;
  KernelNodeParams operator=(KernelNodeParams&&)      = delete;

  ~KernelNodeParams() = default;
};

GCXX_NAMESPACE_DETAILS_BEGIN

template <typename... Args>
class KernelArgPack {
  static_assert((std::is_trivially_copyable_v<Args> && ...),
                "All kernel arguments must be trivially copyable");

 public:
  std::tuple<Args&...> args;
  std::array<void*, sizeof...(Args)> ptrs{};

  GCXX_FHC KernelArgPack(Args&... a) : args(a...) {
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
  std::vector<void*> arg_ptrs_{};

 public:
  GCXX_FH static auto create() -> KernelParamsBuilder { return {}; }

  template <typename Kernel>
  GCXX_FHC auto setKernel(Kernel k) -> KernelParamsBuilder& {
    static_assert(details_::is_void_function_pointer<Kernel>::value,
                  "Passed value must be a function pointer and function should "
                  "return void");
    kernel_ = reinterpret_cast<void*>(k); // NOLINT
    return *this;
  }

  GCXX_FHC auto setGridDim(dim3 g) -> KernelParamsBuilder& {
    grid_ = g;
    return *this;
  }

  GCXX_FHC auto setGridDim(unsigned int x = 1, unsigned int y = 1,
                           unsigned int z = 1) -> KernelParamsBuilder& {

    return setGridDim({x, y, z});
  }

  GCXX_FHC auto setBlockDim(dim3 b) -> KernelParamsBuilder& {
    block_ = b;
    return *this;
  }

  GCXX_FHC auto setBlockDim(unsigned int x = 1, unsigned int y = 1,
                            unsigned int z = 1) -> KernelParamsBuilder& {
    return setBlockDim({x, y, z});
  }

  GCXX_FHC auto setSharedMemBytes(unsigned s) -> KernelParamsBuilder& {
    shmem_ = s;
    return *this;
  }

  template <typename VT>
  GCXX_FHC auto setSharedMem(std::size_t numElems) -> KernelParamsBuilder& {
    return setSharedMemBytes(numElems * sizeof(VT));
  }

  GCXX_FHC auto addSharedMemBytes(unsigned s) -> KernelParamsBuilder& {
    shmem_ += s;
    return *this;
  }

  template <typename VT>
  GCXX_FHC auto addSharedMem(std::size_t numElems) -> KernelParamsBuilder& {
    return addSharedMemBytes(numElems * sizeof(VT));
  }

  template <typename... Args>
  GCXX_FHC auto setArgs(Args&... args) -> KernelParamsBuilder& {
    static_assert((std::is_trivially_copyable_v<Args> && ...),
                  "All kernel args must be trivially copyable");

    arg_ptrs_.clear();
    arg_ptrs_.reserve(sizeof...(Args));
    (arg_ptrs_.push_back(&args), ...);

    return *this;
  }

  template <std::size_t NumParams>
  GCXX_FHC gcxx::KernelNodeParams<NumParams> build() {
    // static_assert(NumParams > 0, "Kernel must have at least one argument");
    assert(arg_ptrs_.size() == NumParams);
    assert(kernel_ != nullptr);

    std::array<void*, NumParams> final_args{};
    std::copy_n(arg_ptrs_.begin(), NumParams, final_args.begin());

    return KernelNodeParams<NumParams>(kernel_, grid_, block_, shmem_,
                                       final_args);
  }
};

GCXX_NAMESPACE_DETAILS_END

// an helper to simply while using it
GCXX_FH auto KernelParamsBuilder() -> details_::KernelParamsBuilder {
  return details_::KernelParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END


#endif