// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_KERNEL_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_KERNEL_NODE_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>


// TODO : use type safe builder patter to make this robust
// https://github.com/CppCon/CppCon2025/blob/main/Lightning%20Talks/The_Type_Safe_Builder_Design_Pattern.pdf

GCXX_NAMESPACE_MAIN_BEGIN()

// KernelNodeParamsView is a non-owning view. It assumes that
// params_.kernelParams (and any memory it points to) remains valid for the
// lifetime of the view. No mutation is allowed through this interface.
class KernelNodeParamsView {
 public:
  using deviceKernelNodeParams_t = driver::deviceKernelNodeParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceKernelNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getFunc() const -> void* { return m_params.func; }

  GCXX_FHC auto getGridDim() const -> dim3 { return m_params.gridDim; }

  GCXX_FHC auto getBlockDim() const -> dim3 { return m_params.blockDim; }

  GCXX_FHC auto getSharedMemBytes() const -> unsigned int {
    return m_params.sharedMemBytes;
  }

  GCXX_FHC auto getKernelParams() const -> void* const* {
    return m_params.kernelParams;
  }

  GCXX_FHC auto getExtraArgs() const -> void* { return m_params.extra; }

 protected:
  deviceKernelNodeParams_t m_params{};  // NOLINT
};

template <std::size_t NumParams>
class KernelNodeParams : public KernelNodeParamsView {
 public:
  template <typename... Args>
  GCXX_FHC KernelNodeParams(void* func, dim3 grid, dim3 block,
                            unsigned int shmem, Args&... args) {
    static_assert(sizeof...(args) == NumParams, "Arg count mismatch!");

    std::size_t i = 0;
    ((m_kernelargs[i++] = static_cast<void*>(&args)), ...);

    m_params.func           = func;
    m_params.gridDim        = grid;
    m_params.blockDim       = block;
    m_params.sharedMemBytes = shmem;
    m_params.kernelParams   = m_kernelargs.data();  // Points to sibling member
    m_params.extra          = nullptr;
  }

  // Constructor from pre-built array of void* pointers (used by builder)
  GCXX_FHC
  KernelNodeParams(void* func, dim3 grid, dim3 block, unsigned int shmem,
                   std::array<void*, NumParams> arg_ptrs)
      : m_kernelargs(arg_ptrs) {
    m_params.func           = func;
    m_params.gridDim        = grid;
    m_params.blockDim       = block;
    m_params.sharedMemBytes = shmem;
    m_params.kernelParams   = m_kernelargs.data();
    m_params.extra          = nullptr;
  }

  // Disable move/copy to ensure params_.kernelParams never points to an old
  // stack location
  KernelNodeParams(const KernelNodeParams&) = delete;
  KernelNodeParams(KernelNodeParams&&)      = delete;

  KernelNodeParams operator=(const KernelNodeParams&) = delete;
  KernelNodeParams operator=(KernelNodeParams&&)      = delete;

  ~KernelNodeParams() = default;

 private:
  std::array<void*, NumParams> m_kernelargs{};
};

GCXX_NAMESPACE_DETAILS_BEGIN()

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
 public:
  GCXX_FH static auto create() -> KernelParamsBuilder { return {}; }

  template <typename Kernel>
  GCXX_FHC auto setKernel(Kernel k) -> KernelParamsBuilder& {
    static_assert(details_::is_void_function_pointer<Kernel>::value,
                  "Passed value must be a function pointer and function should "
                  "return void");
    m_kernel = reinterpret_cast<void*>(k);  // NOLINT
    return *this;
  }

  GCXX_FHC auto setGridDim(dim3 g) -> KernelParamsBuilder& {
    m_grid = g;
    return *this;
  }

  GCXX_FHC
  auto setGridDim(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1)
    -> KernelParamsBuilder& {

    return setGridDim({x, y, z});
  }

  GCXX_FHC auto setBlockDim(dim3 b) -> KernelParamsBuilder& {
    m_block = b;
    return *this;
  }

  GCXX_FHC
  auto setBlockDim(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1)
    -> KernelParamsBuilder& {
    return setBlockDim({x, y, z});
  }

  GCXX_FHC auto setSharedMemBytes(unsigned s) -> KernelParamsBuilder& {
    m_shmem = s;
    return *this;
  }

  template <typename VT>
  GCXX_FHC auto setSharedMem(std::size_t numElems) -> KernelParamsBuilder& {
    return setSharedMemBytes(numElems * sizeof(VT));
  }

  GCXX_FHC auto addSharedMemBytes(unsigned s) -> KernelParamsBuilder& {
    m_shmem += s;
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

    m_arg_ptrs.clear();
    m_arg_ptrs.reserve(sizeof...(Args));
    (m_arg_ptrs.push_back(&args), ...);

    return *this;
  }

  template <std::size_t NumParams>
  GCXX_FHC gcxx::KernelNodeParams<NumParams> build() {
    GCXX_RUNTIME_EXPECT(m_arg_ptrs.size() == NumParams,
                        "Mismatch between m_arg_ptrs.size() and Numparams");
    GCXX_RUNTIME_EXPECT(m_kernel != nullptr, "Need to set the kernel");

    std::array<void*, NumParams> final_args{};
    std::copy_n(m_arg_ptrs.begin(), NumParams, final_args.begin());

    return KernelNodeParams<NumParams>(m_kernel, m_grid, m_block, m_shmem,
                                       final_args);
  }

 private:
  void* m_kernel{nullptr};
  dim3 m_grid{1, 1, 1};
  dim3 m_block{1, 1, 1};
  unsigned m_shmem{0};
  std::vector<void*> m_arg_ptrs{};
};

GCXX_NAMESPACE_DETAILS_END()

// an helper to simply while using it
GCXX_FH auto KernelParamsBuilder() -> details_::KernelParamsBuilder {
  return details_::KernelParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END()


#endif