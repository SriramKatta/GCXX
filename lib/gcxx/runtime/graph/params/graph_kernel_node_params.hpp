// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_KERNEL_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_KERNEL_NODE_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

// Type-state builder: each setter returns a NEW builder whose tag list
// records the call; build() static_asserts the required tags. Missing or
// duplicate calls are compile-time errors.
//
//   gcxx::KernelParamsBuilder()
//       .setKernel(kern)            // required, exactly once
//       .setGridDim(128)            // required, exactly once (y/z default 1)
//       .setBlockDim(64)            // required, exactly once (y/z default 1)
//       .setSharedMem<float>(256)   // optional
//       .setArgs(a, b)              // optional; arg count becomes part of
//       .build();                   // the type — build() infers it

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

// Argument-pointer storage, inherited BEFORE KernelNodeParamsView so the raw
// params can point into it safely.
template <std::size_t NumParams>
struct KernelArgPtrStore {
  std::array<void*, NumParams> ptrs{};  // NOLINT
};

GCXX_NAMESPACE_DETAILS_END()

// Non-owning view; kernelParams memory must outlive the view.
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
    return static_cast<void* const*>(m_params.kernelParams);
  }

  GCXX_FHC auto getExtraArgs() const -> void** { return m_params.extra; }

 protected:
  GCXX_FH KernelNodeParamsView() = default;

  explicit KernelNodeParamsView(deviceKernelNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceKernelNodeParams_t m_params{};  // NOLINT
};

template <std::size_t NumParams>
class KernelNodeParams : private details_::KernelArgPtrStore<NumParams>,
                         public KernelNodeParamsView {
 public:
  template <typename... Args>
  GCXX_FHC KernelNodeParams(void* func, dim3 grid, dim3 block,
                            unsigned int shmem, Args&... args)
      : KernelNodeParams{func, grid, block, shmem, make_arg_array(args...)} {}

  GCXX_FHC KernelNodeParams(void* func, dim3 grid, dim3 block,
                            unsigned int shmem,
                            std::array<void*, NumParams> arg_ptrs)
      : details_::KernelArgPtrStore<NumParams>{std::move(arg_ptrs)},
        KernelNodeParamsView{make_raw_params(
          func, grid, block, shmem,
          details_::KernelArgPtrStore<NumParams>::ptrs.data())} {}

  // Non-copyable/non-movable so kernelParams cannot dangle to an old object.
  KernelNodeParams(const KernelNodeParams&) = delete;
  KernelNodeParams(KernelNodeParams&&)      = delete;

  auto operator=(const KernelNodeParams&) -> KernelNodeParams& = delete;
  auto operator=(KernelNodeParams&&) -> KernelNodeParams&      = delete;

  ~KernelNodeParams() = default;

 private:
  template <typename... Args>
  GCXX_FHC static auto make_arg_array(Args&... args)
    -> std::array<void*, NumParams> {
    static_assert(sizeof...(args) == NumParams, "Arg count mismatch!");
    std::array<void*, NumParams> ptrs{};
    std::size_t i = 0;
    ((i < NumParams ? static_cast<void>(ptrs[i++] = &args) : void()), ...);
    return ptrs;
  }

  GCXX_FHC static auto make_raw_params(void* func, dim3 grid, dim3 block,
                                       unsigned int shmem, void** kernelParams)
    -> deviceKernelNodeParams_t {
    // Field assignment, not positional aggregate init: the field order of
    // hipKernelNodeParams differs from cudaKernelNodeParams.
    deviceKernelNodeParams_t p{};
    p.func           = func;
    p.gridDim        = grid;
    p.blockDim       = block;
    p.sharedMemBytes = shmem;
    p.kernelParams   = kernelParams;
    p.extra          = nullptr;
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace kernel_params_builder {

  struct kernel_tag {};
  struct grid_tag {};
  struct block_tag {};
  struct args_tag_base {};

  template <std::size_t NumArgs>
  struct args_tag : args_tag_base {};

  template <typename... Ts>
  inline constexpr bool has_args_v =
    (std::is_base_of_v<args_tag_base, Ts> || ...);

  template <typename T>
  struct args_count_of : std::integral_constant<std::size_t, 0> {};

  template <std::size_t NumArgs>
  struct args_count_of<args_tag<NumArgs>>
      : std::integral_constant<std::size_t, NumArgs> {};

  template <typename... Ts>
  inline constexpr std::size_t num_args_v =
    (std::size_t{0} + ... + args_count_of<Ts>::value);

}  // namespace kernel_params_builder

namespace kpb = kernel_params_builder;

template <typename... Set>
class KernelParamsBuilder {
 public:
  KernelParamsBuilder() = default;

  template <typename Kernel>
  GCXX_FHC auto setKernel(Kernel k) const
    -> KernelParamsBuilder<Set..., kpb::kernel_tag> {
    static_assert(!details_::contains_v<kpb::kernel_tag, Set...>,
                  "setKernel() may only be called once");
    static_assert(details_::is_void_function_pointer<Kernel>::value,
                  "Passed value must be a function pointer and function should "
                  "return void");
    KernelParamsBuilder<Set..., kpb::kernel_tag> next = *this;
    next.m_params.func = reinterpret_cast<void*>(k);  // NOLINT
    return next;
  }

  GCXX_FHC auto setGridDim(dim3 g) const
    -> KernelParamsBuilder<Set..., kpb::grid_tag> {
    static_assert(!details_::contains_v<kpb::grid_tag, Set...>,
                  "setGridDim() may only be called once");
    KernelParamsBuilder<Set..., kpb::grid_tag> next = *this;
    next.m_params.gridDim                           = g;
    return next;
  }

  // x explicit; y/z default to 1.
  GCXX_FHC auto setGridDim(unsigned int x, unsigned int y = 1,
                           unsigned int z = 1) const
    -> KernelParamsBuilder<Set..., kpb::grid_tag> {
    return setGridDim(dim3{x, y, z});
  }

  GCXX_FHC auto setBlockDim(dim3 b) const
    -> KernelParamsBuilder<Set..., kpb::block_tag> {
    static_assert(!details_::contains_v<kpb::block_tag, Set...>,
                  "setBlockDim() may only be called once");
    KernelParamsBuilder<Set..., kpb::block_tag> next = *this;
    next.m_params.blockDim                           = b;
    return next;
  }

  // x explicit; y/z default to 1.
  GCXX_FHC auto setBlockDim(unsigned int x, unsigned int y = 1,
                            unsigned int z = 1) const
    -> KernelParamsBuilder<Set..., kpb::block_tag> {
    return setBlockDim(dim3{x, y, z});
  }

  GCXX_FHC auto setSharedMemBytes(unsigned s) const -> KernelParamsBuilder {
    KernelParamsBuilder next     = *this;
    next.m_params.sharedMemBytes = s;
    return next;
  }

  template <typename VT>
  GCXX_FHC auto setSharedMem(std::size_t numElems) const
    -> KernelParamsBuilder {
    return setSharedMemBytes(numElems * sizeof(VT));
  }

  GCXX_FHC auto addSharedMemBytes(unsigned s) const -> KernelParamsBuilder {
    KernelParamsBuilder next = *this;
    next.m_params.sharedMemBytes += s;
    return next;
  }

  template <typename VT>
  GCXX_FHC auto addSharedMem(std::size_t numElems) const
    -> KernelParamsBuilder {
    return addSharedMemBytes(numElems * sizeof(VT));
  }

  template <typename... Args>
  // The static_asserts force every Args to be an lvalue reference, so a
  // forward would be a no-op; only the args' addresses are taken.
  GCXX_FHC auto setArgs(
    Args&&... args) const  // NOLINT(cppcoreguidelines-missing-std-forward)
    -> KernelParamsBuilder<Set..., kpb::args_tag<sizeof...(Args)>> {
    static_assert(!kpb::has_args_v<Set...>,
                  "setArgs() may only be called once; use addArgs() to append");
    static_assert((std::is_lvalue_reference_v<Args> && ...),
                  "Kernel arguments must be lvalues: the graph node stores "
                  "pointers into the caller's storage");
    static_assert(
      (std::is_trivially_copyable_v<std::remove_reference_t<Args>> && ...),
      "All kernel args must be trivially copyable");
    KernelParamsBuilder<Set..., kpb::args_tag<sizeof...(Args)>> next = *this;
    next.m_arg_ptrs.clear();
    next.m_arg_ptrs.reserve(sizeof...(Args));
    // The CUDA/HIP kernel-node API takes void* slots for the arg addresses;
    // the builder only reads through them, never writes.
    (next.m_arg_ptrs.push_back(
       const_cast<void*>(  // NOLINT(cppcoreguidelines-pro-type-const-cast)
         static_cast<const void*>(std::addressof(args)))),
     ...);
    return next;
  }

  template <typename... Args>
  // See setArgs(): lvalue-only by contract, forward is a no-op.
  GCXX_FHC auto addArgs(
    Args&&... args) const  // NOLINT(cppcoreguidelines-missing-std-forward)
    -> KernelParamsBuilder<Set..., kpb::args_tag<sizeof...(Args)>> {
    static_assert((std::is_lvalue_reference_v<Args> && ...),
                  "Kernel arguments must be lvalues: the graph node stores "
                  "pointers into the caller's storage");
    static_assert(
      (std::is_trivially_copyable_v<std::remove_reference_t<Args>> && ...),
      "All kernel args must be trivially copyable");
    KernelParamsBuilder<Set..., kpb::args_tag<sizeof...(Args)>> next = *this;
    // The CUDA/HIP kernel-node API takes void* slots for the arg addresses;
    // the builder only reads through them, never writes.
    (next.m_arg_ptrs.push_back(
       const_cast<void*>(  // NOLINT(cppcoreguidelines-pro-type-const-cast)
         static_cast<const void*>(std::addressof(args)))),
     ...);
    return next;
  }

  // Parameter count deduced from setArgs()/addArgs().
  GCXX_FHC auto build() const
    -> gcxx::KernelNodeParams<kpb::num_args_v<Set...>> {
    static_assert(details_::contains_v<kpb::kernel_tag, Set...>,
                  "setKernel() required before build()");
    static_assert(details_::contains_v<kpb::grid_tag, Set...>,
                  "setGridDim() required before build()");
    static_assert(details_::contains_v<kpb::block_tag, Set...>,
                  "setBlockDim() required before build()");
    constexpr std::size_t num_params = kpb::num_args_v<Set...>;
    std::array<void*, num_params> final_args{};
    std::copy_n(m_arg_ptrs.begin(), num_params, final_args.begin());
    return gcxx::KernelNodeParams<num_params>(
      m_params.func, m_params.gridDim, m_params.blockDim,
      m_params.sharedMemBytes, final_args);
  }

 private:
  template <typename...>
  friend class KernelParamsBuilder;  // states construct each other

  // Field assignment, not positional aggregate init: the field order of
  // hipKernelNodeParams differs from cudaKernelNodeParams.
  GCXX_FHC static auto default_raw_params()
    -> KernelNodeParamsView::deviceKernelNodeParams_t {
    KernelNodeParamsView::deviceKernelNodeParams_t p{};
    p.gridDim  = dim3{1, 1, 1};
    p.blockDim = dim3{1, 1, 1};
    return p;
  }

  // State hand-off between builder states.
  template <typename... Other>
  KernelParamsBuilder(const KernelParamsBuilder<Other...>& other)
      : m_params{other.m_params}, m_arg_ptrs{other.m_arg_ptrs} {}

  // Scalars accumulate directly in the driver struct. kernelParams stays null
  // until build(): it must point into the FINAL object's storage, and copied
  // intermediate builders would otherwise carry pointers into another
  // instance's vector.
  KernelNodeParamsView::deviceKernelNodeParams_t m_params{
    default_raw_params()};          // NOLINT
  std::vector<void*> m_arg_ptrs{};  // NOLINT
};

GCXX_NAMESPACE_DETAILS_END()

using KernelParamsBuilder_t = details_::KernelParamsBuilder<>;

GCXX_FH auto KernelParamsBuilder() -> KernelParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
