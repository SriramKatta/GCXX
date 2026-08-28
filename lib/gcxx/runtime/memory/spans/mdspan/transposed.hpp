// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_TRANSPOSED_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_TRANSPOSED_HPP_

#include <array>
#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// Metafunction: swap the (compile-time) dimensions of a rank-2 extents.
template <class Extents>
struct transposed_extents {
  static_assert(
    Extents::rank() == 2,
    "transposed_extents is only supported for rank-2 (matrix) extents");
};

template <class IndexType, std::size_t E0, std::size_t E1>
struct transposed_extents<gcxx::extents<IndexType, E0, E1>> {
  using type = gcxx::extents<IndexType, E1, E0>;
};

template <class Extents>
using transposed_extents_t = typename transposed_extents<Extents>::type;

// Zero-cost A^T view; blas infers transpose from the swapped mapping.
GCXX_TEMPLATE(class T, class Extents, class Layout, class Accessor)
GCXX_REQUIRES(Extents::rank() == 2)
constexpr auto transposed(const gcxx::mdspan<T, Extents, Layout, Accessor>& v) {
  using new_extents_t = transposed_extents_t<Extents>;
  new_extents_t new_extents(v.extent(1), v.extent(0));

  // The mapping constructors below spell out their template argument instead
  // of relying on CTAD: the vendored mdspan's constructors are __host__
  // __device__ (MDSPAN_IMPL_HAS_CUDA/HIP), and nvcc's front-end forms no
  // implicit deduction guides from __host__ __device__ constructors.
  //
  // The first two arms differ textually only in the mapping type spelled; in
  // this uninstantiated template the checker cannot see that each arm builds
  // a distinct mapping<extents> per instantiation, so it flags a clone.
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if constexpr (std::is_same_v<Layout, gcxx::layout_left>) {
    // column-major (e0,e1) -> row-major (e1,e0)
    return gcxx::mdspan(v.data_handle(),
                        gcxx::layout_right::mapping<new_extents_t>(new_extents),
                        v.accessor());
  } else if constexpr (std::is_same_v<Layout, gcxx::layout_right>) {
    // row-major (e0,e1) -> column-major (e1,e0)
    return gcxx::mdspan(v.data_handle(),
                        gcxx::layout_left::mapping<new_extents_t>(new_extents),
                        v.accessor());
  } else {
    // Padded / strided inputs: explicit swapped-stride mapping.
    std::array<typename Extents::index_type, 2> new_strides{
      v.mapping().stride(1), v.mapping().stride(0)};
    return gcxx::mdspan(
      v.data_handle(),
      gcxx::layout_stride::mapping<new_extents_t>(new_extents, new_strides),
      v.accessor());
  }
}

GCXX_NAMESPACE_MAIN_END()

#endif
