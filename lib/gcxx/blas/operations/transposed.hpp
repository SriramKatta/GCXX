// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_TRANSPOSED_HPP_
#define GCXX_BLAS_OPERATIONS_TRANSPOSED_HPP_

#include <array>
#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

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

// transposed(v) returns a non-owning view of v with extents and strides
// swapped (P1673R13's transposed; renamed from gcxx::blas::transpose). The
// BLAS operations infer the transpose state from the view's mapping, so
// passing transposed(A) to a product computes with A^T at zero cost.
GCXX_TEMPLATE(class T, class Extents, class Layout, class Accessor)
GCXX_REQUIRES(Extents::rank() == 2)
constexpr auto transposed(const gcxx::mdspan<T, Extents, Layout, Accessor>& v) {
  using new_extents_t = transposed_extents_t<Extents>;
  new_extents_t new_extents(v.extent(1), v.extent(0));

  if constexpr (std::is_same_v<Layout, gcxx::layout_left>) {
    // column-major (e0,e1) -> row-major (e1,e0)
    return gcxx::mdspan(v.data_handle(),
                        gcxx::layout_right::mapping(new_extents), v.accessor());
  } else if constexpr (std::is_same_v<Layout, gcxx::layout_right>) {
    // row-major (e0,e1) -> column-major (e1,e0)
    return gcxx::mdspan(v.data_handle(),
                        gcxx::layout_left::mapping(new_extents), v.accessor());
  } else {
    // Padded / strided inputs: explicit swapped-stride mapping.
    std::array<typename Extents::index_type, 2> new_strides{
      v.mapping().stride(1), v.mapping().stride(0)};
    return gcxx::mdspan(v.data_handle(),
                        gcxx::layout_stride::mapping(new_extents, new_strides),
                        v.accessor());
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
