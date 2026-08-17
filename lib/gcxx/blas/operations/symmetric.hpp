// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_SYMMETRIC_HPP_
#define GCXX_BLAS_OPERATIONS_SYMMETRIC_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Tag types selecting which triangle of a symmetric matrix is stored (the
// uplo argument of the symmetric/hermitian BLAS routines). Passing the tag
// objects gcxx::blas::upper / gcxx::blas::lower keeps the raw backend
// fill-mode enum out of the public API, mirroring how gcxx::blas::left /
// gcxx::blas::right hide the side-mode enum (side.hpp).
struct upper_t {};
struct lower_t {};

inline constexpr upper_t upper{};
inline constexpr lower_t lower{};

// symmetric_view<MD, Fill> is defined after the fill-mode machinery in
// details_ below; forward-declared here so those traits can name it.
template <class MD, class Fill>
struct symmetric_view;

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// The two accepted fill-mode tags (compile-time uplo).
template <class>
GCXX_CXPR inline bool is_fill_mode_tag_v = false;

template <>
GCXX_CXPR inline bool is_fill_mode_tag_v<gcxx::blas::upper_t> = true;

template <>
GCXX_CXPR inline bool is_fill_mode_tag_v<gcxx::blas::lower_t> = true;

// Compile-time triangle-tag -> backend fill-mode mapping, so the driver enum
// is touched in exactly one place and an argument that is not one of the two
// tags fails to compile at the call site.
template <class Fill>
struct fill_mode {
  static_assert(gcxx::details_::is_always_false_v<Fill>,
                "BLAS fill-mode argument must be gcxx::blas::upper or "
                "gcxx::blas::lower");
};

template <>
struct fill_mode<gcxx::blas::upper_t> {
  static constexpr driver::deviceBlasFillMode_t value =
    driver::deviceBlasFillModeUpper;
};

template <>
struct fill_mode<gcxx::blas::lower_t> {
  static constexpr driver::deviceBlasFillMode_t value =
    driver::deviceBlasFillModeLower;
};

template <class Fill>
inline constexpr driver::deviceBlasFillMode_t fill_mode_v =
  fill_mode<Fill>::value;

// symmetric_view detection, so the symmetric routines can resolve side from
// WHICH operand is wrapped (wrapped first operand -> left, second -> right).
template <class>
GCXX_CXPR inline bool is_symmetric_view_v = false;

template <class MD, class Fill>
GCXX_CXPR inline bool
  is_symmetric_view_v<gcxx::blas::symmetric_view<MD, Fill>> = true;

// Fill tag carried by a symmetric_view, or void for any other type.
template <class T>
struct symmetric_fill_mode {
  using type = void;
};

template <class MD, class Fill>
struct symmetric_fill_mode<gcxx::blas::symmetric_view<MD, Fill>> {
  using type = Fill;
};

template <class T>
using symmetric_fill_mode_t = typename symmetric_fill_mode<T>::type;

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Operand annotation marking a matrix as symmetric with only the upper or
// lower triangle stored: symmetric(A, upper) wraps the mdspan itself (NOT
// its accessor) in an extra passable type, mirroring std::linalg's
// basic_symmetric_matrix. The triangle is part of the type, so the symmetric
// routines resolve uplo at compile time, and side from which operand carries
// the wrapper. Composes with the other view helpers.
//
// Example:
//   gcxx::blas::symm(h, 1.0, gcxx::blas::symmetric(A, gcxx::blas::upper),
//                    B, 0.0, C);  // C = A * B, A symmetric (upper stored)
template <class MD, class Fill>
struct symmetric_view {
  static_assert(MD::rank() == 2, "symmetric operands must be rank-2");
  static_assert(details_::is_fill_mode_tag_v<Fill>,
                "symmetric fill mode must be gcxx::blas::upper or "
                "gcxx::blas::lower");

  using mdspan_type    = MD;
  using fill_mode_type = Fill;

  // Forwarded so BLAS helpers can query the wrapper without unwrapping.
  using element_type  = typename MD::element_type;
  using index_type    = typename MD::index_type;
  using accessor_type = typename MD::accessor_type;
  static constexpr std::size_t rank = MD::rank();

  // The wrapped mdspan, stored by value (an mdspan is a cheap view: data
  // handle + mapping + accessor).
  MD view;

  // Unwrap back to the plain mdspan (the view helpers and the infer_*
  // machinery in op_inference.hpp operate on this).
  constexpr MD base() const { return view; }

  constexpr auto data_handle() const { return view.data_handle(); }
  constexpr auto mapping() const { return view.mapping(); }
  constexpr auto accessor() const { return view.accessor(); }
};

// symmetric(v, upper|lower) -> symmetric_view<typeof(v), upper_t|lower_t>
GCXX_TEMPLATE(class T, class Extents, class Layout, class Accessor, class Fill)
GCXX_REQUIRES(Extents::rank() == 2)
constexpr auto symmetric(const gcxx::mdspan<T, Extents, Layout, Accessor>& v,
                         Fill) {
  using mdspan_t = gcxx::mdspan<T, Extents, Layout, Accessor>;
  return symmetric_view<mdspan_t, Fill>{v};
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
