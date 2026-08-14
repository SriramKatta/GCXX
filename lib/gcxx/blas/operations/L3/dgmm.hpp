// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_DGMM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_DGMM_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/side.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Diagonal matrix-matrix product C = diag(x) * A (left) or C = A * diag(x)
// (right).
//
// A and C are rank-2 mdspans, x is a rank-1 mdspan. The matrix dimensions and
// leading dimensions are inferred from the mdspan metadata; the side is given
// by the gcxx::blas::left / gcxx::blas::right tag, keeping the raw backend
// side-mode enum out of the public API. With the left tag x scales rows of A
// (x has length m); with the right tag x scales columns of A (x has length
// n).
//
// Example:
//   gcxx::blas::dgmm(h, gcxx::blas::left, A, x, C);    // C = diag(x) * A
//   gcxx::blas::dgmm(h, gcxx::blas::right, A, x, C);   // C = A * diag(x)
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasSdgmm_64 (Ddgmm_64) entry
// point, while all other index_types use the standard 32-bit interface.
//
// A, x, and C must be device views: mdspans carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::device_mdspan, gcxx::make_device_vector).
// Host views are rejected at compile time; in check builds the data handles
// are additionally probed at run time so a mislabeled host pointer fails
// here, not inside the GPU kernel.
template <class Side, class A, class X, class C>
auto dgmm(BlasHandleView h, Side side, const A& a, const X& x, C&& c) -> void {

  // local alias for easier refrence
  using A_t = std::decay_t<A>;
  using X_t = std::decay_t<X>;
  using C_t = std::decay_t<C>;
  using AVt = typename A_t::element_type;
  using XVt = typename X_t::element_type;
  using CVt = typename C_t::element_type;
  using AIt = typename A_t::index_type;
  using XIt = typename X_t::index_type;
  using CIt = typename C_t::index_type;

  // static asserts to verify no funny business
  static_assert(A_t::rank() == 2 && X_t::rank() == 1 && C_t::rank() == 2,
                "dgmm operands must be rank-2 (A, C) and rank-1 (x) mdspans");

  static_assert(gcxx::details_::all_same_v<AIt, XIt, CIt>,
                "dgmm operands A, x, C must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, XVt, CVt>,
                "dgmm operands A, x, C must share a single element type");

  // TODO: support complex element types via cublasCdgmm / cublasZdgmm. The
  //       dispatch macro below only handles float and double; a
  //       std::complex<T> element type hits this assert and must be wired up
  //       (add Cdgmm/Zdgmm branches to GCXX_BLAS_DISPATCH_TYPED).
  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "dgmm currently supports only float/double element types "
                "(complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [m, n, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_x, inc_x]     = details_::infer_blas_vector_view(x);
  const auto [m_c, n_c, ld_c, op_c] = details_::infer_blas_matrix_view(c);

  // unused vars just to supress annoying warnings
  (void)side;
  (void)m_c;
  (void)n_c;
  (void)op_a;
  (void)op_c;

  // the diagonal vector must match the scaled extent for the chosen side
  constexpr driver::deviceBlasSideMode_t mode = details_::side_mode_v<Side>;
  if (mode == driver::deviceBlasSideLeft && len_x != m) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "dgmm left side requires x length == rows of A");
  }
  if (mode == driver::deviceBlasSideRight && len_x != n) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "dgmm right side requires x length == cols of A");
  }

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, dgmm, h.getRawHandle(), mode, m,
                           n, a.data_handle(), ld_a, x.data_handle(), inc_x,
                           c.data_handle(), ld_c);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "dgmm failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
