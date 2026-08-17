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
// n). Each operand is typed as a gcxx::mdspan in the signature, so wrong-rank
// (or non-mdspan) arguments fail overload resolution.
//
// A and C must share storage orientation (both column-major-like or both
// row-major-like): the backend's dgmm entry point takes no transpose flags,
// so it cannot pair, say, a row-major A with a column-major C. Both
// same-orientation cases work — a row-major-like pair is dispatched as the
// transposed problem with the side mode flipped — and mixed orientations are
// rejected with a clear error.
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
GCXX_TEMPLATE(class Side, class TA, class ExtentsA, class LayoutA,
              class AccessorA, class TX, class ExtentsX, class LayoutX,
              class AccessorX, class TC, class ExtentsC, class LayoutC,
              class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsC::rank() == 2)
auto dgmm(BlasHandleView h, Side side,
          const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
          const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
          const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using XVt = TX;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using XIt = typename ExtentsX::index_type;
  using CIt = typename ExtentsC::index_type;

  // static asserts to verify no funny business
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
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "dgmm currently supports only f32_t/f64_t element types "
                "(complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [m, n, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_x, inc_x]     = details_::infer_blas_vector_view(x);
  const auto out                = details_::infer_blas_output_view(c);

  // the tag object itself is unused (the mode comes from its type)
  (void)side;

  // the diagonal vector must match the scaled extent for the chosen side
  constexpr driver::deviceBlasSideMode_t mode = details_::side_mode_v<Side>;
  if (mode == driver::deviceBlasSideLeft && len_x != m) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "dgmm left side requires x length == rows of A");
  }
  if (mode == driver::deviceBlasSideRight && len_x != n) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "dgmm right side requires x length == cols of A");
  }
  if (out.rows != m || out.cols != n) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "dgmm requires C to have the same extents as A");
  }
  // cublasDdgmm takes no transpose flags: A and C are read/written
  // column-major as given, so their orientations must match.
  if ((op_a == driver::deviceBlasOpN) != !out.transposed) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "dgmm requires A and C to share storage orientation: the backend entry "
      "point takes no transpose flags, so a column-major-like A must pair "
      "with a column-major-like C and a row-major-like A with a row-major-"
      "like C");
  }

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    // A and C column-major-like: the problem passes through as declared.
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, dgmm, h.getRawHandle(), mode, m,
                             n, a.data_handle(), ld_a, x.data_handle(), inc_x,
                             c.data_handle(), out.leading_dimension);
  } else {
    // A and C row-major-like: present the transposed problem to the
    // column-major backend. (diag(x) * A)^T = A^T * diag(x) and
    // (A * diag(x))^T = diag(x) * A^T, so the side mode flips along with the
    // swapped m/n; reading the same storage column-major yields the
    // transposes, and the leading dimensions carry over unchanged.
    constexpr driver::deviceBlasSideMode_t flipped_mode =
      mode == driver::deviceBlasSideLeft ? driver::deviceBlasSideRight
                                         : driver::deviceBlasSideLeft;
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, dgmm, h.getRawHandle(),
                             flipped_mode, n, m, a.data_handle(), ld_a,
                             x.data_handle(), inc_x, c.data_handle(),
                             out.leading_dimension);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "dgmm failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
