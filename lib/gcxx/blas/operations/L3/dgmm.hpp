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

// C = diag(x)*A or A*diag(x); A and C must share storage orientation.
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

  // TODO: Wire complex Cdgmm/Zdgmm into GCXX_BLAS_DISPATCH_TYPED.
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

  // The diagonal vector must match the scaled extent for the chosen side.
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
    // A and C row-major-like: transposed problem; side flips with m/n swap.
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
