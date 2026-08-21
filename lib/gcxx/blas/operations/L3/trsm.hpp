// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_TRSM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_TRSM_HPP_

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/L3/geam.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/diagonal.hpp>
#include <gcxx/blas/operations/side.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// X = A^-1*B or B*A^-1, staged into X (backend solves in place).
GCXX_TEMPLATE(class Side, class TA, class ExtentsA, class LayoutA,
              class AccessorA, class Tri, class Diag, class TB, class ExtentsB,
              class LayoutB, class AccessorB, class TX, class ExtentsX,
              class LayoutX, class AccessorX)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsX::rank() == 2)
auto triangular_matrix_matrix_solve(
  BlasHandleView h, Side /*side*/,
  const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a, Tri /*triangle*/,
  Diag /*diagonal_storage*/,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
  const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using BVt = TB;
  using XVt = TX;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using XIt = typename ExtentsX::index_type;
  using Sv  = XVt;

  // static asserts to verify no funny business
  static_assert(!gcxx::is_scaled_accessor_v<AccessorA>,
                "triangular_matrix_matrix_solve does not accept a scaled() "
                "view of A: scaling the operator does not scale the inverse");

  static_assert(!gcxx::is_scaled_accessor_v<AccessorX>,
                "triangular_matrix_matrix_solve outputs cannot be scaled() "
                "views; scale B instead");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, XIt>,
                "triangular_matrix_matrix_solve operands A, B, X must share "
                "the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, XVt>,
                "triangular_matrix_matrix_solve operands A, B, X must share "
                "a single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "triangular_matrix_matrix_solve currently supports only "
                "f32_t/f64_t element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(x, "X");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                          = details_::infer_blas_output_view(x);

  constexpr driver::deviceBlasSideMode_t side = details_::side_mode_v<Side>;

  // dimension gates per side (fail here rather than inside the backend)
  if (rows_a != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "triangular_matrix_matrix_solve requires A to "
                             "be square");
  }
  if (side == driver::deviceBlasSideLeft &&
      (rows_b != rows_a || out.rows != rows_a || out.cols != cols_b)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "triangular_matrix_matrix_solve (left) requires B to be A.extent(0) x N "
      "and X to be A.extent(0) x B.extent(1)");
  }
  if (side == driver::deviceBlasSideRight &&
      (cols_b != rows_a || out.rows != rows_b || out.cols != rows_a)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "triangular_matrix_matrix_solve (right) requires B to be M x "
      "A.extent(0) and X to be B.extent(0) x A.extent(0)");
  }
  // Staging copy is elementwise: B and X must share storage orientation.
  if ((op_b == driver::deviceBlasOpN) == out.transposed) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "triangular_matrix_matrix_solve requires B and X to share storage "
      "orientation: the staging copy is elementwise, so a mixed pair would "
      "transpose the data");
  }

  // Stage B into X (plain copy, or geam when B carries a scaled() factor).
  auto alpha_res = details_::resolve_scaled_alpha<Sv>(b.accessor());
  if (!alpha_res.from_device()) {
    if (alpha_res.host_value == Sv(1)) {
      const auto span_elems = std::min(b.mapping().required_span_size(),
                                       x.mapping().required_span_size());
      Copy(h.getStream(), x.data_handle(), b.data_handle(),
           static_cast<std::size_t>(span_elems));
    } else {
      geam(h, alpha_res.host_value, gcxx::strip_scaled(b), Sv(0), x, x);
    }
  } else {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "triangular_matrix_matrix_solve: a device_scalar scaled() factor on B "
      "is unsupported: the staging copy runs under host pointer mode and the "
      "in-place solve's factor is the host constant 1; use host factors");
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag = details_::fill_mode_v<Tri>;
  constexpr driver::deviceBlasDiagType_t diag = details_::diagonal_type_v<Diag>;

  // Row-major-like A reads as its transpose; mirrored triangle, flipped op.
  const auto uplo      = op_a == driver::deviceBlasOpN
                           ? uplo_tag
                           : (uplo_tag == driver::deviceBlasFillModeUpper
                                ? driver::deviceBlasFillModeLower
                                : driver::deviceBlasFillModeUpper);
  const auto trans     = out.transposed ? details_::flip_blas_op(op_a) : op_a;
  const auto side_mode = out.transposed ? (side == driver::deviceBlasSideLeft
                                             ? driver::deviceBlasSideRight
                                             : driver::deviceBlasSideLeft)
                                        : side;

  // The solve's scalar factor is host constant 1; pin host pointer mode.
  details_::BlasPointerModeGuard guard{h, false};
  const Sv alpha_host = Sv(1);
  const Sv* alpha_ptr = &alpha_host;

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, trsm, h.getRawHandle(),
                             side_mode, uplo, trans, diag, out.rows, out.cols,
                             alpha_ptr, a.data_handle(), ld_a, x.data_handle(),
                             out.leading_dimension);
  } else {
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, trsm, h.getRawHandle(),
                             side_mode, uplo, trans, diag, out.cols, out.rows,
                             alpha_ptr, a.data_handle(), ld_a, x.data_handle(),
                             out.leading_dimension);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "triangular_matrix_matrix_solve failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
