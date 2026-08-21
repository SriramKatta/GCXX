// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L2_TRSV_HPP_
#define GCXX_BLAS_OPERATIONS_L2_TRSV_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/L1/copy.hpp>
#include <gcxx/blas/operations/L1/scal.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/diagonal.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// x = A^-1 b, staged via copy+scale (backend solves in place on x).
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class Tri, class Diag, class TB, class ExtentsB, class LayoutB,
              class AccessorB, class TX, class ExtentsX, class LayoutX,
              class AccessorX)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              1 GCXX_AND ExtentsX::rank() == 1)
auto triangular_matrix_vector_solve(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  Tri /*triangle*/, Diag /*diagonal_storage*/,
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
                "triangular_matrix_vector_solve does not accept a scaled() "
                "view of A: scaling the operator does not scale the inverse");

  static_assert(!gcxx::is_scaled_accessor_v<AccessorX>,
                "triangular_matrix_vector_solve outputs cannot be scaled() "
                "views; scale b instead");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, XIt>,
                "triangular_matrix_vector_solve operands A, b, x must share "
                "the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, XVt>,
                "triangular_matrix_vector_solve operands A, b, x must share "
                "a single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "triangular_matrix_vector_solve currently supports only "
                "f32_t/f64_t element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "b");
  details_::validate_device_view(x, "x");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_b, inc_b] = details_::infer_blas_vector_view(b);
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  if (rows_a != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "triangular_matrix_vector_solve requires A to "
                             "be square");
  }
  if (len_b != cols_a || len_x != rows_a) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "triangular_matrix_vector_solve requires b and x to have A.extent(0) "
      "elements");
  }

  // stage b into x, apply b's scaled() factor to the staged copy (the solve
  // is linear, so the factor carries through), then solve in place
  copy(h, b, x);
  auto alpha_res =
    details_::resolve_scaled_alpha<Sv>(b.accessor());
  if (alpha_res.from_device()) {
    scale(h, gcxx::blas::device_scalar<Sv>{alpha_res.device_ptr}, x);
  } else if (alpha_res.host_value != Sv(1)) {
    scale(h, alpha_res.host_value, x);
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag =
    details_::fill_mode_v<Tri>;
  constexpr driver::deviceBlasDiagType_t diag =
    details_::diagonal_type_v<Diag>;

  // a row-major-like operand is read as its transpose: the mirrored triangle
  // plus the flipped op flag recover the mathematical A
  const auto uplo = op_a == driver::deviceBlasOpN
                      ? uplo_tag
                      : (uplo_tag == driver::deviceBlasFillModeUpper
                           ? driver::deviceBlasFillModeLower
                           : driver::deviceBlasFillModeUpper);
  const auto trans = op_a;  // N stays N (column-major-like); T flips it back

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, trsv, h.getRawHandle(), uplo,
                           trans, diag, rows_a, a.data_handle(), ld_a,
                           x.data_handle(), inc_x);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "triangular_matrix_vector_solve failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
