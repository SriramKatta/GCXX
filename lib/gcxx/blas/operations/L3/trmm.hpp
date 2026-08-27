// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_TRMM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_TRMM_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/diagonal.hpp>
#include <gcxx/blas/operations/side.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Triangular product C = A*B or B*A; B and C must share orientation.
GCXX_TEMPLATE(class Side, class TA, class ExtentsA, class LayoutA,
              class AccessorA, class Tri, class Diag, class TB, class ExtentsB,
              class LayoutB, class AccessorB, class TC, class ExtentsC,
              class LayoutC, class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsC::rank() == 2)
auto triangular_matrix_product(
  BlasHandleView h, Side /*side*/,
  const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a, Tri /*triangle*/,
  Diag /*diagonal_storage*/,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
  const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using BVt = TB;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using CIt = typename ExtentsC::index_type;
  using Sv  = CVt;

  // static asserts to verify no funny business
  static_assert(!gcxx::is_scaled_accessor_v<AccessorC>,
                "triangular_matrix_product outputs cannot be scaled() views; "
                "scale an input");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "triangular_matrix_product operands A, B, C must share the "
                "same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, CVt>,
                "triangular_matrix_product operands A, B, C must share a "
                "single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "triangular_matrix_product currently supports only f32_t/"
                "f64_t element types (complex support is a TODO)");

  // Alpha comes only from scaled() views on the inputs.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()),
    "triangular_matrix_product");

  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;

  // Select the pointer mode for this call and restore the prior mode on
  // scope exit.
  details_::BlasPointerModeGuard guard{h, alpha_res.from_device()};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                          = details_::infer_blas_output_view(c);

  constexpr driver::deviceBlasSideMode_t side = details_::side_mode_v<Side>;

  // dimension gates per side (fail here rather than inside the backend)
  if (rows_a != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "triangular_matrix_product requires A to be "
                             "square");
  }
  if (side == driver::deviceBlasSideLeft &&
      (rows_b != rows_a || out.rows != rows_a || out.cols != cols_b)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_product (left) requires B to be A.extent(0) x N and "
      "C to be A.extent(0) x B.extent(1)");
  }
  if (side == driver::deviceBlasSideRight &&
      (cols_b != rows_a || out.rows != rows_b || out.cols != rows_a)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_product (right) requires B to be M x A.extent(0) and "
      "C to be B.extent(0) x A.extent(0)");
  }
  // the backend entry point takes no transpose flag for B, so B's storage
  // orientation must match C's
  if ((op_b == driver::deviceBlasOpN) == out.transposed) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_product requires B and C to share storage "
      "orientation: the backend entry point takes no transpose flag for B, "
      "so a column-major-like B must pair with a column-major-like C and a "
      "row-major-like B with a row-major-like C");
  }
  // the backend reads B and writes C through distinct pointers
  if (details_::views_alias(b, c)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_product requires B and C to be distinct buffers "
      "(the backend entry point reads B and writes C in one call); "
      "in-place triangular products are unsupported by this entry point");
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag = details_::fill_mode_v<Tri>;
  constexpr driver::deviceBlasDiagType_t diag = details_::diagonal_type_v<Diag>;

  // a row-major-like A is read as its transpose: the mirrored triangle, plus
  // the op flag that recovers the mathematical A from the column-major read.
  // When the OUTPUT is row-major-like the whole problem is transposed and
  // the flag flips again (C^T = B^T*A^T needs op(A_cm) = A^T).
  const auto uplo =
    details_::mirrored_fill_mode(op_a != driver::deviceBlasOpN, uplo_tag);
  const auto trans     = out.transposed ? details_::flip_blas_op(op_a) : op_a;
  const auto side_mode = details_::flipped_blas_side(out.transposed, side);

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, trmm, h.getRawHandle(),
                             side_mode, uplo, trans, diag, out.rows, out.cols,
                             alpha_ptr, a.data_handle(), ld_a, b.data_handle(),
                             ld_b, c.data_handle(), out.leading_dimension);
  } else {
    // C row-major-like: transposed problem with swapped m/n; lds carry over.
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, trmm, h.getRawHandle(),
                             side_mode, uplo, trans, diag, out.cols, out.rows,
                             alpha_ptr, a.data_handle(), ld_a, b.data_handle(),
                             ld_b, c.data_handle(), out.leading_dimension);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status,
                             /*msg*/ "triangular_matrix_product failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
