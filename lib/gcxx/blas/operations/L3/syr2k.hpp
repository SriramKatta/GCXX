// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_SYR2K_HPP_
#define GCXX_BLAS_OPERATIONS_L3_SYR2K_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// C += alpha*(A*B^T + B*A^T); tagged triangle only; beta fixed at 1.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class Tri, class TC, class ExtentsC, class LayoutC,
              class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsC::rank() == 2)
auto symmetric_matrix_rank_2k_update(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b, Tri /*triangle*/,
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
                "symmetric_matrix_rank_2k_update outputs cannot be scaled() "
                "views; scale A or B instead");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "symmetric_matrix_rank_2k_update operands A, B, C must share "
                "the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, CVt>,
                "symmetric_matrix_rank_2k_update operands A, B, C must share "
                "a single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "symmetric_matrix_rank_2k_update currently supports only "
                "f32_t/f64_t element types (complex support is a TODO)");

  // Alpha comes only from scaled() views on the inputs; accumulate weight 1.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()),
    "symmetric_matrix_rank_2k_update");
  if (alpha_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "symmetric_matrix_rank_2k_update: the accumulate weight is the host "
      "constant 1, so a device_scalar scaled() factor cannot pair with it "
      "under one pointer mode; use host factors");
  }
  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr = &alpha_host;
  const Sv beta_host  = Sv(1);
  const Sv* beta_ptr  = &beta_host;

  // Pin host pointer mode for the call (restored on scope exit); both
  // scalars above are host values.
  const details_::BlasPointerModeGuard guard{h, /*device_mode*/ false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                          = details_::infer_blas_output_view(c);

  if (out.rows != out.cols) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "symmetric_matrix_rank_2k_update requires C to "
                             "be square");
  }
  if (rows_a != rows_b || cols_a != cols_b) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "symmetric_matrix_rank_2k_update requires A and "
                             "B to have the same extents");
  }
  if (out.rows != rows_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "symmetric_matrix_rank_2k_update requires A to "
                             "have C.extent(0) rows");
  }
  // the backend entry point reads A and B through one transpose flag, so
  // their storage orientations must match
  if (op_a != op_b) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "symmetric_matrix_rank_2k_update requires A and B to share storage "
      "orientation: the backend entry point reads both through one "
      "transpose flag, so a column-major-like A must pair with a "
      "column-major-like B and a row-major-like A with a row-major-like B");
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag = details_::fill_mode_v<Tri>;
  // A row-major-like C is updated as its transpose, whose stored triangle is
  // the mirror of the tagged one (A*B^T + B*A^T is symmetric, so the update
  // is unchanged by the transposition)
  const auto uplo = details_::mirrored_fill_mode(out.transposed, uplo_tag);

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, syr2k, h.getRawHandle(), uplo,
                           op_a, out.rows, cols_a, alpha_ptr, a.data_handle(),
                           ld_a, b.data_handle(), ld_b, beta_ptr,
                           c.data_handle(), out.leading_dimension);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status,
                             /*msg*/ "symmetric_matrix_rank_2k_update failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
