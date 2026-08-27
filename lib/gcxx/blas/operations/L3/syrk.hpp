// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_SYRK_HPP_
#define GCXX_BLAS_OPERATIONS_L3_SYRK_HPP_

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

// C += alpha*A*A^T into the tagged triangle only; beta fixed at 1.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class Tri, class TC, class ExtentsC, class LayoutC,
              class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsC::rank() == 2)
auto symmetric_matrix_rank_k_update(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  Tri /*triangle*/,
  const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using CIt = typename ExtentsC::index_type;
  using Sv  = CVt;

  // static asserts to verify no funny business
  static_assert(!gcxx::is_scaled_accessor_v<AccessorC>,
                "symmetric_matrix_rank_k_update outputs cannot be scaled() "
                "views; scale A instead");

  static_assert(gcxx::details_::all_same_v<AIt, CIt>,
                "symmetric_matrix_rank_k_update operands A, C must share the "
                "same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(std::is_same_v<AVt, CVt>,
                "symmetric_matrix_rank_k_update operands A, C must share a "
                "single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "symmetric_matrix_rank_k_update currently supports only "
                "f32_t/f64_t element types (complex support is a TODO)");

  // Alpha comes only from scaled() views on A; the accumulate weight is 1.
  auto alpha_res = details_::resolve_scaled_alpha<Sv>(a.accessor());
  if (alpha_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "symmetric_matrix_rank_k_update: the accumulate weight is the host "
      "constant 1, so a device_scalar scaled() factor on A cannot pair with "
      "it under one pointer mode; use host factors");
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
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto out                          = details_::infer_blas_output_view(c);

  if (out.rows != out.cols) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "symmetric_matrix_rank_k_update requires C to "
                             "be square");
  }
  if (out.rows != rows_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "symmetric_matrix_rank_k_update requires A to "
                             "have C.extent(0) rows");
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag = details_::fill_mode_v<Tri>;
  // A row-major-like C is updated as its transpose, whose stored triangle is
  // the mirror of the tagged one (A*A^T is symmetric, so the update is
  // unchanged by the transposition)
  const auto uplo = details_::mirrored_fill_mode(out.transposed, uplo_tag);

  // op_a carries A's storage orientation directly: a column-major-like
  // operand is the backend's trans=N problem, a row-major-like operand the
  // trans=T one (reading the same buffer column-major yields A^T).
  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, syrk, h.getRawHandle(), uplo, op_a,
                           out.rows, cols_a, alpha_ptr, a.data_handle(), ld_a,
                           beta_ptr, c.data_handle(), out.leading_dimension);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status,
                             /*msg*/ "symmetric_matrix_rank_k_update failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
