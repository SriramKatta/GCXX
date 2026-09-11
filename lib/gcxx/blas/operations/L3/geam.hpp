// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_GEAM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_GEAM_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// C = alpha*A + beta*B over the cu/hipBLAS geam extension (not in P1673R13);
// alpha and beta arrive as scaled() factors on the inputs (1 when unscaled).
// The elementwise nature makes C aliasing an input the in-place mode.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TC, class ExtentsC, class LayoutC, class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsC::rank() == 2)
auto matrix_addition(BlasHandleView h,
                     const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
                     const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
                     const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c)
  -> void {

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
                "matrix_addition output cannot be a scaled() view; scale an "
                "input instead");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "matrix_addition operands A, B, C must share the same mdspan "
                "index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, CVt>,
                "matrix_addition operands A, B, C must share a single element "
                "type");

  // TODO: Wire complex Cgeam/Zgeam into GCXX_BLAS_DISPATCH_TYPED.
  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "matrix_addition currently supports only float/double element "
                "types (complex support is a TODO)");

  // A's scaled() factor is alpha, B's is beta (1 when unscaled); a factor may
  // be a host value or a device-resident scalar selecting device pointer mode.
  const auto alpha_res = details_::resolve_scaled_alpha<Sv>(a.accessor());
  const auto beta_res  = details_::resolve_scaled_alpha<Sv>(b.accessor());
  if (alpha_res.from_device() != beta_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "matrix_addition: the backend reads alpha and beta through one pointer "
      "mode, so host and device-resident scalar factors cannot be mixed in "
      "one call");
  }
  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;
  const Sv beta_host = beta_res.host_value;
  const Sv* beta_ptr =
    beta_res.from_device() ? beta_res.device_ptr : &beta_host;

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit; alpha/beta are read from the host parameters or the device pointers
  // carried by the scaled() factors, per the mode.
  details_::BlasPointerModeGuard guard{h, alpha_res.from_device()};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(c, "C");

  // extract problem dimensions; the output's orientation decides how the
  // problem is presented to the column-major backend
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                          = details_::infer_blas_output_view(c);

  if (rows_a != out.rows || cols_a != out.cols || rows_b != out.rows ||
      cols_b != out.cols) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "matrix_addition requires A, B, and C to share the same extents");
  }

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, geam, h.getRawHandle(), op_a,
                             op_b, out.rows, out.cols, alpha_ptr,
                             a.data_handle(), ld_a, beta_ptr, b.data_handle(),
                             ld_b, c.data_handle(), out.leading_dimension);
  } else {
    // C row-major-like: compute C^T = alpha*op(A)^T + beta*op(B)^T instead.
    GCXX_BLAS_DISPATCH_TYPED(
      status, AIt, AVt, geam, h.getRawHandle(), details_::flip_blas_op(op_a),
      details_::flip_blas_op(op_b), out.cols, out.rows, alpha_ptr,
      a.data_handle(), ld_a, beta_ptr, b.data_handle(), ld_b, c.data_handle(),
      out.leading_dimension);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "matrix_addition failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
