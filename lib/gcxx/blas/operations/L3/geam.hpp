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
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// C = alpha*op(A) + beta*op(B); cu/hipBLAS extension, not in P1673R13.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TC, class ExtentsC, class LayoutC, class AccessorC,
              class S = TC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsC::rank() == 2)
auto geam(BlasHandleView h, S alpha,
          const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a, S beta,
          const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
          const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using BVt = TB;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using CIt = typename ExtentsC::index_type;

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T. A
  // device_scalar argument selects device pointer mode; a plain scalar selects
  // host mode.
  using Sv                   = details_::scalar_value_t<S>;
  constexpr bool device_mode = details_::is_device_scalar_v<S>;

  // static asserts to verify no funny business
  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "geam operands A, B, C must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, AVt, BVt, CVt>,
                "geam alpha/beta value type must match the operands' element "
                "type");

  // TODO: Wire complex Cgeam/Zgeam into GCXX_BLAS_DISPATCH_TYPED.
  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "geam currently supports only float/double element types "
                "(complex support is a TODO)");

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit; alpha/beta are read from the host parameters or the device pointers
  // carried by device_scalar, per the mode.
  details_::BlasPointerModeGuard guard{h, device_mode};

  const Sv* alpha_ptr = details_::blas_scalar_ptr(alpha);
  const Sv* beta_ptr  = details_::blas_scalar_ptr(beta);

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
      "geam requires A, B, and C to share the same extents");
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
    details_::throwBlasError(status, "geam failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
