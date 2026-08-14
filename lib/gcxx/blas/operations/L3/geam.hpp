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

// Matrix-matrix addition / transpose-copy C = alpha * op(A) + beta * op(B).
//
// A, B, and C are rank-2 mdspan objects. The effective dimensions, the leading
// dimensions, and the transpose state of each operand are inferred from the
// mdspan metadata and any view wrappers (blas::transpose), so the API takes no
// separate shape or operation arguments.
//
// Example:
//   gcxx::blas::geam(h, 1.0, A, 0.0, B, C);    // computes C = A
//   gcxx::blas::geam(h, 1.0, blas::transpose(A), 0.0, B, C);  // C = A^T
//
// alpha/beta may be passed either as host scalars (host pointer mode) or as
// gcxx::blas::device_scalar<T> wrapping a device pointer (device pointer
// mode). The mode is selected per call from the argument type; the handle's
// prior pointer mode is restored when the call returns.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasSgeam_64 (Dgeam_64) entry
// point, while all other index_types use the standard 32-bit interface.
//
// A, B, and C must be device views: mdspans carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::device_mdspan). Host views are rejected
// at compile time; in check builds the data handles are additionally probed
// at run time so a mislabeled host pointer fails here, not inside the GPU
// kernel.
template <class A, class B, class C,
          class S = typename std::decay_t<C>::element_type>
auto geam(BlasHandleView h, S alpha, const A& a, S beta, const B& b,
          C&& c) -> void {

  // local alias for easier refrence
  using A_t = std::decay_t<A>;
  using B_t = std::decay_t<B>;
  using C_t = std::decay_t<C>;
  using AVt = typename A_t::element_type;
  using BVt = typename B_t::element_type;
  using CVt = typename C_t::element_type;
  using AIt = typename A_t::index_type;
  using BIt = typename B_t::index_type;
  using CIt = typename C_t::index_type;

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T. A
  // device_scalar argument selects device pointer mode; a plain scalar selects
  // host mode.
  using Sv                   = details_::scalar_value_t<S>;
  constexpr bool device_mode = details_::is_device_scalar_v<S>;

  // static asserts to verify no funny business
  static_assert(A_t::rank() == 2 && B_t::rank() == 2 && C_t::rank() == 2,
                "geam operands must be rank-2 mdspans");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "geam operands A, B, C must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, AVt, BVt, CVt>,
                "geam alpha/beta value type must match the operands' element "
                "type");

  // TODO: support complex element types via cublasCgeam / cublasZgeam
  //       (hipBLAS uses them natively). The dispatch macro below only handles
  //       float and double; a std::complex<T> element type hits this assert
  //       and must be wired up (add Cgeam/Zgeam branches to
  //       GCXX_BLAS_DISPATCH_TYPED).
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

  // extract problem dimensions: A's logical (post-op) extent defines (m, n)
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto [rows_c, cols_c, ld_c, op_c] = details_::infer_blas_matrix_view(c);

  // unused vars just to supress annoying warnings
  (void)rows_b;
  (void)cols_b;
  (void)rows_c;
  (void)cols_c;
  (void)op_c;

  const AIt m = op_a == driver::deviceBlasOpN ? rows_a : cols_a;
  const AIt n = op_a == driver::deviceBlasOpN ? cols_a : rows_a;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, geam, h.getRawHandle(), op_a, op_b,
                           m, n, alpha_ptr, a.data_handle(), ld_a, beta_ptr,
                           b.data_handle(), ld_b, c.data_handle(), ld_c);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "geam failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
