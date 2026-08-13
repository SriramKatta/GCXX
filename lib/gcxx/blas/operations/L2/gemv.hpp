// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L2_GEMV_HPP_
#define GCXX_BLAS_OPERATIONS_L2_GEMV_HPP_

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


#define GCXX_BLAS_GEMV_FN(BASE) APPEND_NAME(BLAS_BACKEND, BASE)
#define GCXX_BLAS_GEMV_FN_64(BASE) APPEND_NAME(BLAS_BACKEND, BASE##_64)

// Matrix-vector product y = alpha * op(A) * x + beta * y.
//
// A is a rank-2 mdspan; x and y are rank-1 mdspans. The operation op(A), the
// matrix dimensions (m, n), the leading dimension, and the vector increments
// (incx, incy) are all inferred from the mdspan metadata, so the API takes no
// separate shape or operation arguments.
//
// Example:
//   gcxx::blas::gemv(h, 1.0, A, x, 0.0, y);    // computes y = A * x
//   gcxx::blas::gemv(h, 1.0, blas::transpose(A), x, 0.0, y); // computes y =
//   A^T * x
//
// alpha/beta may be passed either as host scalars (host pointer mode) or as
// gcxx::blas::device_scalar<T> wrapping a device pointer (device pointer mode).
// The mode is selected per call from the argument type; the handle's prior
// pointer mode is restored when the call returns.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cublas*gemv_64 entry point
// (int64_t dimensions), while all other index_types use the standard 32-bit
// interface.
template <class A, class X, class Y,
          class S = typename std::decay_t<Y>::element_type>
auto gemv(BlasHandleView h, S alpha, const A& a, const X& x, S beta,
          Y&& y) -> void {

  // local alias for easier refrence
  using A_t = std::decay_t<A>;
  using X_t = std::decay_t<X>;
  using Y_t = std::decay_t<Y>;
  using AVt = typename A_t::element_type;
  using XVt = typename X_t::element_type;
  using YVt = typename Y_t::element_type;
  using AIt = typename A_t::index_type;
  using XIt = typename X_t::index_type;
  using YIt = typename Y_t::index_type;

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T. A
  // device_scalar argument selects device pointer mode; a plain scalar selects
  // host mode.
  using Sv                   = details_::scalar_value_t<S>;
  constexpr bool device_mode = details_::is_device_scalar_v<S>;

  // static asserts to verify no funny business
  static_assert(A_t::rank() == 2 && X_t::rank() == 1 && Y_t::rank() == 1,
                "gemv operands must be rank-2 (A) and rank-1 (x, y) mdspans");

  static_assert(gcxx::details_::all_same_v<AIt, XIt, YIt>,
                "gemv operands A, x, y must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  // The typed gemv backend routines require A, x, y, alpha and beta to share a
  // single element type.
  static_assert(gcxx::details_::all_same_v<Sv, AVt, XVt, YVt>,
                "gemv alpha/beta value type must match the operands' element "
                "type");

  // TODO: support complex element types via cublasCgemv / cublasZgemv
  //       (cublas_v2.h aliases these to the *_v2 forms; hipBLAS uses them
  //       natively). The dispatch macro below only handles float and double; a
  //       std::complex<T> element type hits this assert and must be wired up
  //       (add Cgemv/Zgemv branches to GCXX_BLAS_DISPATCH_TYPED and route
  //       alpha/beta through native_scalar_t<S>).
  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "gemv currently supports only float/double element types "
                "(complex support is a TODO)");

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit. Host mode reads alpha/beta from the by-value parameters; device mode
  // reads them from the device pointers carried by device_scalar (no host copy,
  // no host-side dereference).
  details_::BlasPointerModeGuard guard{
    h, device_mode ? driver::deviceBlasPointerModeDevice
                   : driver::deviceBlasPointerModeHost};

  const Sv* alpha_ptr{};
  const Sv* beta_ptr{};
  if constexpr (device_mode) {
    alpha_ptr = alpha.ptr;
    beta_ptr  = beta.ptr;
  } else {
    alpha_ptr = &alpha;
    beta_ptr  = &beta;
  }

  // extract problem dimensions
  const auto [m, n, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_x, inc_x]     = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y]     = details_::infer_blas_vector_view(y);

  // unused vars just to supress annoying warnings
  (void)len_x;
  (void)len_y;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, GEMV, h.getRawHandle(), op_a, m, n,
                           alpha_ptr, a.data_handle(), ld_a, x.data_handle(),
                           inc_x, beta_ptr, y.data_handle(), inc_y);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "gemv failed");
  }
}

#undef GCXX_BLAS_GEMV_FN
#undef GCXX_BLAS_GEMV_FN_64

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
