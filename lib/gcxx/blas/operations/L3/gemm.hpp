// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_GEMM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_GEMM_HPP_

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

// Matrix-matrix product C = alpha * op(A) * op(B) + beta * C.
//
// A, B, and C are rank-2 mdspan objects. The effective dimensions, layout, and
// transpose state are inferred from the mdspan metadata and any view wrappers,
// so the API does not take separate shape or operation arguments.
//
// Example:
//   gcxx::blas::gemm(h, 1.0, A, B, 4.0, C);    // computes C = A * B + 4.0 * C
//   gcxx::blas::gemm(h, 1.0, blas::transpose(A), B, 0.0, C); // computes C =
//   A^T * B
//
// Because cu/hipBLAS internally treats matrices in column-major form, the
// conventional usage to compute A * B is to pass the operands in the order
// (B, A, C) when matching the backend's storage convention.
//
// alpha/beta may be passed either as host scalars (host pointer mode) or as
// gcxx::blas::device_scalar<T> wrapping a device pointer (device pointer mode).
// The mode is selected per call from the argument type; the handle's prior
// pointer mode is restored when the call returns.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasGemmEx_64 entry point
// (int64_t dimensions), while all other index_types use the standard 32-bit
// interface.
//
// A, B, and C must be device views: mdspans carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::device_mdspan). Host views are rejected
// at compile time; in check builds the data handles are additionally probed
// at run time so a mislabeled host pointer fails here, not inside the GPU
// kernel.
template <class A, class B, class C,
          class S = typename std::decay_t<C>::element_type>
auto gemm(BlasHandleView h, S alpha, const A& a, const B& b, S beta,
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
                "gemm operands must be rank-2 mdspans");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "gemm operands A, B, C must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, AVt, BVt, CVt>,
                "gemm alpha/beta value type must match the operands' element "
                "type");

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

  // extract problem dimensions
  const auto [m, k, ld_a, op_a]     = details_::infer_blas_matrix_view(a);
  const auto [k_b, n, ld_b, op_b]   = details_::infer_blas_matrix_view(b);
  const auto [m_c, n_c, ld_c, op_c] = details_::infer_blas_matrix_view(c);

  // unused vars just to supress annoying warnings
  (void)m_c;
  (void)n_c;
  (void)k_b;
  (void)op_c;

  driver::deviceBlasStatus_t status;
  GCXX_BLAS_DISPATCH_INT64(
    status, AIt, GemmEx, h.getRawHandle(), op_a, op_b, m, n, k, alpha_ptr,
    a.data_handle(), cuda_datatype_v<AVt>, ld_a, b.data_handle(),
    cuda_datatype_v<BVt>, ld_b, beta_ptr, c.data_handle(), cuda_datatype_v<CVt>,
    ld_c, blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "gemm failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
