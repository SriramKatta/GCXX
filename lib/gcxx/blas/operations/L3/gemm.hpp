// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_GEMM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_GEMM_HPP_

#include <type_traits>

#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/type_map.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Matrix-matrix product C = alpha * op(A) * op(B) + beta * C.
//
// A, B, C are rank-2 mdspan's. Transposition is expressed by passing a view:
//   gcxx::blas::gemm(h, 1.0, A, B, 0.0, C);
//   gcxx::blas::gemm(h, 1.0, blas::transpose(A), B, 0.0, C);   // op(A) = T
//
// since cu/hipblas use the column layout internally and also when storing C so
// to simplify this to actually compute C = A * B call it like
// blas::gemm(handle, alpha, B, A, beta, C);
template <class A, class B, class C,
          class S = typename std::decay_t<C>::element_type>
auto gemm(BlasHandleView h, S alpha, const A& a, const B& b, S beta,
          C&& c) -> void {

  // local alias for easier refrence
  using A_t = std::decay_t<A>;
  using B_t = std::decay_t<B>;
  using C_t = std::decay_t<C>;
  using T   = typename C_t::element_type;

  // static asserts to verify no funny business
  static_assert(A_t::rank() == 2 && B_t::rank() == 2 && C_t::rank() == 2,
                "gemm operands must be rank-2 mdspans");

  static_assert(
    std::is_same_v<typename A_t::element_type, T> &&
      std::is_same_v<typename B_t::element_type, T>,
    "gemm (v1) requires A, B, and C to share the same element type");

  static_assert(std::is_same_v<T, native_scalar_t<T>>,
                "gemm (v1) only supports real element types for now; complex "
                "support is not yet wired up");


  // TODO : LOCAL COPIES CAN BE DELETE AT THE END OF BLOCK BECUASE WE ARE
  // DEFAULTING TO CUBLAS_POINTER_MODE_HOST TO BE CHNAGED LATER
  S alpha_v = alpha;
  S beta_v  = beta;

  // extract problem dimensions
  const auto [m, k, ld_a, op_a]     = details_::infer_blas_matrix_view(a);
  const auto [k_b, n, ld_b, op_b]   = details_::infer_blas_matrix_view(b);
  const auto [m_c, n_c, ld_c, op_c] = details_::infer_blas_matrix_view(c);

  // unused vars just to supress annoying warnings
  (void)m_c;
  (void)n_c;
  (void)k_b;
  (void)op_c;

  const auto status = gemm_ptr_v<T>(
    h.getRawHandle(), op_a, op_b, m, n, k, &alpha_v, a.data_handle(), ld_a,
    b.data_handle(), ld_b, &beta_v, c.data_handle(), ld_c);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "gemm failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
