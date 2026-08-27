// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_GEMM_STRIDED_BATCHED_HPP_
#define GCXX_BLAS_OPERATIONS_L3_GEMM_STRIDED_BATCHED_HPP_

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

// Strided batched C_i = A_i*B_i; rank-3 operands with batch-first extents.
template <class A, class B, class C,
          class S = typename std::decay_t<C>::element_type>
// c is only indexed (never moved); forwarding an rvalue would change nothing.
auto gemm_strided_batched(BlasHandleView h, S alpha, const A& a, const B& b,
                          S beta, C&& c)
  -> void {  // NOLINT(cppcoreguidelines-missing-std-forward)

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

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T.
  using Sv = details_::scalar_value_t<S>;

  // static asserts to verify no funny business
  static_assert(A_t::rank() == 3 && B_t::rank() == 3 && C_t::rank() == 3,
                "gemm_strided_batched operands must be rank-3 (batch, rows, "
                "cols) mdspans");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "gemm_strided_batched operands A, B, C must share the same "
                "mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, AVt, BVt, CVt>,
                "gemm_strided_batched alpha/beta value type must match the "
                "operands' element type");

  static_assert(
    !details_::is_device_scalar_v<S>,
    "gemm_strided_batched only supports host alpha/beta scalars (device "
    "pointer mode would require device-side scalar storage)");

  // Pin host pointer mode for the call (restored on scope exit); alpha/beta
  // below are host scalars.
  const details_::BlasPointerModeGuard guard{h, /*device_mode*/ false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [m, k, ld_a, batch_a, stride_a, op_a, trans_a] =
    details_::infer_blas_batched_matrix_view(a);
  const auto [k_b, n, ld_b, batch_b, stride_b, op_b, trans_b] =
    details_::infer_blas_batched_matrix_view(b);
  const auto [m_c, n_c, ld_c, batch_c, stride_c, op_c, trans_c] =
    details_::infer_blas_batched_matrix_view(c);

  // unused vars just to supress annoying warnings
  (void)k_b;
  (void)m_c;
  (void)n_c;
  (void)op_c;
  (void)trans_a;
  (void)trans_b;

  if (k != k_b || m_c != m || n_c != n || batch_a != batch_b ||
      batch_a != batch_c) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "gemm_strided_batched requires A_i to be (m x k), B_i to be (k x n), "
      "C_i to be (m x n), and all batches to have the same count");
  }

  driver::deviceBlasStatus_t status{};  // NOLINT(misc-const-correctness)
                                        // assigned by the dispatches below
  if (!trans_c) {
    // The macro's two branches spell distinct entry points
    // (GemmStridedBatchedEx_64 vs GemmStridedBatchedEx), which the checker
    // cannot tell apart in uninstantiated code.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmStridedBatchedEx, h.getRawHandle(), op_a, op_b, m, n, k,
      &alpha, a.data_handle(), cuda_datatype_v<AVt>, ld_a, stride_a,
      b.data_handle(), cuda_datatype_v<BVt>, ld_b, stride_b, &beta,
      c.data_handle(), cuda_datatype_v<CVt>, ld_c, stride_c, batch_a,
      blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
  } else {
    // C row-major-like: present the transposed problem C_i^T = B_i^T * A_i^T.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmStridedBatchedEx, h.getRawHandle(),
      details_::flip_blas_op(op_b), details_::flip_blas_op(op_a), n, m, k,
      &alpha, b.data_handle(), cuda_datatype_v<BVt>, ld_b, stride_b,
      a.data_handle(), cuda_datatype_v<AVt>, ld_a, stride_a, &beta,
      c.data_handle(), cuda_datatype_v<CVt>, ld_c, stride_c, batch_a,
      blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "gemm_strided_batched failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
