// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_GEMM_BATCHED_HPP_
#define GCXX_BLAS_OPERATIONS_L3_GEMM_BATCHED_HPP_

#include <cstddef>
#include <type_traits>
#include <vector>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/span/span.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Batched matrix-matrix product C_i = alpha * op(A_i) * op(B_i) + beta * C_i,
// where each A_i, B_i, C_i is a rank-2 mdspan and the operands are span-like
// HOST arrays (gcxx::span, std::vector, ...) of those views — the batch
// matrices may live at unrelated device addresses, which is exactly what the
// cu/hipblasGemmBatchedEx pointer-array entry point is for. For matrices that
// share one contiguous buffer with a uniform batch stride, use
// gemm_strided_batched instead: it passes (base, stride) straight through
// with no per-call pointer materialisation.
//
// The per-batch dimensions, leading dimensions, and transpose state are
// inferred from the mdspan metadata of the first element of each array and
// runtime-checked against the remaining elements (the backend takes a single
// m/n/k/ld/op for the whole batch, so every element must agree).
//
// Example:
//   std::vector<mat2d> aViews{...}, bViews{...}, cViews{...};
//   gcxx::blas::gemm_batched(h, 1.0, aViews, bViews, 0.0, cViews);
//
// alpha/beta are host scalars only: the backend reads the pointer arrays from
// host memory in host pointer mode, which this wrapper sets up, so
// device_scalar arguments are rejected at compile time.
//
// The integer interface is selected from the elements' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasGemmBatchedEx_64 entry
// point, while all other index_types use the standard 32-bit interface.
//
// Every matrix element must be a device view (gcxx::device_mdspan /
// gcxx::managed_mdspan). The HOST arrays holding the views are plain host
// containers by design; the gate applies to the matrices they contain. In
// check builds each matrix's data handle is probed at run time so a
// mislabeled host pointer fails here, not inside the GPU kernel.
template <class A, class B, class C,
          class S = typename std::decay_t<C>::value_type::element_type>
auto gemm_batched(BlasHandleView h, S alpha, const A& a, const B& b, S beta,
                  C&& c) -> void {

  // local alias for easier refrence
  using A_t  = std::decay_t<A>;
  using B_t  = std::decay_t<B>;
  using C_t  = std::decay_t<C>;
  using AMat = typename A_t::value_type;
  using BMat = typename B_t::value_type;
  using CMat = typename C_t::value_type;
  using AVt  = typename AMat::element_type;
  using BVt  = typename BMat::element_type;
  using CVt  = typename CMat::element_type;
  using AIt  = typename AMat::index_type;
  using BIt  = typename BMat::index_type;
  using CIt  = typename CMat::index_type;

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T.
  using Sv = details_::scalar_value_t<S>;

  // static asserts to verify no funny business
  static_assert(gcxx::is_span_like_v<A> && gcxx::is_span_like_v<B> &&
                  gcxx::is_span_like_v<C>,
                "gemm_batched operands must be span-like host arrays of "
                "rank-2 mdspans");

  static_assert(AMat::rank() == 2 && BMat::rank() == 2 && CMat::rank() == 2,
                "gemm_batched array elements must be rank-2 mdspans");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "gemm_batched operands A, B, C must share the same mdspan "
                "index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, AVt, BVt, CVt>,
                "gemm_batched alpha/beta value type must match the operands' "
                "element type");

  static_assert(
    !details_::is_device_scalar_v<S>,
    "gemm_batched requires host pointer mode (its pointer arrays live in "
    "host memory), so alpha/beta must be host scalars");

  // an empty batch has nothing to compute
  if (a.size() == 0) {
    return;
  }

  // extract problem dimensions from the first element of each array; the
  // remaining elements must agree because the backend takes a single
  // m/n/k/ld/op for the whole batch
  const auto [m, k, ld_a, op_a]     = details_::infer_blas_matrix_view(a[0]);
  const auto [k_b, n, ld_b, op_b]   = details_::infer_blas_matrix_view(b[0]);
  const auto [m_c, n_c, ld_c, op_c] = details_::infer_blas_matrix_view(c[0]);

  // unused vars just to supress annoying warnings
  (void)k_b;
  (void)m_c;
  (void)n_c;
  (void)op_c;

  if (a.size() != b.size() || a.size() != c.size()) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "gemm_batched operands must hold the same number of matrices");
  }
  for (std::size_t i = 1; i < a.size(); ++i) {
    const auto va = details_::infer_blas_matrix_view(a[i]);
    const auto vb = details_::infer_blas_matrix_view(b[i]);
    const auto vc = details_::infer_blas_matrix_view(c[i]);
    if (va.rows != m || va.cols != k || va.leading_dimension != ld_a ||
        va.op != op_a || vb.rows != k || vb.cols != n ||
        vb.leading_dimension != ld_b || vb.op != op_b || vc.rows != m ||
        vc.cols != n || vc.leading_dimension != ld_c || vc.op != op_c) {
      throw gcxx::blas::BlasException(
        GCXX_BLAS_STATUS(INVALID_VALUE),
        "gemm_batched requires all matrices in an array to share extents, "
        "leading dimension, and layout");
    }
  }

  // materialise the host pointer arrays the entry point consumes: one device
  // pointer per matrix, gathered from each mdspan view (probing each handle
  // on the way — no-op unless checks are enabled)
  std::vector<const void*> a_ptrs(a.size());
  std::vector<const void*> b_ptrs(a.size());
  std::vector<void*> c_ptrs(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    details_::validate_device_view(a[i], "A[i]");
    details_::validate_device_view(b[i], "B[i]");
    details_::validate_device_view(c[i], "C[i]");
    a_ptrs[i] = a[i].data_handle();
    b_ptrs[i] = b[i].data_handle();
    c_ptrs[i] = c[i].data_handle();
  }

  const AIt batch = static_cast<AIt>(a.size());

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(
    status, AIt, GemmBatchedEx, h.getRawHandle(), op_a, op_b, m, n, k, &alpha,
    a_ptrs.data(), cuda_datatype_v<AVt>, ld_a, b_ptrs.data(),
    cuda_datatype_v<BVt>, ld_b, &beta, c_ptrs.data(), cuda_datatype_v<CVt>,
    ld_c, batch, blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "gemm_batched failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
