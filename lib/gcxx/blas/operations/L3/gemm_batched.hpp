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
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/span/span.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Batched C_i = A_i*B_i from host arrays of views (pointer-array API).
template <class A, class B, class C,
          class S = typename std::decay_t<C>::value_type::element_type>
auto gemm_batched(BlasHandleView h, S alpha, const A& a, const B& b, S beta,
                  C&& c)
  -> void {  // NOLINT(cppcoreguidelines-missing-std-forward)

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

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T.
  using Sv = details_::scalar_value_t<S>;

  // static asserts to verify no funny business
  static_assert(gcxx::is_span_like_v<A> && gcxx::is_span_like_v<B> &&
                  gcxx::is_span_like_v<C>,
                "gemm_batched operands must be span-like host arrays of "
                "rank-2 mdspans");

  static_assert(AMat::rank() == 2 && BMat::rank() == 2 && CMat::rank() == 2,
                "gemm_batched array elements must be rank-2 mdspans");

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

  // All arrays must hold the same matrix count; checked before any access.
  if (a.size() != b.size() || a.size() != c.size()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/ "gemm_batched operands must hold the same number of matrices");
  }

  // an empty batch has nothing to compute
  if (a.size() == 0) {
    return;
  }

  // Pin host pointer mode for the call (restored on scope exit): the pointer
  // arrays materialised below and the alpha/beta scalars are host memory.
  const details_::BlasPointerModeGuard guard{h, /*device_mode*/ false};

  // Extract dims from array[0]; all elements must agree (one m/n/k/ld/op).
  const auto [m, k, ld_a, op_a]   = details_::infer_blas_matrix_view(a[0]);
  const auto [k_b, n, ld_b, op_b] = details_::infer_blas_matrix_view(b[0]);
  const auto out                  = details_::infer_blas_output_view(c[0]);

  if (k != k_b || out.rows != m || out.cols != n) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "gemm_batched requires A_i to be (m x k), B_i to be (k x n), and C_i "
      "to be (m x n)");
  }

  for (std::size_t i = 1; i < a.size(); ++i) {
    const auto va = details_::infer_blas_matrix_view(a[i]);
    const auto vb = details_::infer_blas_matrix_view(b[i]);
    const auto vc = details_::infer_blas_output_view(c[i]);
    if (va.rows != m || va.cols != k || va.leading_dimension != ld_a ||
        va.op != op_a || vb.rows != k || vb.cols != n ||
        vb.leading_dimension != ld_b || vb.op != op_b || vc.rows != out.rows ||
        vc.cols != out.cols || vc.leading_dimension != out.leading_dimension ||
        vc.transposed != out.transposed) {
      details_::throwBlasError(
        GCXX_BLAS_STATUS(INVALID_VALUE),
        /*msg*/
        "gemm_batched requires all matrices in an array to share extents, "
        "leading dimension, and layout");
    }
  }

  // Stage pointer arrays to device memory; backend dereferences them there.
  const gcxx::StreamView stream = h.getStream();
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
  auto d_a_ptrs = gcxx::make_device_unique_ptr<const void*>(stream, a.size());
  auto d_b_ptrs = gcxx::make_device_unique_ptr<const void*>(stream, a.size());
  auto d_c_ptrs = gcxx::make_device_unique_ptr<void*>(stream, a.size());
  gcxx::Copy(stream, d_a_ptrs.get(), a_ptrs.data(), a.size());
  gcxx::Copy(stream, d_b_ptrs.get(), b_ptrs.data(), a.size());
  gcxx::Copy(stream, d_c_ptrs.get(), c_ptrs.data(), a.size());

  const AIt batch = static_cast<AIt>(a.size());

  // The C_i's orientation decides the problem handed to the column-major
  // backend: a row-major-like output takes the transposed problem
  // C_i^T = B_i^T * A_i^T (swapped operand slots, flipped op flags, swapped
  // m/n) — see matrix_product. Hoisted here so every entry point below
  // shares it.
  const auto first_op  = out.transposed ? details_::flip_blas_op(op_b) : op_a;
  const auto second_op = out.transposed ? details_::flip_blas_op(op_a) : op_b;
  const auto m_arg     = out.transposed ? n : m;
  const auto n_arg     = out.transposed ? m : n;
  const void* const* first_ptrs =
    out.transposed ? static_cast<const void* const*>(d_b_ptrs.get())
                   : static_cast<const void* const*>(d_a_ptrs.get());
  const void* const* second_ptrs =
    out.transposed ? static_cast<const void* const*>(d_a_ptrs.get())
                   : static_cast<const void* const*>(d_b_ptrs.get());
  const auto first_ld  = out.transposed ? ld_b : ld_a;
  const auto second_ld = out.transposed ? ld_a : ld_b;

  driver::deviceBlasStatus_t status{};
#if GCXX_HIP_MODE()
  // HIP: typed batched gemm (rocBLAS Ex page-faults); CUDA keeps the Ex call.
  GCXX_BLAS_DISPATCH_TYPED(
    status, AIt, AVt, gemmBatched, h.getRawHandle(), first_op, second_op, m_arg,
    n_arg, k, &alpha, reinterpret_cast<const AVt* const*>(first_ptrs), first_ld,
    reinterpret_cast<const BVt* const*>(second_ptrs), second_ld, &beta,
    reinterpret_cast<CVt* const*>(d_c_ptrs.get()), out.leading_dimension,
    batch);
#else
  GCXX_BLAS_DISPATCH_INT64(
    status, AIt, GemmBatchedEx, h.getRawHandle(), first_op, second_op, m_arg,
    n_arg, k, &alpha, first_ptrs, cuda_datatype_v<AVt>, first_ld, second_ptrs,
    cuda_datatype_v<BVt>, second_ld, &beta, d_c_ptrs.get(),
    cuda_datatype_v<CVt>, out.leading_dimension, batch,
    blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
#endif

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "gemm_batched failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
