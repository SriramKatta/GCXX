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

// Batched matrix-matrix product C_i = alpha * op(A_i) * op(B_i) + beta * C_i,
// where each A_i, B_i, C_i is a rank-2 mdspan and the operands are span-like
// HOST arrays (gcxx::span, std::vector, ...) of those views — the batch
// matrices may live at unrelated device addresses, which is exactly what the
// cu/hipblasGemmBatchedEx pointer-array entry point is for. The entry point
// dereferences the pointer arrays on the DEVICE, so the wrapper stages them
// into stream-ordered device memory before the call. For matrices that share
// one contiguous buffer with a uniform batch stride, use
// gemm_strided_batched instead: it passes (base, stride) straight through
// with no per-call pointer materialisation.
//
// NOT part of P1673R13 proper (batching is the P2901 follow-up, whose
// pure-mdspan design gains the batch dimension instead of taking host arrays
// of views); kept as a cu/hipBLAS extension with its BLAS-style alpha/beta
// parameters.
//
// The per-batch dimensions, leading dimensions, and transpose state are
// inferred from the mdspan metadata of the first element of each array and
// runtime-checked against the remaining elements (the backend takes a single
// m/n/k/ld/op for the whole batch, so every element must agree). As with
// matrix_product, the mathematical result C_i = A_i * B_i holds for ANY mix
// of operand layouts: when the C_i are row-major-like the dispatch presents
// the transposed problem to the column-major backend.
//
// Example:
//   std::vector<mat2d> aViews{...}, bViews{...}, cViews{...};
//   gcxx::blas::gemm_batched(h, 1.0, aViews, bViews, 0.0, cViews);
//
// alpha/beta are host scalars only: the entry point reads them (but not the
// pointer arrays, which live in device memory) in host pointer mode, which
// this wrapper sets up, so device_scalar arguments are rejected at compile
// time.
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

  // every array must hold the same number of matrices, checked BEFORE any
  // element is accessed: reading [0] of an empty or shorter array (e.g. an
  // empty std::vector) is undefined behaviour
  if (a.size() != b.size() || a.size() != c.size()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "gemm_batched operands must hold the same number of matrices");
  }

  // an empty batch has nothing to compute
  if (a.size() == 0) {
    return;
  }

  // Pin host pointer mode for the call (restored on scope exit): the pointer
  // arrays materialised below and the alpha/beta scalars are host memory.
  details_::BlasPointerModeGuard guard{h, false};

  // extract problem dimensions from the first element of each array; the
  // remaining elements must agree because the backend takes a single
  // m/n/k/ld/op for the whole batch
  const auto [m, k, ld_a, op_a]   = details_::infer_blas_matrix_view(a[0]);
  const auto [k_b, n, ld_b, op_b] = details_::infer_blas_matrix_view(b[0]);
  const auto out                  = details_::infer_blas_output_view(c[0]);

  if (k != k_b || out.rows != m || out.cols != n) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
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
        "gemm_batched requires all matrices in an array to share extents, "
        "leading dimension, and layout");
    }
  }

  // materialise the pointer arrays the entry point consumes: one device
  // pointer per matrix, gathered from each mdspan view (probing each handle
  // on the way — no-op unless checks are enabled). cu/hipBLAS dereference
  // these arrays ON THE DEVICE, so they must be staged into device memory;
  // the stream-ordered allocation below frees them only after the batched
  // call has run, and the upload shares the handle's stream so it is ordered
  // before it.
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
  // rocBLAS's type-erased GemmBatchedEx pointer-array entry point page-faults
  // the GPU on double-precision batches ("Memory access fault ... Page not
  // present", observed on an IHPC HIP node with all-column-major operands),
  // while the same problem through the strided entry point and through the
  // classic typed batched gemm runs fine. cuBLAS exposes no typed
  // gemmBatched in cublas_v2, so HIP routes to hipblas<S,D>gemmBatched and
  // CUDA keeps the Ex call; the two entry points are argument-compatible
  // apart from the type-erasure parameters. Both consume the pointer arrays
  // staged into device memory above.
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
    cuda_datatype_v<BVt>, second_ld, &beta, d_c_ptrs.get(), cuda_datatype_v<CVt>,
    out.leading_dimension, batch, blas_compute_type_v<CVt>,
    GCXX_BLAS_GEMM(DEFAULT));
#endif

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "gemm_batched failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
