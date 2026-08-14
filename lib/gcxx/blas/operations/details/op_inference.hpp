// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_OP_INFERENCE_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_OP_INFERENCE_HPP_

#include <string>

#include <gcxx/blas/error/blas_exceptions.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>
#include <gcxx/runtime_backend/backend_memory.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// Debug-mode companion to the is_device_view_v compile-time gate: verifies at
// run time (via cuda/hipPointerGetAttributes) that the operand's data handle
// really points at device or managed memory, so a host pointer mislabeled as
// a device view fails loudly at the BLAS call site instead of faulting
// asynchronously inside the backend kernel. Compiles to nothing under
// GCXX_DISABLE_RUNTIME_CHECKS.
#ifndef GCXX_DISABLE_RUNTIME_CHECKS
template <class MD>
GCXX_FH auto validate_device_view(const MD& v, const char* name) -> void {
  if (!driver::isDeviceOrManagedMemory(v.data_handle())) {
    std::string msg{"BLAS operand '"};
    msg += name;
    msg +=
      "' does not reside in device/managed memory (a device_accessor view was "
      "passed, but the pointer is host memory)";
    throw gcxx::blas::BlasException(GCXX_BLAS_STATUS(INVALID_VALUE), msg);
  }
}
#else
template <class MD>
GCXX_FH auto validate_device_view(const MD&, const char*) -> void {}
#endif

GCXX_TEMPLATE(typename IdxT)
GCXX_REQUIRES(std::is_integral_v<IdxT>)
struct blas_matrix_view {
  IdxT rows;               // rows of matrix
  IdxT cols;               // cols of the matrix
  IdxT leading_dimension;  // num elems between columns
  driver::deviceBlasOp_t op;
};

// Stride pair of a rank-2 (or inner rank-2 of a higher-rank) matrix operand.
GCXX_TEMPLATE(typename IdxT)
GCXX_REQUIRES(std::is_integral_v<IdxT>)
struct blas_op_view {
  IdxT leading_dimension;
  driver::deviceBlasOp_t op;
};

GCXX_TEMPLATE(typename IdxT)
GCXX_REQUIRES(std::is_integral_v<IdxT>)
struct blas_vector_view {
  IdxT length;  // number of elements
  IdxT stride;  // increment between elements (incx / incy)
};

GCXX_TEMPLATE(typename IdxT)
GCXX_REQUIRES(std::is_integral_v<IdxT>)
struct blas_batched_matrix_view {
  IdxT rows;               // rows of each matrix in the batch
  IdxT cols;               // cols of each matrix in the batch
  IdxT leading_dimension;  // num elems between columns of one matrix
  IdxT batch_count;        // number of matrices in the batch
  IdxT batch_stride;       // num elems between batch element 0 of dim 2
  driver::deviceBlasOp_t op;
};

// Resolve the BLAS view (leading dimension + op) based on unit stride on row or
// coloum
template <class IdxT>
constexpr auto infer_blas_op_view(IdxT s0, IdxT s1) -> blas_op_view<IdxT> {
  if (s0 == 1) {
    return {s1, driver::deviceBlasOpN};
  }
  if (s1 == 1) {
    return {s0, driver::deviceBlasOpT};
  }
  throw gcxx::blas::BlasException(
    GCXX_BLAS_STATUS(INVALID_VALUE),
    "BLAS matrix operand must have a unit stride on one axis");
}

template <class MD>
constexpr auto infer_blas_matrix_view(const MD& v)
  -> blas_matrix_view<typename MD::index_type> {
  using idx_t = typename MD::index_type;
  static_assert(MD::rank() == 2, "BLAS matrix operand must be rank-2");

  static_assert(gcxx::is_device_view_v<typename MD::accessor_type>,
                "BLAS matrix operands must view device memory: use "
                "gcxx::device_mdspan / gcxx::managed_mdspan (or an mdspan "
                "carrying gcxx::device_accessor / gcxx::managed_accessor)");

  const auto [ld, op] = infer_blas_op_view(v.stride(0), v.stride(1));
  return {v.extent(0), v.extent(1), ld, op};
}

// Infer the BLAS view (length + increment) of a rank-1 mdspan operand.
template <class VD>
constexpr auto infer_blas_vector_view(const VD& v)
  -> blas_vector_view<typename VD::index_type> {
  static_assert(VD::rank() == 1, "BLAS vector operand must be rank-1");

  static_assert(gcxx::is_device_view_v<typename VD::accessor_type>,
                "BLAS vector operands must view device memory: use "
                "gcxx::make_device_vector, gcxx::device_mdspan / "
                "gcxx::managed_mdspan (or an mdspan carrying "
                "gcxx::device_accessor / gcxx::managed_accessor)");

  return {v.extent(0), v.stride(0)};
}

// Infer the BLAS view of a rank-3 batched-matrix mdspan operand whose batch
// dimension is the LAST one, i.e. extents (rows, cols, batch): a layout_left
// operand yields contiguous column-major matrices (op = N, ld = stride(1)),
// a layout_right operand yields row-major ones (op = T, ld = stride(0)), and
// the batch stride is stride(2) either way. This matches how a single
// cublasXgemmStridedBatchedEx pointer + stride covers the whole batch.
template <class MD>
constexpr auto infer_blas_batched_matrix_view(const MD& v)
  -> blas_batched_matrix_view<typename MD::index_type> {
  using idx_t = typename MD::index_type;
  static_assert(MD::rank() == 3,
                "BLAS batched matrix operand must be rank-3 (rows, cols, "
                "batch)");

  static_assert(gcxx::is_device_view_v<typename MD::accessor_type>,
                "BLAS batched matrix operands must view device memory: use "
                "gcxx::device_mdspan / gcxx::managed_mdspan (or an mdspan "
                "carrying gcxx::device_accessor / gcxx::managed_accessor)");

  const auto [ld, op] = infer_blas_op_view(v.stride(0), v.stride(1));
  return {v.extent(0), v.extent(1), ld, v.extent(2), v.stride(2), op};
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
