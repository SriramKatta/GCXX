// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_OP_INFERENCE_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_OP_INFERENCE_HPP_

#include <string>

#include <gcxx/blas/error/blas_exceptions.hpp>
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/blas/operations/scaled.hpp>
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
  IdxT batch_stride;       // num elems between batch elements
  driver::deviceBlasOp_t op;
  bool transposed;         // inner-matrix storage is row-major-like (see
                           // blas_output_view)
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

// Flip an op flag (N <-> T). Reading the same storage with the flipped flag
// yields the transpose of reading it with the original flag; this is how the
// transposed-output dispatches below present the transposed problem to the
// column-major backend.
constexpr auto flip_blas_op(driver::deviceBlasOp_t op) -> driver::deviceBlasOp_t {
  return op == driver::deviceBlasOpN ? driver::deviceBlasOpT
                                     : driver::deviceBlasOpN;
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
// dimension is the FIRST one, i.e. extents (batch, rows, cols) — the
// leftmost-batch convention of P1673R13's batched future work (P2901). The
// inner rank-2 view of batch element 0 is resolved from the strides of axes
// 1 and 2: a layout_left operand yields contiguous column-major matrices
// (op = N), a layout_right one row-major matrices (op = T, plus transposed
// = true for an OUTPUT operand), and the batch stride is stride(0) either
// way. This matches how a single cublasXgemmStridedBatchedEx pointer +
// stride covers the whole batch.
template <class MD>
constexpr auto infer_blas_batched_matrix_view(const MD& v)
  -> blas_batched_matrix_view<typename MD::index_type> {
  using idx_t = typename MD::index_type;
  static_assert(MD::rank() == 3,
                "BLAS batched matrix operand must be rank-3 (batch, rows, "
                "cols)");

  static_assert(gcxx::is_device_view_v<typename MD::accessor_type>,
                "BLAS batched matrix operands must view device memory: use "
                "gcxx::device_mdspan / gcxx::managed_mdspan (or an mdspan "
                "carrying gcxx::device_accessor / gcxx::managed_accessor)");

  const auto [ld, op] = infer_blas_op_view(v.stride(1), v.stride(2));
  const bool transposed =
    v.stride(1) != 1 && v.stride(2) == 1;  // inner matrix row-major-like
  return {v.extent(1), v.extent(2), ld, v.extent(0), v.stride(0), op,
          transposed};
}

// ── scaled()-view factor resolution ────────────────────────────────────────

// A resolved scalar multiplier for a BLAS call: either a host value or a
// single device-side factor pointer. The P1673R13-shaped calls express alpha
// via scaled(alpha, x) views; this is what the operations unwrap them into so
// the backend's single alpha argument can be fed.
template <class Sv>
struct alpha_resolution {
  Sv        host_value{1};
  const Sv* device_ptr{nullptr};

  GCXX_CXPR bool from_device() const noexcept { return device_ptr != nullptr; }
};

// Resolve the scaling factor carried by an operand's accessor (none by
// default, i.e. the identity factor 1).
template <class Sv, class Accessor>
constexpr auto resolve_scaled_alpha(const Accessor&) -> alpha_resolution<Sv> {
  return {};
}

template <class Sv, class ScalingFactor, class NestedAccessor>
constexpr auto resolve_scaled_alpha(
  const gcxx::blas::scaled_accessor<ScalingFactor, NestedAccessor>& acc)
  -> alpha_resolution<Sv> {
  using factor_t = std::remove_cv_t<ScalingFactor>;
  if constexpr (is_device_scalar_v<factor_t>) {
    static_assert(
      std::is_same_v<typename scalar_traits<factor_t>::value_type, Sv>,
      "device_scalar scaling factors must match the operands' element type");
    return {Sv(1), acc.scaling_factor().ptr};
  } else {
    return {static_cast<Sv>(acc.scaling_factor()), nullptr};
  }
}

// Fold one operand's factor into an accumulated resolution. Host factors
// multiply freely; a device-resident factor must be the only non-unit factor
// (it cannot be multiplied on the host, and the backend takes one scalar).
template <class Sv>
auto combine_scaled_alpha(alpha_resolution<Sv>       total,
                          const alpha_resolution<Sv>& extra, const char* op)
  -> alpha_resolution<Sv> {
  const bool incompatible =
    (extra.from_device() &&
     (total.from_device() || total.host_value != Sv(1))) ||
    (total.from_device() && extra.host_value != Sv(1));
  if (incompatible) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      std::string{op} +
        ": a device_scalar scaled() factor cannot be combined with other "
        "factors (the cu/hipBLAS entry points take a single alpha)");
  }
  if (extra.from_device()) {
    total.device_ptr = extra.device_ptr;
    return total;
  }
  if (!total.from_device()) {
    total.host_value = total.host_value * extra.host_value;
  }
  return total;
}

// Whether two views cover exactly the same elements (same data handle and
// same mapping) — used by the accumulate overloads to decide between the
// backend's in-place beta path and a separate accumulation step. Views with
// different mapping types never alias.
template <class MD1, class MD2>
constexpr auto views_alias(const MD1& e, const MD2& c) -> bool {
  if constexpr (std::is_same_v<typename MD1::mapping_type,
                               typename MD2::mapping_type>) {
    return e.data_handle() == c.data_handle() && e.mapping() == c.mapping();
  } else {
    (void)e;
    (void)c;
    return false;
  }
}

GCXX_TEMPLATE(typename IdxT)
GCXX_REQUIRES(std::is_integral_v<IdxT>)
struct blas_output_view {
  IdxT rows;               // mathematical rows of the output
  IdxT cols;               // mathematical cols of the output
  IdxT leading_dimension;  // stride of the non-unit axis
  bool transposed;         // unit stride on axis 1 (row-major-like storage)
};

// Orientation of a rank-2 BLAS OUTPUT. cu/hipBLAS write results in
// column-major order, so an output whose unit stride is on axis 1 (a
// row-major-like mapping, e.g. layout_right) would receive the TRANSPOSE of
// the mathematical result. `transposed` flags that case; gemm-style callers
// react by presenting the transposed problem to the backend (swapped operand
// slots and m/n) so the mathematical contract C = A * B holds for every
// operand layout, per P1673R13.
template <class MD>
constexpr auto infer_blas_output_view(const MD& v)
  -> blas_output_view<typename MD::index_type> {
  static_assert(MD::rank() == 2, "BLAS output matrix must be rank-2");

  static_assert(gcxx::is_device_view_v<typename MD::accessor_type>,
                "BLAS matrix outputs must view device memory: use "
                "gcxx::device_mdspan / gcxx::managed_mdspan (or an mdspan "
                "carrying gcxx::device_accessor / gcxx::managed_accessor)");

  const auto s0 = v.stride(0);
  const auto s1 = v.stride(1);
  if (s0 == 1) {
    return {v.extent(0), v.extent(1), s1, false};
  }
  if (s1 == 1) {
    return {v.extent(0), v.extent(1), s0, true};
  }
  throw gcxx::blas::BlasException(
    GCXX_BLAS_STATUS(INVALID_VALUE),
    "BLAS matrix output must have a unit stride on one axis");
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
