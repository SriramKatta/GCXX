// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_OP_INFERENCE_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_OP_INFERENCE_HPP_

#include <string>

#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/error/blas_exceptions.hpp>
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>
#include <gcxx/runtime_backend/backend_memory.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// Runtime check that operand pointers are truly device memory (opt-in).
#ifdef GCXX_ENABLE_RUNTIME_CHECKS
template <class MD>
GCXX_FH auto validate_device_view(const MD& v, const char* name) -> void {
  if (!driver::isDeviceUsableMemory(v.data_handle())) {
    std::string msg{"BLAS operand '"};
    msg += name;
    msg +=
      "' does not reside in device-accessible memory (a device_accessor view "
      "was passed, but the pointer is neither device/managed memory nor "
      "pinned host memory with a device mapping)";
    throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE), msg.c_str());
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

GCXX_TEMPLATE(typename IdxT)
GCXX_REQUIRES(std::is_integral_v<IdxT>)
struct blas_vector_view {
  IdxT length;  // number of elements
  IdxT stride;  // increment between elements (incx / incy)
};

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

// Batch-first (batch, rows, cols); layout_left only works for batch <= 1.
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

  if (v.stride(1) != 1 && v.stride(2) != 1) {
    throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "BLAS batched matrix operand's inner matrices must have a unit stride "
      "on one axis: a layout_left (batch, rows, cols) operand with batch > 1 "
      "interleaves the inner matrices (its batch axis is the unit-stride "
      "one), which a single (base, batch stride) pair cannot express; use "
      "layout_right, or a layout_stride packing the matrices contiguously "
      "with the batch outermost");
  }
  const auto [ld, op] = infer_blas_op_view(v.stride(1), v.stride(2));
  const bool transposed =
    v.stride(1) != 1 && v.stride(2) == 1;  // inner matrix row-major-like
  return {v.extent(1), v.extent(2), ld,        v.extent(0),
          v.stride(0), op,          transposed};
}

// Scaled()-view factor resolution.

// A resolved scalar multiplier: host value or one device-side factor pointer.
template <class Sv>
struct alpha_resolution {
  Sv host_value{1};
  const Sv* device_ptr{nullptr};

  GCXX_CXPR bool from_device() const noexcept { return device_ptr != nullptr; }
};

// Unwraps a scaled() factor into an alpha_resolution (identity by default).
template <class Sv, class Accessor>
constexpr auto resolve_scaled_alpha(const Accessor&) -> alpha_resolution<Sv> {
  return {};
}

template <class Sv, class ScalingFactor, class NestedAccessor>
constexpr auto resolve_scaled_alpha(
  const gcxx::scaled_accessor<ScalingFactor, NestedAccessor>& acc)
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

// A device factor must be the sole non-unit factor (one backend alpha).
template <class Sv>
auto combine_scaled_alpha(alpha_resolution<Sv> total,
                          const alpha_resolution<Sv>& extra,
                          const char* op) -> alpha_resolution<Sv> {
  const bool incompatible =
    (extra.from_device() &&
     (total.from_device() || total.host_value != Sv(1))) ||
    (total.from_device() && extra.host_value != Sv(1));
  if (incompatible) {
    throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      (std::string{op} +
       ": a device_scalar scaled() factor cannot be combined with other "
       "factors (the cu/hipBLAS entry points take a single alpha)")
        .c_str());
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

// Same handle+mapping; picks in-place beta vs separate accumulation.
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

// Flags row-major-like OUTPUTs that would store the column-major transpose.
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
  throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                 "BLAS matrix output must have a unit stride on one axis");
}

// Infer the BLAS view (length + increment) of a rank-1 mdspan operand.
template <class VD>
constexpr auto infer_blas_vector_view(const VD& v)
  -> blas_vector_view<typename VD::index_type> {
  static_assert(VD::rank() == 1, "BLAS vector operand must be rank-1");
  return {v.extent(0), v.stride(0)};
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
