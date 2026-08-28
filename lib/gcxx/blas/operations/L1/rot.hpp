// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_ROT_HPP_
#define GCXX_BLAS_OPERATIONS_L1_ROT_HPP_

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

// Givens rotation on (x, y); c/s may be host scalars or device_scalar.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class S = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto apply_givens_rotation(
  BlasHandleView h, const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
  const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y, S c, S s) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using YVt = TY;
  using XIt = typename ExtentsX::index_type;
  using YIt = typename ExtentsY::index_type;

  // Value type carried by c/s: unwraps device_scalar<T> -> T. A
  // device_scalar argument selects device pointer mode; a plain scalar selects
  // host mode.
  using Sv                   = details_::scalar_value_t<S>;
  constexpr bool device_mode = details_::is_device_scalar_v<S>;

  // static asserts to verify no funny business
  static_assert(gcxx::details_::all_same_v<XIt, YIt>,
                "apply_givens_rotation operands x, y must share the same "
                "mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, XVt, YVt>,
                "apply_givens_rotation c/s value type must match the "
                "operands' element type");

  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "apply_givens_rotation currently supports only float/double "
                "element types (complex support is a TODO)");

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit; c/s are read from the host parameters or the device pointers carried
  // by device_scalar, per the mode.
  details_::BlasPointerModeGuard guard{h, device_mode};

  const Sv* c_ptr = details_::blas_scalar_ptr(c);
  const Sv* s_ptr = details_::blas_scalar_ptr(s);

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");
  details_::validate_device_view(y, "y");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y] = details_::infer_blas_vector_view(y);

  // extent compatibility: the backend takes a single n and writes BOTH
  // vectors, so mismatched extents would write past the shorter allocation
  if (len_x != len_y) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/ "apply_givens_rotation requires x and y to have the same length");
  }

  driver::deviceBlasStatus_t
    status{};  // NOLINT(misc-const-correctness) assigned by the dispatch below
  // The macro's two branches spell distinct entry points (RotEx_64 vs
  // RotEx); the checker cannot tell them apart in uninstantiated code.
  // NOLINTNEXTLINE(bugprone-branch-clone)
  GCXX_BLAS_DISPATCH_INT64(status, XIt, RotEx, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           y.data_handle(), cuda_datatype_v<YVt>, inc_y, c_ptr,
                           s_ptr, cuda_datatype_v<Sv>, cuda_datatype_v<XVt>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "apply_givens_rotation failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
