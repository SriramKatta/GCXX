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

// Givens rotation applied to (x, y):
//   x_i = c * x_i + s * y_i
//   y_i = c * y_i - s * x_i   (with the pre-rotation x_i)
//
// x and y are rank-1 mdspans; the length n and the increments (incx, incy) are
// inferred from the mdspan metadata. The type-erased cu/hipblasRotEx entry
// point is used, with the data-type and execution-type enums derived from the
// element type. Each operand is typed as a gcxx::mdspan in the signature, so
// wrong-rank (or non-mdspan) arguments fail overload resolution.
//
// Example:
//   gcxx::blas::rot(h, 0.8, 0.6, x, y);
//
// c/s may be passed either as host scalars (host pointer mode) or as
// gcxx::blas::device_scalar<T> wrapping a device pointer (device pointer
// mode). The mode is selected per call from the argument type; the handle's
// prior pointer mode is restored when the call returns.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasRotEx_64 entry point,
// while all other index_types use the standard 32-bit interface.
//
// x and y must be device views: mdspans carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::make_device_vector). Host views are
// rejected at compile time; in check builds the data handles are
// additionally probed at run time so a mislabeled host pointer fails here,
// not inside the GPU kernel.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class S = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto rot(BlasHandleView h, S c, S s,
         const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
         const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y) -> void {

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
                "rot operands x, y must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<Sv, XVt, YVt>,
                "rot c/s value type must match the operands' element type");

  static_assert(gcxx::blas::details_::is_supported_blas_element_v<XVt>,
                "rot currently supports only f32_t/f64_t element types "
                "(complex support is a TODO)");

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

  // unused vars just to supress annoying warnings
  (void)len_y;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, RotEx, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           y.data_handle(), cuda_datatype_v<YVt>, inc_y, c_ptr,
                           s_ptr, cuda_datatype_v<Sv>, cuda_datatype_v<XVt>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "rot failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
