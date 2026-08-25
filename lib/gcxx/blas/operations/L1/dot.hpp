// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_DOT_HPP_
#define GCXX_BLAS_OPERATIONS_L1_DOT_HPP_

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

// dot: returning forms sync the stream; the device_scalar form is async.
namespace dot_impl_ {
  GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
                class TY, class ExtentsY, class LayoutY, class AccessorY,
                class R = TX)
  GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
  auto sync_dot(BlasHandleView h,
                const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y,
                R* result) -> void {

    // local alias for easier refrence
    using XVt = TX;
    using YVt = TY;
    using XIt = typename ExtentsX::index_type;
    using YIt = typename ExtentsY::index_type;

    // static asserts to verify no funny business
    static_assert(gcxx::details_::all_same_v<XIt, YIt>,
                  "dot operands x, y must share the same mdspan index_type");

    static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                  "BLAS operands must use int32_t or int64_t as their "
                  "mdspan index_type");

    static_assert(gcxx::details_::all_same_v<R, XVt, YVt>,
                  "dot result value type must match the operands' element "
                  "type");

    static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                  "dot currently supports only float/double element types "
                  "(complex support is a TODO)");

    // Pin host pointer mode for the call (restored on scope exit) so the result
    // lands in the host storage below.
    details_::BlasPointerModeGuard guard{h, false};

    // run-time device-memory probe (no-op unless checks are enabled)
    details_::validate_device_view(x, "x");
    details_::validate_device_view(y, "y");

    // extract problem dimensions
    const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);
    const auto [len_y, inc_y] = details_::infer_blas_vector_view(y);

    // extent compatibility: the backend takes a single n for both vectors, so
    // mismatched extents would read y past its allocation
    if (len_x != len_y) {
      details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                               "dot requires x and y to have the same length");
    }

    driver::deviceBlasStatus_t status{};
    GCXX_BLAS_DISPATCH_INT64(
      status, XIt, DotEx, h.getRawHandle(), len_x, x.data_handle(),
      cuda_datatype_v<XVt>, inc_x, y.data_handle(), cuda_datatype_v<YVt>, inc_y,
      static_cast<void*>(result), cuda_datatype_v<R>, cuda_datatype_v<R>);

    if (status != driver::deviceBlasStatusSuccess) {
      details_::throwBlasError(status, "dot failed");
    }

    // The backend's host-mode write may lag the host thread; make the returned
    // value observable before this function returns.
    h.getStream().sync();
  }
}  // namespace dot_impl_

// Returning form: dot(h, x, y) -> x . y (synchronizes).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto dot(BlasHandleView h,
         const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
         const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y) -> TX {
  TX result{};
  dot_impl_::sync_dot(h, x, y, &result);
  return result;
}

// Returning form with host-side accumulation (synchronizes): init + x . y.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto dot(BlasHandleView h,
         const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
         const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y, R init) -> R {
  R result{};
  dot_impl_::sync_dot(h, x, y, &result);
  return init + result;
}

// Async form: writes the result to the device_scalar pointer (device mode).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto dot(BlasHandleView h,
         const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
         const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y,
         gcxx::blas::device_scalar<R> result) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using YVt = TY;
  using XIt = typename ExtentsX::index_type;
  using YIt = typename ExtentsY::index_type;

  // static asserts to verify no funny business
  static_assert(gcxx::details_::all_same_v<XIt, YIt>,
                "dot operands x, y must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<R, XVt, YVt>,
                "dot result value type must match the operands' element "
                "type");

  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "dot currently supports only float/double element types "
                "(complex support is a TODO)");

  // Select device pointer mode for this call; the result is written to the
  // wrapped device pointer asynchronously.
  details_::BlasPointerModeGuard guard{h, true};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");
  details_::validate_device_view(y, "y");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y] = details_::infer_blas_vector_view(y);

  // extent compatibility: the backend takes a single n for both vectors, so
  // mismatched extents would read y past its allocation
  if (len_x != len_y) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "dot requires x and y to have the same length");
  }

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, DotEx, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           y.data_handle(), cuda_datatype_v<YVt>, inc_y,
                           static_cast<void*>(const_cast<R*>(result.ptr)),
                           cuda_datatype_v<R>, cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "dot failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
