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

// Dot product result = x . y, in the P1673R13 dot shapes.
//
// Three overloads:
//   dot(h, x, y)                    -> result, synchronizes the handle's
//                                      stream before returning
//   dot(h, x, y, init)              -> init + x . y (host-side accumulation,
//                                      also synchronizes)
//   dot(h, x, y, device_scalar<R>)  -> void; writes the result to the wrapped
//                                      device pointer asynchronously (device
//                                      pointer mode), no synchronization
//
// The returning forms are P1673R13's interface; a GPU backend cannot return
// an asynchronously computed scalar, so they block on the stream first. The
// device_scalar form is gcxx's asynchronous counterpart (P1673R13 explicitly
// excludes asynchronous scalar results); unlike the raw-pointer form it once
// replaced, the storage space is carried by the argument type instead of
// depending on the handle's ambient pointer mode.
//
// x and y are rank-1 mdspans; the length n and the increments (incx, incy)
// are inferred from the mdspan metadata. The type-erased cu/hipblasDotEx
// entry point is used, with the data-type and execution-type enums derived
// from the element type. Each operand is typed as a gcxx::mdspan in the
// signature, so wrong-rank (or non-mdspan) arguments fail overload
// resolution.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasDotEx_64 entry point,
// while all other index_types use the standard 32-bit interface.
//
// x and y must be device views: mdspans carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::make_device_vector). Host views are
// rejected at compile time; in check builds the data handles are
// additionally probed at run time so a mislabeled host pointer fails here,
// not inside the GPU kernel.
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

  // unused vars just to supress annoying warnings
  (void)len_y;

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
  h.getStream().Synchronize();
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

// Returning form with accumulation: dot(h, x, y, init) -> init + x . y
// (synchronizes; the accumulation happens on the host, matching P1673R13's
// init semantics).
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

// Asynchronous form: dot(h, x, y, device_scalar<R>) writes the result to the
// wrapped device pointer on the handle's stream (device pointer mode; the
// handle's prior mode is restored on return).
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

  // unused vars just to supress annoying warnings
  (void)len_y;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(
    status, XIt, DotEx, h.getRawHandle(), len_x, x.data_handle(),
    cuda_datatype_v<XVt>, inc_x, y.data_handle(), cuda_datatype_v<YVt>, inc_y,
    static_cast<void*>(const_cast<R*>(result.ptr)), cuda_datatype_v<R>,
    cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "dot failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
