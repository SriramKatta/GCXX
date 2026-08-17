// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_NRM2_HPP_
#define GCXX_BLAS_OPERATIONS_L1_NRM2_HPP_

#include <cmath>
#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Euclidean norm result = ||x||_2, in the P1673R13 vector_two_norm shapes.
//
// Three overloads:
//   vector_two_norm(h, x)                    -> result, synchronizes the
//                                              handle's stream first
//   vector_two_norm(h, x, init)              -> sqrt(init^2 + ||x||_2^2)
//                                              (host-side accumulation, also
//                                              synchronizes)
//   vector_two_norm(h, x, device_scalar<R>)  -> void; writes the result to
//                                              the wrapped device pointer
//                                              asynchronously (device pointer
//                                              mode), no synchronization
//
// The returning forms are P1673R13's interface; a GPU backend cannot return
// an asynchronously computed scalar, so they block on the stream first. The
// device_scalar form is gcxx's asynchronous counterpart. Unlike the
// raw-pointer form it once replaced, the storage space is carried by the
// argument type instead of depending on the handle's ambient pointer mode.
//
// x is a rank-1 mdspan; the length n and the increment (incx) are inferred
// from the mdspan metadata. The type-erased cu/hipblasNrm2Ex entry point is
// used, with the data-type and execution-type enums derived from the element
// type. The operand is typed as a gcxx::mdspan in the signature, so wrong-rank
// (or non-mdspan) arguments fail overload resolution.
//
// The integer interface is selected from the operand's mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasNrm2Ex_64 entry point,
// while all other index_types use the standard 32-bit interface.
//
// x must be a device view: an mdspan carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::make_device_vector). Host views are
// rejected at compile time; in check builds the data handle is additionally
// probed at run time so a mislabeled host pointer fails here, not inside the
// GPU kernel.
namespace nrm2_impl_ {
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto sync_nrm2(BlasHandleView h,
               const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
               R* result) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using XIt = typename ExtentsX::index_type;

  // static asserts to verify no funny business
  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(std::is_same_v<R, XVt>,
                "vector_two_norm result value type must match the operand's "
                "element type");

  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "vector_two_norm currently supports only float/double "
                "element types (complex support is a TODO)");

  // Pin host pointer mode for the call (restored on scope exit) so the result
  // lands in the host storage below.
  details_::BlasPointerModeGuard guard{h, false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, Nrm2Ex, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           static_cast<void*>(result), cuda_datatype_v<R>,
                           cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "vector_two_norm failed");
  }

  // The backend's host-mode write may lag the host thread; make the returned
  // value observable before this function returns.
  h.getStream().Synchronize();
}
}  // namespace nrm2_impl_

// Returning form: vector_two_norm(h, x) -> ||x||_2 (synchronizes).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x)
  -> TX {
  TX result{};
  nrm2_impl_::sync_nrm2(h, x, &result);
  return result;
}

// Returning form with accumulation: vector_two_norm(h, x, init) ->
// sqrt(init^2 + ||x||_2^2) (synchronizes; the accumulation happens on the
// host, matching P1673R13's init semantics).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                     R init) -> R {
  R result{};
  nrm2_impl_::sync_nrm2(h, x, &result);
  using std::sqrt;
  return sqrt(init * init + result * result);
}

// Asynchronous form: vector_two_norm(h, x, device_scalar<R>) writes the
// result to the wrapped device pointer on the handle's stream (device
// pointer mode; the handle's prior mode is restored on return).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                     gcxx::blas::device_scalar<R> result) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using XIt = typename ExtentsX::index_type;

  // static asserts to verify no funny business
  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(std::is_same_v<R, XVt>,
                "vector_two_norm result value type must match the operand's "
                "element type");

  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "vector_two_norm currently supports only float/double "
                "element types (complex support is a TODO)");

  // Select device pointer mode for this call; the result is written to the
  // wrapped device pointer asynchronously.
  details_::BlasPointerModeGuard guard{h, true};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, Nrm2Ex, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           static_cast<void*>(const_cast<R*>(result.ptr)),
                           cuda_datatype_v<R>, cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "vector_two_norm failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
