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

// nrm2: returning forms sync the stream; the device-resident scalar form is
// async.
namespace blas_impl {

  // Shared host/device-mode core: device_mode selects the result write's
  // pointer mode; host mode additionally syncs so the caller's stack
  // result is observable on return.
  GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
                class R = TX)
  GCXX_REQUIRES(ExtentsX::rank() == 1)
  auto nrm2_core(BlasHandleView h,
                 const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                 R* result, const bool device_mode) -> void {

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
    // device pointer asynchronously.
    const details_::BlasPointerModeGuard guard{h, device_mode};

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
      details_::throwBlasError(status, /*msg*/ "vector_two_norm failed");
    }
    // Host-mode results are consumed by the caller right after the
    // call; device-mode writes are read out asynchronously.
    if (!device_mode) {
      h.getStream().sync();
    }
  }
  GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
                class R = TX)
  GCXX_REQUIRES(ExtentsX::rank() == 1)
  auto sync_nrm2(BlasHandleView h,
                 const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                 R* result) -> void {
    nrm2_core(h, x, result, /*device_mode*/ false);
  }

}  // namespace blas_impl

// Returning form: vector_two_norm(h, x) -> ||x||_2 (synchronizes).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x)
  -> TX {
  TX result{};
  blas_impl::sync_nrm2(h, x, &result);
  return result;
}

// Returning form: sqrt(init^2 + ||x||^2), host-side accumulation (syncs).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                     R init) -> R {
  R result{};
  blas_impl::sync_nrm2(h, x, &result);
  using std::sqrt;
  return sqrt(init * init + result * result);
}

// Async form: writes the result to the device_scalar_view pointer (device
// mode).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                     gcxx::device_scalar_view<R> result) -> void {
  blas_impl::nrm2_core(h, x, const_cast<R*>(result.ptr),
                       /*device_mode*/ true);
}

// Async form into an owning device-accessible scalar (device mode). The
// scalar must outlive the call; read it via result.value() (which syncs the
// scalar's stream) once the write is expected to be done.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX, class... Properties)
GCXX_REQUIRES(ExtentsX::rank() ==
              1 GCXX_AND is_device_accessible<Properties...>)
auto vector_two_norm(BlasHandleView h,
                     const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                     gcxx::scalar<R, Properties...>& result) -> void {
  blas_impl::nrm2_core(h, x, result.data(), /*device_mode*/ true);
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
