// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_ASUM_HPP_
#define GCXX_BLAS_OPERATIONS_L1_ASUM_HPP_

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

// asum: returning forms sync the stream; the device_scalar form is async.
namespace asum_impl_ {
  GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
                class R = TX)
  GCXX_REQUIRES(ExtentsX::rank() == 1)
  auto sync_asum(BlasHandleView h,
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
                  "vector_abs_sum result value type must match the operand's "
                  "element type");

    // TODO: Support complex element types once C/Z dispatch branches exist.
    static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                  "vector_abs_sum currently supports only float/double "
                  "element types (complex support is a TODO)");

    // Pin host pointer mode for the call (restored on scope exit) so the
    // result lands in the host storage below.
    details_::BlasPointerModeGuard guard{h, false};

    // run-time device-memory probe (no-op unless checks are enabled)
    details_::validate_device_view(x, "x");

    // extract problem dimensions
    const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

    driver::deviceBlasStatus_t status{};
    GCXX_BLAS_DISPATCH_INT64(status, XIt, AsumEx, h.getRawHandle(), len_x,
                             x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                             static_cast<void*>(result), cuda_datatype_v<R>,
                             cuda_datatype_v<R>);

    if (status != driver::deviceBlasStatusSuccess) {
      details_::throwBlasError(status, "vector_abs_sum failed");
    }

    // The backend's host-mode write may lag the host thread; make the
    // returned value observable before this function returns.
    h.getStream().Synchronize();
  }
}  // namespace asum_impl_

// Returning form: vector_abs_sum(h, x) -> ||x||_1 (synchronizes).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_abs_sum(BlasHandleView h,
                    const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x)
  -> TX {
  TX result{};
  asum_impl_::sync_asum(h, x, &result);
  return result;
}

// Returning form with host-side accumulation (synchronizes): init + ||x||_1.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_abs_sum(BlasHandleView h,
                    const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                    R init) -> R {
  R result{};
  asum_impl_::sync_asum(h, x, &result);
  return init + result;
}

// Async form: writes the result to the device_scalar pointer (device mode).
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto vector_abs_sum(BlasHandleView h,
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
                "vector_abs_sum result value type must match the operand's "
                "element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "vector_abs_sum currently supports only float/double "
                "element types (complex support is a TODO)");

  // Select device pointer mode for this call; the result is written to the
  // wrapped device pointer asynchronously.
  details_::BlasPointerModeGuard guard{h, true};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, AsumEx, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           static_cast<void*>(const_cast<R*>(result.ptr)),
                           cuda_datatype_v<R>, cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "vector_abs_sum failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
