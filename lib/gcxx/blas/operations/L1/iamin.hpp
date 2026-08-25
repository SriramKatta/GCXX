// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_IAMIN_HPP_
#define GCXX_BLAS_OPERATIONS_L1_IAMIN_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Zero-based index of min |x[i]|; backend is one-based, translated here.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX)
GCXX_REQUIRES(ExtentsX::rank() == 1)
auto idx_abs_min(BlasHandleView h,
                 const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x) ->
  typename ExtentsX::index_type {

  // local alias for easier refrence
  using XVt = TX;
  using XIt = typename ExtentsX::index_type;

  // static asserts to verify no funny business
  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<XVt>,
                "idx_abs_min currently supports only f32_t/f64_t element "
                "types (complex support is a TODO)");

  // Pin host pointer mode for the call (restored on scope exit) so the result
  // lands in the host storage below.
  details_::BlasPointerModeGuard guard{h, false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  // the backend's result integer matches the interface selected by the index
  // type (int for the 32-bit interface, int64_t for the _64 one)
  XIt result{0};

  driver::deviceBlasStatus_t status{};
  if constexpr (std::is_same_v<XVt, gcxx::f32_t>) {
    GCXX_BLAS_DISPATCH_INT64(status, XIt, Isamin, h.getRawHandle(), len_x,
                             x.data_handle(), inc_x, &result);
  } else {
    GCXX_BLAS_DISPATCH_INT64(status, XIt, Idamin, h.getRawHandle(), len_x,
                             x.data_handle(), inc_x, &result);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "idx_abs_min failed");
  }

  // The backend's host-mode write may lag the host thread; make the returned
  // value observable before this function returns.
  h.getStream().sync();

  // translate the backend's one-based convention (0 = not found) to
  // zero-based indexing
  return result == 0 ? XIt{0} : result - XIt{1};
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
