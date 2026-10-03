// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_HELPERS_BLAS_HELPERS_HPP_
#define GCXX_BLAS_HELPERS_BLAS_HELPERS_HPP_

#include <limits>
#include <string>
#include <type_traits>

#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Host <-> device staging over the handle-less cu/hipBLAS set/get entry
// points. Both operands are accessor-gated mdspans resolved with the
// infer_* machinery from op_inference.hpp — the device side through
// infer_blas_*_view (device_accessor / managed_accessor), the host side
// through its infer_host_*_view twins (host_accessor / managed_accessor).
// Extents, element types and strides are checked before anything copies.
// The async forms queue on the handle's stream (host memory should be
// pinned or the copy degrades to synchronous); the sync forms block the
// host and involve neither the handle nor its stream.
//
// The eight public entry points are a 2x2x2 cube (set/get x vector/matrix
// x sync/async) over two shared cores that carry the checks once.

namespace details_ {

  // The portable set/get entry points are int-sized (hipBLAS ships no _64
  // forms), so a resolved int64 index must be narrowed; refuse rather than
  // silently truncate (a wrong length/stride walks past the buffer).
  template <class It>
  GCXX_FH auto to_backend_int(It value, const char* what) -> int {
    static_assert(std::is_integral_v<It> && std::is_signed_v<It>,
                  "resolved mdspan indices must be signed integers");
    if (value < It(0) ||
        value > static_cast<It>(std::numeric_limits<int>::max())) {
      throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                     (std::string{"BLAS host<->device staging: "} + what +
                      " does not fit the backend's int-sized copy interface")
                       .c_str());
    }
    return static_cast<int>(value);
  }

  // Vector staging core: ToDevice selects set (dev = host) over get
  // (host = dev); Async queues on the handle's stream instead of blocking
  // the host. host and dev are rank-1 views of the same length and element
  // type (either side may be strided).
  GCXX_TEMPLATE(bool ToDevice, bool Async, class TH, class ExtentsH,
                class LayoutH, class AccessorH, class TD, class ExtentsD,
                class LayoutD, class AccessorD)
  GCXX_REQUIRES(ExtentsH::rank() == 1 GCXX_AND ExtentsD::rank() == 1)
  auto stage_vector_core(
    BlasHandleView h,
    const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host,
    const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev) -> void {
    using Vt  = std::remove_cv_t<TD>;
    using HIt = typename ExtentsH::index_type;
    using DIt = typename ExtentsD::index_type;

    // static asserts to verify no funny business (shared by every staging
    // entry point; the destination side flips with the copy direction)
    static_assert(
      !std::is_const_v<std::conditional_t<ToDevice, TD, TH>>,
      "host<->device staging destination must have a writable (non-const) "
      "element type");
    static_assert(!gcxx::is_scaled_accessor_v<AccessorH> &&
                    !gcxx::is_scaled_accessor_v<AccessorD>,
                  "host<->device staging cannot take scaled() views (the "
                  "scaling factor would be dropped)");
    static_assert(gcxx::details_::all_same_v<HIt, DIt>,
                  "staging operands host, dev must share the same mdspan "
                  "index_type");
    static_assert(gcxx::blas::details_::is_supported_blas_index_v<DIt>,
                  "BLAS operands must use int32_t or int64_t as their "
                  "mdspan index_type");
    static_assert(gcxx::details_::all_same_v<std::remove_cv_t<TH>, Vt>,
                  "staging operands host, dev must share a single element "
                  "type");
    static_assert(std::is_trivially_copyable_v<Vt>,
                  "host<->device staging copies raw bytes, so the element "
                  "type must be trivially copyable");

    // the sync entry points involve neither handle nor stream
    if constexpr (!Async) {
      (void)h;
    }

    // run-time device-memory probe (no-op unless checks are enabled)
    validate_device_view(dev, "dev");

    // resolve both views, then forward to the backend copy
    const auto [len_h, inc_h] = infer_host_vector_view(host);
    const auto [len_d, inc_d] = infer_blas_vector_view(dev);

    // extent compatibility: the backend takes one n for both sides, so
    // mismatched lengths would run past the shorter buffer
    if (len_h != len_d) {
      throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                     /*msg*/
                     "host<->device vector staging requires host and dev to "
                     "have the same length");
    }

    const auto n    = to_backend_int(len_h, "vector length");
    const auto incx = to_backend_int(inc_h, "host vector stride");
    const auto incy = to_backend_int(inc_d, "vector stride");
    if constexpr (Async) {
      const auto stream = h.getStream().getRawHandle();
      if constexpr (ToDevice) {
        driver::blasSetVectorAsync(stream, n, static_cast<int>(sizeof(Vt)),
                                   host.data_handle(), incx, dev.data_handle(),
                                   incy);
      } else {
        driver::blasGetVectorAsync(stream, n, static_cast<int>(sizeof(Vt)),
                                   dev.data_handle(), incy, host.data_handle(),
                                   incx);
      }
    } else {
      if constexpr (ToDevice) {
        driver::blasSetVector(n, static_cast<int>(sizeof(Vt)),
                              host.data_handle(), incx, dev.data_handle(),
                              incy);
      } else {
        driver::blasGetVector(n, static_cast<int>(sizeof(Vt)),
                              dev.data_handle(), incy, host.data_handle(),
                              incx);
      }
    }
  }

  // Matrix staging core: ToDevice selects set over get; Async queues on the
  // handle's stream instead of blocking the host. host and dev are rank-2
  // views with equal extents, one element type and one storage orientation
  // (the backend copies the raw column-major image and cannot transpose).
  GCXX_TEMPLATE(bool ToDevice, bool Async, class TH, class ExtentsH,
                class LayoutH, class AccessorH, class TD, class ExtentsD,
                class LayoutD, class AccessorD)
  GCXX_REQUIRES(ExtentsH::rank() == 2 GCXX_AND ExtentsD::rank() == 2)
  auto stage_matrix_core(
    BlasHandleView h,
    const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host,
    const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev) -> void {
    using Vt  = std::remove_cv_t<TD>;
    using HIt = typename ExtentsH::index_type;
    using DIt = typename ExtentsD::index_type;

    // static asserts to verify no funny business (shared by every staging
    // entry point; the destination side flips with the copy direction)
    static_assert(
      !std::is_const_v<std::conditional_t<ToDevice, TD, TH>>,
      "host<->device staging destination must have a writable (non-const) "
      "element type");
    static_assert(!gcxx::is_scaled_accessor_v<AccessorH> &&
                    !gcxx::is_scaled_accessor_v<AccessorD>,
                  "host<->device staging cannot take scaled() views (the "
                  "scaling factor would be dropped)");
    static_assert(gcxx::details_::all_same_v<HIt, DIt>,
                  "staging operands host, dev must share the same mdspan "
                  "index_type");
    static_assert(gcxx::blas::details_::is_supported_blas_index_v<DIt>,
                  "BLAS operands must use int32_t or int64_t as their "
                  "mdspan index_type");
    static_assert(gcxx::details_::all_same_v<std::remove_cv_t<TH>, Vt>,
                  "staging operands host, dev must share a single element "
                  "type");
    static_assert(std::is_trivially_copyable_v<Vt>,
                  "host<->device staging copies raw bytes, so the element "
                  "type must be trivially copyable");

    // the sync entry points involve neither handle nor stream
    if constexpr (!Async) {
      (void)h;
    }

    // run-time device-memory probe (no-op unless checks are enabled)
    validate_device_view(dev, "dev");

    // resolve both views, then forward to the backend copy
    const auto [rows_h, cols_h, ld_h, op_h] = infer_host_matrix_view(host);
    const auto [rows_d, cols_d, ld_d, op_d] = infer_blas_matrix_view(dev);

    // extent compatibility: the backend takes one shape for both sides, so
    // mismatched extents would run past the smaller view
    if (rows_h != rows_d || cols_h != cols_d) {
      throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                     /*msg*/
                     "host<->device matrix staging requires host and dev to "
                     "have the same extents");
    }

    // cu/hipBLAS copies the raw column-major image (no transpose), so both
    // sides must read that image with the same logical coordinates.
    if (op_h != op_d) {
      throwBlasError(
        GCXX_BLAS_STATUS(INVALID_VALUE),
        /*msg*/
        "host<->device matrix staging requires host and dev to share one "
        "storage orientation (both column-major-like or both row-major-like); "
        "the backend copy cannot transpose");
    }

    // A row-major-like (op = T) pair copies through its transposed image, so
    // every element lands in its mathematical position on both sides.
    const auto rows = op_d == driver::deviceBlasOpN ? rows_d : cols_d;
    const auto cols = op_d == driver::deviceBlasOpN ? cols_d : rows_d;

    const auto n_rows = to_backend_int(rows, "matrix rows");
    const auto n_cols = to_backend_int(cols, "matrix cols");
    const auto ldd    = to_backend_int(ld_d, "matrix leading dimension");
    const auto ldh    = to_backend_int(ld_h, "host leading dimension");
    if constexpr (Async) {
      const auto stream = h.getStream().getRawHandle();
      if constexpr (ToDevice) {
        driver::blasSetMatrixAsync(
          stream, n_rows, n_cols, static_cast<int>(sizeof(Vt)),
          host.data_handle(), ldh, dev.data_handle(), ldd);
      } else {
        driver::blasGetMatrixAsync(
          stream, n_rows, n_cols, static_cast<int>(sizeof(Vt)),
          dev.data_handle(), ldd, host.data_handle(), ldh);
      }
    } else {
      if constexpr (ToDevice) {
        driver::blasSetMatrix(n_rows, n_cols, static_cast<int>(sizeof(Vt)),
                              host.data_handle(), ldh, dev.data_handle(), ldd);
      } else {
        driver::blasGetMatrix(n_rows, n_cols, static_cast<int>(sizeof(Vt)),
                              dev.data_handle(), ldd, host.data_handle(), ldh);
      }
    }
  }

}  // namespace details_

// dev = host (blocking); host and dev are rank-1 views of the same length
// and element type (either side may be strided).
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 1 GCXX_AND ExtentsD::rank() == 1)
auto set_vector(BlasHandleView h,
                const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host,
                const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev)
  -> void {
  details_::stage_vector_core</*ToDevice*/ true, /*Async*/ false>(h, host, dev);
}

// host = dev (blocking); host and dev are rank-1 views of the same length
// and element type (either side may be strided).
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 1 GCXX_AND ExtentsD::rank() == 1)
auto get_vector(BlasHandleView h,
                const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev,
                const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host)
  -> void {
  details_::stage_vector_core</*ToDevice*/ false, /*Async*/ false>(h, host,
                                                                   dev);
}

// dev = host, queued on the handle's stream; host and dev are rank-1 views
// of the same length and element type (either side may be strided); host
// should view pinned memory.
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 1 GCXX_AND ExtentsD::rank() == 1)
auto set_vector_async(
  BlasHandleView h, const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host,
  const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev) -> void {
  details_::stage_vector_core</*ToDevice*/ true, /*Async*/ true>(h, host, dev);
}

// host = dev, queued on the handle's stream; host and dev are rank-1 views
// of the same length and element type (either side may be strided); host
// should view pinned memory.
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 1 GCXX_AND ExtentsD::rank() == 1)
auto get_vector_async(
  BlasHandleView h, const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev,
  const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host) -> void {
  details_::stage_vector_core</*ToDevice*/ false, /*Async*/ true>(h, host, dev);
}

// dev = host (blocking); host and dev are rank-2 views with equal extents,
// one element type and one storage orientation (the backend copies the raw
// column-major image and cannot transpose).
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 2 GCXX_AND ExtentsD::rank() == 2)
auto set_matrix(BlasHandleView h,
                const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host,
                const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev)
  -> void {
  details_::stage_matrix_core</*ToDevice*/ true, /*Async*/ false>(h, host, dev);
}

// host = dev (blocking); host and dev are rank-2 views with equal extents,
// one element type and one storage orientation (the backend copies the raw
// column-major image and cannot transpose).
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 2 GCXX_AND ExtentsD::rank() == 2)
auto get_matrix(BlasHandleView h,
                const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev,
                const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host)
  -> void {
  details_::stage_matrix_core</*ToDevice*/ false, /*Async*/ false>(h, host,
                                                                   dev);
}

// dev = host, queued on the handle's stream; host and dev are rank-2 views
// with equal extents, one element type and one storage orientation (the
// backend copies the raw column-major image and cannot transpose); host
// should view pinned memory.
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 2 GCXX_AND ExtentsD::rank() == 2)
auto set_matrix_async(
  BlasHandleView h, const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host,
  const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev) -> void {
  details_::stage_matrix_core</*ToDevice*/ true, /*Async*/ true>(h, host, dev);
}

// host = dev, queued on the handle's stream; host and dev are rank-2 views
// with equal extents, one element type and one storage orientation (the
// backend copies the raw column-major image and cannot transpose); host
// should view pinned memory.
GCXX_TEMPLATE(class TH, class ExtentsH, class LayoutH, class AccessorH,
              class TD, class ExtentsD, class LayoutD, class AccessorD)
GCXX_REQUIRES(ExtentsH::rank() == 2 GCXX_AND ExtentsD::rank() == 2)
auto get_matrix_async(
  BlasHandleView h, const gcxx::mdspan<TD, ExtentsD, LayoutD, AccessorD>& dev,
  const gcxx::mdspan<TH, ExtentsH, LayoutH, AccessorH>& host) -> void {
  details_::stage_matrix_core</*ToDevice*/ false, /*Async*/ true>(h, host, dev);
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
