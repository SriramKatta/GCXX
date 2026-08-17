// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_DOT_HPP_
#define GCXX_BLAS_OPERATIONS_L1_DOT_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Dot product result = x . y.
//
// x and y are rank-1 mdspans; the length n and the increments (incx, incy) are
// inferred from the mdspan metadata. The type-erased cu/hipblasDotEx entry
// point is used, with the data-type and execution-type enums derived from the
// element type. Each operand is typed as a gcxx::mdspan in the signature, so
// wrong-rank (or non-mdspan) arguments fail overload resolution.
//
// Example:
//   double r{};
//   gcxx::blas::dot(h, x, y, &r);   // host pointer: host pointer mode
//
// result must point to storage matching the handle's CURRENT pointer mode:
// host memory in host pointer mode (written synchronously on call return),
// device memory in device pointer mode (written asynchronously). Unlike the
// scalar-taking operations there is no per-call mode switch here, so the
// caller is in charge of the mode/destination pairing.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasDotEx_64 entry point,
// while all other index_types use the standard 32-bit interface.
//
// x and y must be device views: mdspans carrying gcxx::device_accessor /
// gcxx::managed_accessor (e.g. gcxx::make_device_vector). Host views are
// rejected at compile time; in check builds the data handles are
// additionally probed at run time so a mislabeled host pointer fails here,
// not inside the GPU kernel. The raw result pointer is NOT covered by the
// gate — it must match the handle's current pointer mode per the note above.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class R = TX)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto dot(BlasHandleView h,
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

  static_assert(gcxx::blas::details_::is_supported_blas_element_v<XVt>,
                "dot currently supports only f32_t/f64_t element types "
                "(complex support is a TODO)");

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
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
