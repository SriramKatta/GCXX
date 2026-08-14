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
// element type.
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
template <class X, class Y,
          class R = typename std::decay_t<X>::element_type>
auto dot(BlasHandleView h, const X& x, const Y& y, R* result) -> void {

  // local alias for easier refrence
  using X_t = std::decay_t<X>;
  using Y_t = std::decay_t<Y>;
  using XVt = typename X_t::element_type;
  using YVt = typename Y_t::element_type;
  using XIt = typename X_t::index_type;
  using YIt = typename Y_t::index_type;

  // static asserts to verify no funny business
  static_assert(X_t::rank() == 1 && Y_t::rank() == 1,
                "dot operands x, y must be rank-1 mdspans");

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

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y] = details_::infer_blas_vector_view(y);

  // unused vars just to supress annoying warnings
  (void)len_y;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, DotEx, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           y.data_handle(), cuda_datatype_v<YVt>, inc_y,
                           static_cast<void*>(result), cuda_datatype_v<R>,
                           cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "dot failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
