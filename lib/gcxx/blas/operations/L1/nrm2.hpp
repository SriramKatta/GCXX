// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_NRM2_HPP_
#define GCXX_BLAS_OPERATIONS_L1_NRM2_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Euclidean norm result = ||x||_2.
//
// x is a rank-1 mdspan; the length n and the increment (incx) are inferred
// from the mdspan metadata. The type-erased cu/hipblasNrm2Ex entry point is
// used, with the data-type and execution-type enums derived from the element
// type.
//
// Example:
//   double r{};
//   gcxx::blas::nrm2(h, x, &r);
//
// result must point to storage matching the handle's CURRENT pointer mode:
// host memory in host pointer mode (written synchronously on call return),
// device memory in device pointer mode (written asynchronously). Unlike the
// scalar-taking operations there is no per-call mode switch here, so the
// caller is in charge of the mode/destination pairing.
//
// The integer interface is selected from the operand's mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasNrm2Ex_64 entry point,
// while all other index_types use the standard 32-bit interface.
template <class X, class R = typename std::decay_t<X>::element_type>
auto nrm2(BlasHandleView h, const X& x, R* result) -> void {

  // local alias for easier refrence
  using X_t = std::decay_t<X>;
  using XVt = typename X_t::element_type;
  using XIt = typename X_t::index_type;

  // static asserts to verify no funny business
  static_assert(X_t::rank() == 1, "nrm2 operand x must be a rank-1 mdspan");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(std::is_same_v<R, XVt>,
                "nrm2 result value type must match the operand's element "
                "type");

  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "nrm2 currently supports only float/double element types "
                "(complex support is a TODO)");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, Nrm2Ex, h.getRawHandle(), len_x,
                           x.data_handle(), cuda_datatype_v<XVt>, inc_x,
                           static_cast<void*>(result), cuda_datatype_v<R>,
                           cuda_datatype_v<R>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "nrm2 failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
