// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_SCAL_HPP_
#define GCXX_BLAS_OPERATIONS_L1_SCAL_HPP_

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

// Vector scaling x = alpha * x.
//
// x is a rank-1 mdspan; the length n and the increment (incx) are inferred
// from the mdspan metadata. The type-erased cu/hipblasScalEx entry point is
// used, with the data-type and execution-type enums derived from the element
// type.
//
// Example:
//   gcxx::blas::scal(h, 3.0, x);
//
// alpha may be passed either as a host scalar (host pointer mode) or as a
// gcxx::blas::device_scalar<T> wrapping a device pointer (device pointer
// mode). The mode is selected per call from the argument type; the handle's
// prior pointer mode is restored when the call returns.
//
// The integer interface is selected from the operand's mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasScalEx_64 entry point,
// while all other index_types use the standard 32-bit interface.
template <class X, class S = typename std::decay_t<X>::element_type>
auto scal(BlasHandleView h, S alpha, X&& x) -> void {

  // local alias for easier refrence
  using X_t = std::decay_t<X>;
  using XVt = typename X_t::element_type;
  using XIt = typename X_t::index_type;

  // Value type carried by alpha: unwraps device_scalar<T> -> T. A
  // device_scalar argument selects device pointer mode; a plain scalar selects
  // host mode.
  using Sv                   = details_::scalar_value_t<S>;
  constexpr bool device_mode = details_::is_device_scalar_v<S>;

  // static asserts to verify no funny business
  static_assert(X_t::rank() == 1, "scal operand x must be a rank-1 mdspan");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(std::is_same_v<Sv, XVt>,
                "scal alpha value type must match the operand's element "
                "type");

  static_assert(std::is_same_v<XVt, float> || std::is_same_v<XVt, double>,
                "scal currently supports only float/double element types "
                "(complex support is a TODO)");

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit; alpha is read from the host parameter or the device pointer carried
  // by device_scalar, per the mode.
  details_::BlasPointerModeGuard guard{h, device_mode};

  const Sv* alpha_ptr = details_::blas_scalar_ptr(alpha);

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_INT64(status, XIt, ScalEx, h.getRawHandle(), len_x,
                           alpha_ptr, cuda_datatype_v<Sv>, x.data_handle(),
                           cuda_datatype_v<XVt>, inc_x, cuda_datatype_v<Sv>);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "scal failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
