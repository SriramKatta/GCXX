// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L2_GER_HPP_
#define GCXX_BLAS_OPERATIONS_L2_GER_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// A += alpha*x*y^T; alpha from scaled() views; always accumulates into A.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class TA, class ExtentsA, class LayoutA, class AccessorA)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() ==
              1 GCXX_AND ExtentsA::rank() == 2)
auto matrix_rank_1_update(
  BlasHandleView h, const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
  const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y,
  const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using YVt = TY;
  using AVt = TA;
  using XIt = typename ExtentsX::index_type;
  using YIt = typename ExtentsY::index_type;
  using AIt = typename ExtentsA::index_type;
  using Sv  = AVt;

  // static asserts to verify no funny business
  static_assert(!gcxx::is_scaled_accessor_v<AccessorA>,
                "matrix_rank_1_update outputs cannot be scaled() views; "
                "scale x or y instead");

  static_assert(gcxx::details_::all_same_v<XIt, YIt, AIt>,
                "matrix_rank_1_update operands x, y, A must share the same "
                "mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<XVt, YVt, AVt>,
                "matrix_rank_1_update operands x, y, A must share a single "
                "element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "matrix_rank_1_update currently supports only f32_t/f64_t "
                "element types (complex support is a TODO)");

  // Alpha comes only from scaled() views on the inputs.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(x.accessor()),
    details_::resolve_scaled_alpha<Sv>(y.accessor()), "matrix_rank_1_update");

  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit.
  details_::BlasPointerModeGuard guard{h, alpha_res.from_device()};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");
  details_::validate_device_view(y, "y");
  details_::validate_device_view(a, "A");

  // extract problem dimensions
  const auto [len_x, inc_x]               = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y]               = details_::infer_blas_vector_view(y);
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);

  if (len_x != rows_a || len_y != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "matrix_rank_1_update requires x to have "
                             "A.extent(0) elements and y to have "
                             "A.extent(1) elements");
  }

  driver::deviceBlasStatus_t status{};
  if (op_a == driver::deviceBlasOpN) {
    // A column-major-like: the problem passes through as declared.
    GCXX_BLAS_DISPATCH_TYPED(status, XIt, XVt, ger, h.getRawHandle(), rows_a,
                             cols_a, alpha_ptr, x.data_handle(), inc_x,
                             y.data_handle(), inc_y, a.data_handle(), ld_a);
  } else {
    // A row-major-like: A^T += alpha*y*x^T; slots and m/n swapped.
    GCXX_BLAS_DISPATCH_TYPED(status, XIt, XVt, ger, h.getRawHandle(), cols_a,
                             rows_a, alpha_ptr, y.data_handle(), inc_y,
                             x.data_handle(), inc_x, a.data_handle(), ld_a);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "matrix_rank_1_update failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
