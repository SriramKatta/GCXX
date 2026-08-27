// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_ROTM_HPP_
#define GCXX_BLAS_OPERATIONS_L1_ROTM_HPP_

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

// (x,y) <- H*(x,y); param = 5 device elements as produced by the setup.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY,
              class TP, class ExtentsP, class LayoutP, class AccessorP)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() ==
              1 GCXX_AND ExtentsP::rank() == 1)
auto apply_modified_givens_rotation(
  BlasHandleView h, const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
  const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y,
  const gcxx::mdspan<TP, ExtentsP, LayoutP, AccessorP>& param) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using YVt = TY;
  using PVt = TP;
  using XIt = typename ExtentsX::index_type;
  using YIt = typename ExtentsY::index_type;
  using PIt = typename ExtentsP::index_type;

  // static asserts to verify no funny business
  static_assert(gcxx::details_::all_same_v<XIt, YIt, PIt>,
                "apply_modified_givens_rotation operands x, y, param must "
                "share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<XVt, YVt, PVt>,
                "apply_modified_givens_rotation operands x, y, param must "
                "share a single element type");

  // rotm has no complex entry point; f32_t/f64_t are the whole backend set.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<XVt>,
                "apply_modified_givens_rotation supports only f32_t/f64_t "
                "element types");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(x, "x");
  details_::validate_device_view(y, "y");
  details_::validate_device_view(param, "param");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y] = details_::infer_blas_vector_view(y);
  const auto [len_p, inc_p] = details_::infer_blas_vector_view(param);

  // extent compatibility: the backend takes a single n and writes BOTH
  // vectors, so mismatched extents would write past the shorter allocation
  if (len_x != len_y) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "apply_modified_givens_rotation requires x and y to have the same "
      "length");
  }
  if (len_p != 5) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "apply_modified_givens_rotation requires param to have exactly 5 "
      "elements (flag + four stored H entries)");
  }
  // the backend reads the five entries densely; a strided param view would
  // read past the mapping
  if (inc_p != 1) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "apply_modified_givens_rotation requires param to have unit stride");
  }

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, XIt, XVt, rotm, h.getRawHandle(), len_x,
                           x.data_handle(), inc_x, y.data_handle(), inc_y,
                           param.data_handle());

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status,
                             /*msg*/ "apply_modified_givens_rotation failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
