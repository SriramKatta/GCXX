// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_COPY_HPP_
#define GCXX_BLAS_OPERATIONS_L1_COPY_HPP_

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

// to = from on the handle's stream (async); no pointer mode involved.
GCXX_TEMPLATE(class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY)
GCXX_REQUIRES(ExtentsX::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto copy(BlasHandleView h, const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& from,
          const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& to) -> void {

  // local alias for easier refrence
  using XVt = TX;
  using YVt = TY;
  using XIt = typename ExtentsX::index_type;
  using YIt = typename ExtentsY::index_type;

  // static asserts to verify no funny business
  static_assert(gcxx::details_::all_same_v<XIt, YIt>,
                "copy operands from, to must share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<XIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<XVt, YVt>,
                "copy operands from, to must share a single element type");

  // TODO: Wire complex Ccopy/Zcopy into GCXX_BLAS_DISPATCH_TYPED.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<XVt>,
                "copy currently supports only f32_t/f64_t element types "
                "(complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(from, "from");
  details_::validate_device_view(to, "to");

  // extract problem dimensions
  const auto [len_x, inc_x] = details_::infer_blas_vector_view(from);
  const auto [len_y, inc_y] = details_::infer_blas_vector_view(to);

  // extent compatibility: the backend takes a single n for both vectors, so
  // mismatched extents would run to past its allocation
  if (len_x != len_y) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "copy requires from and to to have the same "
                             "length");
  }

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, XIt, XVt, copy, h.getRawHandle(), len_x,
                           from.data_handle(), inc_x, to.data_handle(), inc_y);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "copy failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
