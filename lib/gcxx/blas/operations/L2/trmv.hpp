// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L2_TRMV_HPP_
#define GCXX_BLAS_OPERATIONS_L2_TRMV_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/L1/axpy.hpp>
#include <gcxx/blas/operations/L1/copy.hpp>
#include <gcxx/blas/operations/L1/scal.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/diagonal.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// y = A*x, staged (backend trmv is in-place); only the tagged triangle read.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class Tri, class Diag, class TX, class ExtentsX, class LayoutX,
              class AccessorX, class TY, class ExtentsY, class LayoutY,
              class AccessorY)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsY::rank() == 1)
auto triangular_matrix_vector_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  Tri /*triangle*/, Diag /*diagonal_storage*/,
  const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
  const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using XVt = TX;
  using YVt = TY;
  using AIt = typename ExtentsA::index_type;
  using XIt = typename ExtentsX::index_type;
  using YIt = typename ExtentsY::index_type;
  using Sv  = YVt;

  // static asserts to verify no funny business
  static_assert(!gcxx::is_scaled_accessor_v<AccessorY>,
                "triangular_matrix_vector_product outputs cannot be scaled() "
                "views; scale x instead");

  static_assert(gcxx::details_::all_same_v<AIt, XIt, YIt>,
                "triangular_matrix_vector_product operands A, x, y must "
                "share the same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, XVt, YVt>,
                "triangular_matrix_vector_product operands A, x, y must "
                "share a single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "triangular_matrix_vector_product currently supports only "
                "f32_t/f64_t element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(y, "y");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_x, inc_x]               = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y]               = details_::infer_blas_vector_view(y);

  if (rows_a != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             /*msg*/
                             "triangular_matrix_vector_product requires A "
                             "to be square");
  }
  if (len_x != cols_a || len_y != rows_a) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_vector_product requires x and y to have "
      "A.extent(0) elements");
  }

  // stage x into y, apply any scaled() factor from x and A to the staged
  // copy (the backend's trmv has no alpha), then apply the rotation in place
  copy(h, x, y);
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(x.accessor()),
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    "triangular_matrix_vector_product");
  if (alpha_res.from_device()) {
    scale(h, gcxx::blas::device_scalar<Sv>{alpha_res.device_ptr}, y);
  } else if (alpha_res.host_value != Sv(1)) {
    scale(h, alpha_res.host_value, y);
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag = details_::fill_mode_v<Tri>;
  constexpr driver::deviceBlasDiagType_t diag = details_::diagonal_type_v<Diag>;

  // a row-major-like operand is read as its transpose: the mirrored triangle
  // plus the flipped op flag recover the mathematical A
  const auto uplo =
    details_::mirrored_fill_mode(op_a != driver::deviceBlasOpN, uplo_tag);
  const auto trans = op_a;  // N stays N (column-major-like); T flips it back

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, trmv, h.getRawHandle(), uplo,
                           trans, diag, rows_a, a.data_handle(), ld_a,
                           y.data_handle(), inc_y);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status,
                             /*msg*/
                             "triangular_matrix_vector_product "
                             "failed");
  }
}

// Accumulate form: split path only; b must NOT alias y (in-place staging).
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class Tri, class Diag, class TX, class ExtentsX, class LayoutX,
              class AccessorX, class TB, class ExtentsB, class LayoutB,
              class AccessorB, class TY, class ExtentsY, class LayoutY,
              class AccessorY)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsB::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto triangular_matrix_vector_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  Tri triangle, Diag diagonal_storage,
  const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
  const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y) -> void {

  using AVt = TA;
  using XVt = TX;
  using BVt = TB;
  using YVt = TY;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using YIt = typename ExtentsY::index_type;
  using Sv  = YVt;

  static_assert(gcxx::details_::all_same_v<AIt, BIt, YIt>,
                "triangular_matrix_vector_product operands must share the "
                "same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, XVt, BVt, YVt>,
                "triangular_matrix_vector_product operands A, x, b, y must "
                "share a single element type");

  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "triangular_matrix_vector_product currently supports only "
                "f32_t/f64_t element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(b, "b");
  details_::validate_device_view(y, "y");

  if (b.extent(0) != y.extent(0)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_vector_product addend b must have the same extent "
      "as y");
  }
  if (details_::views_alias(b, y)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "triangular_matrix_vector_product: the addend b must not alias y "
      "(the in-place trmv staging would destroy it); pass a distinct "
      "addend view");
  }

  triangular_matrix_vector_product(h, a, triangle, diagonal_storage, x, y);

  auto beta_res = details_::resolve_scaled_alpha<Sv>(b.accessor());
  if (beta_res.from_device()) {
    axpy(h, gcxx::blas::device_scalar<Sv>{beta_res.device_ptr},
         gcxx::strip_scaled(b), y);
  } else {
    axpy(h, beta_res.host_value, gcxx::strip_scaled(b), y);
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
