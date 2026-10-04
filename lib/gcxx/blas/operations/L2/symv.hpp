// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L2_SYMV_HPP_
#define GCXX_BLAS_OPERATIONS_L2_SYMV_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/L1/axpy.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// y = A*x with only the tagged triangle read; scaling via scaled() inputs.
namespace blas_impl {

  // Shared dispatch core: both public forms resolve their alpha/beta factors
  // first, then hand the raw scalar pointers and the matching pointer mode to
  // this core.
  GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
                class Tri, class TX, class ExtentsX, class LayoutX,
                class AccessorX, class TY, class ExtentsY, class LayoutY,
                class AccessorY)
  GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
                1 GCXX_AND ExtentsY::rank() == 1)
  auto symv_core(BlasHandleView h,
                 const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
                 Tri /*triangle*/,
                 const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
                 const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y,
                 const TY* alpha_ptr, const TY* beta_ptr,
                 const bool device_mode) -> void {

    // local alias for easier refrence
    using AVt = TA;
    using XVt = TX;
    using YVt = TY;
    using AIt = typename ExtentsA::index_type;
    using XIt = typename ExtentsX::index_type;
    using YIt = typename ExtentsY::index_type;

    // static asserts to verify no funny business
    static_assert(!gcxx::is_scaled_accessor_v<AccessorY>,
                  "symmetric_matrix_vector_product outputs cannot be scaled() "
                  "views; scale an input, or use the accumulate form with a "
                  "scaled addend");

    static_assert(gcxx::details_::all_same_v<AIt, XIt, YIt>,
                  "symmetric_matrix_vector_product operands A, x, y must share "
                  "the same mdspan index_type");

    static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                  "BLAS operands must use int32_t or int64_t as their "
                  "mdspan index_type");

    static_assert(gcxx::details_::all_same_v<AVt, XVt, YVt>,
                  "symmetric_matrix_vector_product operands A, x, y must share "
                  "a single element type");

    // TODO: Support complex element types once the C/Z dispatch branches exist.
    static_assert(
      gcxx::blas::details_::is_supported_blas_element_v<AVt>,
      "symmetric_matrix_vector_product currently supports only "
      "float32_t/float64_t element types (complex support is a TODO)");

    // Pin the caller-chosen pointer mode for the call (restored on scope
    // exit); alpha_ptr and beta_ptr must both live in that space.
    const details_::BlasPointerModeGuard guard{h, device_mode};

    // run-time device-memory probe (no-op unless checks are enabled)
    details_::validate_device_view(a, "A");
    details_::validate_device_view(x, "x");
    details_::validate_device_view(y, "y");

    // extract problem dimensions
    const auto [rows_a, cols_a, ld_a, op_a] =
      details_::infer_blas_matrix_view(a);
    const auto [len_x, inc_x] = details_::infer_blas_vector_view(x);
    const auto [len_y, inc_y] = details_::infer_blas_vector_view(y);

    if (rows_a != cols_a) {
      details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                               /*msg*/
                               "symmetric_matrix_vector_product requires A to "
                               "be square");
    }
    if (len_x != cols_a || len_y != rows_a) {
      details_::throwBlasError(
        GCXX_BLAS_STATUS(INVALID_VALUE),
        /*msg*/
        "symmetric_matrix_vector_product requires x and y to have "
        "A.extent(0) elements");
    }

    // The tag object itself is unused (the mode comes from its type).
    constexpr driver::deviceBlasFillMode_t uplo_tag =
      details_::fill_mode_v<Tri>;
    // a row-major-like operand is read as its transpose, whose stored triangle
    // is the mirror of the tagged one
    const auto uplo =
      details_::mirrored_fill_mode(op_a != driver::deviceBlasOpN, uplo_tag);

    driver::deviceBlasStatus_t status{};
    GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, symv, h.getRawHandle(), uplo,
                             rows_a, alpha_ptr, a.data_handle(), ld_a,
                             x.data_handle(), inc_x, beta_ptr, y.data_handle(),
                             inc_y);

    if (status != driver::deviceBlasStatusSuccess) {
      details_::throwBlasError(
        status,
        /*msg*/ "symmetric_matrix_vector_product failed");
    }
  }

}  // namespace blas_impl

// Write-only form: y = A * x.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class Tri, class TX, class ExtentsX, class LayoutX,
              class AccessorX, class TY, class ExtentsY, class LayoutY,
              class AccessorY)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsY::rank() == 1)
auto symmetric_matrix_vector_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  Tri /*triangle*/, const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
  const gcxx::mdspan<TY, ExtentsY, LayoutY, AccessorY>& y) -> void {

  using Sv = TY;

  // Alpha comes only from scaled() views on the inputs; beta is host zero.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(x.accessor()),
    "symmetric_matrix_vector_product");
  if (alpha_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "symmetric_matrix_vector_product: the write-only form has no "
      "device-resident beta, so a device-resident scalar scaled() factor is "
      "unsupported here; use the accumulate form (with a device-resident "
      "scalar zero "
      "addend) or host factors");
  }
  const Sv alpha_host = alpha_res.host_value;
  const Sv beta_host  = Sv(0);

  blas_impl::symv_core(h, a, Tri{}, x, y, &alpha_host, &beta_host,
                       /*device_mode*/ false);
}

// Accumulate form: b aliases y -> in-place beta path, else split via axpy.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class Tri, class TX, class ExtentsX, class LayoutX,
              class AccessorX, class TB, class ExtentsB, class LayoutB,
              class AccessorB, class TY, class ExtentsY, class LayoutY,
              class AccessorY)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsB::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto symmetric_matrix_vector_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  Tri /*triangle*/, const gcxx::mdspan<TX, ExtentsX, LayoutX, AccessorX>& x,
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
                "symmetric_matrix_vector_product operands must share the "
                "same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, XVt, BVt, YVt>,
                "symmetric_matrix_vector_product operands A, x, b, y must "
                "share a single element type");

  static_assert(
    gcxx::blas::details_::is_supported_blas_element_v<AVt>,
    "symmetric_matrix_vector_product currently supports only "
    "float32_t/float64_t element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(b, "b");
  details_::validate_device_view(y, "y");

  if (b.extent(0) != y.extent(0)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "symmetric_matrix_vector_product addend b must have the same extent "
      "as y");
  }

  // The addend's factor doubles as beta (aliased) or axpy alpha (split).
  auto beta_res = details_::resolve_scaled_alpha<Sv>(b.accessor());

  if (!details_::views_alias(b, y)) {
    // Split path: write A*x into y, then accumulate b.
    symmetric_matrix_vector_product(h, a, Tri{}, x, y);
    if (beta_res.from_device()) {
      axpy(h, gcxx::device_scalar_view<Sv>{beta_res.device_ptr},
           gcxx::strip_scaled(b), y);
    } else {
      axpy(h, beta_res.host_value, gcxx::strip_scaled(b), y);
    }
    return;
  }

  // In-place path: one backend call with beta read from b's factor.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(x.accessor()),
    "symmetric_matrix_vector_product");
  if (alpha_res.from_device() != beta_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "symmetric_matrix_vector_product: the backend reads alpha and beta "
      "through one pointer mode, so host and device-resident scalar factors "
      "cannot be "
      "mixed in one call");
  }

  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;
  const Sv beta_host = beta_res.host_value;
  const Sv* beta_ptr =
    beta_res.from_device() ? beta_res.device_ptr : &beta_host;

  blas_impl::symv_core(h, a, Tri{}, x, y, alpha_ptr, beta_ptr,
                       alpha_res.from_device());
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
