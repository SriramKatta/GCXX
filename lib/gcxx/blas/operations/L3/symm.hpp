// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_SYMM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_SYMM_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/L3/geam.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/side.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// C = A*B | B*A with only the tagged triangle of symmetric A read.
namespace symm_impl_ {

// Flip a fill mode (Upper <-> Lower): the mirror of a stored triangle.
constexpr auto flip_fill_mode(driver::deviceBlasFillMode_t f)
  -> driver::deviceBlasFillMode_t {
  return f == driver::deviceBlasFillModeUpper ? driver::deviceBlasFillModeLower
                                              : driver::deviceBlasFillModeUpper;
}

// Flips Left <-> Right when presenting the transposed problem.
constexpr auto flip_side_mode(driver::deviceBlasSideMode_t s)
  -> driver::deviceBlasSideMode_t {
  return s == driver::deviceBlasSideLeft ? driver::deviceBlasSideRight
                                         : driver::deviceBlasSideLeft;
}

}  // namespace symm_impl_

// Write-only form: C = A*B (left) or C = B*A (right).
GCXX_TEMPLATE(class Side, class TA, class ExtentsA, class LayoutA,
              class AccessorA, class Tri, class TB, class ExtentsB, class LayoutB,
              class AccessorB, class TC, class ExtentsC, class LayoutC,
              class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsC::rank() == 2)
auto symmetric_matrix_product(
  BlasHandleView h, Side /*side*/,
  const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a, Tri /*triangle*/,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
  const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {

  // local alias for easier refrence
  using AVt = TA;
  using BVt = TB;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using CIt = typename ExtentsC::index_type;
  using Sv  = CVt;

  // static asserts to verify no funny business
  static_assert(!gcxx::is_scaled_accessor_v<AccessorC>,
                "symmetric_matrix_product outputs cannot be scaled() views; "
                "scale an input, or use the accumulate form with a scaled "
                "addend");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "symmetric_matrix_product operands A, B, C must share the "
                "same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, CVt>,
                "symmetric_matrix_product operands A, B, C must share a "
                "single element type");

  // TODO: Support complex element types once the C/Z dispatch branches exist.
  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "symmetric_matrix_product currently supports only f32_t/"
                "f64_t element types (complex support is a TODO)");

  // Alpha comes only from scaled() views on the inputs; beta is host zero.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()),
    "symmetric_matrix_product");
  if (alpha_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product: the write-only form has no device-resident "
      "beta, so a device_scalar scaled() factor is unsupported here; use the "
      "accumulate form (with a device_scalar zero addend) or host factors");
  }
  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr = &alpha_host;
  const Sv beta_host  = Sv(0);
  const Sv* beta_ptr  = &beta_host;

  // Pin host pointer mode for the call (restored on scope exit); both
  // scalars above are host values in this form.
  details_::BlasPointerModeGuard guard{h, false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                          = details_::infer_blas_output_view(c);

  constexpr driver::deviceBlasSideMode_t side = details_::side_mode_v<Side>;

  // dimension gates per side (fail here rather than inside the backend)
  if (rows_a != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "symmetric_matrix_product requires A to be "
                             "square");
  }
  if (side == driver::deviceBlasSideLeft &&
      (rows_b != rows_a || out.rows != rows_a || out.cols != cols_b)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product (left) requires B to be A.extent(0) x N and "
      "C to be A.extent(0) x B.extent(1)");
  }
  if (side == driver::deviceBlasSideRight &&
      (cols_b != rows_a || out.rows != rows_b || out.cols != rows_a)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product (right) requires B to be M x A.extent(0) and "
      "C to be B.extent(0) x A.extent(0)");
  }
  // the backend entry point takes no transpose flag for B, so B's storage
  // orientation must match C's (the row-major pair is dispatched as the
  // transposed problem with the side mode flipped)
  if ((op_b == driver::deviceBlasOpN) == out.transposed) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product requires B and C to share storage "
      "orientation: the backend entry point takes no transpose flag for B, "
      "so a column-major-like B must pair with a column-major-like C and a "
      "row-major-like B with a row-major-like C");
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag =
    details_::fill_mode_v<Tri>;
  // a row-major-like A is read as its transpose, whose stored triangle is
  // the mirror of the tagged one
  const auto uplo = op_a == driver::deviceBlasOpN
                      ? uplo_tag
                      : symm_impl_::flip_fill_mode(uplo_tag);

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    // Column-major-like B and C: the problem passes through as declared.
    GCXX_BLAS_DISPATCH_TYPED(
      status, AIt, AVt, symm, h.getRawHandle(), side, uplo, out.rows,
      out.cols, alpha_ptr, a.data_handle(), ld_a, b.data_handle(), ld_b,
      beta_ptr, c.data_handle(), out.leading_dimension);
  } else {
    // Row-major-like B and C: transposed problem; side flips, lds carry over.
    GCXX_BLAS_DISPATCH_TYPED(
      status, AIt, AVt, symm, h.getRawHandle(),
      symm_impl_::flip_side_mode(side), uplo, out.cols, out.rows, alpha_ptr,
      a.data_handle(), ld_a, b.data_handle(), ld_b, beta_ptr,
      c.data_handle(), out.leading_dimension);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "symmetric_matrix_product failed");
  }
}

// Accumulate form: E aliases C -> in-place beta path, else split via geam.
GCXX_TEMPLATE(class Side, class TA, class ExtentsA, class LayoutA,
              class AccessorA, class Tri, class TB, class ExtentsB, class LayoutB,
              class AccessorB, class TE, class ExtentsE, class LayoutE,
              class AccessorE, class TC, class ExtentsC, class LayoutC,
              class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsE::rank() == 2 GCXX_AND ExtentsC::rank() == 2)
auto symmetric_matrix_product(
  BlasHandleView h, Side side,
  const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a, Tri triangle,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
  const gcxx::mdspan<TE, ExtentsE, LayoutE, AccessorE>& e,
  const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {

  using AVt = TA;
  using BVt = TB;
  using EVt = TE;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using EIt = typename ExtentsE::index_type;
  using CIt = typename ExtentsC::index_type;
  using Sv  = CVt;

  static_assert(gcxx::details_::all_same_v<AIt, BIt, EIt, CIt>,
                "symmetric_matrix_product operands A, B, E, C must share the "
                "same mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, EVt, CVt>,
                "symmetric_matrix_product operands A, B, E, C must share a "
                "single element type");

  static_assert(gcxx::blas::details_::is_supported_blas_element_v<AVt>,
                "symmetric_matrix_product currently supports only f32_t/"
                "f64_t element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(e, "E");
  details_::validate_device_view(c, "C");

  if (e.extent(0) != c.extent(0) || e.extent(1) != c.extent(1)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product addend E must have the same extents as C");
  }

  // The addend's factor doubles as beta (aliased) or geam alpha (split).
  auto beta_res = details_::resolve_scaled_alpha<Sv>(e.accessor());

  if (!details_::views_alias(e, c)) {
    // Split path: write A*B into C, then accumulate E in place.
    if (beta_res.from_device()) {
      details_::throwBlasError(
        GCXX_BLAS_STATUS(INVALID_VALUE),
        "symmetric_matrix_product: a non-aliased addend with a "
        "device_scalar scaled() factor is unsupported: the in-place geam "
        "accumulation would have to read a device-resident alpha and a host "
        "beta through one pointer mode; use the aliased form "
        "symmetric_matrix_product(h, side, A, t, B, scaled(f, C), C) with a "
        "device-resident zero addend, or host factors");
    }
    symmetric_matrix_product(h, side, a, triangle, b, c);
    geam(h, beta_res.host_value, gcxx::strip_scaled(e), Sv(1), c, c);
    return;
  }

  // In-place path: one backend call with beta read from E's factor.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()),
    "symmetric_matrix_product");
  if (alpha_res.from_device() != beta_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product: the backend reads alpha and beta through "
      "one pointer mode, so host and device_scalar factors cannot be mixed "
      "in one call");
  }

  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;
  const Sv beta_host = beta_res.host_value;
  const Sv* beta_ptr =
    beta_res.from_device() ? beta_res.device_ptr : &beta_host;

  details_::BlasPointerModeGuard guard{h, alpha_res.from_device()};

  const auto [rows_a, cols_a, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [rows_b, cols_b, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                          = details_::infer_blas_output_view(c);

  constexpr driver::deviceBlasSideMode_t side_mode = details_::side_mode_v<Side>;

  if (rows_a != cols_a) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "symmetric_matrix_product requires A to be "
                             "square");
  }
  if (side_mode == driver::deviceBlasSideLeft &&
      (rows_b != rows_a || out.rows != rows_a || out.cols != cols_b)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product (left) requires B to be A.extent(0) x N and "
      "C to be A.extent(0) x B.extent(1)");
  }
  if (side_mode == driver::deviceBlasSideRight &&
      (cols_b != rows_a || out.rows != rows_b || out.cols != rows_a)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product (right) requires B to be M x A.extent(0) and "
      "C to be B.extent(0) x A.extent(0)");
  }
  if ((op_b == driver::deviceBlasOpN) == out.transposed) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "symmetric_matrix_product requires B and C to share storage "
      "orientation (see the write-only form)");
  }

  constexpr driver::deviceBlasFillMode_t uplo_tag =
    details_::fill_mode_v<Tri>;
  const auto uplo = op_a == driver::deviceBlasOpN
                      ? uplo_tag
                      : symm_impl_::flip_fill_mode(uplo_tag);

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_TYPED(
      status, AIt, AVt, symm, h.getRawHandle(), side_mode, uplo, out.rows,
      out.cols, alpha_ptr, a.data_handle(), ld_a, b.data_handle(), ld_b,
      beta_ptr, c.data_handle(), out.leading_dimension);
  } else {
    GCXX_BLAS_DISPATCH_TYPED(
      status, AIt, AVt, symm, h.getRawHandle(),
      symm_impl_::flip_side_mode(side_mode), uplo, out.cols, out.rows,
      alpha_ptr, a.data_handle(), ld_a, b.data_handle(), ld_b, beta_ptr,
      c.data_handle(), out.leading_dimension);
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "symmetric_matrix_product failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
