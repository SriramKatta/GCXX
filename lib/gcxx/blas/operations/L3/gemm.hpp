// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L3_GEMM_HPP_
#define GCXX_BLAS_OPERATIONS_L3_GEMM_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/L3/geam.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Matrix-matrix product, the P1673R13 matrix_product shape.
//
//   matrix_product(h, A, B, C)      ->  C = A * B      (write-only)
//   matrix_product(h, A, B, E, C)   ->  C = A * B + E  (accumulate)
//
// A, B, C (and the addend E) are rank-2 mdspans. The effective dimensions,
// layouts, and transpose state are inferred from the mdspan metadata and view
// wrappers, and the mathematical result C = A * B holds for ANY mix of
// operand layouts (column-major, row-major, or transposed views): when C's
// storage is row-major-like the dispatch presents the transposed problem
// (swapped operand slots, flipped op flags, swapped m/n) to the column-major
// backend. There is no alpha/beta parameter; per P1673R13 10.3, scaling is
// expressed with scaled(alpha, x) views, whose factors are unwrapped and
// folded into the backend's single alpha:
//
//   matrix_product(h, scaled(2.0, A), B, C);         // C = 2*A*B
//   matrix_product(h, A, B, scaled(0.5, C), C);      // C = A*B + 0.5*C
//
// The 3-argument form never reads C (the backend's beta is a host zero), so
// C may hold uninitialized or NaN data. The 5-argument accumulate form reads
// E: when E aliases C exactly (the canonical scaled(beta, C), C form) the
// backend's in-place beta path computes everything in one call; otherwise
// the product is written first and E is accumulated with a second, in-place
// geam call (the documented C = alpha*op(A) + beta*C mode), the two steps
// staying ordered on the handle's stream.
//
// At most one non-unit scaled() factor may be a gcxx::blas::device_scalar,
// and a device_scalar factor is only supported where it can pair with the
// call's other scalars under one pointer mode: the ALIASED accumulate form
// (all factors device-resident) — not the write-only form (implicit zero
// beta) and not a device factor on a NON-aliased addend (the in-place geam
// step needs a host beta). Violations throw with a message naming the
// supported alternative.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasGemmEx_64 entry point,
// while all other index_types use the standard 32-bit interface.
//
// A, B, C, and E must be device views: mdspans carrying
// gcxx::device_accessor / gcxx::managed_accessor (e.g. gcxx::device_mdspan).
// Host views are rejected at compile time; in check builds the data handles
// are additionally probed at run time so a mislabeled host pointer fails
// here, not inside the GPU kernel.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TC, class ExtentsC, class LayoutC, class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() == 2 GCXX_AND
              ExtentsC::rank() == 2)
auto matrix_product(BlasHandleView h,
                    const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
                    const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
                    const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c)
  -> void {

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
                "matrix_product outputs cannot be scaled() views; scale an "
                "input, or use the accumulate form with a scaled addend");

  static_assert(gcxx::details_::all_same_v<AIt, BIt, CIt>,
                "matrix_product operands A, B, C must share the same mdspan "
                "index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, CVt>,
                "matrix_product operands A, B, C must share a single element "
                "type");

  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "matrix_product currently supports only float/double element "
                "types (complex support is a TODO)");

  // alpha comes only from scaled() views on the inputs; beta is a host zero
  // in this write-only form, so C is never read.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()), "matrix_product");
  if (alpha_res.from_device()) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product: the write-only form has no device-resident beta, so a "
      "device_scalar scaled() factor is unsupported here; use the accumulate "
      "form (with a device_scalar zero addend) or host factors");
  }
  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr = &alpha_host;
  const Sv  beta_host = Sv(0);
  const Sv* beta_ptr  = &beta_host;

  // Pin host pointer mode for the call (restored on scope exit); both
  // scalars above are host values in this form.
  details_::BlasPointerModeGuard guard{h, false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(c, "C");

  // extract problem dimensions
  const auto [m, k, ld_a, op_a]   = details_::infer_blas_matrix_view(a);
  const auto [k_b, n, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                  = details_::infer_blas_output_view(c);

  // extent compatibility (mandated by P1673R13; fail here rather than inside
  // the backend with a confusing status)
  if (k != k_b) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product requires A.extent(1) == B.extent(0)");
  }
  if (out.rows != m || out.cols != n) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product requires C to be A.extent(0) x B.extent(1)");
  }

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmEx, h.getRawHandle(), op_a, op_b, m, n, k, alpha_ptr,
      a.data_handle(), cuda_datatype_v<AVt>, ld_a, b.data_handle(),
      cuda_datatype_v<BVt>, ld_b, beta_ptr, c.data_handle(),
      cuda_datatype_v<CVt>, out.leading_dimension, blas_compute_type_v<CVt>,
      GCXX_BLAS_GEMM(DEFAULT));
  } else {
    // C's storage is row-major-like: present the transposed problem
    // C^T = B^T * A^T to the column-major backend.
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmEx, h.getRawHandle(), details_::flip_blas_op(op_b),
      details_::flip_blas_op(op_a), n, m, k, alpha_ptr, b.data_handle(),
      cuda_datatype_v<BVt>, ld_b, a.data_handle(), cuda_datatype_v<AVt>, ld_a,
      beta_ptr, c.data_handle(), cuda_datatype_v<CVt>, out.leading_dimension,
      blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "matrix_product failed");
  }
}

// Accumulate form: C = A * B + E (P1673R13's read-and-write version). E may
// carry a scaled() factor. When E aliases C exactly the backend's in-place
// beta path computes everything in one call; otherwise the product is
// written to C first and E is accumulated with an in-place geam (which also
// applies E's factor). E must either alias C exactly or not overlap it at
// all.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TE, class ExtentsE, class LayoutE, class AccessorE,
              class TC, class ExtentsC, class LayoutC, class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() == 2 GCXX_AND
              ExtentsE::rank() == 2 GCXX_AND ExtentsC::rank() == 2)
auto matrix_product(BlasHandleView h,
                    const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
                    const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
                    const gcxx::mdspan<TE, ExtentsE, LayoutE, AccessorE>& e,
                    const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c)
  -> void {

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
                "matrix_product operands A, B, E, C must share the same "
                "mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, BVt, EVt, CVt>,
                "matrix_product operands A, B, E, C must share a single "
                "element type");

  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "matrix_product currently supports only float/double element "
                "types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(b, "B");
  details_::validate_device_view(e, "E");
  details_::validate_device_view(c, "C");

  if (e.extent(0) != c.extent(0) || e.extent(1) != c.extent(1)) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product addend E must have the same extents as C");
  }

  // The addend's factor (identity 1 for a plain E) doubles as the backend's
  // beta in the aliased case, and as geam's alpha in the split case.
  auto beta_res = details_::resolve_scaled_alpha<Sv>(e.accessor());

  if (!details_::views_alias(e, c)) {
    // Split path: write A*B into C, then accumulate E in place
    // (C = factor*E + 1*C, the documented in-place geam mode).
    if (beta_res.from_device()) {
      throw gcxx::blas::BlasException(
        GCXX_BLAS_STATUS(INVALID_VALUE),
        "matrix_product: a non-aliased addend with a device_scalar scaled() "
        "factor is unsupported: the in-place geam accumulation would have to "
        "read a device-resident alpha and a host beta through one pointer "
        "mode; use the aliased form matrix_product(h, A, B, scaled(f, C), C) "
        "with a device-resident zero addend, or host factors");
    }
    matrix_product(h, a, b, c);
    geam(h, beta_res.host_value, gcxx::strip_scaled(e), Sv(1), c, c);
    return;
  }

  // In-place path: one backend call with beta read from E's factor.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()), "matrix_product");
  if (alpha_res.from_device() != beta_res.from_device()) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product: the backend reads alpha and beta through one pointer "
      "mode, so host and device_scalar factors cannot be mixed in one call");
  }

  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;
  const Sv beta_host = beta_res.host_value;
  const Sv* beta_ptr =
    beta_res.from_device() ? beta_res.device_ptr : &beta_host;

  details_::BlasPointerModeGuard guard{h, alpha_res.from_device()};

  const auto [m, k, ld_a, op_a]   = details_::infer_blas_matrix_view(a);
  const auto [k_b, n, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                  = details_::infer_blas_output_view(c);

  if (k != k_b) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product requires A.extent(1) == B.extent(0)");
  }
  if (out.rows != m || out.cols != n) {
    throw gcxx::blas::BlasException(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product requires C to be A.extent(0) x B.extent(1)");
  }

  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmEx, h.getRawHandle(), op_a, op_b, m, n, k, alpha_ptr,
      a.data_handle(), cuda_datatype_v<AVt>, ld_a, b.data_handle(),
      cuda_datatype_v<BVt>, ld_b, beta_ptr, c.data_handle(),
      cuda_datatype_v<CVt>, out.leading_dimension, blas_compute_type_v<CVt>,
      GCXX_BLAS_GEMM(DEFAULT));
  } else {
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmEx, h.getRawHandle(), details_::flip_blas_op(op_b),
      details_::flip_blas_op(op_a), n, m, k, alpha_ptr, b.data_handle(),
      cuda_datatype_v<BVt>, ld_b, a.data_handle(), cuda_datatype_v<AVt>, ld_a,
      beta_ptr, c.data_handle(), cuda_datatype_v<CVt>, out.leading_dimension,
      blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
  }

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "matrix_product failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
