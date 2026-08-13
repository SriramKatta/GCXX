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
<<<<<<< HEAD
#include <gcxx/blas/operations/L3/geam.hpp>
=======
>>>>>>> f6989c9 (Amending to new examples)
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/blas/operations/details/scalar.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
<<<<<<< HEAD
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
=======
>>>>>>> f6989c9 (Amending to new examples)
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

<<<<<<< HEAD
// C = A*B; scaling via scaled() input views, any mix of operand layouts.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TC, class ExtentsC, class LayoutC, class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsC::rank() == 2)
auto matrix_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
  const gcxx::mdspan<TB, ExtentsB, LayoutB, AccessorB>& b,
  const gcxx::mdspan<TC, ExtentsC, LayoutC, AccessorC>& c) -> void {
=======
#define GCXX_BLAS_GEMM(name) \
  GCXX_DIRECT_BACKEND_ALT(CUBLAS_GEMM_##name, HIPBLAS_GEMM_##name)

// Matrix-matrix product C = alpha * op(A) * op(B) + beta * C.
//
// A, B, and C are rank-2 mdspan objects. The effective dimensions, layout, and
// transpose state are inferred from the mdspan metadata and any view wrappers,
// so the API does not take separate shape or operation arguments.
//
// Example:
//   gcxx::blas::gemm(h, 1.0, A, B, 4.0, C);    // computes C = A * B + 4.0 * C
//   gcxx::blas::gemm(h, 1.0, blas::transpose(A), B, 0.0, C); // computes C =
//   A^T * B
//
// Because cu/hipBLAS internally treats matrices in column-major form, the
// conventional usage to compute A * B is to pass the operands in the order
// (B, A, C) when matching the backend's storage convention.
//
// alpha/beta may be passed either as host scalars (host pointer mode) or as
// gcxx::blas::device_scalar<T> wrapping a device pointer (device pointer mode).
// The mode is selected per call from the argument type; the handle's prior
// pointer mode is restored when the call returns.
//
// The integer interface is selected from the operands' mdspan index_type: an
// int64_t index_type routes to the 64-bit cu/hipblasGemmEx_64 entry point
// (int64_t dimensions), while all other index_types use the standard 32-bit
// interface.
template <class A, class B, class C,
          class S = typename std::decay_t<C>::element_type>
auto gemm(BlasHandleView h, S alpha, const A& a, const B& b, S beta,
          C&& c) -> void {
>>>>>>> f6989c9 (Amending to new examples)

  // local alias for easier refrence
  using AVt = TA;
  using BVt = TB;
  using CVt = TC;
  using AIt = typename ExtentsA::index_type;
  using BIt = typename ExtentsB::index_type;
  using CIt = typename ExtentsC::index_type;
  using Sv  = CVt;

  // Value type carried by alpha/beta: unwraps device_scalar<T> -> T. A
  // device_scalar argument selects device pointer mode; a plain scalar selects
  // host mode.
  using Sv                   = details_::scalar_value_t<S>;
  constexpr bool device_mode = details_::is_device_scalar_v<S>;

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

<<<<<<< HEAD
  static_assert(gcxx::details_::all_same_v<AVt, BVt, CVt>,
                "matrix_product operands A, B, C must share a single element "
                "type");

  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "matrix_product currently supports only float/double element "
                "types (complex support is a TODO)");

  // Alpha comes only from scaled() views on the inputs; beta is host zero.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(b.accessor()), "matrix_product");
  if (alpha_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product: the write-only form has no device-resident beta, so a "
      "device_scalar scaled() factor is unsupported here; use the accumulate "
      "form (with a device_scalar zero addend) or host factors");
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
=======
  static_assert(gcxx::details_::all_same_v<Sv, AVt, BVt, CVt>,
                "gemm alpha/beta value type must match the operands' element "
                "type");

  // Select the pointer mode for this call and restore the prior mode on scope
  // exit. Host mode reads alpha/beta from the by-value parameters; device mode
  // reads them from the device pointers carried by device_scalar (no host copy,
  // no host-side dereference).
  details_::BlasPointerModeGuard guard{
    h, device_mode ? driver::deviceBlasPointerModeDevice
                   : driver::deviceBlasPointerModeHost};

  const Sv* alpha_ptr{};
  const Sv* beta_ptr{};
  if constexpr (device_mode) {
    alpha_ptr = alpha.ptr;
    beta_ptr  = beta.ptr;
  } else {
    alpha_ptr = &alpha;
    beta_ptr  = &beta;
  }
>>>>>>> f6989c9 (Amending to new examples)

  // extract problem dimensions
  const auto [m, k, ld_a, op_a]   = details_::infer_blas_matrix_view(a);
  const auto [k_b, n, ld_b, op_b] = details_::infer_blas_matrix_view(b);
  const auto out                  = details_::infer_blas_output_view(c);

  // Extent compatibility (P1673R13); fail here rather than in the backend.
  if (k != k_b) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "matrix_product requires A.extent(1) == "
                             "B.extent(0)");
  }
  if (out.rows != m || out.cols != n) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "matrix_product requires C to be A.extent(0) x "
                             "B.extent(1)");
  }

<<<<<<< HEAD
  driver::deviceBlasStatus_t status{};
  if (!out.transposed) {
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmEx, h.getRawHandle(), op_a, op_b, m, n, k, alpha_ptr,
      a.data_handle(), cuda_datatype_v<AVt>, ld_a, b.data_handle(),
      cuda_datatype_v<BVt>, ld_b, beta_ptr, c.data_handle(),
      cuda_datatype_v<CVt>, out.leading_dimension, blas_compute_type_v<CVt>,
      GCXX_BLAS_GEMM(DEFAULT));
  } else {
    // C row-major-like: present the transposed problem C^T = B^T * A^T.
    GCXX_BLAS_DISPATCH_INT64(
      status, AIt, GemmEx, h.getRawHandle(), details_::flip_blas_op(op_b),
      details_::flip_blas_op(op_a), n, m, k, alpha_ptr, b.data_handle(),
      cuda_datatype_v<BVt>, ld_b, a.data_handle(), cuda_datatype_v<AVt>, ld_a,
      beta_ptr, c.data_handle(), cuda_datatype_v<CVt>, out.leading_dimension,
      blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
  }
=======
  driver::deviceBlasStatus_t status;
  GCXX_BLAS_DISPATCH_INT64(
    status, AIt, GemmEx, h.getRawHandle(), op_a, op_b, m, n, k, alpha_ptr,
    a.data_handle(), cuda_datatype_v<AVt>, ld_a, b.data_handle(),
    cuda_datatype_v<BVt>, ld_b, beta_ptr, c.data_handle(), cuda_datatype_v<CVt>,
    ld_c, blas_compute_type_v<CVt>, GCXX_BLAS_GEMM(DEFAULT));
>>>>>>> f6989c9 (Amending to new examples)

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, "matrix_product failed");
  }
}

// Accumulate form: E aliases C -> in-place beta path, else split via geam.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TE, class ExtentsE, class LayoutE, class AccessorE,
              class TC, class ExtentsC, class LayoutC, class AccessorC)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsB::rank() ==
              2 GCXX_AND ExtentsE::rank() == 2 GCXX_AND ExtentsC::rank() == 2)
auto matrix_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
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
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      "matrix_product addend E must have the same extents as C");
  }

  // The addend's factor doubles as beta (aliased) or geam alpha (split).
  auto beta_res = details_::resolve_scaled_alpha<Sv>(e.accessor());

  if (!details_::views_alias(e, c)) {
    // Split path: write A*B into C, then accumulate E in place
    // (C = factor*E + 1*C, the documented in-place geam mode).
    if (beta_res.from_device()) {
      details_::throwBlasError(
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
    details_::throwBlasError(
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
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "matrix_product requires A.extent(1) == "
                             "B.extent(0)");
  }
  if (out.rows != m || out.cols != n) {
    details_::throwBlasError(GCXX_BLAS_STATUS(INVALID_VALUE),
                             "matrix_product requires C to be A.extent(0) x "
                             "B.extent(1)");
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

#undef GCXX_BLAS_GEMM

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
