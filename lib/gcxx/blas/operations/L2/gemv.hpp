// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L2_GEMV_HPP_
#define GCXX_BLAS_OPERATIONS_L2_GEMV_HPP_

#include <type_traits>

#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas/operations/L1/axpy.hpp>
#include <gcxx/blas/operations/details/integer_interface.hpp>
#include <gcxx/blas/operations/details/op_inference.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// y = A*x; scaling via scaled() views on the inputs (P1673 10.3).
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TY, class ExtentsY, class LayoutY, class AccessorY)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsY::rank() == 1)
auto matrix_vector_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
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
                "matrix_vector_product outputs cannot be scaled() views; "
                "scale an input, or use the accumulate form with a scaled "
                "addend");

  static_assert(gcxx::details_::all_same_v<AIt, XIt, YIt>,
                "matrix_vector_product operands A, x, y must share the same "
                "mdspan index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, XVt, YVt>,
                "matrix_vector_product operands A, x, y must share a single "
                "element type");

  // TODO: Wire complex Cgemv/Zgemv into GCXX_BLAS_DISPATCH_TYPED.
  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "matrix_vector_product currently supports only float/double "
                "element types (complex support is a TODO)");

  // Alpha comes only from scaled() views on the inputs; beta is host zero.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(x.accessor()), "matrix_vector_product");
  if (alpha_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "matrix_vector_product: the write-only form has no device-resident "
      "beta, so a device_scalar scaled() factor is unsupported here; use the "
      "accumulate form (with a device_scalar zero addend) or host factors");
  }
  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr = &alpha_host;
  const Sv beta_host  = Sv(0);
  const Sv* beta_ptr  = &beta_host;

  // Pin host pointer mode for the call (restored on scope exit); both
  // scalars above are host values in this form.
  const details_::BlasPointerModeGuard guard{h, /*device_mode*/ false};

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(y, "y");

  // extract problem dimensions
  const auto [m, n, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_x, inc_x]     = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y]     = details_::infer_blas_vector_view(y);

  if (len_x != n || len_y != m) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "matrix_vector_product requires x to have A.extent(1) elements and y "
      "to have A.extent(0) elements");
  }

  // cu/hipBLAS gemv follows the Fortran convention: the STORED matrix is
  // always m x n column-major and op = T computes y = A_st^T * x (an
  // m-contraction, n-long result). An operand inferred as op = T (row-major
  // or transposed storage) therefore swaps the m/n arguments relative to its
  // extents, unlike the gemm family where op = T keeps m/n and swaps the
  // stored dimensions instead.
  const auto m_arg = op_a == driver::deviceBlasOpN ? m : n;
  const auto n_arg = op_a == driver::deviceBlasOpN ? n : m;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, gemv, h.getRawHandle(), op_a,
                           m_arg, n_arg, alpha_ptr, a.data_handle(), ld_a,
                           x.data_handle(), inc_x, beta_ptr, y.data_handle(),
                           inc_y);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "matrix_vector_product failed");
  }
}

// Accumulate form: b aliases y -> in-place beta path, else split via axpy.
GCXX_TEMPLATE(class TA, class ExtentsA, class LayoutA, class AccessorA,
              class TX, class ExtentsX, class LayoutX, class AccessorX,
              class TB, class ExtentsB, class LayoutB, class AccessorB,
              class TY, class ExtentsY, class LayoutY, class AccessorY)
GCXX_REQUIRES(ExtentsA::rank() == 2 GCXX_AND ExtentsX::rank() ==
              1 GCXX_AND ExtentsB::rank() == 1 GCXX_AND ExtentsY::rank() == 1)
auto matrix_vector_product(
  BlasHandleView h, const gcxx::mdspan<TA, ExtentsA, LayoutA, AccessorA>& a,
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
                "matrix_vector_product operands must share the same mdspan "
                "index_type");

  static_assert(gcxx::blas::details_::is_supported_blas_index_v<AIt>,
                "BLAS operands must use int32_t or int64_t as their "
                "mdspan index_type");

  static_assert(gcxx::details_::all_same_v<AVt, XVt, BVt, YVt>,
                "matrix_vector_product operands A, x, b, y must share a "
                "single element type");

  static_assert(std::is_same_v<AVt, float> || std::is_same_v<AVt, double>,
                "matrix_vector_product currently supports only float/double "
                "element types (complex support is a TODO)");

  // run-time device-memory probe (no-op unless checks are enabled)
  details_::validate_device_view(a, "A");
  details_::validate_device_view(x, "x");
  details_::validate_device_view(b, "b");
  details_::validate_device_view(y, "y");

  if (b.extent(0) != y.extent(0)) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/ "matrix_vector_product addend b must have the same extent as y");
  }

  // The addend's factor doubles as beta (aliased) or axpy alpha (split).
  auto beta_res = details_::resolve_scaled_alpha<Sv>(b.accessor());

  if (!details_::views_alias(b, y)) {
    // Split path: write A*x into y, then accumulate b.
    matrix_vector_product(h, a, x, y);
    if (beta_res.from_device()) {
      axpy(h, gcxx::blas::device_scalar<Sv>{beta_res.device_ptr},
           gcxx::strip_scaled(b), y);
    } else {
      axpy(h, beta_res.host_value, gcxx::strip_scaled(b), y);
    }
    return;
  }

  // In-place path: one backend call with beta read from b's factor.
  auto alpha_res = details_::combine_scaled_alpha(
    details_::resolve_scaled_alpha<Sv>(a.accessor()),
    details_::resolve_scaled_alpha<Sv>(x.accessor()), "matrix_vector_product");
  if (alpha_res.from_device() != beta_res.from_device()) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "matrix_vector_product: the backend reads alpha and beta through one "
      "pointer mode, so host and device_scalar factors cannot be mixed in "
      "one call");
  }

  const Sv alpha_host = alpha_res.host_value;
  const Sv* alpha_ptr =
    alpha_res.from_device() ? alpha_res.device_ptr : &alpha_host;
  const Sv beta_host = beta_res.host_value;
  const Sv* beta_ptr =
    beta_res.from_device() ? beta_res.device_ptr : &beta_host;

  details_::BlasPointerModeGuard guard{h, alpha_res.from_device()};

  const auto [m, n, ld_a, op_a] = details_::infer_blas_matrix_view(a);
  const auto [len_x, inc_x]     = details_::infer_blas_vector_view(x);
  const auto [len_y, inc_y]     = details_::infer_blas_vector_view(y);

  if (len_x != n || len_y != m) {
    details_::throwBlasError(
      GCXX_BLAS_STATUS(INVALID_VALUE),
      /*msg*/
      "matrix_vector_product requires x to have A.extent(1) elements and y "
      "to have A.extent(0) elements");
  }

  // See the write-only form: op = T operands swap the gemv m/n arguments.
  const auto m_arg = op_a == driver::deviceBlasOpN ? m : n;
  const auto n_arg = op_a == driver::deviceBlasOpN ? n : m;

  driver::deviceBlasStatus_t status{};
  GCXX_BLAS_DISPATCH_TYPED(status, AIt, AVt, gemv, h.getRawHandle(), op_a,
                           m_arg, n_arg, alpha_ptr, a.data_handle(), ld_a,
                           x.data_handle(), inc_x, beta_ptr, y.data_handle(),
                           inc_y);

  if (status != driver::deviceBlasStatusSuccess) {
    details_::throwBlasError(status, /*msg*/ "matrix_vector_product failed");
  }
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
