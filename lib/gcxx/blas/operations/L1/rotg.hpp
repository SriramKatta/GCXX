// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_ROTG_HPP_
#define GCXX_BLAS_OPERATIONS_L1_ROTG_HPP_

#include <cmath>
#include <type_traits>

#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/types/scalar_types.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Host-only rotg; overflow-safe, r sign per the LAPACK convention.
GCXX_TEMPLATE(class T)
GCXX_REQUIRES(true)
auto setup_givens_rotation(const T& a, const T& b, T& c, T& s, T& r) -> void {
  static_assert(std::is_same_v<T, gcxx::f32_t> || std::is_same_v<T, gcxx::f64_t>,
                "setup_givens_rotation currently supports only float/double "
                "element types (complex support is a TODO)");

  // Scale by |a| + |b| (std::hypot's trick) so neither square overflows for
  // finite inputs and denormals underflow gracefully.
  const T scale = std::fabs(a) + std::fabs(b);
  if (scale == T(0)) {
    c = T(1);
    s = T(0);
    r = T(0);
    return;
  }

  using std::copysign;
  using std::sqrt;
  const T norm = scale * sqrt((a / scale) * (a / scale) +
                              (b / scale) * (b / scale));

  // sign(r) = sign(a) when |a| >= |b|, sign(b) otherwise (LAPACK convention)
  r = (std::fabs(a) > std::fabs(b)) ? copysign(norm, a)
                                    : copysign(norm, b);

  c = a / r;
  s = b / r;
}

GCXX_TEMPLATE(class T)
GCXX_REQUIRES(true)
auto setup_givens_rotation(const T& a, const T& b, T& c, T& s) -> void {
  T r{};
  setup_givens_rotation(a, b, c, s, r);
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
