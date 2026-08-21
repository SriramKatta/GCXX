// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_L1_ROTMG_HPP_
#define GCXX_BLAS_OPERATIONS_L1_ROTMG_HPP_

#include <array>
#include <type_traits>

#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/types/scalar_types.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Host-only rotmg; d1/d2/x1 in-out, param[0] flag selects stored H entries.
GCXX_TEMPLATE(class T)
GCXX_REQUIRES(true)
auto setup_modified_givens_rotation(T& d1, T& d2, T& x1, const T& y1,
                                    std::array<T, 5>& param) -> void {
  static_assert(
    std::is_same_v<T, gcxx::f32_t> || std::is_same_v<T, gcxx::f64_t>,
    "setup_modified_givens_rotation currently supports only "
    "float/double element types");

  using std::fabs;

  // Rescale guard band (reference values: GAM = 4096, GAM^2, 1/GAM^2).
  constexpr T gam    = T(4096);
  constexpr T gamsq  = gam * gam;
  constexpr T rgamsq = T(1) / gamsq;

  T flag{};
  T h11{}, h12{}, h21{}, h22{};

  if (d1 < T(0)) {
    // zero H, d, and x1
    flag = T(-1);
    h11 = h12 = h21 = h22 = T(0);
    d1                    = T(0);
    d2                    = T(0);
    x1                    = T(0);
  } else {
    const T dp2 = d2 * y1;
    if (dp2 == T(0)) {
      // H = I: only the flag is stored, param[1..4] are left untouched
      param[0] = T(-2);
      return;
    }

    // regular case
    const T dp1 = d1 * x1;
    const T dq2 = dp2 * y1;
    const T dq1 = dp1 * x1;

    if (fabs(dq1) > fabs(dq2)) {
      h21 = -y1 / x1;
      h12 = dp2 / dp1;

      const T du = T(1) - h12 * h21;

      if (du > T(0)) {
        flag = T(0);
        d1 /= du;
        d2 /= du;
        x1 *= du;
      } else {
        // rounding-error edge case (see the reference routine's note,
        // Hammarling's modified Givens): zero H, d, and x1
        flag = T(-1);
        h11 = h12 = h21 = h22 = T(0);
        d1                    = T(0);
        d2                    = T(0);
        x1                    = T(0);
      }
    } else {
      if (dq2 < T(0)) {
        // zero H, d, and x1
        flag = T(-1);
        h11 = h12 = h21 = h22 = T(0);
        d1                    = T(0);
        d2                    = T(0);
        x1                    = T(0);
      } else {
        flag          = T(1);
        h11           = dp1 / dp2;
        h22           = x1 / y1;
        const T du    = T(1) + h11 * h22;
        const T dtemp = d2 / du;
        d2            = d1 / du;
        d1            = dtemp;
        x1            = y1 * du;
      }
    }

    // scale check: keep d1 inside the guard band, folding rescales into H
    if (d1 != T(0)) {
      while (d1 <= rgamsq || d1 >= gamsq) {
        if (flag == T(0)) {
          h11  = T(1);
          h22  = T(1);
          flag = T(-1);
        } else {
          h21  = T(-1);
          h12  = T(1);
          flag = T(-1);
        }
        if (d1 <= rgamsq) {
          d1 *= gamsq;
          x1 /= gam;
          h11 /= gam;
          h12 /= gam;
        } else {
          d1 /= gamsq;
          x1 *= gam;
          h11 *= gam;
          h12 *= gam;
        }
      }
    }

    if (d2 != T(0)) {
      while (fabs(d2) <= rgamsq || fabs(d2) >= gamsq) {
        if (flag == T(0)) {
          h11  = T(1);
          h22  = T(1);
          flag = T(-1);
        } else {
          h21  = T(-1);
          h12  = T(1);
          flag = T(-1);
        }
        if (fabs(d2) <= rgamsq) {
          d2 *= gamsq;
          h21 /= gam;
          h22 /= gam;
        } else {
          d2 /= gamsq;
          h21 *= gam;
          h22 *= gam;
        }
      }
    }
  }

  // store the entries the flag does not imply; the other slots keep the
  // caller's values, matching the reference routine
  if (flag < T(0)) {
    param[1] = h11;
    param[2] = h21;
    param[3] = h12;
    param[4] = h22;
  } else if (flag == T(0)) {
    param[2] = h21;
    param[3] = h12;
  } else {
    param[1] = h11;
    param[4] = h22;
  }

  param[0] = flag;
}

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
