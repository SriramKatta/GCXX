// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Marks alpha/beta as device-resident; selects backend device pointer mode.
template <class T>
struct device_scalar {
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const T* ptr;
};

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// Maps plain T / device_scalar<T> to value_type and device-mode flag.
template <class S>
struct scalar_traits {
  using value_type                = std::decay_t<S>;
  static constexpr bool is_device = false;
};

template <class T>
struct scalar_traits<device_scalar<T>> {
  using value_type                = T;
  static constexpr bool is_device = true;
};

template <class S>
using scalar_value_t = typename scalar_traits<std::remove_cv_t<S>>::value_type;

template <class S>
GCXX_CXPR inline bool is_device_scalar_v =
  scalar_traits<std::remove_cv_t<S>>::is_device;

// Pointer the backend reads the scalar from; pair with a mode guard.
template <class S>
GCXX_CXPR auto blas_scalar_ptr(const S& s) -> const scalar_value_t<S>* {
  if constexpr (is_device_scalar_v<S>) {
    return s.ptr;
  } else {
    return &s;
  }
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
