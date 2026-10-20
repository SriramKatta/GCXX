// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/scalars/device_scalar_view.hpp>
#include <gcxx/runtime/memory/scalars/scalar.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// Maps plain T / device_scalar_view<T> / owning gcxx scalars to value_type
// and device-mode flag.
template <class S>
struct scalar_traits {
  using value_type                = std::decay_t<S>;
  static constexpr bool is_device = false;
};

template <class T>
struct scalar_traits<gcxx::device_scalar_view<T>> {
  using value_type                = T;
  static constexpr bool is_device = true;
};

template <class T, class... Properties>
struct scalar_traits<gcxx::scalar<T, Properties...>> {
  using value_type                = T;
  static constexpr bool is_device = gcxx::is_device_accessible<Properties...>;
};

template <class S>
using scalar_value_t = typename scalar_traits<std::remove_cv_t<S>>::value_type;

template <class S>
GCXX_CXPR inline bool is_device_scalar_v =
  scalar_traits<std::remove_cv_t<S>>::is_device;

// Pointer the backend reads the scalar from; pair with a mode guard.
template <class S>
GCXX_CXPR auto blas_scalar_ptr(const S& s) -> const scalar_value_t<S>* {
  if constexpr (gcxx::details_::is_owning_scalar_v<S>) {
    return s.data();
  } else if constexpr (is_device_scalar_v<S>) {
    return s.ptr;
  } else {
    return &s;
  }
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
