// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Tag wrapper marking a scalar BLAS argument (alpha/beta) as residing in device
// memory. Passing a device_scalar to gemm/gemv selects device pointer mode
// (cublasSetPointerMode(..., _DEVICE)); a plain scalar selects host mode. It
// holds a single pointer to one device-side element and is passed by value.
//
// Example (the scaled() view forwards the factor to the operation, which
// selects device pointer mode from its type):
//   gcxx::blas::device_scalar<double> alpha_d{dAlpha.get()};
//   gcxx::blas::device_scalar<double> beta_d{dBeta.get()};
//   gcxx::blas::matrix_product(h, gcxx::blas::scaled(alpha_d, A), B,
//                              gcxx::blas::scaled(beta_d, C), C);
template <class T>
struct device_scalar {
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const T* ptr;
};

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// Maps a scalar argument type (plain T or device_scalar<T>) to its underlying
// value type and to whether it selects device pointer mode.
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

// Resolves a scalar argument to the pointer the backend routine reads it
// from: the device pointer carried by device_scalar<T> (device pointer mode),
// or the address of the host scalar itself (host pointer mode). Pair with a
// BlasPointerModeGuard constructed from is_device_scalar_v<S>.
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
