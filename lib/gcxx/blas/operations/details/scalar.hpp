// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_SCALAR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

<<<<<<< HEAD
// Marks alpha/beta as device-resident; selects backend device pointer mode.
=======
// Tag wrapper marking a scalar BLAS argument (alpha/beta) as residing in device
// memory. Passing a device_scalar to gemm/gemv selects device pointer mode
// (cublasSetPointerMode(..., _DEVICE)); a plain scalar selects host mode. It
// holds a single pointer to one device-side element and is passed by value.
//
// Example:
//   gcxx::blas::device_scalar<double> alpha_d{dAlpha.get()};
//   gcxx::blas::gemm(h, alpha_d, A, B,
//                    gcxx::blas::device_scalar<double>{dBeta.get()}, C);
>>>>>>> f6989c9 (Amending to new examples)
template <class T>
struct device_scalar {
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const T* ptr;
};

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

<<<<<<< HEAD
// Maps plain T / device_scalar<T> to value_type and device-mode flag.
=======
// Maps a scalar argument type (plain T or device_scalar<T>) to its underlying
// value type and to whether it selects device pointer mode.
>>>>>>> f6989c9 (Amending to new examples)
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

<<<<<<< HEAD
// Pointer the backend reads the scalar from; pair with a mode guard.
template <class S>
GCXX_CXPR auto blas_scalar_ptr(const S& s) -> const scalar_value_t<S>* {
  if constexpr (is_device_scalar_v<S>) {
    return s.ptr;
  } else {
    return &s;
  }
}

=======
>>>>>>> f6989c9 (Amending to new examples)
GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
