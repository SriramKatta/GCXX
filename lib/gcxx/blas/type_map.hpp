// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_TYPE_MAP_HPP_
#define GCXX_BLAS_TYPE_MAP_HPP_

#include <complex>

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

template <class T>
struct native_scalar {
  using type = T;
};
template <>
struct native_scalar<std::complex<float>> {
  using type = GCXX_DIRECT_BACKEND_ALT(cuComplex, hipComplex);
};
template <>
struct native_scalar<std::complex<double>> {
  using type = GCXX_DIRECT_BACKEND_ALT(cuDoubleComplex, hipDoubleComplex);
};
template <class T>
using native_scalar_t = typename native_scalar<T>::type;

// Compile-time dispatch table: maps a C++ element type to the address of the
// matching typed backend symbol (e.g. gemm_ptr_v<float> == &cublasSgemm).
#define GCXX_BLAS_REGISTER_OP(name, S, D, C, Z)               \
  template <class T>                                          \
  struct name##_ptr {                                         \
    static_assert(gcxx::details_::is_always_false_v<T>,       \
                  "Unsupported BLAS type for " #name "_ptr"); \
  };                                                          \
  template <>                                                 \
  struct name##_ptr<float> {                                  \
    static constexpr auto value = &GCXX_BLAS_BACKEND(S);      \
  };                                                          \
  template <>                                                 \
  struct name##_ptr<double> {                                 \
    static constexpr auto value = &GCXX_BLAS_BACKEND(D);      \
  };                                                          \
  template <>                                                 \
  struct name##_ptr<std::complex<float>> {                    \
    static constexpr auto value = &GCXX_BLAS_BACKEND(C);      \
  };                                                          \
  template <>                                                 \
  struct name##_ptr<std::complex<double>> {                   \
    static constexpr auto value = &GCXX_BLAS_BACKEND(Z);      \
  };                                                          \
  template <class T>                                          \
  inline constexpr auto name##_ptr_v = name##_ptr<T>::value

GCXX_BLAS_REGISTER_OP(gemm, Sgemm, Dgemm, Cgemm, Zgemm);

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
