// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_TYPE_MAP_HPP_
#define GCXX_BLAS_TYPE_MAP_HPP_

#include <complex>

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/types/scalar_types.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

template <class T>
struct native_scalar {
  using type = T;
};
template <>
struct native_scalar<gcxx::cf32_t> {
  using type = GCXX_DIRECT_BACKEND_ALT(cuComplex, hipComplex);
};
template <>
struct native_scalar<gcxx::cf64_t> {
  using type = GCXX_DIRECT_BACKEND_ALT(cuDoubleComplex, hipDoubleComplex);
};
template <class T>
using native_scalar_t = typename native_scalar<T>::type;

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
