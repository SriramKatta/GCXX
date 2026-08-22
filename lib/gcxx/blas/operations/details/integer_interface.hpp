// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_INTEGER_INTERFACE_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_INTEGER_INTERFACE_HPP_

#include <cstdint>
#include <type_traits>

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/types/scalar_types.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

template <class IdxT>
inline constexpr bool is_supported_blas_index_v =
  std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>;

template <class IdxT>
inline constexpr bool uses_64bit_interface_v =
  std::is_same_v<IdxT, std::int64_t>;

// Centralized so widening the accepted element set is a one-line change.
template <class ElemT>
inline constexpr bool is_supported_blas_element_v =
  std::is_same_v<ElemT, gcxx::f32_t> || std::is_same_v<ElemT, gcxx::f64_t>;


GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

// for version that have have no type-erased Ex extry point like cublasgemmex
// cublasaxpyex etc etc
#define GCXX_BLAS_DISPATCH_INT64(OUT, IDX_TYPE, FN, ...)                    \
  do {                                                                      \
    if constexpr (gcxx::blas::details_::uses_64bit_interface_v<IDX_TYPE>) { \
      OUT = ::GCXX_BLAS_BACKEND(FN##_64)(__VA_ARGS__);                      \
    } else {                                                                \
      OUT = ::GCXX_BLAS_BACKEND(FN)(__VA_ARGS__);                           \
    }                                                                       \
  } while (0)

// General typed routines (gemv, ger, symv, ...) have no type-erased Ex entry
// point,
#define GCXX_BLAS_DISPATCH_TYPED(OUT, IDX_TYPE, ELEM_TYPE, OP, ...)           \
  do {                                                                        \
    if constexpr (std::is_same_v<ELEM_TYPE, float>) {                         \
      if constexpr (gcxx::blas::details_::uses_64bit_interface_v<IDX_TYPE>) { \
        OUT = ::GCXX_BLAS_##OP##_FN_64(Sgemv)(__VA_ARGS__);                   \
      } else {                                                                \
        OUT = ::GCXX_BLAS_##OP##_FN(Sgemv)(__VA_ARGS__);                      \
      }                                                                       \
    } else if constexpr (std::is_same_v<ELEM_TYPE, double>) {                 \
      if constexpr (gcxx::blas::details_::uses_64bit_interface_v<IDX_TYPE>) { \
        OUT = ::GCXX_BLAS_##OP##_FN_64(Dgemv)(__VA_ARGS__);                   \
      } else {                                                                \
        OUT = ::GCXX_BLAS_##OP##_FN(Dgemv)(__VA_ARGS__);                      \
      }                                                                       \
    }                                                                         \
  } while (0)

#endif
