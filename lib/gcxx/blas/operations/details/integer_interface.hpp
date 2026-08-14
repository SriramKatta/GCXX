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

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

template <class IdxT>
inline constexpr bool is_supported_blas_index_v =
  std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>;

template <class IdxT>
inline constexpr bool uses_64bit_interface_v =
  std::is_same_v<IdxT, std::int64_t>;


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

// GCXX_BLAS_TYPED_FN(S, gemv) expands to cublasSgemv / hipblasSgemv.
#define GCXX_BLAS_TYPED_FN(PREFIX, OP) APPEND_NAME(BLAS_BACKEND, PREFIX##OP)
// GCXX_BLAS_TYPED_FN_64(S, gemv) expands to cublasSgemv_64 / hipblasSgemv_64.
#define GCXX_BLAS_TYPED_FN_64(PREFIX, OP) \
  APPEND_NAME(BLAS_BACKEND, PREFIX##OP##_64)

// General typed routines (gemv, ger, symv, ...)
#define GCXX_BLAS_DISPATCH_TYPED(OUT, IDX_TYPE, ELEM_TYPE, OP, ...)           \
  do {                                                                        \
    if constexpr (std::is_same_v<ELEM_TYPE, float>) {                         \
      if constexpr (gcxx::blas::details_::uses_64bit_interface_v<IDX_TYPE>) { \
        OUT = ::GCXX_BLAS_TYPED_FN_64(S, OP)(__VA_ARGS__);                    \
      } else {                                                                \
        OUT = ::GCXX_BLAS_TYPED_FN(S, OP)(__VA_ARGS__);                       \
      }                                                                       \
    } else if constexpr (std::is_same_v<ELEM_TYPE, double>) {                 \
      if constexpr (gcxx::blas::details_::uses_64bit_interface_v<IDX_TYPE>) { \
        OUT = ::GCXX_BLAS_TYPED_FN_64(D, OP)(__VA_ARGS__);                    \
      } else {                                                                \
        OUT = ::GCXX_BLAS_TYPED_FN(D, OP)(__VA_ARGS__);                       \
      }                                                                       \
    } else {                                                                  \
      static_assert(gcxx::details_::is_always_false_v<ELEM_TYPE>,             \
                    "GCXX_BLAS_DISPATCH_TYPED: unsupported element type "     \
                    "(float/double only until C/Z branches are added)");      \
    }                                                                         \
  } while (0)

#endif
