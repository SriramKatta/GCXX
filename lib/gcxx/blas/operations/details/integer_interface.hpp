// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DETAILS_INTEGER_INTERFACE_HPP_
#define GCXX_BLAS_OPERATIONS_DETAILS_INTEGER_INTERFACE_HPP_

#include <cstdint>
#include <type_traits>

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

template <class IdxT>
inline constexpr bool is_supported_blas_index_v =
  std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>;

template <class IdxT>
inline constexpr bool uses_64bit_interface_v =
  std::is_same_v<IdxT, std::int64_t>;


GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#define GCXX_BLAS_DISPATCH_INT64(OUT, IDX_TYPE, FN, ...)                    \
  do {                                                                      \
    if constexpr (gcxx::blas::details_::uses_64bit_interface_v<IDX_TYPE>) { \
      OUT = ::GCXX_BLAS_BACKEND(FN##_64)(__VA_ARGS__);                      \
    } else {                                                                \
      OUT = ::GCXX_BLAS_BACKEND(FN)(__VA_ARGS__);                           \
    }                                                                       \
  } while (0)

#endif