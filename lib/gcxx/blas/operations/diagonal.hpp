// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_DIAGONAL_HPP_
#define GCXX_BLAS_OPERATIONS_DIAGONAL_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// explicit_diagonal reads the stored diagonal; implicit assumes it is 1.
struct implicit_diagonal_t {};
struct explicit_diagonal_t {};

inline constexpr implicit_diagonal_t implicit_diagonal{};
inline constexpr explicit_diagonal_t explicit_diagonal{};

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// The two accepted diagonal-storage tags (compile-time diag).
template <class>
GCXX_CXPR inline bool is_diagonal_storage_tag_v = false;

template <>
GCXX_CXPR inline bool
  is_diagonal_storage_tag_v<gcxx::blas::implicit_diagonal_t> = true;

template <>
GCXX_CXPR inline bool
  is_diagonal_storage_tag_v<gcxx::blas::explicit_diagonal_t> = true;

// Tag -> backend diag enum in one place; other tags fail at the call site.
template <class Diag>
struct diagonal_type {
  static_assert(gcxx::details_::is_always_false_v<Diag>,
                "BLAS diagonal-storage argument must be "
                "gcxx::blas::implicit_diagonal or "
                "gcxx::blas::explicit_diagonal");
};

template <>
struct diagonal_type<gcxx::blas::implicit_diagonal_t> {
  static constexpr driver::deviceBlasDiagType_t value =
    driver::deviceBlasDiagUnit;
};

template <>
struct diagonal_type<gcxx::blas::explicit_diagonal_t> {
  static constexpr driver::deviceBlasDiagType_t value =
    driver::deviceBlasDiagNonUnit;
};

template <class Diag>
inline constexpr driver::deviceBlasDiagType_t diagonal_type_v =
  diagonal_type<Diag>::value;

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
