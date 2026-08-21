// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_SYMMETRIC_HPP_
#define GCXX_BLAS_OPERATIONS_SYMMETRIC_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Triangle accessed; other side assumed symmetric (P1673R13 10.5).
struct upper_t {};
struct lower_t {};

inline constexpr upper_t upper{};
inline constexpr lower_t lower{};

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// The two accepted fill-mode tags (compile-time uplo).
template <class>
GCXX_CXPR inline bool is_fill_mode_tag_v = false;

template <>
GCXX_CXPR inline bool is_fill_mode_tag_v<gcxx::blas::upper_t> = true;

template <>
GCXX_CXPR inline bool is_fill_mode_tag_v<gcxx::blas::lower_t> = true;

// Tag -> backend fill enum in one place; other tags fail at the call site.
template <class Fill>
struct fill_mode {
  static_assert(gcxx::details_::is_always_false_v<Fill>,
                "BLAS fill-mode argument must be gcxx::blas::upper or "
                "gcxx::blas::lower");
};

template <>
struct fill_mode<gcxx::blas::upper_t> {
  static constexpr driver::deviceBlasFillMode_t value =
    driver::deviceBlasFillModeUpper;
};

template <>
struct fill_mode<gcxx::blas::lower_t> {
  static constexpr driver::deviceBlasFillMode_t value =
    driver::deviceBlasFillModeLower;
};

template <class Fill>
inline constexpr driver::deviceBlasFillMode_t fill_mode_v =
  fill_mode<Fill>::value;

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
