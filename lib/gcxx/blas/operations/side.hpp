// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_SIDE_HPP_
#define GCXX_BLAS_OPERATIONS_SIDE_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Tags selecting which side (rows vs cols) a vector operand applies to.
struct left_t {};
struct right_t {};

inline constexpr left_t left{};
inline constexpr right_t right{};

GCXX_NAMESPACE_MAIN_BLAS_END()

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// Tag -> backend side enum in one place; other tags fail at the call site.
template <class Side>
struct side_mode {
  static_assert(gcxx::details_::is_always_false_v<Side>,
                "BLAS side argument must be gcxx::blas::left or "
                "gcxx::blas::right");
};

template <>
struct side_mode<gcxx::blas::left_t> {
  static constexpr driver::deviceBlasSideMode_t value =
    driver::deviceBlasSideLeft;
};

template <>
struct side_mode<gcxx::blas::right_t> {
  static constexpr driver::deviceBlasSideMode_t value =
    driver::deviceBlasSideRight;
};

template <class Side>
inline constexpr driver::deviceBlasSideMode_t side_mode_v =
  side_mode<Side>::value;

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
