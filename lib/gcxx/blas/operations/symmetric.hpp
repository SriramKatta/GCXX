// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_SYMMETRIC_HPP_
#define GCXX_BLAS_OPERATIONS_SYMMETRIC_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Tag types selecting which triangle of a symmetric/Hermitian matrix is
// stored (the uplo argument of the symmetric/hermitian BLAS routines),
// following P1673R13's design (10.4, "Approach 1"): the triangle is a TAG
// PARAMETER of the algorithm — e.g. symmetric_matrix_product(h, A, upper,
// B, C) — not a property of a wrapper type. The tag names (upper/lower)
// mirror r13's upper_triangle/lower_triangle objects. The symmetric and
// hermitian routines themselves (symm/symv/herk/...) are a deferred follow-up
// that will consume these tags plus the fill_mode machinery below.
//
// Passing the tag objects gcxx::blas::upper / gcxx::blas::lower keeps the
// raw backend fill-mode enum out of the public API, mirroring how
// gcxx::blas::left / gcxx::blas::right hide the side-mode enum (side.hpp).
//
// Note the P1673R13 semantics (10.5): the triangle names the part of the
// matrix the algorithm ACCESSES; the routine computes as if the other side
// satisfied the symmetry, without asserting any mathematical property.
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

// Compile-time triangle-tag -> backend fill-mode mapping, so the driver enum
// is touched in exactly one place and an argument that is not one of the two
// tags fails to compile at the call site.
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
