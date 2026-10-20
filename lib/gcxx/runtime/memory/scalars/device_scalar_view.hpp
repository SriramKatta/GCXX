// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SCALARS_DEVICE_SCALAR_VIEW_HPP_
#define GCXX_RUNTIME_MEMORY_SCALARS_DEVICE_SCALAR_VIEW_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// Non-owning, copyable marker: "read this scalar from device memory".
// Selects device pointer mode in BLAS dispatch and is the stored factor type
// of scaled_accessor. gcxx::blas::device_scalar aliases this type.
// Lifetime contract: the pointed-to value must outlive every view (and every
// BLAS call consuming it) — same as a raw pointer.
template <class T>
struct device_scalar_view {
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const T* ptr;
};

GCXX_NAMESPACE_MAIN_END()

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// Duck-typed detection of the owning gcxx::scalar, which carries a public
// `static constexpr bool is_gcxx_scalar = true`. void_t SFINAE on purpose:
// the concept DSL's typename(TYPE) fails for dependent nested members in the
// C++17/EDG branch. Distinguishes scalar from buffer/uninit_buffer (those
// also carry a `properties` alias, so property detection would be ambiguous).
template <class T, class = void>
struct is_owning_scalar : std::false_type {};

template <class T>
struct is_owning_scalar<T, std::void_t<decltype(T::is_gcxx_scalar)>>
    : std::bool_constant<T::is_gcxx_scalar> {};

template <class T>
GCXX_CXPR inline bool is_owning_scalar_v =
  is_owning_scalar<std::remove_cv_t<std::remove_reference_t<T>>>::value;

GCXX_NAMESPACE_MAIN_DETAILS_END()

#endif
