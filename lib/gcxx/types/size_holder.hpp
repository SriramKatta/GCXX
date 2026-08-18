// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TYPES_SIZE_HOLDER_HPP_
#define GCXX_TYPES_SIZE_HOLDER_HPP_

#include <cstddef>
#include <limits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// ─────────────────────────────────────────────────────────────────────────────
// gcxx::details_::size_holder<N>
//
// A compile-time-or-runtime std::size_t value: the primary template bakes N
// into the type (empty, static accessor); the dynamic_size specialization
// stores the value as a member. Extracted from span.hpp (span_storage) so
// stride_iterator can reuse the same mechanism for a compile-time-settable
// stride.
//
// The dynamic sentinel is numerically identical to gcxx::dynamic_extent
// (std::dynamic_extent), so instantiating with either name selects the
// dynamic specialization.
// ─────────────────────────────────────────────────────────────────────────────
GCXX_NAMESPACE_DETAILS_BEGIN()

inline constexpr std::size_t dynamic_size =
  std::numeric_limits<std::size_t>::max();

/// Fixed value: empty, static accessor (EBO-friendly).
template <std::size_t N>
struct size_holder {
  GCXX_FHDC size_holder() noexcept = default;

  GCXX_FHDC explicit size_holder(std::size_t) noexcept {}

  static GCXX_FHDC std::size_t size() noexcept { return N; }
};

/// Dynamic value: stored at runtime.
template <>
struct size_holder<dynamic_size> {
  std::size_t m_size{0};

  size_holder() noexcept = default;

  GCXX_FHDC explicit size_holder(std::size_t n) noexcept : m_size(n) {}

  GCXX_FHDC std::size_t size() const noexcept { return m_size; }
};

GCXX_NAMESPACE_DETAILS_END()


GCXX_NAMESPACE_MAIN_END()

#endif
