// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TYPES_SIZE_HOLDER_HPP_
#define GCXX_TYPES_SIZE_HOLDER_HPP_

#include <cstddef>
#include <limits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// gcxx::details_::size_holder<N>: compile-time or runtime std::size_t value.
GCXX_NAMESPACE_DETAILS_BEGIN()

inline constexpr std::size_t dynamic_size =
  std::numeric_limits<std::size_t>::max();

template <std::size_t N>
struct size_holder {
  GCXX_FHDC size_holder() noexcept = default;

  GCXX_FHDC explicit size_holder(std::size_t) noexcept {}

  static GCXX_FHDC std::size_t size() noexcept { return N; }
};

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
