// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TESTING_COMMON_TESTING_COMMON_HPP
#define GCXX_TESTING_COMMON_TESTING_COMMON_HPP

#include <gtest/gtest.h>

#include <type_traits>

// Stamps out a SFINAE detection trait: std::true_type iff __VA_ARGS__ is
// well-formed for the given Args... Write the expression using Args...
// where the candidate types go. Enables both positive AND negative asserts
// (the latter is impossible with a plain decltype(...) type check).
//
// Implementation: the std::is_detected (n4502) detector pattern. The pack
// is kept LAST in both the primary template and the partial specialization —
// NVCC's EDG frontend rejects a pack followed by another parameter, so the
// common `template <typename... Args, typename = void>` idiom (legal on
// GCC/Clang as an extension) does not compile here.
//
// Example:
//   GCXX_DEFINE_IS_CALLABLE(is_memset_callable,
//       gcxx::memory::Memset(std::declval<Args>()..., 0, std::size_t{0}));
//   static_assert( is_memset_callable_v<int*>);
//   static_assert(!is_memset_callable_v<NotAHandle>);
#define GCXX_DEFINE_IS_CALLABLE(Name, ...)                                    \
  template <typename... Args>                                                 \
  using Name##_detail = decltype(__VA_ARGS__);                                \
  template <template <typename...> class Op, typename, typename... Args>      \
  struct Name##_detector : std::false_type {};                                \
  template <template <typename...> class Op, typename... Args>                \
  struct Name##_detector<Op, std::void_t<Op<Args...>>, Args...>               \
      : std::true_type {};                                                    \
  template <typename... VT>                                                   \
  struct Name : Name##_detector<Name##_detail, void, VT...> {};               \
  template <typename... VT>                                                   \
  static constexpr bool Name##_v = Name<VT...>::value;


#endif