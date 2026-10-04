// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_FUNCTIONAL_FUNCTIONAL_HPP
#define GCXX_FUNCTIONAL_FUNCTIONAL_HPP

#include <gcxx/internal/prologue.hpp>
#include <type_traits>

// Reusable host/device arithmetic operation functors, in the spirit of
// cuda::std::plus (libcudacxx <cuda/std/__functional/operations.h>). Like the
// cuda::std transparent specializations, each call operator is generic over
// both operand types and returns decltype(a <op> b), so mixed operand types
// work (gcxx::plus{}(1, 2.5)). Unlike cuda::std there is no typed plus<T>
// form — the single generic form covers both roles.

GCXX_NAMESPACE_MAIN_BEGIN()

struct plus {
  template <typename A, typename B>
  GCXX_FHDC auto operator()(const A& a, const B& b) const -> decltype(a + b) {
    return a + b;
  }
};

struct minus {
  template <typename A, typename B>
  GCXX_FHDC auto operator()(const A& a, const B& b) const -> decltype(a - b) {
    return a - b;
  }
};

struct multiplies {
  template <typename A, typename B>
  GCXX_FHDC auto operator()(const A& a, const B& b) const -> decltype(a * b) {
    return a * b;
  }
};

struct divides {
  template <typename A, typename B>
  GCXX_FHDC auto operator()(const A& a, const B& b) const -> decltype(a / b) {
    return a / b;
  }
};

struct modulus {
  GCXX_TEMPLATE(typename A, typename B)
  GCXX_REQUIRES(std::is_integral_v<A>&& std::is_integral_v<B>)
  GCXX_FHDC auto operator()(const A& a, const B& b) const -> decltype(a % b) {
    return a % b;
  }
};

GCXX_NAMESPACE_MAIN_END()

#endif
