// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__concepts/concept_macros.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#ifndef GCXX_MACROS_TEMPLATE_HELPER_MACROS_HPP_
#define GCXX_MACROS_TEMPLATE_HELPER_MACROS_HPP_

#include <type_traits>

#include <gcxx/macros/namespace_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
template <class _Tp, bool _Bp>
using requires_t = typename std::enable_if<_Bp, _Tp>::type;
GCXX_NAMESPACE_MAIN_DETAILS_END

////////////////////////////////////////////////////////////////////////////////
// GCXX_TEMPLATE / GCXX_REQUIRES
//
// Usage (Template Constraints):
//   GCXX_TEMPLATE(typename A, typename B)
//     GCXX_REQUIRES( Concept1<A> GCXX_AND Concept2<B> )
//   void foo(A a, B b)
//   {}
//
// Usage (Trailing Return Type Constraints):
//   template<typename T>
//   auto bar(T x)
//     GCXX_TRAILING_REQUIRES(<return_type>)( Concept<T> ){}
//
// Notes:
// - Use GCXX_AND instead of && inside GCXX_REQUIRES.
// - Works with C++20 concepts and pre-C++20 SFINAE.
// - Trailing requires is useful when constraints depend on the return type.
//
////////////////////////////////////////////////////////////////////////////////

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define GCXX_TEMPLATE(...) template <__VA_ARGS__>
#define GCXX_REQUIRES(...) \
  requires(__VA_ARGS__)  // some operators cannot appear at the top level
                         // (without parentheses) in a requires-clause
#define GCXX_AND &&
#define GCXX_TRAILING_REQUIRES_IMPL_(...) requires __VA_ARGS__
#define GCXX_TRAILING_REQUIRES(...) ->__VA_ARGS__ GCXX_TRAILING_REQUIRES_IMPL_
#define GCXX_CONCEPT concept
#else
#define GCXX_TEMPLATE(...) template <__VA_ARGS__
#define GCXX_REQUIRES(...)        \
  , bool gcxx_always_true = true, \
         std::enable_if_t < __VA_ARGS__ && gcxx_always_true, int > = 0 >
#define GCXX_AND &&gcxx_always_true, int > = 0, std::enable_if_t <
#define GCXX_TRAILING_REQUIRES(...) \
  ->gcxx::details_::requires_t < __VA_ARGS__ GCXX_TRAILING_REQUIRES_IMPL_
#define GCXX_TRAILING_REQUIRES_IMPL_(...) , __VA_ARGS__ >
#define GCXX_CONCEPT constexpr inline bool
#endif

#endif
