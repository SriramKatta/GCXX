#pragma once
#ifndef GCXX_MACROS_TEMPLATE_HELPER_MACROS_HPP_
#define GCXX_MACROS_TEMPLATE_HELPER_MACROS_HPP_

#include <type_traits>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define GCXX_TEMPLATE(...) template <__VA_ARGS__>
#define GCXX_REQUIRES(...) requires __VA_ARGS__
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
  ->gcxx_requires_t < __VA_ARGS__ GCXX_TRAILING_REQUIRES_IMPL_
#define GCXX_TRAILING_REQUIRES_IMPL_(...) , __VA_ARGS__ >
#define GCXX_CONCEPT inline constexpr bool
#endif

#endif
