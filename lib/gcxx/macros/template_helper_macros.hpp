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

#include <gcxx/macros/function_decorator_macros.hpp>
#include <gcxx/macros/namespace_macros.hpp>
#include <gcxx/macros/preprocessor_macros.hpp>

// GCC < 14 cannot mangle noexcept expressions (gcc.gnu.org/bugzilla 70790).
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 14
#define GCXX_HAS_NOEXCEPT_MANGLING() 0
#else
#define GCXX_HAS_NOEXCEPT_MANGLING() 1
#endif

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define GCXX_HAS_CONCEPTS() 1
#else
#define GCXX_HAS_CONCEPTS() 0
#endif

#if GCXX_HAS_CONCEPTS()
// std::same_as is needed by the {EXPR} -> same_as<TYPE> requirement form.
#include <concepts>
#endif

// Helper types for the concept DSL (SFINAE), mirroring CCCL __cccl_* helpers.
GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// enable_if-style selection.
template <bool>
struct select {};
template <>
struct select<true> {
  template <class _Tp>
  using type = _Tp;
};
template <bool _Bp, class _Tp = void>
using enable_if_t = typename select<_Bp>::template type<_Tp>;
template <class _Tp, bool _Bp>
using requires_t = typename select<_Bp>::template type<_Tp>;

// Lightweight type-list threading parameter packs through SFINAE probes.
template <class...>
struct tag;

// A constexpr true that depends on _Tp, for use inside decltype.
template <class>
GCXX_FHDC bool is_true() {
  return true;
}

// requires_<BOOL>: well-formed only when BOOL is true (SFINAE).
#if defined(_MSC_VER) && !defined(__clang__)
template <bool _Bp>
GCXX_HD enable_if_t<_Bp> requires_() {}
#else
template <bool _Bp, enable_if_t<_Bp, int> = 0>
inline constexpr int requires_ = 0;
#endif

// Never defined; unevaluated decltype use; extern prevents a definition.
template <class _Tp, class... _Args>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern _Tp make_dependent;

template <class _Impl, class... _Args>
using requires_expr_impl = decltype(make_dependent<_Impl, _Args...>);


// Deliberate sink for forwarding refs; suppresses unused-value warnings.
template <typename _Tp>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
GCXX_FHDC void unused(_Tp&&) noexcept {}

#if !GCXX_HAS_CONCEPTS()
// Pre-C++20 _Same_as needs bool same_as<T, U>; C++20 uses ::std (VSTD).
template <class _Tp, class _Up>
inline constexpr bool same_as = std::is_same_v<_Tp, _Up>;
#endif

GCXX_NAMESPACE_MAIN_DETAILS_END()

// GCXX_TEMPLATE/GCXX_REQUIRES: portable constraints; use GCXX_AND not &&.
#if GCXX_HAS_CONCEPTS()
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

// Concept DSL: authors a requires-body once for C++20 and pre-C++20.
#if GCXX_HAS_CONCEPTS()
// {EXPR} -> Concept<T> needs a concept: point at ::std (CUDA 12.8 floor).
#define GCXX_CONCEPT_VSTD ::std
#endif

// gcc < 10 warns on ignored [[nodiscard]] required expressions; discard.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 10
#define GCXX_CONCEPT_IGNORE_RESULT_(...) static_cast<void>(__VA_ARGS__)
#else
#define GCXX_CONCEPT_IGNORE_RESULT_(...) __VA_ARGS__
#endif

// Silences nvcc #177-D on SFINAE probes; [[maybe_unused]]/_Pragma split.
#if defined(__CUDACC__)
#define GCXX_DIAG_SUPPRESS_177_ _Pragma("nv_diag_suppress 177")
#define GCXX_DIAG_RESTORE_177_ _Pragma("nv_diag_default 177")
#elif defined(__NVCOMPILER)
#define GCXX_DIAG_SUPPRESS_177_ _Pragma("diag_suppress 177")
#define GCXX_DIAG_RESTORE_177_ _Pragma("diag_default 177")
#else
#define GCXX_DIAG_SUPPRESS_177_
#define GCXX_DIAG_RESTORE_177_
#endif

// The "0" or "1" suffixes indicate whether _REQ is parenthesized or not.
#define GCXX_CONCEPT_REQUIREMENT_0(_REQ) \
  GCXX_PP_SWITCH(GCXX_CONCEPT_REQUIREMENT, _REQ)
#define GCXX_CONCEPT_REQUIREMENT_1(_REQ) GCXX_CONCEPT_IGNORE_RESULT_ _REQ

// Dispatch table for special requirement forms; SWITCH tokens are bare.
#define GCXX_CONCEPT_REQUIREMENT_SWITCH_requires \
  GCXX_PP_CASE(GCXX_SWITCH_REQUIRES)
#define GCXX_CONCEPT_REQUIREMENT_SWITCH_noexcept \
  GCXX_PP_CASE(GCXX_SWITCH_NOEXCEPT)
#define GCXX_CONCEPT_REQUIREMENT_SWITCH_typename \
  GCXX_PP_CASE(GCXX_SWITCH_TYPENAME)
#define GCXX_CONCEPT_REQUIREMENT_SWITCH__Same_as \
  GCXX_PP_CASE(GCXX_SWITCH_SAME_AS)
#define GCXX_CONCEPT_REQUIREMENT_SWITCH__Satisfies \
  GCXX_PP_CASE(GCXX_SWITCH_SATISFIES)

// Converts "requires(ARGS...)" to "ARGS...".
#define GCXX_CONCEPT_EAT_REQUIRES_(...) \
  GCXX_PP_CAT(GCXX_CONCEPT_EAT_REQUIRES_, __VA_ARGS__)
#define GCXX_CONCEPT_EAT_REQUIRES_requires(...) __VA_ARGS__

// Converts "noexcept(ARGS...)" to "ARGS...".
#define GCXX_CONCEPT_EAT_NOEXCEPT_(...) \
  GCXX_PP_CAT(GCXX_CONCEPT_EAT_NOEXCEPT_, __VA_ARGS__)
#define GCXX_CONCEPT_EAT_NOEXCEPT_noexcept(...) __VA_ARGS__

// Converts "typename(TYPE...)" to "TYPE...".
#define GCXX_CONCEPT_EAT_TYPENAME_(_REQ) \
  GCXX_PP_CAT2(GCXX_CONCEPT_EAT_TYPENAME_, _REQ)
#define GCXX_CONCEPT_EAT_TYPENAME_typename(...) __VA_ARGS__

// Converts "[typename]opt TYPE..." to "typename TYPE...".
#define GCXX_CONCEPT_TRY_ADD_TYPENAME_(...) \
  GCXX_PP_SWITCH2(GCXX_CONCEPT_TRY_ADD_TYPENAME, __VA_ARGS__)
#define GCXX_CONCEPT_TRY_ADD_TYPENAME_SWITCH_typename \
  GCXX_PP_CASE(GCXX_SWITCH_TYPENAME)
#define GCXX_CONCEPT_TRY_ADD_TYPENAME_CASE_GCXX_SWITCH_DEFAULT(...) \
  typename __VA_ARGS__
#define GCXX_CONCEPT_TRY_ADD_TYPENAME_CASE_GCXX_SWITCH_TYPENAME(...) __VA_ARGS__

// Converts "_Same_as(TYPE) EXPR..." to "EXPR...".
#define GCXX_CONCEPT_EAT_SAME_AS_(...) \
  GCXX_PP_CAT(GCXX_CONCEPT_EAT_SAME_AS_, __VA_ARGS__)
#define GCXX_CONCEPT_EAT_SAME_AS__Same_as(...)

// Converts "_Same_as(TYPE) EXPR..." to "TYPE" (placemarker rescan, MSVC).
#define GCXX_CONCEPT_GET_TYPE_FROM_SAME_AS_(...)                            \
  GCXX_PP_CAT(GCXX_PP_PLACEMARKER,                                          \
              GCXX_PP_EVAL(GCXX_PP_FIRST,                                   \
                           GCXX_PP_CAT(GCXX_CONCEPT_GET_TYPE_FROM_SAME_AS_, \
                                       __VA_ARGS__)))
#define GCXX_CONCEPT_GET_TYPE_FROM_SAME_AS__Same_as(...) \
  GCXX_PP_EXPAND(__VA_ARGS__),

// Converts "_Satisfies(TYPE) EXPR..." to "EXPR...".
#define GCXX_CONCEPT_EAT_SATISFIES_(...) \
  GCXX_PP_CAT(GCXX_CONCEPT_EAT_SATISFIES_, __VA_ARGS__)
#define GCXX_CONCEPT_EAT_SATISFIES__Satisfies(...)

// Converts "_Satisfies(TYPE) EXPR..." to "TYPE".
#define GCXX_CONCEPT_GET_CONCEPT_FROM_SATISFIES_(...) \
  GCXX_PP_CAT(                                        \
    GCXX_PP_PLACEMARKER,                              \
    GCXX_PP_EVAL(                                     \
      GCXX_PP_FIRST,                                  \
      GCXX_PP_CAT(GCXX_CONCEPT_GET_CONCEPT_FROM_SATISFIES_, __VA_ARGS__)))
#define GCXX_CONCEPT_GET_CONCEPT_FROM_SATISFIES__Satisfies(...) \
  GCXX_PP_EXPAND(__VA_ARGS__),

// C++20 concept path.
#if GCXX_HAS_CONCEPTS()

// GCXX_CONCEPT_FRAGMENT(NAME, requires(ARGS)(REQS)) -> concept NAME = ...
#define GCXX_CONCEPT_FRAGMENT(_NAME, ...) \
  concept _NAME = GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_##__VA_ARGS__
#define GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_requires(...) \
  requires(__VA_ARGS__) GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_
#define GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_(...) \
  { GCXX_PP_FOR_EACH(GCXX_CONCEPT_REQUIREMENT_, __VA_ARGS__) }

// Converts "EXPR" and "(EXPR)" to REQUIREMENT_0/REQUIREMENT_1 forms.
#define GCXX_CONCEPT_REQUIREMENT_(_REQ)                          \
  GCXX_PP_CAT(GCXX_CONCEPT_REQUIREMENT_, GCXX_PP_IS_PAREN(_REQ)) \
  (_REQ);

// Requirement handlers for the C++20 path.
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_DEFAULT(_REQ) _REQ
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_REQUIRES(_REQ) \
  requires GCXX_CONCEPT_EAT_REQUIRES_(_REQ)
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_NOEXCEPT(_REQ) \
  GCXX_PP_EXPAND({ GCXX_CONCEPT_EAT_NOEXCEPT_(_REQ) } noexcept)
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_TYPENAME(_REQ) \
  GCXX_CONCEPT_TRY_ADD_TYPENAME_(GCXX_CONCEPT_EAT_TYPENAME_(_REQ))
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_SAME_AS(_REQ) \
  {                                                             \
    GCXX_CONCEPT_EAT_SAME_AS_(_REQ)                             \
  } -> GCXX_CONCEPT_VSTD::same_as<GCXX_CONCEPT_GET_TYPE_FROM_SAME_AS_(_REQ)>
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_SATISFIES(_REQ) \
  {                                                               \
    GCXX_CONCEPT_EAT_SATISFIES_(_REQ)                             \
  } -> GCXX_CONCEPT_GET_CONCEPT_FROM_SATISFIES_(_REQ)

#define GCXX_FRAGMENT(_NAME, ...) _NAME<__VA_ARGS__>

#else  // ^^^ GCXX_HAS_CONCEPTS() ^^^ / vvv !GCXX_HAS_CONCEPTS() vvv

// GCXX_CONCEPT_FRAGMENT = sizeof probe; true iff all reqs well-formed.
#define GCXX_CONCEPT_FRAGMENT(_NAME, ...)                                  \
  [[maybe_unused]] GCXX_HD inline auto _NAME##_GCXX_CONCEPT_FRAGMENT_impl_ \
      GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_##__VA_ARGS__ > {}                \
  template <class... _As>                                                  \
  GCXX_HD inline auto _NAME##_GCXX_CONCEPT_FRAGMENT_(                      \
    ::gcxx::details_::tag<_As...>*,                                        \
    decltype(&_NAME##_GCXX_CONCEPT_FRAGMENT_impl_<_As...>))                \
    ->char(&)[1];                                                          \
  GCXX_HD inline auto _NAME##_GCXX_CONCEPT_FRAGMENT_(...)->char(&)[2]
#define GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_requires(...) \
  (__VA_ARGS__)->::gcxx::details_::enable_if_t <         \
    GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_IMPL_
#define GCXX_CONCEPT_FRAGMENT_REQUIREMENTS_IMPL_(...)  \
  ::gcxx::details_::is_true<decltype(GCXX_PP_FOR_EACH( \
    GCXX_CONCEPT_REQUIREMENT_, __VA_ARGS__) void())>()

// Called with each individual requirement in the list of requirements.
#define GCXX_CONCEPT_REQUIREMENT_(_REQ) \
  void(), GCXX_PP_CAT(GCXX_CONCEPT_REQUIREMENT_, GCXX_PP_IS_PAREN(_REQ))(_REQ),

// Requirement handlers for the pre-C++20 emulation path.
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_DEFAULT(_REQ) \
  GCXX_CONCEPT_IGNORE_RESULT_(_REQ)
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_REQUIRES(_REQ) \
  ::gcxx::details_::requires_<GCXX_CONCEPT_EAT_REQUIRES_(_REQ)>
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_NOEXCEPT(_REQ) \
  GCXX_CONCEPT_NOEXCEPT_REQUIREMENT_(_REQ)
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_TYPENAME(_REQ) \
  static_cast<::gcxx::details_::tag<GCXX_CONCEPT_EAT_TYPENAME_(_REQ)>*>(nullptr)
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_SAME_AS(_REQ) \
  ::gcxx::details_::requires_<                                  \
    ::gcxx::details_::same_as<GCXX_CONCEPT_SAME_AS_REQUIREMENT_(_REQ)>>
#define GCXX_CONCEPT_REQUIREMENT_CASE_GCXX_SWITCH_SATISFIES(_REQ) \
  ::gcxx::details_::requires_ <                                   \
    GCXX_CONCEPT_GET_CONCEPT_FROM_SATISFIES_(_REQ) <              \
    decltype(GCXX_CONCEPT_EAT_SATISFIES_(_REQ)) >>

// Converts "_Same_as(TYPE) EXPR..." to "TYPE, decltype(EXPR...)".
#define GCXX_CONCEPT_SAME_AS_REQUIREMENT_(_REQ) \
  GCXX_CONCEPT_GET_TYPE_FROM_SAME_AS_(_REQ),    \
    decltype(GCXX_CONCEPT_EAT_SAME_AS_(_REQ))

#if GCXX_HAS_NOEXCEPT_MANGLING()
// Converts "noexcept(EXPR)" to "::gcxx::details_::requires_<noexcept(EXPR)>".
#define GCXX_CONCEPT_NOEXCEPT_REQUIREMENT_(_REQ) \
  ::gcxx::details_::requires_<_REQ>
#else
// No noexcept mangling: check well-formedness via IGNORE_RESULT only.
#define GCXX_CONCEPT_NOEXCEPT_REQUIREMENT_(_REQ) \
  GCXX_CONCEPT_IGNORE_RESULT_(GCXX_CONCEPT_EAT_NOEXCEPT_(_REQ))
#endif

// GCXX_FRAGMENT(NAME, Args...) -> "(1 == sizeof(probe(tag<Args...>*)))".
#define GCXX_FRAGMENT(_NAME, ...)         \
  (1 ==                                   \
   sizeof(_NAME##_GCXX_CONCEPT_FRAGMENT_( \
     static_cast<::gcxx::details_::tag<__VA_ARGS__>*>(nullptr), nullptr)))

#endif  // ^^^ !GCXX_HAS_CONCEPTS() ^^^

// GCXX_REQUIRES_EXPR((T), args...)(reqs...) — last requirement only.
#if GCXX_HAS_CONCEPTS()

#define GCXX_REQUIRES_EXPR(_TY, ...) \
  requires(__VA_ARGS__) GCXX_REQUIRES_EXPR_IMPL_
#define GCXX_REQUIRES_EXPR_IMPL_(...) \
  { GCXX_PP_FOR_EACH(GCXX_CONCEPT_REQUIREMENT_, __VA_ARGS__) }

#else  // ^^^ GCXX_HAS_CONCEPTS() ^^^ / vvv !GCXX_HAS_CONCEPTS() vvv

#define GCXX_REQUIRES_EXPR(_TY, ...) \
  GCXX_REQUIRES_EXPR_IMPL(_TY, GCXX_REQUIRES_EXPR_ID(_TY), __VA_ARGS__)
#define GCXX_REQUIRES_EXPR_IMPL(_TY, _ID, ...)                             \
  ::gcxx::details_::requires_expr_impl<struct GCXX_PP_CAT(                 \
    gcxx_requires_expr_detail_, _ID) GCXX_REQUIRES_EXPR_TPARAM_REFS _TY>:: \
    is_satisfied(                                                          \
      static_cast<                                                         \
        ::gcxx::details_::tag<void GCXX_REQUIRES_EXPR_TPARAM_REFS _TY>*>(  \
        nullptr),                                                          \
      0);                                                                  \
  GCXX_DIAG_SUPPRESS_177_                                                  \
  struct GCXX_PP_CAT(gcxx_requires_expr_detail_, _ID) {                    \
    using self_t = GCXX_PP_CAT(gcxx_requires_expr_detail_, _ID);           \
    template <class GCXX_REQUIRES_EXPR_TPARAM_DEFNS _TY>                   \
    GCXX_HD inline static auto well_formed(__VA_ARGS__)                    \
      GCXX_REQUIRES_EXPR_REQUIREMENTS_

// Expands "T1, T2, variadic T3" to ", class T1, class T2, class... T3".
#define GCXX_REQUIRES_EXPR_TPARAM_DEFNS(...) \
  GCXX_PP_FOR_EACH(GCXX_REQUIRES_EXPR_TPARAM_DEFN, __VA_ARGS__)

// Expands "TY" to ", class TY" and "variadic TY" to ", class... TY".
#define GCXX_REQUIRES_EXPR_TPARAM_DEFN(_TY) \
  , GCXX_PP_SWITCH2(GCXX_REQUIRES_EXPR_TPARAM_DEFN, _TY)
#define GCXX_REQUIRES_EXPR_TPARAM_DEFN_SWITCH_variadic \
  GCXX_PP_CASE(GCXX_SWITCH_VARIADIC)
#define GCXX_REQUIRES_EXPR_TPARAM_DEFN_CASE_GCXX_SWITCH_DEFAULT(_TY) class _TY
#define GCXX_REQUIRES_EXPR_TPARAM_DEFN_CASE_GCXX_SWITCH_VARIADIC(_TY) \
  class... GCXX_PP_CAT(GCXX_REQUIRES_EXPR_EAT_VARIADIC_, _TY)

// Expands "T1, T2, variadic T3" to ", T1, T2, T3...".
#define GCXX_REQUIRES_EXPR_TPARAM_REFS(...) \
  GCXX_PP_FOR_EACH(GCXX_REQUIRES_EXPR_TPARAM_REF, __VA_ARGS__)

// Expands "TY" to ", TY" and "variadic TY" to ", TY...".
#define GCXX_REQUIRES_EXPR_TPARAM_REF(_TY) \
  , GCXX_PP_SWITCH2(GCXX_REQUIRES_EXPR_TPARAM_REF, _TY)
#define GCXX_REQUIRES_EXPR_TPARAM_REF_SWITCH_variadic \
  GCXX_PP_CASE(GCXX_SWITCH_VARIADIC)
#define GCXX_REQUIRES_EXPR_TPARAM_REF_CASE_GCXX_SWITCH_DEFAULT(_TY) _TY
#define GCXX_REQUIRES_EXPR_TPARAM_REF_CASE_GCXX_SWITCH_VARIADIC(_TY) \
  GCXX_PP_CAT(GCXX_REQUIRES_EXPR_EAT_VARIADIC_, _TY)...

// NVRTC has no __COUNTER__; synthesize an id from type params + line.
#if defined(__CUDACC_RTC__)

// Expands ((Ty...), Ty...) into GCXX_REQUIRES_EXPR_ID_NO_PAREN(Ty...).
#define GCXX_REQUIRES_EXPR_ID(_TY, ...) GCXX_REQUIRES_EXPR_ID_NO_PAREN _TY

// Expands "T1, T2, variadic T3" to "T1T2T3_##GCXX_PP_COUNTER()".
#define GCXX_REQUIRES_EXPR_ID_NO_PAREN(...)                    \
  GCXX_REQUIRES_EXPR_ID_CONCAT_ALL(                            \
    GCXX_PP_FOR_EACH(GCXX_REQUIRES_EXPR_ID_IMPL, __VA_ARGS__), \
    GCXX_PP_COUNTER())

// Expands "T1, T2, T3" to "T1T2T3".
#define GCXX_REQUIRES_EXPR_ID_CONCAT_ALL_IMPL(_0, _1, _2, _3, _4, _5, _6, _7, \
                                              _8, _9, ...)                    \
  _0##_1##_2##_3##_4##_5##_6##_7##_8##_9
#define GCXX_REQUIRES_EXPR_ID_CONCAT_ALL(...)                                  \
  GCXX_PP_EVAL(GCXX_REQUIRES_EXPR_ID_CONCAT_ALL_IMPL, __VA_ARGS__, , , , , , , \
               , , )

// Expands "TY" to "TY" and "variadic TY" to "TY".
#define GCXX_REQUIRES_EXPR_ID_IMPL(_TY) \
  , GCXX_PP_SWITCH2(GCXX_REQUIRES_EXPR_ID_IMPL, _TY)
#define GCXX_REQUIRES_EXPR_ID_IMPL_SWITCH_variadic \
  GCXX_PP_CASE(GCXX_SWITCH_VARIADIC)
#define GCXX_REQUIRES_EXPR_ID_IMPL_CASE_GCXX_SWITCH_DEFAULT(_TY) _TY
#define GCXX_REQUIRES_EXPR_ID_IMPL_CASE_GCXX_SWITCH_VARIADIC(_TY) \
  GCXX_PP_CAT(GCXX_REQUIRES_EXPR_EAT_VARIADIC_, _TY)

#else  // ^^^ __CUDACC_RTC__ ^^^ / vvv !__CUDACC_RTC__ vvv
#define GCXX_REQUIRES_EXPR_ID(...) GCXX_PP_COUNTER()
#endif  // !__CUDACC_RTC__

#define GCXX_REQUIRES_EXPR_EAT_VARIADIC_variadic

#define GCXX_REQUIRES_EXPR_REQUIREMENTS_(...)                                  \
  ->decltype(GCXX_PP_FOR_EACH(GCXX_CONCEPT_REQUIREMENT_,                       \
                              __VA_ARGS__) void()) {}                          \
  template <class... _Args, class = decltype(&self_t::well_formed<_Args...>)>  \
  GCXX_HD static constexpr bool is_satisfied(::gcxx::details_::tag<_Args...>*, \
                                             int) {                            \
    return true;                                                               \
  }                                                                            \
  GCXX_HD static constexpr bool is_satisfied(void*, long) {                    \
    return false;                                                              \
  }                                                                            \
  }                                                                            \
  GCXX_DIAG_RESTORE_177_
#endif  // ^^^ !GCXX_HAS_CONCEPTS() ^^^

#endif
