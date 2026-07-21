// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__cccl/preprocessor.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Minimal preprocessor-metaprogramming library. Only the primitives required by
// the concept DSL in template_helper_macros.hpp
#pragma once
#ifndef GCXX_MACROS_PREPROCESSOR_MACROS_HPP_
#define GCXX_MACROS_PREPROCESSOR_MACROS_HPP_

// check taken from google benchmark
#if defined(__COUNTER__) && (__COUNTER__ + 1 == __COUNTER__ + 0)
#define GCXX_PP_COUNTER() __COUNTER__
#else
#define GCXX_PP_COUNTER() __LINE__
#endif

// Convert parameter to a string literal
#define GCXX_PP_TO_STRING2(_STR) #_STR
#define GCXX_PP_TO_STRING(_STR) GCXX_PP_TO_STRING2(_STR)

#define GCXX_PP_FIRST(_FIRST, ...) _FIRST
#define GCXX_PP_SECOND(_, _SECOND, ...) _SECOND
#define GCXX_PP_THIRD(_1, _2, _THIRD) _THIRD

#define GCXX_PP_EXPAND(...) __VA_ARGS__
#define GCXX_PP_EAT(...)

#define GCXX_PP_CAT_(_Xp, ...) _Xp##__VA_ARGS__
#define GCXX_PP_CAT(_Xp, ...) GCXX_PP_CAT_(_Xp, __VA_ARGS__)

#define GCXX_PP_CAT2_(_Xp, ...) _Xp##__VA_ARGS__
#define GCXX_PP_CAT2(_Xp, ...) GCXX_PP_CAT2_(_Xp, __VA_ARGS__)

#define GCXX_PP_EVAL_(_Xp, _ARGS) _Xp _ARGS
#define GCXX_PP_EVAL(_Xp, ...) GCXX_PP_EVAL_(_Xp, (__VA_ARGS__))

#define GCXX_PP_CHECK(...) GCXX_PP_EXPAND(GCXX_PP_CHECK_N(__VA_ARGS__, 0, ))
#define GCXX_PP_CHECK_N(_Xp, _Num, ...) _Num
#define GCXX_PP_PROBE(_Xp) _Xp, 1,
#define GCXX_PP_PROBE_N(_Xp, _Num) _Xp, _Num,

#define GCXX_PP_IS_PAREN(_Xp) GCXX_PP_CHECK(GCXX_PP_IS_PAREN_PROBE _Xp)
#define GCXX_PP_IS_PAREN_PROBE(...) GCXX_PP_PROBE(~)

#define GCXX_PP_LPAREN (
#define GCXX_PP_RPAREN )

// Empty object-like placemarker (mirrors CCCL's empty _CCCL token). Pasting it
// with another token yields that token: GCXX_PP_CAT(GCXX_PP_PLACEMARKER, X) ->
// X. Used by the concept DSL's _Same_as / _Satisfies type extractors, where it
// also works around MSVC's traditional-preprocessor rescan limitations.
#define GCXX_PP_PLACEMARKER

#define GCXX_PP_EMPTY()
#define GCXX_PP_COMMA() ,
#define GCXX_PP_LBRACE() {
#define GCXX_PP_RBRACE() }

// GCXX_PP_CASE(ARG) yields a probe whose second field is ARG. The ARG token is
// then recovered by GCXX_PP_CHECK and used to select a *_CASE_<ARG> handler.
// ARG is a bare placeholder token (e.g. GCXX_SWITCH_DEFAULT); it is never
// defined as a macro.
#define GCXX_PP_CASE(_ARG) GCXX_PP_PROBE_N(~, _ARG)

// GCXX_PP_SWITCH / GCXX_PP_SWITCH2 dispatch on the first token of __VA_ARGS__:
//   GCXX_PP_SWITCH(PREFIX, X...) ->
//     PREFIX_CASE_<label>(X...)
// where <label> is ARG if `PREFIX_SWITCH_<first-token>` is defined as
// GCXX_PP_CASE(ARG), otherwise GCXX_SWITCH_DEFAULT.
#define GCXX_PP_SWITCH(_PREFIX, ...)                                      \
  GCXX_PP_CAT(_PREFIX##_CASE_, GCXX_PP_CASE_LABEL_(_PREFIX, __VA_ARGS__)) \
  (__VA_ARGS__)
#define GCXX_PP_SWITCH2(_PREFIX, ...)                                     \
  GCXX_PP_CAT(_PREFIX##_CASE_, GCXX_PP_CASE_LABEL_(_PREFIX, __VA_ARGS__)) \
  (__VA_ARGS__)
#define GCXX_PP_CASE_LABEL_(_PREFIX, ...)                                  \
  GCXX_PP_EVAL(GCXX_PP_CHECK,                                              \
               GCXX_PP_CAT(_PREFIX##_SWITCH_, GCXX_PP_FIRST(__VA_ARGS__)), \
               GCXX_SWITCH_DEFAULT, )

///////////////////////////////////////////////////////////////////////////////
// GCXX_PP_FOR_EACH
//
// Applies the macro _Mp to each argument: GCXX_PP_FOR_EACH(_Mp, a, b, c) ->
// _Mp(a) _Mp(b) _Mp(c). Up to 19 arguments are supported.
#define GCXX_PP_FOR_EACH(_Mp, ...) \
  GCXX_PP_FOR_EACH_N(GCXX_PP_COUNT(__VA_ARGS__), _Mp, __VA_ARGS__)
#define GCXX_PP_FOR_EACH_N(_Np, _Mp, ...) \
  GCXX_PP_CAT2(GCXX_PP_FOR_EACH_, _Np)(_Mp, __VA_ARGS__)
#define GCXX_PP_FOR_EACH_1(_Mp, _1) _Mp(_1)
#define GCXX_PP_FOR_EACH_2(_Mp, _1, _2) _Mp(_1) _Mp(_2)
#define GCXX_PP_FOR_EACH_3(_Mp, _1, _2, _3) _Mp(_1) _Mp(_2) _Mp(_3)
#define GCXX_PP_FOR_EACH_4(_Mp, _1, _2, _3, _4) _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4)
#define GCXX_PP_FOR_EACH_5(_Mp, _1, _2, _3, _4, _5) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5)
#define GCXX_PP_FOR_EACH_6(_Mp, _1, _2, _3, _4, _5, _6) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6)
#define GCXX_PP_FOR_EACH_7(_Mp, _1, _2, _3, _4, _5, _6, _7) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7)
#define GCXX_PP_FOR_EACH_8(_Mp, _1, _2, _3, _4, _5, _6, _7, _8) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8)
#define GCXX_PP_FOR_EACH_9(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)
#define GCXX_PP_FOR_EACH_10(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9) \
    _Mp(_10)
#define GCXX_PP_FOR_EACH_11(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11)
#define GCXX_PP_FOR_EACH_12(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12)                                               \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12)
#define GCXX_PP_FOR_EACH_13(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13)                                          \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13)
#define GCXX_PP_FOR_EACH_14(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13, _14)                                     \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13) _Mp(_14)
#define GCXX_PP_FOR_EACH_15(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13, _14, _15)                                \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13) _Mp(_14) _Mp(_15)
#define GCXX_PP_FOR_EACH_16(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13, _14, _15, _16)                           \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13) _Mp(_14) _Mp(_15) _Mp(_16)
#define GCXX_PP_FOR_EACH_17(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13, _14, _15, _16, _17)                      \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13) _Mp(_14) _Mp(_15) _Mp(_16) _Mp(_17)
#define GCXX_PP_FOR_EACH_18(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13, _14, _15, _16, _17, _18)                 \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13) _Mp(_14) _Mp(_15) _Mp(_16) _Mp(_17)    \
      _Mp(_18)
#define GCXX_PP_FOR_EACH_19(_Mp, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, \
                            _12, _13, _14, _15, _16, _17, _18, _19)            \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8) _Mp(_9)      \
    _Mp(_10) _Mp(_11) _Mp(_12) _Mp(_13) _Mp(_14) _Mp(_15) _Mp(_16) _Mp(_17)    \
      _Mp(_18) _Mp(_19)

///////////////////////////////////////////////////////////////////////////////
// GCXX_PP_COUNT
//
// Count the number of arguments.
// clang-format off
#define GCXX_PP_COUNT_IMPL(                                                                      \
  _125, _124, _123, _122, _121, _120, _119, _118, _117, _116, _115, _114, _113, _112, _111, _110, \
  _109, _108, _107, _106, _105, _104, _103, _102, _101, _100, _99, _98, _97, _96, _95, _94,       \
  _93, _92, _91, _90, _89, _88, _87, _86, _85, _84, _83, _82, _81, _80, _79, _78,                 \
  _77, _76, _75, _74, _73, _72, _71, _70, _69, _68, _67, _66, _65, _64, _63, _62,                 \
  _61, _60, _59, _58, _57, _56, _55, _54, _53, _52, _51, _50, _49, _48, _47, _46,                 \
  _45, _44, _43, _42, _41, _40, _39, _38, _37, _36, _35, _34, _33, _32, _31, _30,                 \
  _29, _28, _27, _26, _25, _24, _23, _22, _21, _20, _19, _18, _17, _16, _15, _14,                 \
  _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3, _2, _1, _0, ...) _0

#define GCXX_PP_COUNT(...)                                                         \
  GCXX_PP_EXPAND(GCXX_PP_COUNT_IMPL( __VA_ARGS__,                                 \
    125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, \
    109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94,       \
    93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78,                 \
    77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62,                 \
    61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,                 \
    45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,                 \
    29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,                 \
    13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
// clang-format on

#endif
