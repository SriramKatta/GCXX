#pragma once
#ifndef GCXX_TYPES_VECTOR_TYPES_HPP
#define GCXX_TYPES_VECTOR_TYPES_HPP

#include <type_traits>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

// Helper to trigger static_assert for unsupported combinations
template <typename>
struct always_false : std::false_type {};

template <typename T, int N>
struct vec {
  static_assert(always_false<T>::value, "vec: unsupported type or dimension");
};

// Macro to define specializations
#define DEFINE_VEC_MAPPING(T, N, VEC_TYPE) \
  template <>                              \
  struct vec<T, N> {                       \
    using type = VEC_TYPE;                 \
  }

// --------------- DEFINE ALL MAPPINGS -----------------

#define MULTI_DEFINE_VEC_MAPPING(TYPE, NAME) \
  DEFINE_VEC_MAPPING(TYPE, 1, NAME##1);      \
  DEFINE_VEC_MAPPING(TYPE, 2, NAME##2);      \
  DEFINE_VEC_MAPPING(TYPE, 3, NAME##3);      \
  DEFINE_VEC_MAPPING(TYPE, 4, NAME##4);


// char
MULTI_DEFINE_VEC_MAPPING(char, char);

// unsigned char
MULTI_DEFINE_VEC_MAPPING(unsigned char, uchar);

// short
MULTI_DEFINE_VEC_MAPPING(short, short);

// unsigned short
MULTI_DEFINE_VEC_MAPPING(unsigned short, ushort);

// int
MULTI_DEFINE_VEC_MAPPING(int, int);

// unsigned int
MULTI_DEFINE_VEC_MAPPING(unsigned int, uint);

// long
MULTI_DEFINE_VEC_MAPPING(long, long);

// unsigned long
MULTI_DEFINE_VEC_MAPPING(unsigned long, ulong);

// long long
MULTI_DEFINE_VEC_MAPPING(long long, longlong);

// unsigned long long
MULTI_DEFINE_VEC_MAPPING(unsigned long long, ulonglong);

// float
MULTI_DEFINE_VEC_MAPPING(float, float);

// double
MULTI_DEFINE_VEC_MAPPING(double, double);

GCXX_NAMESPACE_DETAILS_END

// --------------- TYPE ALIASES TO USE -----------------

template <typename T>
using vec1_t = typename details_::vec<T, 1>::type;

template <typename T>
using vec2_t = typename details_::vec<T, 2>::type;

template <typename T>
using vec3_t = typename details_::vec<T, 3>::type;

template <typename T>
using vec4_t = typename details_::vec<T, 4>::type;

GCXX_NAMESPACE_MAIN_END


#endif
