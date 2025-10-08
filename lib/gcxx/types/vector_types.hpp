#pragma once
#ifndef GCXX_TYPES_VECTOR_TYPES_HPP
#define GCXX_TYPES_VECTOR_TYPES_HPP

#include <type_traits>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

// Helper to trigger static_assert for unsupported combinations
template <class>
inline constexpr bool is_always_false_v = false;

template <typename VT, int N>
struct vec {
  GCXX_STATIC_EXPECT(is_always_false_v<VT>,
                     "vec: unsupported type and/or dimension");
};

// Macro to define specializations
#define DEFINE_VEC_MAPPING(VTYPE, VN, VEC_TYPE) \
  template <>                                   \
  struct vec<VTYPE, VN> {                       \
    using type = VEC_TYPE;                      \
  }

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                  DEFINE ALL MAPPINGS                   █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█


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

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                  TYPE ALIASES TO USE                   █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

/// A 1-component vector of type VT.
/// Equivalent to: struct [DEVICE SPECIFIC ATTIRIBUTES] VT1 { VT x; };
template <typename VT>
using vec1_t = typename details_::vec<VT, 1>::type;

/// A 2-component vector of type VT.
/// Equivalent to: struct [DEVICE SPECIFIC ATTIRIBUTES] VT2 { VT x, y; };
template <typename VT>
using vec2_t = typename details_::vec<VT, 2>::type;

/// A 3-component vector of type VT.
/// Equivalent to: struct [DEVICE SPECIFIC ATTIRIBUTES] VT3 { VT x, y, z; };
template <typename VT>
using vec3_t = typename details_::vec<VT, 3>::type;

/// A 4-component vector of type VT.
/// Equivalent to: struct [DEVICE SPECIFIC ATTIRIBUTES] VT4 { VT x, y, z, w; };
template <typename VT>
using vec4_t = typename details_::vec<VT, 4>::type;

GCXX_NAMESPACE_MAIN_END


#endif
