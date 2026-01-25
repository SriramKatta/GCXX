#pragma once
#ifndef GCXX_TYPES_VECTOR_TYPES_HPP
#define GCXX_TYPES_VECTOR_TYPES_HPP

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <type_traits>

GCXX_NAMESPACE_MAIN_BEGIN
GCXX_NAMESPACE_DETAILS_BEGIN

template <typename VT, int N, int ALIGN = 0>
struct vec {
  GCXX_STATIC_EXPECT(is_always_false_v<VT>,
                     "vec: unsupported type and/or dimension and/or alignment");
};

template <typename VT>
struct vec_traits : std::false_type {};

#define DEFINE_VEC_TRAITS(BASETYPE, N, VECTYPE) \
  template <>                                   \
  struct vec_traits<VECTYPE> : std::true_type { \
    using value_type          = BASETYPE;       \
    static constexpr int size = N;              \
  }

// Helper variable template
template <typename VT>
inline constexpr bool is_vectype_v = vec_traits<VT>::value;

#define DEFINE_VEC(BASETYPE, N, VECTYPE) \
  template <>                            \
  struct vec<BASETYPE, N> {              \
    using type = VECTYPE;                \
  }

#define DEFINE_VEC_ALIGNED(BASETYPE, N, A, VECTYPE) \
  template <>                                       \
  struct vec<BASETYPE, N, A> {                      \
    using type = VECTYPE;                           \
  }

#define DEFINE_VEC_VEC_TRAITS(BASETYPE, N, VECTYPE) \
  DEFINE_VEC(BASETYPE, N, VECTYPE);                 \
  DEFINE_VEC_TRAITS(BASETYPE, N, VECTYPE)

#define DEFINE_VEC_ALIGNED_VEC_TRAITS(BASETYPE, N, A, VECTYPE) \
  DEFINE_VEC_ALIGNED(BASETYPE, N, A, VECTYPE);                 \
  DEFINE_VEC_TRAITS(BASETYPE, N, VECTYPE)

#define MAP_1_3(BASETYPE, BASE)                \
  DEFINE_VEC_VEC_TRAITS(BASETYPE, 1, BASE##1); \
  DEFINE_VEC_VEC_TRAITS(BASETYPE, 2, BASE##2); \
  DEFINE_VEC_VEC_TRAITS(BASETYPE, 3, BASE##3)

#define MAP_1_4(BASETYPE, BASE) \
  MAP_1_3(BASETYPE, BASE);      \
  DEFINE_VEC_VEC_TRAITS(BASETYPE, 4, BASE##4)

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █    Standard type mappings (CUDA ≤ 12.x base types)     █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
MAP_1_4(char, char);
MAP_1_4(unsigned char, uchar);
MAP_1_4(short, short);
MAP_1_4(unsigned short, ushort);
MAP_1_4(int, int);
MAP_1_4(unsigned int, uint);
MAP_1_4(float, float);

MAP_1_3(long, long);
MAP_1_3(unsigned long, ulong);
MAP_1_3(long long, longlong);
MAP_1_3(unsigned long long, ulonglong);
MAP_1_3(double, double);

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █       CUDA 13.0+ alignment-aware vec4 overrides        █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__) && \
  (__CUDACC_VER_MAJOR__ >= 13)

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                          long                          █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
DEFINE_VEC_ALIGNED_VEC_TRAITS(long, 4, 16, long4_16a);
DEFINE_VEC_ALIGNED_VEC_TRAITS(long, 4, 32, long4_32a);
DEFINE_VEC_VEC_TRAITS(long, 4, long4_16a);  // default vec<long,4> -> 16a

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                     unsigned long                      █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
DEFINE_VEC_ALIGNED_VEC_TRAITS(unsigned long, 4, 16, ulong4_16a);
DEFINE_VEC_ALIGNED_VEC_TRAITS(unsigned long, 4, 32, ulong4_32a);
DEFINE_VEC_VEC_TRAITS(unsigned long, 4, ulong4_16a);

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                       long long                        █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
DEFINE_VEC_ALIGNED_VEC_TRAITS(long long, 4, 16, longlong4_16a);
DEFINE_VEC_ALIGNED_VEC_TRAITS(long long, 4, 32, longlong4_32a);
DEFINE_VEC_VEC_TRAITS(long long, 4, longlong4_16a);

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                   unsigned long long                   █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
DEFINE_VEC_ALIGNED_VEC_TRAITS(unsigned long long, 4, 16, ulonglong4_16a);
DEFINE_VEC_ALIGNED_VEC_TRAITS(unsigned long long, 4, 32, ulonglong4_32a);
DEFINE_VEC_VEC_TRAITS(unsigned long long, 4, ulonglong4_16a);

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                         double                         █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
DEFINE_VEC_ALIGNED_VEC_TRAITS(double, 4, 16, double4_16a);
DEFINE_VEC_ALIGNED_VEC_TRAITS(double, 4, 32, double4_32a);
DEFINE_VEC_VEC_TRAITS(double, 4, double4_16a);

#else

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █ Legacy CUDA or non-CUDA device: use legacy 4-wide names █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
DEFINE_VEC(long, 4, long4);
DEFINE_VEC(unsigned long, 4, ulong4);
DEFINE_VEC(long long, 4, longlong4);
DEFINE_VEC(unsigned long long, 4, ulonglong4);
DEFINE_VEC(double, 4, double4);
#endif

GCXX_NAMESPACE_DETAILS_END

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                  User-facing aliases                   █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <typename VT>
using vec1_t = typename details_::vec<VT, 1>::type;

template <typename VT>
using vec2_t = typename details_::vec<VT, 2>::type;

template <typename VT>
using vec3_t = typename details_::vec<VT, 3>::type;

#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__) && \
  (__CUDACC_VER_MAJOR__ >= 13)
// Alignment-aware vec4 variants (only valid if mapped above)
template <typename VT>
using vec4_16a_t = typename details_::vec<VT, 4, 16>::type;

template <typename VT>
using vec4_32a_t = typename details_::vec<VT, 4, 32>::type;
#endif
// Default vec4_t — will be 16a on CUDA 13+ (due to override above)
template <typename VT>
using vec4_t = typename details_::vec<VT, 4>::type;


// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                  Cleanup local macros                  █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
#undef DEFINE_VEC
#undef DEFINE_VEC_ALIGNED
#undef MAP_1_3
#undef MAP_1_4

GCXX_NAMESPACE_MAIN_END
#endif
