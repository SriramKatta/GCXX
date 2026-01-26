#include <type_traits>

#include "tests_common.hpp"

#include <gcxx/types/vector_types.hpp>

// Test that GCXX vector types are the same as CUDA vector types using
// std::is_same_v

// Test char vector types
TEST(VectorTypesTest, CharVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<char>, char1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<char>, char2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<char>, char3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<char>, char4>));
}

// Test unsigned char vector types
TEST(VectorTypesTest, UCharVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<unsigned char>, uchar1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<unsigned char>, uchar2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<unsigned char>, uchar3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<unsigned char>, uchar4>));
}

// Test short vector types
TEST(VectorTypesTest, ShortVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<short>, short1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<short>, short2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<short>, short3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<short>, short4>));
}

// Test unsigned short vector types
TEST(VectorTypesTest, UShortVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<unsigned short>, ushort1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<unsigned short>, ushort2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<unsigned short>, ushort3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<unsigned short>, ushort4>));
}

// Test int vector types
TEST(VectorTypesTest, IntVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<int>, int1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<int>, int2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<int>, int3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<int>, int4>));
}

// Test unsigned int vector types
TEST(VectorTypesTest, UIntVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<unsigned int>, uint1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<unsigned int>, uint2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<unsigned int>, uint3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<unsigned int>, uint4>));
}

// Test float vector types
TEST(VectorTypesTest, FloatVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<float>, float1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<float>, float2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<float>, float3>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<float>, float4>));
}

// Test long vector types (only 1-3 dimensions)
TEST(VectorTypesTest, LongVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<long>, long1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<long>, long2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<long>, long3>));

#if GCXX_CUDA_MAJOR_GREATER_EQUAL(13)
  // CUDA 13+ uses alignment-aware types
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<long>, long4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_16a_t<long>, long4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_32a_t<long>, long4_32a>));
#else
  // Legacy CUDA uses standard long4
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<long>, long4>));
#endif
}

// Test unsigned long vector types (only 1-3 dimensions)
TEST(VectorTypesTest, ULongVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<unsigned long>, ulong1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<unsigned long>, ulong2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<unsigned long>, ulong3>));

#if GCXX_CUDA_MAJOR_GREATER_EQUAL(13)
  // CUDA 13+ uses alignment-aware types
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<unsigned long>, ulong4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_16a_t<unsigned long>, ulong4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_32a_t<unsigned long>, ulong4_32a>));
#else
  // Legacy CUDA uses standard ulong4
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<unsigned long>, ulong4>));
#endif
}

// Test long long vector types (only 1-3 dimensions)
TEST(VectorTypesTest, LongLongVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<long long>, longlong1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<long long>, longlong2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<long long>, longlong3>));

#if GCXX_CUDA_MAJOR_GREATER_EQUAL(13)
  // CUDA 13+ uses alignment-aware types
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<long long>, longlong4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_16a_t<long long>, longlong4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_32a_t<long long>, longlong4_32a>));
#else
  // Legacy CUDA uses standard longlong4
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<long long>, longlong4>));
#endif
}

// Test unsigned long long vector types (only 1-3 dimensions)
TEST(VectorTypesTest, ULongLongVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<unsigned long long>, ulonglong1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<unsigned long long>, ulonglong2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<unsigned long long>, ulonglong3>));

#if GCXX_CUDA_MAJOR_GREATER_EQUAL(13)
  // CUDA 13+ uses alignment-aware types
  EXPECT_TRUE(
    (std::is_same_v<gcxx::vec4_t<unsigned long long>, ulonglong4_16a>));
  EXPECT_TRUE(
    (std::is_same_v<gcxx::vec4_16a_t<unsigned long long>, ulonglong4_16a>));
  EXPECT_TRUE(
    (std::is_same_v<gcxx::vec4_32a_t<unsigned long long>, ulonglong4_32a>));
#else
  // Legacy CUDA uses standard ulonglong4
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<unsigned long long>, ulonglong4>));
#endif
}

// Test double vector types (only 1-3 dimensions)
TEST(VectorTypesTest, DoubleVectorTypesSame) {
  EXPECT_TRUE((std::is_same_v<gcxx::vec1_t<double>, double1>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec2_t<double>, double2>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec3_t<double>, double3>));

#if GCXX_CUDA_MAJOR_GREATER_EQUAL(13)
  // CUDA 13+ uses alignment-aware types
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<double>, double4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_16a_t<double>, double4_16a>));
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_32a_t<double>, double4_32a>));
#else
  // Legacy CUDA uses standard double4
  EXPECT_TRUE((std::is_same_v<gcxx::vec4_t<double>, double4>));
#endif
}
