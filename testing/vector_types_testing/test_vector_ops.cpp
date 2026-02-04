#include <type_traits>

#include "tests_common.hpp"

#include <gcxx/types/vector_types.hpp>
#include <gcxx/types/vector_types_op.hpp>

// ========================================
// Test Vector Operations
// ========================================

// Test helper for comparing floating point vectors
template <typename T>
bool almost_equal(T a, T b, T epsilon = 1e-5) {
  return std::abs(a - b) < epsilon;
}

// ========================================
// Test Addition Operations
// ========================================

TEST(VectorOps, AddInt2) {
  int2 a      = {1, 2};
  int2 b      = {3, 4};
  int2 result = a + b;
  EXPECT_EQ(result.x, 4);
  EXPECT_EQ(result.y, 6);
}

TEST(VectorOps, AddInt3) {
  int3 a      = {1, 2, 3};
  int3 b      = {4, 5, 6};
  int3 result = a + b;
  EXPECT_EQ(result.x, 5);
  EXPECT_EQ(result.y, 7);
  EXPECT_EQ(result.z, 9);
}

TEST(VectorOps, AddInt4) {
  int4 a      = {1, 2, 3, 4};
  int4 b      = {5, 6, 7, 8};
  int4 result = a + b;
  EXPECT_EQ(result.x, 6);
  EXPECT_EQ(result.y, 8);
  EXPECT_EQ(result.z, 10);
  EXPECT_EQ(result.w, 12);
}

TEST(VectorOps, AddFloat2) {
  float2 a      = {1.5f, 2.5f};
  float2 b      = {3.5f, 4.5f};
  float2 result = a + b;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 7.0f));
}

TEST(VectorOps, AddFloat3) {
  float3 a      = {1.0f, 2.0f, 3.0f};
  float3 b      = {4.0f, 5.0f, 6.0f};
  float3 result = a + b;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 7.0f));
  EXPECT_TRUE(almost_equal(result.z, 9.0f));
}

TEST(VectorOps, AddFloat4) {
  float4 a      = {1.0f, 2.0f, 3.0f, 4.0f};
  float4 b      = {5.0f, 6.0f, 7.0f, 8.0f};
  float4 result = a + b;
  EXPECT_TRUE(almost_equal(result.x, 6.0f));
  EXPECT_TRUE(almost_equal(result.y, 8.0f));
  EXPECT_TRUE(almost_equal(result.z, 10.0f));
  EXPECT_TRUE(almost_equal(result.w, 12.0f));
}

TEST(VectorOps, AddDouble2) {
  double2 a      = {1.5, 2.5};
  double2 b      = {3.5, 4.5};
  double2 result = a + b;
  EXPECT_TRUE(almost_equal(result.x, 5.0));
  EXPECT_TRUE(almost_equal(result.y, 7.0));
}

TEST(VectorOps, AddDouble3) {
  double3 a      = {1.0, 2.0, 3.0};
  double3 b      = {4.0, 5.0, 6.0};
  double3 result = a + b;
  EXPECT_TRUE(almost_equal(result.x, 5.0));
  EXPECT_TRUE(almost_equal(result.y, 7.0));
  EXPECT_TRUE(almost_equal(result.z, 9.0));
}

// ========================================
// Test Subtraction Operations
// ========================================

TEST(VectorOps, SubtractInt2) {
  int2 a      = {5, 8};
  int2 b      = {3, 4};
  int2 result = a - b;
  EXPECT_EQ(result.x, 2);
  EXPECT_EQ(result.y, 4);
}

TEST(VectorOps, SubtractInt3) {
  int3 a      = {10, 15, 20};
  int3 b      = {4, 5, 6};
  int3 result = a - b;
  EXPECT_EQ(result.x, 6);
  EXPECT_EQ(result.y, 10);
  EXPECT_EQ(result.z, 14);
}

TEST(VectorOps, SubtractInt4) {
  int4 a      = {10, 20, 30, 40};
  int4 b      = {5, 6, 7, 8};
  int4 result = a - b;
  EXPECT_EQ(result.x, 5);
  EXPECT_EQ(result.y, 14);
  EXPECT_EQ(result.z, 23);
  EXPECT_EQ(result.w, 32);
}

TEST(VectorOps, SubtractFloat3) {
  float3 a      = {10.0f, 20.0f, 30.0f};
  float3 b      = {4.0f, 5.0f, 6.0f};
  float3 result = a - b;
  EXPECT_TRUE(almost_equal(result.x, 6.0f));
  EXPECT_TRUE(almost_equal(result.y, 15.0f));
  EXPECT_TRUE(almost_equal(result.z, 24.0f));
}

// ========================================
// Test Multiplication Operations
// ========================================

TEST(VectorOps, MultiplyInt2) {
  int2 a      = {2, 3};
  int2 b      = {4, 5};
  int2 result = a * b;
  EXPECT_EQ(result.x, 8);
  EXPECT_EQ(result.y, 15);
}

TEST(VectorOps, MultiplyInt3) {
  int3 a      = {2, 3, 4};
  int3 b      = {5, 6, 7};
  int3 result = a * b;
  EXPECT_EQ(result.x, 10);
  EXPECT_EQ(result.y, 18);
  EXPECT_EQ(result.z, 28);
}

TEST(VectorOps, MultiplyInt4) {
  int4 a      = {2, 3, 4, 5};
  int4 b      = {6, 7, 8, 9};
  int4 result = a * b;
  EXPECT_EQ(result.x, 12);
  EXPECT_EQ(result.y, 21);
  EXPECT_EQ(result.z, 32);
  EXPECT_EQ(result.w, 45);
}

TEST(VectorOps, MultiplyFloat2) {
  float2 a      = {2.5f, 3.5f};
  float2 b      = {2.0f, 4.0f};
  float2 result = a * b;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 14.0f));
}

TEST(VectorOps, MultiplyFloat4) {
  float4 a      = {2.0f, 3.0f, 4.0f, 5.0f};
  float4 b      = {1.5f, 2.5f, 3.5f, 4.5f};
  float4 result = a * b;
  EXPECT_TRUE(almost_equal(result.x, 3.0f));
  EXPECT_TRUE(almost_equal(result.y, 7.5f));
  EXPECT_TRUE(almost_equal(result.z, 14.0f));
  EXPECT_TRUE(almost_equal(result.w, 22.5f));
}

// ========================================
// Test Division Operations
// ========================================

TEST(VectorOps, DivideInt2) {
  int2 a      = {8, 15};
  int2 b      = {2, 3};
  int2 result = a / b;
  EXPECT_EQ(result.x, 4);
  EXPECT_EQ(result.y, 5);
}

TEST(VectorOps, DivideInt3) {
  int3 a      = {10, 20, 30};
  int3 b      = {2, 4, 5};
  int3 result = a / b;
  EXPECT_EQ(result.x, 5);
  EXPECT_EQ(result.y, 5);
  EXPECT_EQ(result.z, 6);
}

TEST(VectorOps, DivideFloat2) {
  float2 a      = {10.0f, 20.0f};
  float2 b      = {2.0f, 4.0f};
  float2 result = a / b;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 5.0f));
}

TEST(VectorOps, DivideFloat3) {
  float3 a      = {12.0f, 18.0f, 24.0f};
  float3 b      = {3.0f, 6.0f, 4.0f};
  float3 result = a / b;
  EXPECT_TRUE(almost_equal(result.x, 4.0f));
  EXPECT_TRUE(almost_equal(result.y, 3.0f));
  EXPECT_TRUE(almost_equal(result.z, 6.0f));
}

TEST(VectorOps, DivideFloat4) {
  float4 a      = {10.0f, 20.0f, 30.0f, 40.0f};
  float4 b      = {2.0f, 4.0f, 5.0f, 8.0f};
  float4 result = a / b;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 5.0f));
  EXPECT_TRUE(almost_equal(result.z, 6.0f));
  EXPECT_TRUE(almost_equal(result.w, 5.0f));
}

// ========================================
// Test Modulo Operations (integers only)
// ========================================

TEST(VectorOps, ModuloInt2) {
  int2 a      = {10, 15};
  int2 b      = {3, 4};
  int2 result = a % b;
  EXPECT_EQ(result.x, 1);
  EXPECT_EQ(result.y, 3);
}

TEST(VectorOps, ModuloInt3) {
  int3 a      = {10, 17, 23};
  int3 b      = {3, 5, 7};
  int3 result = a % b;
  EXPECT_EQ(result.x, 1);
  EXPECT_EQ(result.y, 2);
  EXPECT_EQ(result.z, 2);
}

TEST(VectorOps, ModuloInt4) {
  int4 a      = {10, 15, 20, 25};
  int4 b      = {3, 4, 6, 7};
  int4 result = a % b;
  EXPECT_EQ(result.x, 1);
  EXPECT_EQ(result.y, 3);
  EXPECT_EQ(result.z, 2);
  EXPECT_EQ(result.w, 4);
}

// ========================================
// Test Vector-Scalar Operations
// ========================================

TEST(VectorOps, AddVectorScalarInt2) {
  int2 a      = {1, 2};
  int scalar  = 5;
  int2 result = a + scalar;
  EXPECT_EQ(result.x, 6);
  EXPECT_EQ(result.y, 7);
}

TEST(VectorOps, AddScalarVectorInt3) {
  int scalar  = 10;
  int3 b      = {1, 2, 3};
  int3 result = scalar + b;
  EXPECT_EQ(result.x, 11);
  EXPECT_EQ(result.y, 12);
  EXPECT_EQ(result.z, 13);
}

TEST(VectorOps, MultiplyVectorScalarFloat2) {
  float2 a      = {2.0f, 3.0f};
  float scalar  = 2.5f;
  float2 result = a * scalar;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 7.5f));
}

TEST(VectorOps, MultiplyScalarVectorFloat3) {
  float scalar  = 3.0f;
  float3 b      = {2.0f, 4.0f, 6.0f};
  float3 result = scalar * b;
  EXPECT_TRUE(almost_equal(result.x, 6.0f));
  EXPECT_TRUE(almost_equal(result.y, 12.0f));
  EXPECT_TRUE(almost_equal(result.z, 18.0f));
}

TEST(VectorOps, SubtractVectorScalarInt4) {
  int4 a      = {10, 20, 30, 40};
  int scalar  = 5;
  int4 result = a - scalar;
  EXPECT_EQ(result.x, 5);
  EXPECT_EQ(result.y, 15);
  EXPECT_EQ(result.z, 25);
  EXPECT_EQ(result.w, 35);
}

TEST(VectorOps, SubtractScalarVectorInt3) {
  int scalar  = 50;
  int3 b      = {5, 10, 15};
  int3 result = scalar - b;
  EXPECT_EQ(result.x, 45);
  EXPECT_EQ(result.y, 40);
  EXPECT_EQ(result.z, 35);
}

TEST(VectorOps, DivideVectorScalarFloat2) {
  float2 a      = {10.0f, 20.0f};
  float scalar  = 2.0f;
  float2 result = a / scalar;
  EXPECT_TRUE(almost_equal(result.x, 5.0f));
  EXPECT_TRUE(almost_equal(result.y, 10.0f));
}

TEST(VectorOps, DivideScalarVectorFloat3) {
  float scalar  = 12.0f;
  float3 b      = {2.0f, 3.0f, 4.0f};
  float3 result = scalar / b;
  EXPECT_TRUE(almost_equal(result.x, 6.0f));
  EXPECT_TRUE(almost_equal(result.y, 4.0f));
  EXPECT_TRUE(almost_equal(result.z, 3.0f));
}

TEST(VectorOps, ModuloVectorScalarInt2) {
  int2 a      = {10, 17};
  int scalar  = 3;
  int2 result = a % scalar;
  EXPECT_EQ(result.x, 1);
  EXPECT_EQ(result.y, 2);
}

TEST(VectorOps, ModuloScalarVectorInt3) {
  int scalar  = 20;
  int3 b      = {3, 6, 7};
  int3 result = scalar % b;
  EXPECT_EQ(result.x, 2);
  EXPECT_EQ(result.y, 2);
  EXPECT_EQ(result.z, 6);
}

// ========================================
// Test In-Place Operations (+=, -=, *=, /=, %=)
// ========================================

TEST(VectorOps, AddAssignInt2) {
  int2 a = {1, 2};
  int2 b = {3, 4};
  a += b;
  EXPECT_EQ(a.x, 4);
  EXPECT_EQ(a.y, 6);
}

TEST(VectorOps, AddAssignInt3) {
  int3 a = {1, 2, 3};
  int3 b = {4, 5, 6};
  a += b;
  EXPECT_EQ(a.x, 5);
  EXPECT_EQ(a.y, 7);
  EXPECT_EQ(a.z, 9);
}

TEST(VectorOps, AddAssignFloat4) {
  float4 a = {1.0f, 2.0f, 3.0f, 4.0f};
  float4 b = {5.0f, 6.0f, 7.0f, 8.0f};
  a += b;
  EXPECT_TRUE(almost_equal(a.x, 6.0f));
  EXPECT_TRUE(almost_equal(a.y, 8.0f));
  EXPECT_TRUE(almost_equal(a.z, 10.0f));
  EXPECT_TRUE(almost_equal(a.w, 12.0f));
}

TEST(VectorOps, SubtractAssignInt2) {
  int2 a = {10, 15};
  int2 b = {3, 5};
  a -= b;
  EXPECT_EQ(a.x, 7);
  EXPECT_EQ(a.y, 10);
}

TEST(VectorOps, SubtractAssignFloat3) {
  float3 a = {10.0f, 20.0f, 30.0f};
  float3 b = {4.0f, 5.0f, 6.0f};
  a -= b;
  EXPECT_TRUE(almost_equal(a.x, 6.0f));
  EXPECT_TRUE(almost_equal(a.y, 15.0f));
  EXPECT_TRUE(almost_equal(a.z, 24.0f));
}

TEST(VectorOps, MultiplyAssignInt2) {
  int2 a = {2, 3};
  int2 b = {4, 5};
  a *= b;
  EXPECT_EQ(a.x, 8);
  EXPECT_EQ(a.y, 15);
}

TEST(VectorOps, MultiplyAssignInt4) {
  int4 a = {2, 3, 4, 5};
  int4 b = {6, 7, 8, 9};
  a *= b;
  EXPECT_EQ(a.x, 12);
  EXPECT_EQ(a.y, 21);
  EXPECT_EQ(a.z, 32);
  EXPECT_EQ(a.w, 45);
}

TEST(VectorOps, DivideAssignInt3) {
  int3 a = {10, 20, 30};
  int3 b = {2, 4, 5};
  a /= b;
  EXPECT_EQ(a.x, 5);
  EXPECT_EQ(a.y, 5);
  EXPECT_EQ(a.z, 6);
}

TEST(VectorOps, DivideAssignFloat2) {
  float2 a = {10.0f, 20.0f};
  float2 b = {2.0f, 4.0f};
  a /= b;
  EXPECT_TRUE(almost_equal(a.x, 5.0f));
  EXPECT_TRUE(almost_equal(a.y, 5.0f));
}

TEST(VectorOps, ModuloAssignInt2) {
  int2 a = {10, 15};
  int2 b = {3, 4};
  a %= b;
  EXPECT_EQ(a.x, 1);
  EXPECT_EQ(a.y, 3);
}

TEST(VectorOps, ModuloAssignInt4) {
  int4 a = {10, 15, 20, 25};
  int4 b = {3, 4, 6, 7};
  a %= b;
  EXPECT_EQ(a.x, 1);
  EXPECT_EQ(a.y, 3);
  EXPECT_EQ(a.z, 2);
  EXPECT_EQ(a.w, 4);
}

// ========================================
// Test In-Place Operations with Scalars
// ========================================

TEST(VectorOps, AddAssignScalarInt2) {
  int2 a     = {1, 2};
  int scalar = 5;
  a += scalar;
  EXPECT_EQ(a.x, 6);
  EXPECT_EQ(a.y, 7);
}

TEST(VectorOps, AddAssignScalarFloat3) {
  float3 a     = {1.0f, 2.0f, 3.0f};
  float scalar = 2.5f;
  a += scalar;
  EXPECT_TRUE(almost_equal(a.x, 3.5f));
  EXPECT_TRUE(almost_equal(a.y, 4.5f));
  EXPECT_TRUE(almost_equal(a.z, 5.5f));
}

TEST(VectorOps, SubtractAssignScalarInt4) {
  int4 a     = {10, 20, 30, 40};
  int scalar = 5;
  a -= scalar;
  EXPECT_EQ(a.x, 5);
  EXPECT_EQ(a.y, 15);
  EXPECT_EQ(a.z, 25);
  EXPECT_EQ(a.w, 35);
}

TEST(VectorOps, MultiplyAssignScalarFloat2) {
  float2 a     = {2.0f, 3.0f};
  float scalar = 2.5f;
  a *= scalar;
  EXPECT_TRUE(almost_equal(a.x, 5.0f));
  EXPECT_TRUE(almost_equal(a.y, 7.5f));
}

TEST(VectorOps, MultiplyAssignScalarInt3) {
  int3 a     = {2, 4, 6};
  int scalar = 3;
  a *= scalar;
  EXPECT_EQ(a.x, 6);
  EXPECT_EQ(a.y, 12);
  EXPECT_EQ(a.z, 18);
}

TEST(VectorOps, DivideAssignScalarFloat4) {
  float4 a     = {10.0f, 20.0f, 30.0f, 40.0f};
  float scalar = 2.0f;
  a /= scalar;
  EXPECT_TRUE(almost_equal(a.x, 5.0f));
  EXPECT_TRUE(almost_equal(a.y, 10.0f));
  EXPECT_TRUE(almost_equal(a.z, 15.0f));
  EXPECT_TRUE(almost_equal(a.w, 20.0f));
}

TEST(VectorOps, DivideAssignScalarInt2) {
  int2 a     = {10, 20};
  int scalar = 2;
  a /= scalar;
  EXPECT_EQ(a.x, 5);
  EXPECT_EQ(a.y, 10);
}

TEST(VectorOps, ModuloAssignScalarInt3) {
  int3 a     = {10, 17, 23};
  int scalar = 3;
  a %= scalar;
  EXPECT_EQ(a.x, 1);
  EXPECT_EQ(a.y, 2);
  EXPECT_EQ(a.z, 2);
}

// ========================================
// Test Mixed Type Operations
// ========================================

TEST(VectorOps, MixedTypeAddShortInt2) {
  short2 a      = {1, 2};
  short2 b      = {3, 4};
  short2 result = a + b;
  EXPECT_EQ(result.x, 4);
  EXPECT_EQ(result.y, 6);
}

TEST(VectorOps, MixedTypeUnsignedInt3) {
  uint3 a      = {10, 20, 30};
  uint3 b      = {5, 6, 7};
  uint3 result = a - b;
  EXPECT_EQ(result.x, 5u);
  EXPECT_EQ(result.y, 14u);
  EXPECT_EQ(result.z, 23u);
}

TEST(VectorOps, MixedTypeChar4) {
  char4 a      = {10, 20, 30, 40};
  char4 b      = {2, 4, 5, 8};
  char4 result = a / b;
  EXPECT_EQ(result.x, 5);
  EXPECT_EQ(result.y, 5);
  EXPECT_EQ(result.z, 6);
  EXPECT_EQ(result.w, 5);
}

TEST(VectorOps, MixedTypeUChar2) {
  uchar2 a      = {10, 15};
  uchar2 b      = {3, 4};
  uchar2 result = a % b;
  EXPECT_EQ(result.x, 1);
  EXPECT_EQ(result.y, 3);
}

// ========================================
// Test Complex Expressions
// ========================================

TEST(VectorOps, ComplexExpressionInt2) {
  int2 a      = {2, 3};
  int2 b      = {4, 5};
  int2 c      = {1, 1};
  int2 result = (a + b) * c;
  EXPECT_EQ(result.x, 6);
  EXPECT_EQ(result.y, 8);
}

TEST(VectorOps, ComplexExpressionFloat3) {
  float3 a      = {2.0f, 4.0f, 6.0f};
  float3 b      = {1.0f, 2.0f, 3.0f};
  float scalar  = 2.0f;
  float3 result = (a + b) / scalar;
  EXPECT_TRUE(almost_equal(result.x, 1.5f));
  EXPECT_TRUE(almost_equal(result.y, 3.0f));
  EXPECT_TRUE(almost_equal(result.z, 4.5f));
}

TEST(VectorOps, ComplexExpressionInt4) {
  int4 a      = {10, 20, 30, 40};
  int4 b      = {2, 4, 5, 8};
  int scalar  = 5;
  int4 result = (a / b) + scalar;
  EXPECT_EQ(result.x, 10);
  EXPECT_EQ(result.y, 10);
  EXPECT_EQ(result.z, 11);
  EXPECT_EQ(result.w, 10);
}

TEST(VectorOps, ChainedInPlaceOps) {
  int3 a = {10, 20, 30};
  int3 b = {5, 5, 5};
  a += b;
  a *= 2;
  int3 c = {10, 10, 10};
  a -= c;
  EXPECT_EQ(a.x, 20);
  EXPECT_EQ(a.y, 40);
  EXPECT_EQ(a.z, 60);
}

// ========================================
// Test Edge Cases
// ========================================

TEST(VectorOps, ZeroVectorAddition) {
  int2 a      = {5, 10};
  int2 zero   = {0, 0};
  int2 result = a + zero;
  EXPECT_EQ(result.x, 5);
  EXPECT_EQ(result.y, 10);
}

TEST(VectorOps, NegativeValues) {
  int3 a      = {-5, -10, -15};
  int3 b      = {3, 4, 5};
  int3 result = a + b;
  EXPECT_EQ(result.x, -2);
  EXPECT_EQ(result.y, -6);
  EXPECT_EQ(result.z, -10);
}

TEST(VectorOps, NegativeMultiplication) {
  int2 a      = {-2, 3};
  int2 b      = {4, -5};
  int2 result = a * b;
  EXPECT_EQ(result.x, -8);
  EXPECT_EQ(result.y, -15);
}

TEST(VectorOps, OneMultiplication) {
  float4 a      = {2.0f, 4.0f, 6.0f, 8.0f};
  float scalar  = 1.0f;
  float4 result = a * scalar;
  EXPECT_TRUE(almost_equal(result.x, 2.0f));
  EXPECT_TRUE(almost_equal(result.y, 4.0f));
  EXPECT_TRUE(almost_equal(result.z, 6.0f));
  EXPECT_TRUE(almost_equal(result.w, 8.0f));
}

TEST(VectorOps, SelfAssignment) {
  int2 a = {5, 10};
  a += a;
  EXPECT_EQ(a.x, 10);
  EXPECT_EQ(a.y, 20);
}
