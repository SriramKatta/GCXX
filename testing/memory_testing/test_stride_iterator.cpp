// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Coverage for gcxx::stride_iterator: stride-stepped iteration over a raw
// range, random-access ops, distance measured in stride steps, the
// make_stride_iterator factory, and std-algorithm interop. Host-side (raw
// array); no GPU needed.
#include "tests_common.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <type_traits>

#include <gcxx/api.hpp>

TEST(StrideIteratorTest, IteratesWithStride) {
  // 0,1,2,...,8 — stride 3 visits 0,3,6
  int a[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  gcxx::stride_iterator<int*> begin(a, 3);
  gcxx::stride_iterator<int*> end(a + 9, 3);

  EXPECT_EQ(*begin, 0);
  EXPECT_EQ(*(begin + 1), 3);
  EXPECT_EQ(*(begin + 2), 6);
  EXPECT_EQ(end - begin, 3);  // 3 stride steps
}

TEST(StrideIteratorTest, IncrementAndDereference) {
  int a[9] = {0, 10, 20, 30, 40, 50, 60, 70, 80};
  auto it  = gcxx::make_stride_iterator(a, 3);
  EXPECT_EQ(*it, 0);
  ++it;
  EXPECT_EQ(*it, 30);
  ++it;
  EXPECT_EQ(*it, 60);

  --it;
  EXPECT_EQ(*it, 30);
  it += 1;
  EXPECT_EQ(*it, 60);
  it -= 2;
  EXPECT_EQ(*it, 0);
}

TEST(StrideIteratorTest, SubscriptAdvancesByStride) {
  int a[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto it   = gcxx::make_stride_iterator(a, 4);  // 0,4,8
  EXPECT_EQ(it[0], 0);
  EXPECT_EQ(it[1], 4);
  EXPECT_EQ(it[2], 8);
}

TEST(StrideIteratorTest, StdAlgorithmInteropStatic) {
  // Every 2nd element from a[0]: 0,2,4,6. The end sentinel (a+8) is stride-
  // aligned with begin (a): (8-0)/2 == 4 stride steps. operator-(end,begin)
  // requires aligned bases; an unaligned end would truncate on integer
  // division.
  int a[8]   = {0, 1, 2, 3, 4, 5, 6, 7};
  auto begin = gcxx::make_stride_iterator<2>(a);
  auto end   = gcxx::make_stride_iterator<2>(a + 8);
  EXPECT_EQ(end - begin, 4);

  const int total = std::accumulate(begin, end, 0);  // 0+2+4+6
  EXPECT_EQ(total, 12);

  EXPECT_EQ(std::count(begin, end, 4), 1);
}

TEST(StrideIteratorTest, StdAlgorithmInteropDynamic) {
  // Every 2nd element from a[0]: 0,2,4,6. The end sentinel (a+8) is stride-
  // aligned with begin (a): (8-0)/2 == 4 stride steps. operator-(end,begin)
  // requires aligned bases; an unaligned end would truncate on integer
  // division.
  int a[8]   = {0, 1, 2, 3, 4, 5, 6, 7};
  auto begin = gcxx::make_stride_iterator(a, 2);
  auto end   = gcxx::make_stride_iterator(a + 8, 2);
  EXPECT_EQ(end - begin, 4);

  const int total = std::accumulate(begin, end, 0);  // 0+2+4+6
  EXPECT_EQ(total, 12);

  EXPECT_EQ(std::count(begin, end, 4), 1);
}

TEST(StrideIteratorTest, IteratorTraits) {
  using it_t = gcxx::stride_iterator<int*>;
  static_assert(
    std::is_same_v<it_t::iterator_category, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<it_t::value_type, int>);
  static_assert(std::is_same_v<it_t::reference, int&>);
  static_assert(std::is_same_v<it_t::pointer, int*>);
}

// Compile-time stride (size_holder mechanism shared with span's extent):
// stride_iterator<T, N> bakes the stride into the type — no stored stride —
// while the default (dynamic) keeps the runtime constructor.
TEST(StrideIteratorTest, CompileTimeStride) {
  using fixed_t   = gcxx::stride_iterator<int*, 3>;
  using dynamic_t = gcxx::stride_iterator<int*>;

  static_assert(fixed_t::stride_extent == 3);
  static_assert(dynamic_t::stride_extent ==
                std::numeric_limits<std::size_t>::max());
  static_assert(
    gcxx::stride_iterator<int*, gcxx::dynamic_extent>::stride_extent ==
    std::numeric_limits<std::size_t>::max());

  // Fixed stride stores nothing beyond the pointer (empty size_holder,
  // [[no_unique_address]]); dynamic carries the runtime value.
  static_assert(sizeof(fixed_t) == sizeof(int*));
  static_assert(sizeof(dynamic_t) == 2 * sizeof(int*));

  // stride() is the baked constant for fixed, the ctor value for dynamic.
  static_assert(fixed_t{nullptr}.stride() == 3);

  int data[9]{0, 1, 2, 3, 4, 5, 6, 7, 8};
  fixed_t begin{data}, end{data + 9};
  EXPECT_EQ(*begin, 0);
  EXPECT_EQ(begin[1], 3);
  ++begin;
  EXPECT_EQ(*begin, 3);
  EXPECT_EQ(end - begin, 2);  // (9 - 3) / 3 logical steps

  // Same traversal with the factory and the dynamic flavor.
  auto fbegin = gcxx::make_stride_iterator<3>(data);
  EXPECT_EQ(*fbegin, 0);
  EXPECT_EQ(fbegin[2], 6);
  auto dbegin = gcxx::make_stride_iterator(data, 3);
  EXPECT_EQ(fbegin.base() + 6, dbegin.base() + 6);
}

// The adapter wraps ANY random-access iterator, not just pointers: striding a
// heterogeneous_iterator keeps its space-restricted dereference (host here),
// and adapters compose (stride over reverse).
TEST(StrideIteratorTest, WrapsArbitraryRandomAccessIterators) {
  using hetero_t = gcxx::heterogeneous_iterator<int, gcxx::host_accessible>;
  static_assert(
    std::is_same_v<gcxx::stride_iterator<hetero_t>::value_type, int>);
  static_assert(
    std::is_same_v<gcxx::stride_iterator<hetero_t>::reference, int&>);
  static_assert(
    std::is_same_v<gcxx::stride_iterator<hetero_t>::iterator_category,
                   std::random_access_iterator_tag>);

  int data[9]{0, 1, 2, 3, 4, 5, 6, 7, 8};
  gcxx::stride_iterator<hetero_t> begin{hetero_t{data}, 3};
  gcxx::stride_iterator<hetero_t> end{hetero_t{data + 9}, 3};
  EXPECT_EQ(*begin, 0);
  EXPECT_EQ(begin[2], 6);
  EXPECT_EQ(end - begin, 3);
  EXPECT_EQ(std::accumulate(begin, end, 0), 0 + 3 + 6);

  // Compile-time stride over a wrapped heterogeneous_iterator too.
  gcxx::stride_iterator<hetero_t, 4> fbegin{hetero_t{data}};
  EXPECT_EQ(*fbegin, 0);
  EXPECT_EQ(fbegin[2], 8);
  ++fbegin;
  EXPECT_EQ(*fbegin, 4);

  // Adapter composition: stride over reverse over a raw pointer.
  using rev_t = gcxx::reverse_iterator<int*>;
  gcxx::stride_iterator<rev_t> rbegin{rev_t{data + 9}, 3};
  EXPECT_EQ(*rbegin, 8);  // last, then every 3rd backwards
  EXPECT_EQ(rbegin[1], 5);
  ++rbegin;
  EXPECT_EQ(*rbegin, 5);
}
