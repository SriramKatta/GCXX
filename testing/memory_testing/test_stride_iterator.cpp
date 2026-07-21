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
#include <numeric>
#include <type_traits>

#include <gcxx/api.hpp>

TEST(StrideIteratorTest, IteratesWithStride) {
  // 0,1,2,...,8 — stride 3 visits 0,3,6
  int a[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  gcxx::stride_iterator<int> begin(a, 3);
  gcxx::stride_iterator<int> end(a + 9, 3);

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

TEST(StrideIteratorTest, StdAlgorithmInterop) {
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
  using it_t = gcxx::stride_iterator<int>;
  static_assert(
    std::is_same_v<it_t::iterator_category, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<it_t::value_type, int>);
  static_assert(std::is_same_v<it_t::reference, int&>);
  static_assert(std::is_same_v<it_t::pointer, int*>);
}
