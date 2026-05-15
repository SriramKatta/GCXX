// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <array>
#include <type_traits>
#include <vector>

#include <gcxx/api.hpp>

TEST(SpanDeductionGuide, IteratorAndCountDeduceDynamicExtent) {
  int values[]{1, 2, 3, 4, 5};

  gcxx::span s1{std::begin(values), std::end(values)};
  gcxx::span s2{std::begin(values), std::size(values)};

  static_assert(std::is_same_v<decltype(s1), gcxx::span<int>>);
  static_assert(std::is_same_v<decltype(s2), gcxx::span<int>>);

  EXPECT_EQ(s1.data(), values);
  EXPECT_EQ(s1.size(), std::size(values));
  EXPECT_EQ(s1.extent, gcxx::dynamic_extent);

  EXPECT_EQ(s2.data(), values);
  EXPECT_EQ(s2.size(), std::size(values));
  EXPECT_EQ(s2.extent, gcxx::dynamic_extent);
}

TEST(SpanDeductionGuide, CArrayDeducesElementTypeAndStaticExtent) {
  int values[]{1, 2, 3, 4, 5};
  const int const_values[]{1, 2, 3, 4, 5};

  gcxx::span s1{values};
  gcxx::span s2{const_values};
  gcxx::span<int> dynamic{values};

  static_assert(std::is_same_v<decltype(s1), gcxx::span<int, 5>>);
  static_assert(std::is_same_v<decltype(s2), gcxx::span<const int, 5>>);
  static_assert(std::is_same_v<decltype(dynamic), gcxx::span<int>>);

  EXPECT_EQ(s1.data(), values);
  EXPECT_EQ(s1.size(), std::size(values));
  EXPECT_EQ(s1.extent, std::size(values));

  EXPECT_EQ(s2.data(), const_values);
  EXPECT_EQ(s2.size(), std::size(const_values));
  EXPECT_EQ(s2.extent, std::size(const_values));

  EXPECT_EQ(dynamic.data(), values);
  EXPECT_EQ(dynamic.size(), std::size(values));
  EXPECT_EQ(dynamic.extent, gcxx::dynamic_extent);
}

TEST(SpanDeductionGuide, StdArrayDeducesConstnessAndStaticExtent) {
  std::array values{6, 7, 8};
  const std::array const_values{9, 10, 11};

  gcxx::span s1{values};
  gcxx::span s2{const_values};

  static_assert(std::is_same_v<decltype(s1), gcxx::span<int, 3>>);
  static_assert(std::is_same_v<decltype(s2), gcxx::span<const int, 3>>);

  EXPECT_EQ(s1.data(), values.data());
  EXPECT_EQ(s1.size(), values.size());
  EXPECT_EQ(s1.extent, values.size());

  EXPECT_EQ(s2.data(), const_values.data());
  EXPECT_EQ(s2.size(), const_values.size());
  EXPECT_EQ(s2.extent, const_values.size());
}

TEST(SpanDeductionGuide, ContiguousRangeDeducesDynamicExtent) {
  std::vector values{66, 69, 99};
  const std::vector const_values{1, 2, 3};

  gcxx::span s1{values};
  gcxx::span s2{const_values};

  static_assert(std::is_same_v<decltype(s1), gcxx::span<int>>);
  static_assert(std::is_same_v<decltype(s2), gcxx::span<const int>>);

  EXPECT_EQ(s1.data(), values.data());
  EXPECT_EQ(s1.size(), values.size());
  EXPECT_EQ(s1.extent, gcxx::dynamic_extent);

  EXPECT_EQ(s2.data(), const_values.data());
  EXPECT_EQ(s2.size(), const_values.size());
  EXPECT_EQ(s2.extent, gcxx::dynamic_extent);
}

TEST(SpanDeductionGuide, ObjectSizeTracksStaticVersusDynamicExtentStorage) {
  int values[]{1, 2, 3, 4, 5};
  std::array arr{6, 7, 8};
  std::vector vec{66, 69, 99};

  gcxx::span from_iterators{std::begin(values), std::end(values)};
  gcxx::span from_c_array{values};
  gcxx::span<int> explicit_dynamic{values};
  gcxx::span from_std_array{arr};
  gcxx::span from_range{vec};

  EXPECT_EQ(sizeof(from_iterators), sizeof(int*) + sizeof(std::size_t));
  EXPECT_EQ(sizeof(from_c_array), sizeof(int*));
  EXPECT_EQ(sizeof(explicit_dynamic), sizeof(int*) + sizeof(std::size_t));
  EXPECT_EQ(sizeof(from_std_array), sizeof(int*));
  EXPECT_EQ(sizeof(from_range), sizeof(int*) + sizeof(std::size_t));
}
