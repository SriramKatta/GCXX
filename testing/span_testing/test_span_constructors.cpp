// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <array>
#include <type_traits>
#include <vector>

#include <gcxx/api.hpp>

TEST(SpanConstructors, DefaultConstructorIsAvailableForDynamicAndZeroExtent) {
  static_assert(std::is_default_constructible_v<gcxx::span<int>>);
  static_assert(std::is_default_constructible_v<gcxx::span<int, 0>>);
  static_assert(!std::is_default_constructible_v<gcxx::span<int, 1>>);

  gcxx::span<int> dynamic;
  gcxx::span<int, 0> zero;

  EXPECT_EQ(dynamic.data(), nullptr);
  EXPECT_EQ(dynamic.size(), 0U);
  EXPECT_TRUE(dynamic.empty());

  EXPECT_EQ(zero.data(), nullptr);
  EXPECT_EQ(zero.size(), 0U);
  EXPECT_TRUE(zero.empty());
}

TEST(SpanConstructors, PointerAndCountConstructDynamicAndStaticExtents) {
  int values[]{1, 2, 3, 4, 5};

  gcxx::span<int> dynamic{values, std::size(values)};
  gcxx::span<int, 5> fixed{values, std::size(values)};

  EXPECT_EQ(dynamic.data(), values);
  EXPECT_EQ(dynamic.size(), std::size(values));
  EXPECT_EQ(dynamic.front(), 1);
  EXPECT_EQ(dynamic.back(), 5);

  EXPECT_EQ(fixed.data(), values);
  EXPECT_EQ(fixed.size(), std::size(values));
  EXPECT_EQ(fixed.extent, std::size(values));
}

TEST(SpanConstructors, IteratorAndSentinelConstructDynamicAndStaticExtents) {
  std::array values{1, 2, 3, 4, 5};

  gcxx::span<int> dynamic{values.begin(), values.end()};
  gcxx::span<int, 5> fixed{values.begin(), values.end()};

  EXPECT_EQ(dynamic.data(), values.data());
  EXPECT_EQ(dynamic.size(), values.size());
  EXPECT_EQ(fixed.data(), values.data());
  EXPECT_EQ(fixed.size(), values.size());
}

TEST(SpanConstructors, CArrayConstructorPreservesDataAndChecksExtent) {
  int values[]{1, 2, 3, 4, 5};

  static_assert(std::is_constructible_v<gcxx::span<int>, int(&)[5]>);
  static_assert(std::is_constructible_v<gcxx::span<int, 5>, int(&)[5]>);
  static_assert(!std::is_constructible_v<gcxx::span<int, 4>, int(&)[5]>);
  static_assert(std::is_constructible_v<gcxx::span<const int>, int(&)[5]>);
  static_assert(!std::is_constructible_v<gcxx::span<int>, const int(&)[5]>);

  gcxx::span<int> dynamic{values};
  gcxx::span<int, 5> fixed{values};
  gcxx::span<const int, 5> readonly{values};

  EXPECT_EQ(dynamic.data(), std::data(values));
  EXPECT_EQ(dynamic.size(), std::size(values));
  EXPECT_EQ(fixed.data(), std::data(values));
  EXPECT_EQ(fixed.size(), std::size(values));
  EXPECT_EQ(readonly.data(), std::data(values));
  EXPECT_EQ(readonly.size(), std::size(values));
}

TEST(SpanConstructors, StdArrayConstructorHandlesMutableAndConstArrays) {
  std::array values{1, 2, 3, 4, 5};
  const std::array const_values{1, 2, 3, 4, 5};

  static_assert(std::is_constructible_v<gcxx::span<int>, decltype(values)&>);
  static_assert(std::is_constructible_v<gcxx::span<int, 5>, decltype(values)&>);
  static_assert(
    !std::is_constructible_v<gcxx::span<int, 4>, decltype(values)&>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int>, decltype(values)&>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int>, const decltype(values)&>);
  static_assert(
    !std::is_constructible_v<gcxx::span<int>, const decltype(values)&>);

  gcxx::span<int> dynamic{values};
  gcxx::span<int, 5> fixed{values};
  gcxx::span<const int, 5> readonly{const_values};

  EXPECT_EQ(dynamic.data(), values.data());
  EXPECT_EQ(dynamic.size(), values.size());
  EXPECT_EQ(fixed.data(), values.data());
  EXPECT_EQ(fixed.size(), values.size());
  EXPECT_EQ(readonly.data(), const_values.data());
  EXPECT_EQ(readonly.size(), const_values.size());
}

TEST(SpanConstructors, ContiguousSizedRangeConstructorUsesDataAndSize) {
  std::vector values{1, 2, 3, 4, 5};
  const std::vector const_values{1, 2, 3, 4, 5};

  static_assert(std::is_constructible_v<gcxx::span<int>, decltype(values)&>);
  static_assert(std::is_constructible_v<gcxx::span<int, 5>, decltype(values)&>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int>, decltype(values)&>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int>, const decltype(values)&>);
  static_assert(
    !std::is_constructible_v<gcxx::span<int>, const decltype(values)&>);

  gcxx::span<int> dynamic{values};
  gcxx::span<int, 5> fixed{values};
  gcxx::span<const int> readonly{const_values};

  EXPECT_EQ(dynamic.data(), values.data());
  EXPECT_EQ(dynamic.size(), values.size());
  EXPECT_EQ(fixed.data(), values.data());
  EXPECT_EQ(fixed.size(), values.size());
  EXPECT_EQ(readonly.data(), const_values.data());
  EXPECT_EQ(readonly.size(), const_values.size());
}

TEST(SpanConstructors,
     SpanConversionConstructorAllowsOnlyCompatibleExtentsAndCv) {
  int values[]{1, 2, 3, 4, 5};
  gcxx::span<int, 5> fixed{values};
  gcxx::span<int> dynamic{values};

  static_assert(std::is_constructible_v<gcxx::span<int>, gcxx::span<int, 5>&>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int>, gcxx::span<int, 5>&>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int, 5>, gcxx::span<int, 5>&>);
  static_assert(std::is_constructible_v<gcxx::span<int, 5>, gcxx::span<int>&>);
  static_assert(
    !std::is_constructible_v<gcxx::span<int, 4>, gcxx::span<int, 5>&>);
  static_assert(
    !std::is_constructible_v<gcxx::span<int>, gcxx::span<const int, 5>&>);

  gcxx::span<int> from_fixed{fixed};
  gcxx::span<const int> readonly{fixed};
  gcxx::span<int, 5> from_dynamic{dynamic};

  EXPECT_EQ(from_fixed.data(), values);
  EXPECT_EQ(from_fixed.size(), std::size(values));
  EXPECT_EQ(readonly.data(), values);
  EXPECT_EQ(readonly.size(), std::size(values));
  EXPECT_EQ(from_dynamic.data(), values);
  EXPECT_EQ(from_dynamic.size(), std::size(values));
}

TEST(SpanConstructors, RestrictSpanUsesSharedConversionConstructors) {
  int values[]{1, 2, 3, 4, 5};
  gcxx::restrict_span<int, 5> fixed{values};
  gcxx::restrict_span<int> dynamic{values};

  static_assert(std::is_constructible_v<gcxx::restrict_span<int>,
                                        gcxx::restrict_span<int, 5>&>);
  static_assert(std::is_constructible_v<gcxx::restrict_span<const int>,
                                        gcxx::restrict_span<int, 5>&>);
  static_assert(std::is_constructible_v<gcxx::restrict_span<const int, 5>,
                                        gcxx::restrict_span<int, 5>&>);
  static_assert(std::is_constructible_v<gcxx::restrict_span<int, 5>,
                                        gcxx::restrict_span<int>&>);
  static_assert(!std::is_constructible_v<gcxx::restrict_span<int, 4>,
                                         gcxx::restrict_span<int, 5>&>);
  static_assert(!std::is_constructible_v<gcxx::restrict_span<int>,
                                         gcxx::restrict_span<const int, 5>&>);

  gcxx::restrict_span<int> from_fixed{fixed};
  gcxx::restrict_span<const int> readonly{fixed};
  gcxx::restrict_span<int, 5> from_dynamic{dynamic};

  EXPECT_EQ(from_fixed.data(), values);
  EXPECT_EQ(from_fixed.size(), std::size(values));
  EXPECT_EQ(readonly.data(), values);
  EXPECT_EQ(readonly.size(), std::size(values));
  EXPECT_EQ(from_dynamic.data(), values);
  EXPECT_EQ(from_dynamic.size(), std::size(values));
}

TEST(SpanObserversAndElementAccess, MatchCppreferenceObserverSemantics) {
  int values[]{1, 2, 3, 4, 5};
  gcxx::span<int> s{values};

  EXPECT_FALSE(s.empty());
  EXPECT_EQ(s.size(), std::size(values));
  EXPECT_EQ(s.size_bytes(), sizeof(values));
  EXPECT_EQ(s.data(), values);
  EXPECT_EQ(s.front(), 1);
  EXPECT_EQ(s.back(), 5);
  EXPECT_EQ(s[2], 3);
}

TEST(SpanSubviews, FirstLastAndSubspanExposeExpectedSlices) {
  int values[]{1, 2, 3, 4, 5};
  gcxx::span<int> s{values};

  auto first_two = s.first(2);
  auto last_two  = s.last(2);
  auto middle    = s.subspan(1, 3);

  static_assert(std::is_same_v<decltype(first_two), gcxx::span<int>>);
  static_assert(std::is_same_v<decltype(last_two), gcxx::span<int>>);
  static_assert(std::is_same_v<decltype(middle), gcxx::span<int>>);

  EXPECT_EQ(first_two.data(), values);
  EXPECT_EQ(first_two.size(), 2U);
  EXPECT_EQ(last_two.data(), values + 3);
  EXPECT_EQ(last_two.size(), 2U);
  EXPECT_EQ(middle.data(), values + 1);
  EXPECT_EQ(middle.size(), 3U);
}

TEST(SpanSubviews, StaticSubviewsReturnPublicWrapperTypes) {
  int values[]{1, 2, 3, 4, 5};
  gcxx::span<int, 5> s{values};
  gcxx::restrict_span<int, 5> rs{values};

  auto first_two       = s.first<2>();
  auto last_two        = s.last<2>();
  auto middle          = s.subspan<1, 3>();
  auto restrict_middle = rs.subspan<1, 3>();

  static_assert(std::is_same_v<decltype(first_two), gcxx::span<int, 2>>);
  static_assert(std::is_same_v<decltype(last_two), gcxx::span<int, 2>>);
  static_assert(std::is_same_v<decltype(middle), gcxx::span<int, 3>>);
  static_assert(
    std::is_same_v<decltype(restrict_middle), gcxx::restrict_span<int, 3>>);

  EXPECT_EQ(first_two.data(), values);
  EXPECT_EQ(first_two.size(), 2U);
  EXPECT_EQ(last_two.data(), values + 3);
  EXPECT_EQ(last_two.size(), 2U);
  EXPECT_EQ(middle.data(), values + 1);
  EXPECT_EQ(middle.size(), 3U);
  EXPECT_EQ(restrict_middle.data(), values + 1);
  EXPECT_EQ(restrict_middle.size(), 3U);
}
