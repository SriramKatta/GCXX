// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include <array>
#include <vector>

#include "tests_common.hpp"

#include <gcxx/api.hpp>

TEST(SpanDeductionGuide, BasicDeductionGuides) {
  int a[]{1, 2, 3, 4, 5};

  gcxx::span s1{std::begin(a),
                std::end(a)};       // guide (1): iter + end  → dynamic
  gcxx::span s2{std::begin(a), 3};  // guide (1): iter + count → dynamic
  gcxx::span s4{a};       // guide (2): C-array      → static extent N
  gcxx::span<int> s5{a};  // no guide, explicit <int> → dynamic_extent

  std::array arr{6, 7, 8};
  gcxx::span s6{arr};  // guide (3): std::array& → static extent N

  const std::array arr2{9, 10, 11};
  gcxx::span s7{arr2};  // guide (4): const std::array& → static extent N

  std::vector v{66, 69, 99};
  [[maybe_unused]] gcxx::span s8{v};  // guide (5): range → dynamic

  // ── sizeof ──────────────────────────────────────────────────────────────
  EXPECT_EQ(sizeof(s1), 16) << "FAIL : pointer + stored size";
  EXPECT_EQ(sizeof(s2), 16) << "FAIL : pointer + stored size";
  EXPECT_EQ(sizeof(s4), 8) << "FAIL : pointer only (extent baked in)";
  EXPECT_EQ(sizeof(s5), 16) << "FAIL : pointer + stored size";
  EXPECT_EQ(sizeof(s6), 8) << "FAIL : pointer only (extent baked in)";
  EXPECT_EQ(sizeof(s7), 8) << "FAIL : pointer only (extent baked in)";
  EXPECT_EQ(sizeof(s8), 16) << "FAIL : pointer + stored size";

  // ── extent ──────────────────────────────────────────────────────────────
  EXPECT_EQ(s1.extent, gcxx::dynamic_extent) << "FAIL : pointer + stored size";
  EXPECT_EQ(s2.extent, gcxx::dynamic_extent) << "FAIL : pointer + stored size";
  EXPECT_EQ(s4.extent, 5) << "FAIL : pointer only (extent baked in)";
  EXPECT_EQ(s5.extent, gcxx::dynamic_extent) << "FAIL : pointer + stored size";
  EXPECT_EQ(s6.extent, 3) << "FAIL : pointer only (extent baked in)";
  EXPECT_EQ(s7.extent, 3) << "FAIL : pointer only (extent baked in)";
  EXPECT_EQ(s8.extent, gcxx::dynamic_extent) << "FAIL : pointer + stored size";
}
