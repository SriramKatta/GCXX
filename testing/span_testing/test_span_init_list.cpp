// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <type_traits>

#include <gcxx/api.hpp>

// Braced-list construction: the initializer_list constructor takes
// value_type (not element_type), so const-qualified spans bind non-const
// lists and lists of derived classes slice in — the dependency-list shape
// used by GraphView::addNode(p, {nodeA, nodeB}).
//
// The backing array lives only for the FULL EXPRESSION, so these tests use
// the safe form: a braced list as an argument to a span PARAMETER. Storing
// `span<const T> s = {a, b};` and reading it on a later statement is UB —
// empirically it "works" under nvcc and reads garbage under HIP/clang.

namespace {

  struct Payload {};
  struct PayloadA : Payload {};
  struct PayloadB : Payload {};

  auto countInts(gcxx::span<const int> s) -> std::size_t {
    return s.size();
  }

  auto sumInts(gcxx::span<const int> s) -> int {
    int total = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
      total += s[i];
    }
    return total;
  }

}  // namespace

TEST(SpanInitList, HomogeneousConstElement) {
  EXPECT_EQ(sumInts({1, 2, 3, 4}), 10);
  EXPECT_EQ(countInts({1, 2, 3, 4}), std::size_t{4});
}

TEST(SpanInitList, NonConstElementIsRejected) {
  // initializer_list yields const pointers; a mutable span over the
  // temporary backing array is not constructible (nor meaningful).
  static_assert(
    !std::is_constructible_v<gcxx::span<int>, std::initializer_list<int>>);
  static_assert(
    std::is_constructible_v<gcxx::span<const int>, std::initializer_list<int>>);
}

TEST(SpanInitList, EmptyList) {
  EXPECT_EQ(countInts({}), std::size_t{0});
  EXPECT_EQ(sumInts({}), 0);
}

TEST(SpanInitList, MixedDerivedElementsSliceIn) {
  PayloadA a{};
  PayloadB b{};

  auto countPayloads = [](gcxx::span<const Payload> s) {
    return s.size();
  };
  EXPECT_EQ(countPayloads({a, b}), std::size_t{2});
}

// std::initializer_list is usable from __device__ code (trivial constexpr
// type; implicitly host+device under nvcc and HIP/clang), so the constructor
// is GCXX_FHDC like its siblings. Same lifetime rule: use the span within
// the full expression of the braced list.
namespace {

  __global__ void initListKernel(int* out) {
    *out = [](gcxx::span<const int> s) {
      int total = 0;
      for (std::size_t i = 0; i < s.size(); ++i) {
        total += s[i];
      }
      return total;
    }({1, 2, 3});
  }

}  // namespace

TEST(SpanInitList, DeviceSideConstruction) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No device available";
  }

  auto dOutRaii = gcxx::make_device_unique_ptr<int>(1);
  int* dOut     = dOutRaii.get();

  gcxx::Stream stream;
  gcxx::launch::Kernel(stream, {1}, {1}, 0, initListKernel, dOut);
  stream.Synchronize();

  int host = 0;
  gcxx::Copy(&host, dOut, 1);
  EXPECT_EQ(host, 6);
}
