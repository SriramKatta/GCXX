// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Tier 1 coverage: initializer_list/range ctors; runtime tests use zero
// sizes (no GPU Copy) while non-zero callability is checked statically.
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include <gcxx/api.hpp>

namespace {

  struct host_mock_resource {
    void* allocate(gcxx::StreamView, std::size_t num_bytes) {
      return std::malloc(num_bytes);
    }
    void deallocate(gcxx::StreamView, void* ptr) { std::free(ptr); }

    // Advertise host_accessible to satisfy buffer's static_assert.
    using properties = gcxx::TypeSet<gcxx::host_accessible>;
  };

  template <typename VT>
  using mock_buffer = gcxx::buffer<VT, gcxx::host_accessible>;

  // SFINAE boundary: integral args must hit the size ctor, not range.
  GCXX_DEFINE_IS_CALLABLE(is_range_ctor_callable,
                          mock_buffer<int>(std::declval<gcxx::StreamView>(),
                                           std::declval<host_mock_resource>(),
                                           std::declval<Args>()...));

}  // namespace

TEST(BufferRangeCtorTest, InitializerListEmptyHasZeroSize) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::initializer_list<int>{});
  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.size_bytes(), 0);
}

TEST(BufferRangeCtorTest, EmptyVectorRangeHasZeroSize) {
  std::vector<int> v;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, v);
  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.size_bytes(), 0);
}

TEST(BufferRangeCtorTest, EmptyStdArrayRangeHasZeroSize) {
  // std::array with size 0 is a valid empty range; .size() == 0.
  std::array<int, 0> a;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, a);
  EXPECT_EQ(buf.size(), 0);
}

// No rejection assert: integrals still legally call the size ctor.
TEST(BufferRangeCtorSfinaeTest, AcceptsSizedRanges) {
  static_assert(is_range_ctor_callable_v<std::vector<int>&>);
  static_assert(is_range_ctor_callable_v<std::vector<int>>);
  static_assert(is_range_ctor_callable_v<std::array<int, 4>&>);
}

TEST(MakeBufferTest, MakeBufferWithSize) {
  auto buf = gcxx::make_buffer<int, gcxx::host_accessible>(
    gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
    gcxx::no_init);
  EXPECT_EQ(buf.size(), 8);
}

TEST(MakeBufferTest, MakeBufferWithInitializerList) {
  auto buf = gcxx::make_buffer<int, gcxx::host_accessible>(
    gcxx::StreamView::Null(), host_mock_resource{},
    std::initializer_list<int>{});
  EXPECT_EQ(buf.size(), 0);
}
