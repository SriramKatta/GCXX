// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tier 1 coverage for the initializer_list and range constructors.
// The Copy call inside both ctors needs a GPU, so:
//   * Runtime tests use zero-size inputs (Copy is skipped via an early return).
//   * Callability for the non-zero path is verified via static_assert below.
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include <gcxx/api.hpp>

namespace {

  struct host_mock_resource {
    void* allocate(std::size_t num_bytes, gcxx::StreamView) {
      return std::malloc(num_bytes);
    }
    void deallocate(void* ptr, gcxx::StreamView) { std::free(ptr); }

    // Advertise host_accessible to satisfy buffer's static_assert.
    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible>;
  };

  template <typename VT>
  using mock_buffer = gcxx::memory::buffer<VT, gcxx::memory::host_accessible>;

  // Detects whether `buffer(stream, resource, T{})` resolves to the range
  // ctor. Used to assert the SFINAE boundary: integral types must hit the
  // size ctor, not the range ctor.
  GCXX_DEFINE_IS_CALLABLE(is_range_ctor_callable,
                          mock_buffer<int>(std::declval<gcxx::StreamView>(),
                                           std::declval<host_mock_resource>(),
                                           std::declval<Args>()...));

}  // namespace

// =============================================================================
// Initializer-list ctor: zero-size at runtime (Copy is skipped), size tracked.
// =============================================================================
TEST(BufferRangeCtorTest, InitializerListEmptyHasZeroSize) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::initializer_list<int>{});
  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.size_bytes(), 0);
}

// =============================================================================
// Range ctor: empty vector at runtime (Copy is skipped), size tracked.
// =============================================================================
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

// =============================================================================
// SFINAE: range ctor accepts sized ranges. Integral types still resolve to
// the existing size ctor (also callable), so we don't assert rejection here —
// the positive asserts below prove the range ctor is in the overload set.
// =============================================================================
TEST(BufferRangeCtorSfinaeTest, AcceptsSizedRanges) {
  static_assert(is_range_ctor_callable_v<std::vector<int>&>);
  static_assert(is_range_ctor_callable_v<std::vector<int>>);
  static_assert(is_range_ctor_callable_v<std::array<int, 4>&>);
}

// =============================================================================
// make_buffer factory: forwards to the matching ctor.
// =============================================================================
TEST(MakeBufferTest, MakeBufferWithSize) {
  auto buf = gcxx::memory::make_buffer<int, gcxx::memory::host_accessible>(
    gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
    gcxx::memory::no_init);
  EXPECT_EQ(buf.size(), 8);
}

TEST(MakeBufferTest, MakeBufferWithInitializerList) {
  auto buf = gcxx::memory::make_buffer<int, gcxx::memory::host_accessible>(
    gcxx::StreamView::Null(), host_mock_resource{},
    std::initializer_list<int>{});
  EXPECT_EQ(buf.size(), 0);
}
