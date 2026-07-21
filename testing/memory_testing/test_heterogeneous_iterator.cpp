// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Phase 5 coverage for heterogeneous_iterator: the buffer's iterator type, its
// random-access mechanics, dereference from host code (host_accessible buffer),
// and interop with std algorithms (std::fill / std::accumulate). Uses a host
// mock so the suite runs without a GPU.
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  struct host_mock_resource {
    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible>;
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
  };

  using host_buf = gcxx::memory::buffer<int, gcxx::memory::host_accessible>;

  void raw_fill(host_buf& b, int start) {
    for (std::size_t i = 0; i < b.size(); ++i)
      b.data()[i] = start + static_cast<int>(i);
  }

}  // namespace

// =============================================================================
// buffer<VT, Properties...>::iterator is heterogeneous_iterator<VT,
// Properties...>.
// =============================================================================
TEST(HeterogeneousIteratorTest, BufferIteratorIsHeterogeneous) {
  static_assert(
    std::is_same_v<host_buf::iterator, gcxx::heterogeneous_iterator<
                                         int, gcxx::memory::host_accessible>>);
  static_assert(
    std::is_same_v<
      host_buf::const_iterator,
      gcxx::heterogeneous_iterator<const int, gcxx::memory::host_accessible>>);
}

// =============================================================================
// Dereference + iteration via begin()/end() from host code.
// =============================================================================
TEST(HeterogeneousIteratorTest, IterateAndDereference) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{4},
             gcxx::memory::no_init);
  raw_fill(b, 10);

  int sum = 0;
  for (auto it = b.begin(); it != b.end(); ++it)
    sum += *it;
  EXPECT_EQ(sum, 10 + 11 + 12 + 13);
}

// =============================================================================
// Random-access mechanics: [], +, -, ++/--.
// =============================================================================
TEST(HeterogeneousIteratorTest, RandomAccessOps) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
             gcxx::memory::no_init);
  raw_fill(b, 0);

  auto it = b.begin();
  EXPECT_EQ(*it, 0);
  EXPECT_EQ(it[3], 3);
  EXPECT_EQ(*(it + 5), 5);
  EXPECT_EQ((b.end() - b.begin()), 8);

  ++it;
  EXPECT_EQ(*it, 1);
  --it;
  EXPECT_EQ(*it, 0);
  it += 4;
  EXPECT_EQ(*it, 4);
  it -= 2;
  EXPECT_EQ(*it, 2);
}

// =============================================================================
// std algorithms work through the iterators (contiguous random-access).
// =============================================================================
TEST(HeterogeneousIteratorTest, StdAlgorithmsInterop) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
             gcxx::memory::no_init);
  std::fill(b.begin(), b.end(), 7);
  EXPECT_EQ(*b.begin(), 7);
  EXPECT_EQ(*(b.end() - 1), 7);

  const int total = std::accumulate(b.begin(), b.end(), 0);
  EXPECT_EQ(total, 7 * 8);
}
