// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>

#include <gcxx/api.hpp>

namespace {

  // Host-only malloc/free mock; properties satisfy buffer's static_assert.
  struct host_mock_resource {
    void* allocate(gcxx::StreamView, std::size_t num_bytes) {
      return std::malloc(num_bytes);
    }

    void deallocate(gcxx::StreamView, void* ptr) { std::free(ptr); }

    using properties = gcxx::TypeSet<gcxx::host_accessible>;
  };

  template <typename VT>
  using mock_buffer = gcxx::buffer<VT, gcxx::host_accessible>;

}  // namespace

// Regression: size_bytes() once returned the element count.
TEST(BufferSizeTest, ReportsElementCountAndByteCountForInt) {
  constexpr std::size_t N = 1000;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, N);

  EXPECT_EQ(buf.size(), N);
  EXPECT_EQ(buf.size_bytes(), N * sizeof(int));
}

TEST(BufferSizeTest, ReportsElementCountAndByteCountForFloat) {
  constexpr std::size_t N = 500;
  mock_buffer<float> buf(gcxx::StreamView::Null(), host_mock_resource{}, N);

  EXPECT_EQ(buf.size(), N);
  EXPECT_EQ(buf.size_bytes(), N * sizeof(float));
}

TEST(BufferSizeTest, SpanFromBufferPreservesElementCount) {
  constexpr std::size_t N = 1000;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, N);
  gcxx::span<int> s(buf);

  EXPECT_EQ(s.size(), N);
  EXPECT_EQ(s.size_bytes(), N * sizeof(int));
}

// Regression: span size N/sizeof(VT) broke subspan offsets in [N/2, N).
TEST(BufferSizeTest, SpanFromBufferAcceptsFullRangeSubspanOffsets) {
  constexpr std::size_t N = 1000;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, N);
  gcxx::span<int> s(buf);

  ASSERT_EQ(s.size(), N);

  auto first_half  = s.subspan(0, N / 2);
  auto second_half = s.subspan(N / 2, N / 2);
  EXPECT_EQ(first_half.size(), N / 2);
  EXPECT_EQ(second_half.size(), N / 2);
  EXPECT_EQ(second_half.data(), s.data() + N / 2);
}

TEST(BufferSizeTest, ResizeReallocatesThroughStoredResource) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, 10);
  ASSERT_EQ(buf.size(), 10);

  buf.resize(100);
  EXPECT_EQ(buf.size(), 100);
  EXPECT_EQ(buf.size_bytes(), 100 * sizeof(int));
  buf[99] = 7;  // storage is host_accessible: writable after resize
  EXPECT_EQ(buf[99], 7);

  buf.resize(4);
  EXPECT_EQ(buf.size(), 4);
  buf[0] = 1;
  EXPECT_EQ(buf[0], 1);

  buf.resize(0);
  EXPECT_EQ(buf.size(), 0);
  EXPECT_TRUE(buf.empty());
}

// Regression: driver Fill failed on host memory; now the std::fill_n path.
TEST(BufferSizeTest, ValueInitOnHostMemoryUsesHostFill) {
  constexpr std::size_t N = 100;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, N, 42);
  ASSERT_EQ(buf.size(), N);
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_EQ(buf[i], 42) << "element " << i;
  }

  // Zero value (previously the driver-memset path) must work too.
  mock_buffer<int> zeros(gcxx::StreamView::Null(), host_mock_resource{}, N, 0);
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_EQ(zeros[i], 0) << "element " << i;
  }
}
