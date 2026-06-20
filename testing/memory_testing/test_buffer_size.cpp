// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>

#include <gcxx/api.hpp>

namespace {

  // Host-only resource (malloc/free) so buffer size logic can be exercised
  // without a GPU or the CUDA runtime.
  struct host_mock_resource {
    void* allocate(std::size_t num_bytes, gcxx::StreamView) {
      return std::malloc(num_bytes);
    }

    void deallocate(void* ptr, gcxx::StreamView) { std::free(ptr); }
  };

  template <typename VT>
  using mock_buffer = gcxx::memory::buffer<VT, host_mock_resource>;

}  // namespace

// Regression: buffer_storage::size_bytes() previously returned the element
// count instead of the byte count, so buffer::size() reported N/sizeof(VT)
// elements for an allocation of N.
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

// Regression for the vector_add_overlap crash: a span built from the buffer
// previously had size N/sizeof(VT), so offsets in [N/sizeof(VT), N) tripped
// the subspan contract. With the fix these offsets are valid.
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
