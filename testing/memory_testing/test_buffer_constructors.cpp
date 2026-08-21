// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
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

  using device_ptr = gcxx::device_ptr<std::uint32_t>;
  using device_buf = gcxx::device_buffer<std::uint32_t>;

  // Satisfies no handle/span trait: universal negative case.
  struct NotAHandle {};

  // Args... is the candidate pack; stream/resource/count are concrete.
  GCXX_DEFINE_IS_CALLABLE(
    is_buf_value_init_callable,
    mock_buffer<std::uint32_t>(std::declval<gcxx::StreamView>(),
                               std::declval<host_mock_resource>(),
                               std::size_t{4}, std::declval<Args>()...));

  GCXX_DEFINE_IS_CALLABLE(is_fill_ptr_callable,
                          gcxx::Fill(std::declval<Args>()..., std::uint32_t{0},
                                     std::size_t{4}));

  GCXX_DEFINE_IS_CALLABLE(is_fill_span_callable,
                          gcxx::Fill(std::declval<Args>()...,
                                     std::uint32_t{0}));

}  // namespace

TEST(BufferSfinaeTest, AcceptsValidHandleShapes) {
  static_assert(is_buf_value_init_callable_v<std::uint32_t>);
  static_assert(is_fill_ptr_callable_v<std::uint32_t*&>);
  static_assert(is_fill_ptr_callable_v<device_ptr&>);
  static_assert(is_fill_span_callable_v<gcxx::span<std::uint32_t>&>);
  static_assert(is_fill_span_callable_v<device_buf&>);
}

// Negative asserts impossible with the old decltype type check.
TEST(BufferSfinaeTest, RejectsInvalidHandleShapes) {
  static_assert(!is_buf_value_init_callable_v<NotAHandle>);

  // Pointer Fill rejects spans (no .get()), NotAHandle.
  static_assert(!is_fill_ptr_callable_v<gcxx::span<std::uint32_t>&>);
  static_assert(!is_fill_ptr_callable_v<NotAHandle>);

  // Span Fill rejects raw pointers (no .data()/.size() members), NotAHandle.
  static_assert(!is_fill_span_callable_v<std::uint32_t*>);
  static_assert(!is_fill_span_callable_v<NotAHandle>);
}

TEST(BufferCtorTest, StreamResourceCtorIsEmpty) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{});

  EXPECT_TRUE(buf.empty());
  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.size_bytes(), 0);
}

TEST(BufferCtorTest, StreamResourceCtorKeepsResourceAndStream) {
  host_mock_resource res{};
  mock_buffer<int> buf(gcxx::StreamView::Null(), res);

  EXPECT_EQ(buf.size_bytes(), 0);
  // Stream round-trips through the storage.
  EXPECT_EQ(buf.stream().getRawHandle(),
            gcxx::StreamView::Null().getRawHandle());
}

TEST(BufferCtorTest, NoInitCtorAllocatesRequestedSize) {
  constexpr std::size_t N = 1024;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, N,
                       gcxx::no_init);

  EXPECT_FALSE(buf.empty());
  EXPECT_EQ(buf.size(), N);
  EXPECT_EQ(buf.size_bytes(), N * sizeof(int));
}

TEST(BufferCtorTest, NoInitCtorZeroSizeHasZeroSize) {
  // Zero-size allocations are valid handles; assert counts, not empty().
  mock_buffer<float> buf(gcxx::StreamView::Null(), host_mock_resource{},
                         std::size_t{0}, gcxx::no_init);

  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.size_bytes(), 0);
}

// n=0 still compiles/links the whole fill path; fill_dispatch early-returns.
TEST(BufferCtorTest, ValueInitZeroSizeInstantiatesFillPath) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{0}, 0);
  EXPECT_EQ(buf.size(), 0);
}
