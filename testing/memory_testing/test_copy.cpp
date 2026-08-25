// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gcxx/api.hpp>

// Raw-handle contract (see tests_common.hpp); no dedicated stream test yet.
GCXX_ASSERT_RAW_HANDLE(StreamView, gcxx::driver::deviceStream_t);
GCXX_ASSERT_RAW_HANDLE(Stream, gcxx::driver::deviceStream_t);

namespace {

  using u32        = std::uint32_t;
  using device_ptr = gcxx::device_ptr<u32>;
  using device_buf = gcxx::device_buffer<u32>;

  // Satisfies no handle/span trait: universal negative case.
  struct NotAHandle {};

  // Args... is the candidate pack; stream/count are concrete.

  // Copy(dst, src, count) — sync, pointer/smart-pointer.
  GCXX_DEFINE_IS_CALLABLE(is_copy_ptrs_sync_callable,
                          gcxx::Copy(std::declval<Args>()..., std::size_t{4}));

  // Copy(stream, dst, src, count) — async, pointer/smart-pointer.
  GCXX_DEFINE_IS_CALLABLE(is_copy_ptrs_async_callable,
                          gcxx::Copy(std::declval<const gcxx::StreamView&>(),
                                     std::declval<Args>()..., std::size_t{4}));

  // Copy(stream, dstSpan, srcSpan) — async, span-like.
  GCXX_DEFINE_IS_CALLABLE(is_copy_spans_async_callable,
                          gcxx::Copy(std::declval<const gcxx::StreamView&>(),
                                     std::declval<Args>()...));

}  // namespace

// Lvalue pointers (u32*&) are the regression net for the uncvref fix.
TEST(CopySfinaeTest, AcceptsValidArgumentShapes) {
  static_assert(is_copy_ptrs_sync_callable_v<u32*&, u32*&>);
  static_assert(is_copy_ptrs_sync_callable_v<u32*, u32*>);
  static_assert(is_copy_ptrs_sync_callable_v<device_ptr&, device_ptr&>);

  static_assert(is_copy_ptrs_async_callable_v<u32*&, u32*&>);
  static_assert(is_copy_ptrs_async_callable_v<device_ptr&, device_ptr&>);

  static_assert(
    is_copy_spans_async_callable_v<gcxx::span<u32>&, gcxx::span<u32>&>);
  static_assert(is_copy_spans_async_callable_v<device_buf&, device_buf&>);
}

// Negative asserts impossible with the old decltype type check.
TEST(CopySfinaeTest, RejectsInvalidArgumentShapes) {
  // Pointer overloads reject spans (no .get()), NotAHandle, and plain values.
  static_assert(
    !is_copy_ptrs_sync_callable_v<gcxx::span<u32>&, gcxx::span<u32>&>);
  static_assert(!is_copy_ptrs_sync_callable_v<NotAHandle, NotAHandle>);
  static_assert(
    !is_copy_ptrs_async_callable_v<gcxx::span<u32>&, gcxx::span<u32>&>);

  // Span overloads reject raw pointers and NotAHandle.
  static_assert(!is_copy_spans_async_callable_v<u32*, u32*>);
  static_assert(!is_copy_spans_async_callable_v<NotAHandle, NotAHandle>);
}

TEST(CopyTest, RawPointerSyncRoundTrip) {
  if (!gcxx::testing::haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  constexpr std::size_t N = 1024;
  std::vector<u32> h_src(N), h_dst(N);
  std::iota(h_src.begin(), h_src.end(), u32{0});

  auto d = gcxx::make_device_unique_ptr<u32>(N);

  // lvalue pointers on purpose: exercises the Ptr = T*& deduction path.
  u32* d_raw   = d.get();
  u32* src_raw = h_src.data();
  u32* dst_raw = h_dst.data();
  gcxx::Copy(d_raw, src_raw, N);  // H2D (sync, blocks)
  gcxx::Copy(dst_raw, d_raw, N);  // D2H (sync, blocks)

  EXPECT_EQ(h_src, h_dst);
}

TEST(CopyTest, RawPointerAsyncRoundTrip) {
  if (!gcxx::testing::haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  constexpr std::size_t N = 1024;
  std::vector<u32> h_src(N), h_dst(N);
  std::iota(h_src.begin(), h_src.end(), u32{7});

  auto d = gcxx::make_device_unique_ptr<u32>(N);
  gcxx::Stream str;

  u32* d_raw   = d.get();
  u32* src_raw = h_src.data();
  u32* dst_raw = h_dst.data();
  gcxx::Copy(str, d_raw, src_raw, N);  // H2D async
  gcxx::Copy(str, dst_raw, d_raw, N);  // D2H async
  str.sync();

  EXPECT_EQ(h_src, h_dst);
}

TEST(CopyTest, SpanAsyncRoundTrip) {
  if (!gcxx::testing::haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  constexpr std::size_t N = 512;
  std::vector<u32> h_src(N), h_dst(N);
  std::iota(h_src.begin(), h_src.end(), u32{1});

  auto d = gcxx::make_device_unique_ptr<u32>(N);
  gcxx::Stream str;

  gcxx::span<u32> d_span(d.get(), N);
  gcxx::span<u32> src_span(h_src.data(), N);
  gcxx::span<u32> dst_span(h_dst.data(), N);
  gcxx::Copy(str, d_span, src_span);  // H2D async
  gcxx::Copy(str, dst_span, d_span);  // D2H async
  str.sync();

  EXPECT_EQ(h_src, h_dst);
}
