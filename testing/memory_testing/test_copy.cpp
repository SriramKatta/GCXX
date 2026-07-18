// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gcxx/api.hpp>

namespace {

  using u32        = std::uint32_t;
  using device_ptr = gcxx::memory::device_ptr<u32>;
  using device_buf = gcxx::memory::device_buffer<u32>;

  // Satisfies neither is_pointer_or_has_get_v nor is_span_like_v — the
  // universal negative case for every overload's SFINAE constraint.
  struct NotAHandle {};

  // ─────────────────────────────────────────────────────────────────────────────
  // Detection traits per overload shape. Args... is the pack of candidate
  // types; the rest of the call (stream, count) is concrete. Positive asserts
  // check each accepted shape; negative asserts check rejection — something
  // the old decltype(...) type check could not do at all.
  // ─────────────────────────────────────────────────────────────────────────────

  // Copy(dst, src, count) — sync, pointer/smart-pointer.
  GCXX_DEFINE_IS_CALLABLE(
    is_copy_ptrs_sync_callable,
    gcxx::memory::Copy(std::declval<Args>()..., std::size_t{4}));

  // Copy(stream, dst, src, count) — async, pointer/smart-pointer.
  GCXX_DEFINE_IS_CALLABLE(
    is_copy_ptrs_async_callable,
    gcxx::memory::Copy(std::declval<const gcxx::StreamView&>(),
                       std::declval<Args>()..., std::size_t{4}));

  // Copy(stream, dstSpan, srcSpan) — async, span-like.
  GCXX_DEFINE_IS_CALLABLE(
    is_copy_spans_async_callable,
    gcxx::memory::Copy(std::declval<const gcxx::StreamView&>(),
                       std::declval<Args>()...));

  // gcxx wraps every runtime call in GCXX_SAFE_RUNTIME_CALL, which (with
  // GCXX_WITH_EXCEPTIONS off) std::abort()s on failure. So a plain device-count
  // query would kill the test binary on a GPU-less host. Probe the raw backend
  // call directly: it returns an error code instead of aborting, letting us
  // skip GPU-dependent tests gracefully.
  auto haveCudaDevice() -> bool {
    int count      = 0;
    const auto err = ::GCXX_RUNTIME_BACKEND(GetDeviceCount)(&count);
    return err == gcxx::driver::deviceErrSuccess && count > 0;
  }

}  // namespace

// =============================================================================
// Positive: every overload resolves for the argument shapes it accepts.
// Passing lvalue pointers (u32*&) is the regression net for the
// forwarding-reference trait fix — lvalue pointers used to deduce Ptr = T*&
// and fail is_pointer_or_has_get_v before the uncvref fix.
// =============================================================================

TEST(CopySfinaeTest, AcceptsValidArgumentShapes) {
  static_assert( is_copy_ptrs_sync_callable_v<u32*&, u32*&>);
  static_assert( is_copy_ptrs_sync_callable_v<u32*, u32*>);
  static_assert( is_copy_ptrs_sync_callable_v<device_ptr&, device_ptr&>);

  static_assert( is_copy_ptrs_async_callable_v<u32*&, u32*&>);
  static_assert( is_copy_ptrs_async_callable_v<device_ptr&, device_ptr&>);

  static_assert( is_copy_spans_async_callable_v<gcxx::span<u32>&,
                                                gcxx::span<u32>&>);
  static_assert( is_copy_spans_async_callable_v<device_buf&, device_buf&>);
}

// =============================================================================
// Negative: each overload rejects the wrong argument shape. This is the part
// the old decltype(...) type check could not do at all.
// =============================================================================

TEST(CopySfinaeTest, RejectsInvalidArgumentShapes) {
  // Pointer overloads reject spans (no .get()), NotAHandle, and plain values.
  static_assert(!is_copy_ptrs_sync_callable_v<gcxx::span<u32>&,
                                             gcxx::span<u32>&>);
  static_assert(!is_copy_ptrs_sync_callable_v<NotAHandle, NotAHandle>);
  static_assert(!is_copy_ptrs_async_callable_v<gcxx::span<u32>&,
                                              gcxx::span<u32>&>);

  // Span overloads reject raw pointers (no .data()/.size() members),
  // NotAHandle.
  static_assert(!is_copy_spans_async_callable_v<u32*, u32*>);
  static_assert(!is_copy_spans_async_callable_v<NotAHandle, NotAHandle>);
}

// =============================================================================
// Runtime round-trips: real H2D / D2H copies, verified on the host. Skipped on
// GPU-less hosts so the suite stays green in GPU-free CI.
// =============================================================================

TEST(CopyTest, RawPointerSyncRoundTrip) {
  if (!haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  constexpr std::size_t N = 1024;
  std::vector<u32> h_src(N), h_dst(N);
  std::iota(h_src.begin(), h_src.end(), u32{0});

  auto d = gcxx::memory::make_device_unique_ptr<u32>(N);

  // lvalue pointers on purpose: exercises the Ptr = T*& deduction path.
  u32* d_raw   = d.get();
  u32* src_raw = h_src.data();
  u32* dst_raw = h_dst.data();
  gcxx::memory::Copy(d_raw, src_raw, N);  // H2D (sync, blocks)
  gcxx::memory::Copy(dst_raw, d_raw, N);  // D2H (sync, blocks)

  EXPECT_EQ(h_src, h_dst);
}

TEST(CopyTest, RawPointerAsyncRoundTrip) {
  if (!haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  constexpr std::size_t N = 1024;
  std::vector<u32> h_src(N), h_dst(N);
  std::iota(h_src.begin(), h_src.end(), u32{7});

  auto d = gcxx::memory::make_device_unique_ptr<u32>(N);
  gcxx::Stream str;

  u32* d_raw   = d.get();
  u32* src_raw = h_src.data();
  u32* dst_raw = h_dst.data();
  gcxx::memory::Copy(str, d_raw, src_raw, N);  // H2D async
  gcxx::memory::Copy(str, dst_raw, d_raw, N);  // D2H async
  str.Synchronize();

  EXPECT_EQ(h_src, h_dst);
}

TEST(CopyTest, SpanAsyncRoundTrip) {
  if (!haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  constexpr std::size_t N = 512;
  std::vector<u32> h_src(N), h_dst(N);
  std::iota(h_src.begin(), h_src.end(), u32{1});

  auto d = gcxx::memory::make_device_unique_ptr<u32>(N);
  gcxx::Stream str;

  gcxx::span<u32> d_span(d.get(), N);
  gcxx::span<u32> src_span(h_src.data(), N);
  gcxx::span<u32> dst_span(h_dst.data(), N);
  gcxx::memory::Copy(str, d_span, src_span);  // H2D async
  gcxx::memory::Copy(str, dst_span, d_span);  // D2H async
  str.Synchronize();

  EXPECT_EQ(h_src, h_dst);
}
