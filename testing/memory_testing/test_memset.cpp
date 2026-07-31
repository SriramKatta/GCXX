// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gcxx/api.hpp>

namespace {

  template <typename T>
  auto memset_expected_value(const int byte_value) -> T {
    T value{};
    std::memset(&value, byte_value, sizeof(T));
    return value;
  }

  template <typename T, std::size_t N>
  void expect_device_contents_equal(T* device_ptr, const int byte_value) {
    std::array<T, N> host_values{};
    gcxx::Copy(host_values.data(), device_ptr, N);

    const T expected = memset_expected_value<T>(byte_value);
    for (const auto& actual : host_values) {
      EXPECT_EQ(actual, expected);
    }
  }

  using device_ptr = gcxx::device_ptr<std::uint32_t>;
  using device_buf = gcxx::device_buffer<std::uint32_t>;

  // Satisfies neither is_pointer_or_has_get_v nor is_span_like_v — the
  // universal negative case for every overload's SFINAE constraint.
  struct NotAHandle {};

  // ─────────────────────────────────────────────────────────────────────────────
  // Detection traits per overload shape. Args... is the handle candidate;
  // value/count are concrete. Positive asserts check each accepted shape;
  // negative asserts check rejection — something the old decltype(...) type
  // check could not do at all.
  // ─────────────────────────────────────────────────────────────────────────────

  // Memset(handle, value, count) — sync, pointer/smart-pointer.
  GCXX_DEFINE_IS_CALLABLE(is_memset_ptr_callable,
                          gcxx::Memset(std::declval<Args>()..., 0,
                                       std::size_t{4}));

  // Memset(spanLike, value) — sync, span-like.
  GCXX_DEFINE_IS_CALLABLE(is_memset_span_callable,
                          gcxx::Memset(std::declval<Args>()..., 0));

}  // namespace

// =============================================================================
// Positive: every tested overload resolves for the handle shapes it accepts.
// =============================================================================

TEST(MemsetSfinaeTest, AcceptsValidHandleShapes) {
  static_assert(is_memset_ptr_callable_v<std::uint32_t*&>);
  static_assert(is_memset_ptr_callable_v<device_ptr&>);
  static_assert(is_memset_span_callable_v<gcxx::span<std::uint32_t>&>);
  static_assert(is_memset_span_callable_v<device_buf&>);
}

// =============================================================================
// Negative: each overload rejects the wrong handle shape. This is the part
// the old decltype(...) type check could not do at all.
// =============================================================================

TEST(MemsetSfinaeTest, RejectsInvalidHandleShapes) {
  // Pointer overload rejects spans (no .get()), NotAHandle, plain values.
  static_assert(!is_memset_ptr_callable_v<gcxx::span<std::uint32_t>&>);
  static_assert(!is_memset_ptr_callable_v<NotAHandle>);

  // Span overload rejects raw pointers (no .data()/.size() members),
  // NotAHandle.
  static_assert(!is_memset_span_callable_v<std::uint32_t*>);
  static_assert(!is_memset_span_callable_v<NotAHandle>);
}
