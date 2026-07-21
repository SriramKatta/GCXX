// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gcxx/api.hpp>

namespace {

  // Host-only resource (malloc/free) so ctor/size logic can be exercised
  // without a GPU or the CUDA runtime. T3: must advertise host_accessible
  // to satisfy buffer's static_assert on execution-space properties.
  struct host_mock_resource {
    void* allocate(std::size_t num_bytes, gcxx::StreamView) {
      return std::malloc(num_bytes);
    }

    void deallocate(void* ptr, gcxx::StreamView) { std::free(ptr); }

    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible>;
  };

  template <typename VT>
  using mock_buffer = gcxx::memory::buffer<VT, gcxx::memory::host_accessible>;

  using device_ptr = gcxx::memory::device_ptr<std::uint32_t>;
  using device_buf = gcxx::memory::device_buffer<std::uint32_t>;

  // Satisfies neither is_pointer_or_has_get_v nor is_span_like_v — the
  // universal negative case for every overload's SFINAE constraint.
  struct NotAHandle {};

  // ─────────────────────────────────────────────────────────────────────────────
  // Detection traits per overload shape. Args... is the candidate pack; the
  // rest of the call (stream, resource, count) is concrete. Positive asserts
  // check each accepted shape; negative asserts check rejection — something
  // the old decltype(...) type check could not do at all.
  //
  // value-init ctor: Args = the value type (last ctor arg).
  // ─────────────────────────────────────────────────────────────────────────────

  GCXX_DEFINE_IS_CALLABLE(
    is_buf_value_init_callable,
    mock_buffer<std::uint32_t>(std::declval<gcxx::StreamView>(),
                               std::declval<host_mock_resource>(),
                               std::size_t{4}, std::declval<Args>()...));

  GCXX_DEFINE_IS_CALLABLE(is_fill_ptr_callable,
                          gcxx::memory::Fill(std::declval<Args>()...,
                                             std::uint32_t{0}, std::size_t{4}));

  GCXX_DEFINE_IS_CALLABLE(is_fill_span_callable,
                          gcxx::memory::Fill(std::declval<Args>()...,
                                             std::uint32_t{0}));

}  // namespace

// =============================================================================
// Positive: value-init ctor + each Fill overload resolve for accepted shapes.
// =============================================================================
TEST(BufferSfinaeTest, AcceptsValidHandleShapes) {
  static_assert(is_buf_value_init_callable_v<std::uint32_t>);
  static_assert(is_fill_ptr_callable_v<std::uint32_t*&>);
  static_assert(is_fill_ptr_callable_v<device_ptr&>);
  static_assert(is_fill_span_callable_v<gcxx::span<std::uint32_t>&>);
  static_assert(is_fill_span_callable_v<device_buf&>);
}

// =============================================================================
// Negative: each overload rejects the wrong handle / value shape. This is the
// part the old decltype(...) type check could not do at all.
// =============================================================================
TEST(BufferSfinaeTest, RejectsInvalidHandleShapes) {
  static_assert(!is_buf_value_init_callable_v<NotAHandle>);

  // Pointer Fill rejects spans (no .get()), NotAHandle.
  static_assert(!is_fill_ptr_callable_v<gcxx::span<std::uint32_t>&>);
  static_assert(!is_fill_ptr_callable_v<NotAHandle>);

  // Span Fill rejects raw pointers (no .data()/.size() members), NotAHandle.
  static_assert(!is_fill_span_callable_v<std::uint32_t*>);
  static_assert(!is_fill_span_callable_v<NotAHandle>);
}

// ─────────────────────────────────────────────────────────────────────────────
// buffer(stream, resource) — empty buffer bound to a stream + resource.
// ─────────────────────────────────────────────────────────────────────────────
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
  EXPECT_EQ(buf.stream().getRawStream(),
            gcxx::StreamView::Null().getRawStream());
}

// ─────────────────────────────────────────────────────────────────────────────
// buffer(stream, resource, n, no_init) — allocate n elements, uninitialized.
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferCtorTest, NoInitCtorAllocatesRequestedSize) {
  constexpr std::size_t N = 1024;
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{}, N,
                       gcxx::memory::no_init);

  EXPECT_FALSE(buf.empty());
  EXPECT_EQ(buf.size(), N);
  EXPECT_EQ(buf.size_bytes(), N * sizeof(int));
}

TEST(BufferCtorTest, NoInitCtorZeroSizeHasZeroSize) {
  // A zero-size allocation is a valid (possibly non-null) handle with a count
  // of zero — empty() is ptr-based and implementation-defined for malloc(0),
  // so assert on the element/byte counts instead.
  mock_buffer<float> buf(gcxx::StreamView::Null(), host_mock_resource{},
                         std::size_t{0}, gcxx::memory::no_init);

  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.size_bytes(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// buffer(stream, resource, n, value) — value-initialized. The dispatch
// (memset for zero, kernel for non-zero) needs a GPU to execute, so callability
// is asserted at namespace scope above (is_buf_value_init_callable); here we
// only run the zero-size path.
// ─────────────────────────────────────────────────────────────────────────────

// n == 0 actually constructs via the value-init ctor, which forces the whole
// Fill / fill_dispatch template to instantiate (so the fill_kernel + launch
// path is compiled and linked) while performing no GPU operation —
// fill_dispatch returns early on a zero element count.
TEST(BufferCtorTest, ValueInitZeroSizeInstantiatesFillPath) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{0}, 0);
  EXPECT_EQ(buf.size(), 0);
}
