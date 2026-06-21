// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  // Host-only resource (malloc/free) so ctor/size logic can be exercised
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
// (memset for zero, kernel for non-zero) needs a GPU to execute, so here we
// only assert the constructors are callable (compile-time), mirroring the
// Memset callable-tests.
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferCtorTest, ValueInitCtorIsCallableWithZero) {
  using buf_t = mock_buffer<std::uint32_t>;
  EXPECT_TRUE((std::is_same_v<decltype(buf_t(std::declval<gcxx::StreamView>(),
                                             std::declval<host_mock_resource>(),
                                             std::size_t{4}, std::uint32_t{0})),
                              buf_t>));
}

TEST(BufferCtorTest, ValueInitCtorIsCallableWithNonZero) {
  using buf_t = mock_buffer<std::uint32_t>;
  EXPECT_TRUE((std::is_same_v<decltype(buf_t(std::declval<gcxx::StreamView>(),
                                             std::declval<host_mock_resource>(),
                                             std::size_t{4}, std::uint32_t{7})),
                              buf_t>));
}

// n == 0 actually constructs via the value-init ctor, which forces the whole
// Fill / fill_dispatch template to instantiate (so the fill_kernel + launch
// path is compiled and linked) while performing no GPU operation —
// fill_dispatch returns early on a zero element count.
TEST(BufferCtorTest, ValueInitZeroSizeInstantiatesFillPath) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{0}, 0);
  EXPECT_EQ(buf.size(), 0);
}

TEST(BufferCtorTest, FillOverloadsAreCallable) {
  using device_ptr = gcxx::memory::device_ptr<std::uint32_t>;
  using device_buf = gcxx::memory::device_buffer<std::uint32_t>;

  EXPECT_TRUE((std::is_same_v<decltype(gcxx::memory::Fill(
                                std::declval<std::uint32_t*&>(),
                                std::uint32_t{0}, std::size_t{4})),
                              void>));
  EXPECT_TRUE((std::is_same_v<decltype(gcxx::memory::Fill(
                                std::declval<device_ptr&>(), std::uint32_t{0},
                                std::size_t{4})),
                              void>));
  EXPECT_TRUE((std::is_same_v<decltype(gcxx::memory::Fill(
                                std::declval<gcxx::span<std::uint32_t>&>(),
                                std::uint32_t{0})),
                              void>));
  EXPECT_TRUE((std::is_same_v<decltype(gcxx::memory::Fill(
                                std::declval<device_buf&>(), std::uint32_t{0})),
                              void>));
}
