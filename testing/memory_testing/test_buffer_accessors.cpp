// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tier 1 coverage for buffer's element accessors and slicing members.
// Uses host_mock_resource so the suite runs without a GPU.
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>

#include <gcxx/api.hpp>

namespace {

  // Host-only resource (malloc/free) so accessor logic can be exercised
  // without a GPU or the CUDA runtime. Advertises host_accessible via
  // `using properties` so the buffer's SFINAE-gated element accessors
  // (operator[], at, front, back) are visible.
  struct host_mock_resource {
    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible>;

    void* allocate(std::size_t num_bytes, gcxx::StreamView) {
      return std::malloc(num_bytes);
    }

    void deallocate(void* ptr, gcxx::StreamView) { std::free(ptr); }
  };

  template <typename VT>
  using mock_buffer = gcxx::memory::buffer<VT, gcxx::memory::host_accessible>;

}  // namespace

// =============================================================================
// Reference: a buffer freshly allocated with no_init has uninitialized storage.
// We hand-fill it before reading back so the assertions are deterministic.
// =============================================================================
namespace {

  void fill_buffer(mock_buffer<int>& buf, int start) {
    for (std::size_t i = 0; i < buf.size(); ++i)
      buf.data()[i] = start + static_cast<int>(i);
  }

}  // namespace

// =============================================================================
// operator[] — bounds-checked (debug-build assert) element access.
// =============================================================================
TEST(BufferAccessorsTest, SubscriptReturnsNthElement) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 100);

  EXPECT_EQ(buf[0], 100);
  EXPECT_EQ(buf[1], 101);
  EXPECT_EQ(buf[7], 107);
}

TEST(BufferAccessorsTest, SubscriptConstReturnsNthElement) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 100);
  const mock_buffer<int>& cbuf = buf;

  EXPECT_EQ(cbuf[0], 100);
  EXPECT_EQ(cbuf[7], 107);
}

TEST(BufferAccessorsTest, SubscriptIsMutable) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{4}, gcxx::memory::no_init);
  fill_buffer(buf, 0);

  buf[2] = 999;
  EXPECT_EQ(buf[2], 999);
}

// =============================================================================
// at() — throws std::out_of_range on bad index.
// =============================================================================
TEST(BufferAccessorsTest, AtReturnsNthElement) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);

  EXPECT_EQ(buf.at(0), 0);
  EXPECT_EQ(buf.at(7), 7);
}

TEST(BufferAccessorsTest, AtThrowsOutOfRange) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{4}, gcxx::memory::no_init);

  EXPECT_THROW({ (void)buf.at(4); }, std::out_of_range);
  EXPECT_THROW({ (void)buf.at(100); }, std::out_of_range);
}

TEST(BufferAccessorsTest, AtConstThrowsOutOfRange) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{4}, gcxx::memory::no_init);
  const mock_buffer<int>& cbuf = buf;

  EXPECT_THROW({ (void)cbuf.at(4); }, std::out_of_range);
}

// =============================================================================
// front() / back() — assert-guarded (empty buffer is UB).
// =============================================================================
TEST(BufferAccessorsTest, FrontBackOnSingleElement) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{1}, gcxx::memory::no_init);
  buf.data()[0] = 42;

  EXPECT_EQ(buf.front(), 42);
  EXPECT_EQ(buf.back(), 42);
}

TEST(BufferAccessorsTest, FrontBackOnMultipleElements) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{4}, gcxx::memory::no_init);
  fill_buffer(buf, 10);

  EXPECT_EQ(buf.front(), 10);
  EXPECT_EQ(buf.back(), 13);
}

TEST(BufferAccessorsTest, FrontBackConst) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{4}, gcxx::memory::no_init);
  fill_buffer(buf, 10);
  const mock_buffer<int>& cbuf = buf;

  EXPECT_EQ(cbuf.front(), 10);
  EXPECT_EQ(cbuf.back(), 13);
}

// =============================================================================
// first(n) / last(n) — span views.
// =============================================================================
TEST(BufferAccessorsTest, FirstReturnsPrefixSpan) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);

  gcxx::span<int> s = buf.first(3);
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s.data(), buf.data());
  EXPECT_EQ(s[0], 0);
  EXPECT_EQ(s[2], 2);
}

TEST(BufferAccessorsTest, FirstFullSizeIsWholeBuffer) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  gcxx::span<int> s = buf.first(8);
  EXPECT_EQ(s.size(), buf.size());
  EXPECT_EQ(s.data(), buf.data());
}

TEST(BufferAccessorsTest, FirstConstReturnsConstSpan) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);
  const mock_buffer<int>& cbuf = buf;

  gcxx::span<const int> s = cbuf.first(3);
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s[0], 0);
}

TEST(BufferAccessorsTest, LastReturnsSuffixSpan) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);

  gcxx::span<int> s = buf.last(3);
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s.data(), buf.data() + 5);
  EXPECT_EQ(s[0], 5);
  EXPECT_EQ(s[2], 7);
}

TEST(BufferAccessorsTest, LastFullSizeIsWholeBuffer) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  gcxx::span<int> s = buf.last(8);
  EXPECT_EQ(s.size(), buf.size());
  EXPECT_EQ(s.data(), buf.data());
}

// =============================================================================
// subspan(offset, count) — middle-window views.
// =============================================================================
TEST(BufferAccessorsTest, SubspanWithCount) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);

  gcxx::span<int> s = buf.subspan(2, 3);
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s.data(), buf.data() + 2);
  EXPECT_EQ(s[0], 2);
  EXPECT_EQ(s[2], 4);
}

TEST(BufferAccessorsTest, SubspanDefaultCountIsRestOfBuffer) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);

  gcxx::span<int> s = buf.subspan(3);
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.data(), buf.data() + 3);
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[4], 7);
}

TEST(BufferAccessorsTest, SubspanAtZeroReturnsWholeBuffer) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);

  gcxx::span<int> s = buf.subspan(0);
  EXPECT_EQ(s.size(), buf.size());
  EXPECT_EQ(s.data(), buf.data());
}

TEST(BufferAccessorsTest, SubspanAtSizeIsEmpty) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);

  gcxx::span<int> s = buf.subspan(8);
  EXPECT_EQ(s.size(), 0);
}

TEST(BufferAccessorsTest, SubspanConstReturnsConstSpan) {
  mock_buffer<int> buf(gcxx::StreamView::Null(), host_mock_resource{},
                       std::size_t{8}, gcxx::memory::no_init);
  fill_buffer(buf, 0);
  const mock_buffer<int>& cbuf = buf;

  gcxx::span<const int> s = cbuf.subspan(2, 3);
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s[0], 2);
}

// =============================================================================
// Typedef sanity: reference / const_reference must be exposed.
// =============================================================================
TEST(BufferAccessorsTest, ExposesReferenceTypedefs) {
  static_assert(std::is_same_v<mock_buffer<int>::reference, int&>);
  static_assert(std::is_same_v<mock_buffer<int>::const_reference, const int&>);
}
