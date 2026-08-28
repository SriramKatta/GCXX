// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Phase 3/4 coverage: any_resource dispatch, copy independence, move,
// empty state; properties live on buffers (see test_buffer_properties).
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <gcxx/api.hpp>

namespace {

  // Clone keeps counting into the SAME shared counters.
  struct counting_mock_resource {
    using properties = gcxx::TypeSet<gcxx::host_accessible>;
    int* alloc_count_;
    int* free_count_;
    explicit counting_mock_resource(int* a = nullptr, int* f = nullptr)
        : alloc_count_(a), free_count_(f) {}
    void* allocate(gcxx::StreamView, std::size_t n) {
      if (alloc_count_)
        ++*alloc_count_;
      return std::malloc(n);
    }
    void deallocate(gcxx::StreamView, void* p) {
      if (free_count_)
        ++*free_count_;
      std::free(p);
    }
  };

}  // namespace

TEST(AnyResourceTest, DispatchesToHeldResource) {
  int allocs = 0;
  int frees  = 0;
  gcxx::any_resource ar(counting_mock_resource{&allocs, &frees});

  EXPECT_TRUE(static_cast<bool>(ar));
  void* p = ar.allocate(gcxx::StreamView::Null(), std::size_t{32});
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(allocs, 1);

  ar.deallocate(gcxx::StreamView::Null(), p);
  EXPECT_EQ(frees, 1);
}

TEST(AnyResourceTest, CopyIsIndependent) {
  int allocs = 0;
  int frees  = 0;
  gcxx::any_resource ar(counting_mock_resource{&allocs, &frees});

  auto ar2 = ar;  // copy
  EXPECT_TRUE(static_cast<bool>(ar));
  EXPECT_TRUE(static_cast<bool>(ar2));

  void* p2 = ar2.allocate(gcxx::StreamView::Null(), std::size_t{16});
  EXPECT_EQ(allocs, 1);  // ar2 dispatched to its own copy of the resource
  ar2.deallocate(gcxx::StreamView::Null(), p2);
  EXPECT_EQ(frees, 1);

  // Original is unaffected and still usable.
  void* p1 = ar.allocate(gcxx::StreamView::Null(), std::size_t{16});
  EXPECT_EQ(allocs, 2);
  ar.deallocate(gcxx::StreamView::Null(), p1);
  EXPECT_EQ(frees, 2);
}

TEST(AnyResourceTest, MoveEmptiesSource) {
  int allocs = 0;
  gcxx::any_resource ar(counting_mock_resource{&allocs, nullptr});
  auto ar2 = std::move(ar);
  EXPECT_FALSE(static_cast<bool>(ar));  // moved-from
  EXPECT_TRUE(static_cast<bool>(ar2));
  void* p = ar2.allocate(gcxx::StreamView::Null(), std::size_t{8});
  EXPECT_EQ(allocs, 1);
  ar2.deallocate(gcxx::StreamView::Null(), p);
}

TEST(AnyResourceTest, DefaultIsEmpty) {
  gcxx::any_resource ar;
  EXPECT_FALSE(static_cast<bool>(ar));
}
