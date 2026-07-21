// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Phase 3/4 coverage for any_resource (the non-templated type-erased
// allocator): type-erasure round-trip (dispatch to the held resource), copy
// (clone) independence, move (source empties), empty state. Properties live on
// the buffer type, not any_resource; the resource_has_all_v contract is
// enforced by the buffer ctor (covered in test_buffer_properties).
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <gcxx/api.hpp>

namespace {

  // Copyable mock: counts allocate/deallocate via shared pointers (a clone
  // keeps counting into the same counters).
  struct counting_mock_resource {
    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible>;
    int* alloc_count_;
    int* free_count_;
    explicit counting_mock_resource(int* a = nullptr, int* f = nullptr)
        : alloc_count_(a), free_count_(f) {}
    void* allocate(std::size_t n, gcxx::StreamView) {
      if (alloc_count_)
        ++*alloc_count_;
      return std::malloc(n);
    }
    void deallocate(void* p, gcxx::StreamView) {
      if (free_count_)
        ++*free_count_;
      std::free(p);
    }
  };

}  // namespace

// =============================================================================
// Type-erasure round-trip: allocate/deallocate dispatch to the held resource.
// =============================================================================
TEST(AnyResourceTest, DispatchesToHeldResource) {
  int allocs = 0;
  int frees  = 0;
  gcxx::memory::any_resource ar(counting_mock_resource{&allocs, &frees});

  EXPECT_TRUE(static_cast<bool>(ar));
  void* p = ar.allocate(std::size_t{32}, gcxx::StreamView::Null());
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(allocs, 1);

  ar.deallocate(p, gcxx::StreamView::Null());
  EXPECT_EQ(frees, 1);
}

// =============================================================================
// Copy: a clone is an independent holder that dispatches to its own copy of
// the resource (the shared counters still advance).
// =============================================================================
TEST(AnyResourceTest, CopyIsIndependent) {
  int allocs = 0;
  int frees  = 0;
  gcxx::memory::any_resource ar(counting_mock_resource{&allocs, &frees});

  auto ar2 = ar;  // copy
  EXPECT_TRUE(static_cast<bool>(ar));
  EXPECT_TRUE(static_cast<bool>(ar2));

  void* p2 = ar2.allocate(std::size_t{16}, gcxx::StreamView::Null());
  EXPECT_EQ(allocs, 1);  // ar2 dispatched to its own copy of the resource
  ar2.deallocate(p2, gcxx::StreamView::Null());
  EXPECT_EQ(frees, 1);

  // Original is unaffected and still usable.
  void* p1 = ar.allocate(std::size_t{16}, gcxx::StreamView::Null());
  EXPECT_EQ(allocs, 2);
  ar.deallocate(p1, gcxx::StreamView::Null());
  EXPECT_EQ(frees, 2);
}

// =============================================================================
// Move: the source becomes empty.
// =============================================================================
TEST(AnyResourceTest, MoveEmptiesSource) {
  int allocs = 0;
  gcxx::memory::any_resource ar(counting_mock_resource{&allocs, nullptr});
  auto ar2 = std::move(ar);
  EXPECT_FALSE(static_cast<bool>(ar));  // moved-from
  EXPECT_TRUE(static_cast<bool>(ar2));
  void* p = ar2.allocate(std::size_t{8}, gcxx::StreamView::Null());
  EXPECT_EQ(allocs, 1);
  ar2.deallocate(p, gcxx::StreamView::Null());
}

// =============================================================================
// Empty default-constructed any_resource.
// =============================================================================
TEST(AnyResourceTest, DefaultIsEmpty) {
  gcxx::memory::any_resource ar;
  EXPECT_FALSE(static_cast<bool>(ar));
}
