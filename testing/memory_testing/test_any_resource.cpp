// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Phase 3 coverage for any_resource<Properties...>: type-erasure round-trip
// (dispatch to the held resource), copy (clone) independence, move (source
// empties), the `using properties` advertisement, resource_accessibility_v,
// and the construction-time property contract (a resource missing a claimed
// Property is rejected — verified via the resource_has_all_v trait, since gtest
// can't assert a static_assert failure at runtime).
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <gcxx/api.hpp>

namespace {

  // Copyable mock: counts allocate/deallocate via shared pointers (a clone
  // keeps counting into the same counters). Advertises host_accessible.
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
  gcxx::memory::any_resource<gcxx::memory::host_accessible> ar(
    counting_mock_resource{&allocs, &frees});

  EXPECT_TRUE(static_cast<bool>(ar));
  void* p = ar.allocate(std::size_t{32}, gcxx::StreamView::Null());
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(allocs, 1);

  ar.deallocate(p, gcxx::StreamView::Null());
  EXPECT_EQ(frees, 1);
}

// =============================================================================
// Copy: a clone is an independent holder but dispatches to its own copy of the
// resource (the shared counters still advance).
// =============================================================================
TEST(AnyResourceTest, CopyIsIndependent) {
  int allocs = 0;
  int frees  = 0;
  gcxx::memory::any_resource<gcxx::memory::host_accessible> ar(
    counting_mock_resource{&allocs, &frees});

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
  gcxx::memory::any_resource<gcxx::memory::host_accessible> ar(
    counting_mock_resource{&allocs, nullptr});
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
  gcxx::memory::any_resource<gcxx::memory::host_accessible> ar;
  EXPECT_FALSE(static_cast<bool>(ar));
}

// =============================================================================
// Properties: any_resource advertises its Properties via `using properties`,
// and resource_accessibility_v reflects them.
// =============================================================================
TEST(AnyResourceTest, AdvertisesProperties) {
  using gcxx::memory::any_resource;
  using gcxx::memory::device_accessible;
  using gcxx::memory::host_accessible;
  using gcxx::memory::memory_accessibility;
  using gcxx::memory::resource_accessibility_v;

  static_assert(gcxx::memory::has_property_v<any_resource<device_accessible>,
                                             device_accessible>);
  static_assert(!gcxx::memory::has_property_v<any_resource<device_accessible>,
                                              host_accessible>);

  static_assert(resource_accessibility_v<any_resource<device_accessible>> ==
                memory_accessibility::device);
  static_assert(resource_accessibility_v<any_resource<host_accessible>> ==
                memory_accessibility::host);
  static_assert(resource_accessibility_v<
                  any_resource<host_accessible, device_accessible>> ==
                memory_accessibility::host_device);
}

// =============================================================================
// Construction contract: a resource must advertise ⊇ Properties. The
// any_resource ctor static_asserts resource_has_all_v; here we confirm the
// trait is false for a mismatched resource (so the ctor would be rejected).
// =============================================================================
TEST(AnyResourceTest, RejectsMismatchedResourceTrait) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::resource_has_all_v;
  // counting_mock_resource advertises host_accessible only:
  static_assert(
    resource_has_all_v<counting_mock_resource, gcxx::memory::host_accessible>);
  static_assert(!resource_has_all_v<counting_mock_resource, device_accessible>);
}
