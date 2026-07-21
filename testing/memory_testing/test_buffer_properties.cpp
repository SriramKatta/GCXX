// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Phase 4 coverage: buffer<VT, Properties...> property system.
//
//   * has_property_v / is_*_v / contains_execution_space_property_v behave
//     correctly for resources and buffers advertising properties.
//   * synchronous_resource<...> exposes Properties via `using properties`.
//   * buffer<VT, Properties...> carries its OWN Properties (not derived from a
//     resource): has_property_v<buffer<VT, P...>, Q> reads the buffer's pack.
//   * device_buffer<T> / host_buffer<T> are property-based aliases; the SAME
//     type regardless of which allocator backed construction (decoupling win).
//   * The ctor rejects a resource missing a claimed Property (verified via the
//     resource_has_all_v trait — gtest can't assert a static_assert at
//     runtime).
//   * Cross-properties copy ctor (CCCL __properties_match: narrowing copy from
//     a more-accessible buffer); same-type copy ctor (deep copy).
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  // Mock that advertises host_accessible.
  struct host_mock_resource {
    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible>;
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
  };

  // Mock that advertises device_accessible only.
  struct device_mock_resource {
    using properties = gcxx::memory::TypeSet<gcxx::memory::device_accessible>;
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
  };

  // Mock that advertises BOTH (e.g. managed memory).
  struct host_device_mock_resource {
    using properties = gcxx::memory::TypeSet<gcxx::memory::host_accessible,
                                             gcxx::memory::device_accessible>;
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
  };

  template <typename VT, typename... Ps>
  using buf = gcxx::memory::buffer<VT, Ps...>;

}  // namespace

// =============================================================================
// Property detection on resources.
// =============================================================================
TEST(PropertyTest, HasPropertyDetectsAdvertisedProperty) {
  static_assert(gcxx::memory::has_property_v<host_mock_resource,
                                             gcxx::memory::host_accessible>);
  static_assert(!gcxx::memory::has_property_v<host_mock_resource,
                                              gcxx::memory::device_accessible>);
  static_assert(gcxx::memory::has_property_v<device_mock_resource,
                                             gcxx::memory::device_accessible>);
  static_assert(!gcxx::memory::has_property_v<device_mock_resource,
                                              gcxx::memory::host_accessible>);
}

TEST(PropertyTest, HasPropertyHandlesBothAdvertisements) {
  using R = host_device_mock_resource;
  static_assert(gcxx::memory::has_property_v<R, gcxx::memory::host_accessible>);
  static_assert(
    gcxx::memory::has_property_v<R, gcxx::memory::device_accessible>);
}

TEST(PropertyTest, ConvenienceTraitsMatchHasProperty) {
  static_assert(gcxx::memory::is_host_accessible_v<host_mock_resource>);
  static_assert(!gcxx::memory::is_host_accessible_v<device_mock_resource>);
  static_assert(gcxx::memory::is_device_accessible_v<device_mock_resource>);
  static_assert(!gcxx::memory::is_device_accessible_v<host_mock_resource>);
  static_assert(gcxx::memory::is_host_accessible_v<host_device_mock_resource>);
  static_assert(
    gcxx::memory::is_device_accessible_v<host_device_mock_resource>);
}

TEST(PropertyTest, ContainsExecutionSpacePropertyForAllVariants) {
  static_assert(
    gcxx::memory::contains_execution_space_property_v<host_mock_resource>);
  static_assert(
    gcxx::memory::contains_execution_space_property_v<device_mock_resource>);
  static_assert(gcxx::memory::contains_execution_space_property_v<
                host_device_mock_resource>);
}

// =============================================================================
// synchronous_resource exposes Properties via `using properties` (read by
// has_property_v). pooled_device_resource is hand-written but advertises the
// same way.
// =============================================================================
TEST(PropertyTest, ResourcesAdvertiseProperties) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::host_accessible;
  using gcxx::memory::managed_device_resource;
  using gcxx::memory::pooled_device_resource;
  using gcxx::memory::sync_device_resource;
  using gcxx::memory::sync_host_resource;

  static_assert(
    gcxx::memory::has_property_v<sync_device_resource, device_accessible>);
  static_assert(
    !gcxx::memory::has_property_v<sync_device_resource, host_accessible>);

  static_assert(
    gcxx::memory::has_property_v<sync_host_resource, host_accessible>);
  static_assert(
    !gcxx::memory::has_property_v<sync_host_resource, device_accessible>);

  // managed_device_resource advertises both
  static_assert(
    gcxx::memory::has_property_v<managed_device_resource, device_accessible>);
  static_assert(
    gcxx::memory::has_property_v<managed_device_resource, host_accessible>);

  // pooled_device_resource is hand-written (not a synchronous_resource alias)
  // but advertises device_accessible the same way.
  static_assert(
    gcxx::memory::has_property_v<pooled_device_resource, device_accessible>);
  static_assert(
    !gcxx::memory::has_property_v<pooled_device_resource, host_accessible>);
  static_assert(
    gcxx::memory::contains_execution_space_property_v<pooled_device_resource>);
}

// =============================================================================
// buffer<VT, Properties...> carries its OWN Properties (exposed via `using
// properties`), independent of which resource backed construction.
// =============================================================================
TEST(PropertyTest, BufferHasItsOwnProperties) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::host_accessible;

  static_assert(
    gcxx::memory::has_property_v<buf<int, host_accessible>, host_accessible>);
  static_assert(!gcxx::memory::has_property_v<buf<int, host_accessible>,
                                              device_accessible>);

  static_assert(gcxx::memory::has_property_v<buf<int, device_accessible>,
                                             device_accessible>);
  static_assert(!gcxx::memory::has_property_v<buf<int, device_accessible>,
                                              host_accessible>);

  // both (managed-style)
  static_assert(
    gcxx::memory::has_property_v<buf<int, host_accessible, device_accessible>,
                                 host_accessible>);
  static_assert(
    gcxx::memory::has_property_v<buf<int, host_accessible, device_accessible>,
                                 device_accessible>);
}

// =============================================================================
// Decoupling win: device_buffer<T> / host_buffer<T> are ONE type regardless of
// the allocator. A buffer built from sync_device_resource and one from
// pooled_device_resource are the SAME type (buffer<int, device_accessible>).
// =============================================================================
TEST(PropertyTest, AliasesArePropertyBased) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::device_buffer;
  using gcxx::memory::host_accessible;
  using gcxx::memory::host_buffer;

  static_assert(std::is_same_v<device_buffer<int>,
                               gcxx::memory::buffer<int, device_accessible>>);
  static_assert(std::is_same_v<host_buffer<int>,
                               gcxx::memory::buffer<int, host_accessible>>);

  // Same type across allocators (zero-size so no pool/driver call):
  using d_from_sync = decltype(gcxx::memory::buffer<int, device_accessible>(
    gcxx::StreamView::Null(), gcxx::memory::sync_device_resource{},
    std::size_t{0}, gcxx::memory::no_init));
  using d_from_pool = decltype(gcxx::memory::buffer<int, device_accessible>(
    gcxx::StreamView::Null(), gcxx::memory::pooled_device_resource{},
    std::size_t{0}, gcxx::memory::no_init));
  static_assert(std::is_same_v<d_from_sync, d_from_pool>);
}

// =============================================================================
// Ctor contract: a resource must advertise ⊇ the buffer's Properties. A
// host-only resource cannot back a device_accessible buffer — the buffer ctor
// static_asserts resource_has_all_v. Here we confirm the trait is false for the
// mismatch (so the ctor would be rejected).
// =============================================================================
TEST(PropertyTest, RejectsMismatchedResourceTrait) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::resource_has_all_v;
  using gcxx::memory::sync_host_resource;

  static_assert(!resource_has_all_v<sync_host_resource, device_accessible>);
  static_assert(
    resource_has_all_v<sync_host_resource, gcxx::memory::host_accessible>);
}

// =============================================================================
// Cross-properties copy ctor (CCCL __properties_match — narrowing copy from a
// more-accessible buffer). Zero-size path so no GPU is needed. Host↔device is
// NOT supported (neither is a superset); only managed→host / managed→device.
// =============================================================================
TEST(BufferCrossPropertiesCtorTest, NarrowsManagedToHostZeroSize) {
  buf<int, gcxx::memory::host_accessible, gcxx::memory::device_accessible> src(
    gcxx::StreamView::Null(), host_device_mock_resource{}, std::size_t{0},
    gcxx::memory::no_init);
  buf<int, gcxx::memory::host_accessible> dst(src);  // narrowing copy
  EXPECT_EQ(dst.size(), 0);
}

TEST(BufferCrossPropertiesCtorTest, NarrowsManagedToDeviceZeroSize) {
  buf<int, gcxx::memory::host_accessible, gcxx::memory::device_accessible> src(
    gcxx::StreamView::Null(), host_device_mock_resource{}, std::size_t{0},
    gcxx::memory::no_init);
  buf<int, gcxx::memory::device_accessible> dst(
    std::move(src));  // narrowing move
  EXPECT_EQ(dst.size(), 0);
}

// =============================================================================
// Same-type copy ctor (deep copy). buffer is copyable.
// =============================================================================
TEST(BufferCopyCtorTest, SameTypeDeepCopyZeroSize) {
  buf<int, gcxx::memory::host_accessible> src(
    gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{0},
    gcxx::memory::no_init);
  buf<int, gcxx::memory::host_accessible> dst(src);  // copy ctor
  EXPECT_EQ(dst.size(), 0);
}
