// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Phase 4 coverage: buffer<VT, Properties...> property system.
//
//   * has_property_v / is_*_v / contains_execution_space_property_v behave
//     correctly for resources and buffers advertising properties.
//   * pool views (DeviceMemPoolView/PinnedMemPoolView/ManagedMemPoolView)
//     expose Properties via `using properties`.
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
    using properties = gcxx::TypeSet<gcxx::host_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  // Mock that advertises device_accessible only.
  struct device_mock_resource {
    using properties = gcxx::TypeSet<gcxx::device_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  // Mock that advertises BOTH (e.g. managed memory).
  struct host_device_mock_resource {
    using properties =
      gcxx::TypeSet<gcxx::host_accessible, gcxx::device_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  template <typename VT, typename... Ps>
  using buf = gcxx::buffer<VT, Ps...>;

}  // namespace

// =============================================================================
// Property detection on resources.
// =============================================================================
TEST(PropertyTest, HasPropertyDetectsAdvertisedProperty) {
  static_assert(
    gcxx::has_property_v<host_mock_resource, gcxx::host_accessible>);
  static_assert(
    !gcxx::has_property_v<host_mock_resource, gcxx::device_accessible>);
  static_assert(
    gcxx::has_property_v<device_mock_resource, gcxx::device_accessible>);
  static_assert(
    !gcxx::has_property_v<device_mock_resource, gcxx::host_accessible>);
}

TEST(PropertyTest, HasPropertyHandlesBothAdvertisements) {
  using R = host_device_mock_resource;
  static_assert(gcxx::has_property_v<R, gcxx::host_accessible>);
  static_assert(gcxx::has_property_v<R, gcxx::device_accessible>);
}

TEST(PropertyTest, ConvenienceTraitsMatchHasProperty) {
  static_assert(gcxx::is_host_accessible_v<host_mock_resource>);
  static_assert(!gcxx::is_host_accessible_v<device_mock_resource>);
  static_assert(gcxx::is_device_accessible_v<device_mock_resource>);
  static_assert(!gcxx::is_device_accessible_v<host_mock_resource>);
  static_assert(gcxx::is_host_accessible_v<host_device_mock_resource>);
  static_assert(gcxx::is_device_accessible_v<host_device_mock_resource>);
}

TEST(PropertyTest, ContainsExecutionSpacePropertyForAllVariants) {
  static_assert(gcxx::contains_execution_space_property_v<host_mock_resource>);
  static_assert(
    gcxx::contains_execution_space_property_v<device_mock_resource>);
  static_assert(
    gcxx::contains_execution_space_property_v<host_device_mock_resource>);
}

// =============================================================================
// Pool views expose Properties via `using properties` (read by has_property_v):
// device_default_memory_pool() -> DeviceMemPoolView,
// pinned_default_memory_pool()
// -> PinnedMemPoolView, managed_default_memory_pool() -> ManagedMemPoolView.
// =============================================================================
TEST(PropertyTest, ResourcesAdvertiseProperties) {
  using gcxx::device_accessible;
  using gcxx::DeviceMemPoolView;
  using gcxx::host_accessible;
  using gcxx::PinnedMemPoolView;

  // device_default_memory_pool() returns DeviceMemPoolView — device only.
  static_assert(gcxx::has_property_v<DeviceMemPoolView, device_accessible>);
  static_assert(!gcxx::has_property_v<DeviceMemPoolView, host_accessible>);
  static_assert(gcxx::contains_execution_space_property_v<DeviceMemPoolView>);

  // pinned_default_memory_pool() returns PinnedMemPoolView. Pinned memory is
  // reachable from both host and device, so it advertises BOTH.
  static_assert(gcxx::has_property_v<PinnedMemPoolView, host_accessible>);
  static_assert(gcxx::has_property_v<PinnedMemPoolView, device_accessible>);

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  using gcxx::ManagedMemPoolView;
  // managed_default_memory_pool() returns ManagedMemPoolView (host and device).
  static_assert(gcxx::has_property_v<ManagedMemPoolView, device_accessible>);
  static_assert(gcxx::has_property_v<ManagedMemPoolView, host_accessible>);
#endif
}

// =============================================================================
// buffer<VT, Properties...> carries its OWN Properties (exposed via `using
// properties`), independent of which resource backed construction.
// =============================================================================
TEST(PropertyTest, BufferHasItsOwnProperties) {
  using gcxx::device_accessible;
  using gcxx::host_accessible;

  static_assert(
    gcxx::has_property_v<buf<int, host_accessible>, host_accessible>);
  static_assert(
    !gcxx::has_property_v<buf<int, host_accessible>, device_accessible>);

  static_assert(
    gcxx::has_property_v<buf<int, device_accessible>, device_accessible>);
  static_assert(
    !gcxx::has_property_v<buf<int, device_accessible>, host_accessible>);

  // both (managed-style)
  static_assert(
    gcxx::has_property_v<buf<int, host_accessible, device_accessible>,
                         host_accessible>);
  static_assert(
    gcxx::has_property_v<buf<int, host_accessible, device_accessible>,
                         device_accessible>);
}

// =============================================================================
// Decoupling win: device_buffer<T> / host_buffer<T> are ONE type regardless of
// the allocator. A buffer built from the default device pool and one from a
// hand-constructed DeviceMemPoolView are the SAME type
// (buffer<int, device_accessible>).
// =============================================================================
TEST(PropertyTest, AliasesArePropertyBased) {
  using gcxx::device_accessible;
  using gcxx::device_buffer;
  using gcxx::host_accessible;
  using gcxx::host_buffer;

  static_assert(
    std::is_same_v<device_buffer<int>, gcxx::buffer<int, device_accessible>>);
  static_assert(
    std::is_same_v<host_buffer<int>, gcxx::buffer<int, host_accessible>>);

  // Same type across allocators (zero-size; the pool call is in an unevaluated
  // decltype, so no driver call occurs):
  using d_from_default_pool = decltype(gcxx::buffer<int, device_accessible>(
    gcxx::StreamView::Null(),
    gcxx::device_default_memory_pool(gcxx::DeviceHandle{0}), std::size_t{0},
    gcxx::no_init));
  using d_from_pool_view    = decltype(gcxx::buffer<int, device_accessible>(
    gcxx::StreamView::Null(),
    gcxx::DeviceMemPoolView{gcxx::driver::deviceMemPool_t{}}, std::size_t{0},
    gcxx::no_init));
  static_assert(std::is_same_v<d_from_default_pool, d_from_pool_view>);
}

// =============================================================================
// Ctor contract: a resource must advertise ⊇ the buffer's Properties. A
// host-only resource cannot back a device_accessible buffer — the buffer ctor
// static_asserts resource_has_all_v. Here we confirm the trait is false for the
// mismatch (so the ctor would be rejected).
// =============================================================================
TEST(PropertyTest, RejectsMismatchedResourceTrait) {
  using gcxx::device_accessible;
  using gcxx::resource_has_all_v;

  // host_mock_resource (host-only) cannot back a device_accessible buffer.
  static_assert(!resource_has_all_v<host_mock_resource, device_accessible>);
  static_assert(resource_has_all_v<host_mock_resource, gcxx::host_accessible>);
}

// =============================================================================
// Cross-properties copy ctor (CCCL __properties_match — narrowing copy from a
// more-accessible buffer). Zero-size path so no GPU is needed. Host↔device is
// NOT supported (neither is a superset); only managed→host / managed→device.
// =============================================================================
TEST(BufferCrossPropertiesCtorTest, NarrowsManagedToHostZeroSize) {
  buf<int, gcxx::host_accessible, gcxx::device_accessible> src(
    gcxx::StreamView::Null(), host_device_mock_resource{}, std::size_t{0},
    gcxx::no_init);
  buf<int, gcxx::host_accessible> dst(src);  // narrowing copy
  EXPECT_EQ(dst.size(), 0);
}

TEST(BufferCrossPropertiesCtorTest, NarrowsManagedToDeviceZeroSize) {
  buf<int, gcxx::host_accessible, gcxx::device_accessible> src(
    gcxx::StreamView::Null(), host_device_mock_resource{}, std::size_t{0},
    gcxx::no_init);
  buf<int, gcxx::device_accessible> dst(std::move(src));  // narrowing move
  EXPECT_EQ(dst.size(), 0);
}

// =============================================================================
// Same-type copy ctor (deep copy). buffer is copyable.
// =============================================================================
TEST(BufferCopyCtorTest, SameTypeDeepCopyZeroSize) {
  buf<int, gcxx::host_accessible> src(gcxx::StreamView::Null(),
                                      host_mock_resource{}, std::size_t{0},
                                      gcxx::no_init);
  buf<int, gcxx::host_accessible> dst(src);  // copy ctor
  EXPECT_EQ(dst.size(), 0);
}
