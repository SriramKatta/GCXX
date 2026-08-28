// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Phase 4 coverage: buffer property traits, pool views, aliases, ctor
// resource contract, cross-properties narrowing copy, same-type deep copy.
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  struct host_mock_resource {
    using properties = gcxx::TypeSet<gcxx::host_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  struct device_mock_resource {
    using properties = gcxx::TypeSet<gcxx::device_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  struct host_device_mock_resource {
    using properties =
      gcxx::TypeSet<gcxx::host_accessible, gcxx::device_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  template <typename VT, typename... Ps>
  using buf = gcxx::buffer<VT, Ps...>;

}  // namespace

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

TEST(PropertyTest, ResourcesAdvertiseProperties) {
  using gcxx::device_accessible;
  using gcxx::DeviceMemPoolView;
  using gcxx::host_accessible;
  using gcxx::PinnedMemPoolView;

  // device_default_memory_pool() returns DeviceMemPoolView — device only.
  static_assert(gcxx::has_property_v<DeviceMemPoolView, device_accessible>);
  static_assert(!gcxx::has_property_v<DeviceMemPoolView, host_accessible>);
  static_assert(gcxx::contains_execution_space_property_v<DeviceMemPoolView>);

  // pinned_default_memory_pool() returns PinnedMemPoolView: host + device.
  static_assert(gcxx::has_property_v<PinnedMemPoolView, host_accessible>);
  static_assert(gcxx::has_property_v<PinnedMemPoolView, device_accessible>);

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  using gcxx::ManagedMemPoolView;
  // managed_default_memory_pool() returns ManagedMemPoolView (host and device).
  static_assert(gcxx::has_property_v<ManagedMemPoolView, device_accessible>);
  static_assert(gcxx::has_property_v<ManagedMemPoolView, host_accessible>);
#endif
}

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

// device_buffer<T> is one type regardless of the backing allocator.
TEST(PropertyTest, AliasesArePropertyBased) {
  using gcxx::device_accessible;
  using gcxx::device_buffer;
  using gcxx::host_accessible;
  using gcxx::host_buffer;

  static_assert(
    std::is_same_v<device_buffer<int>, gcxx::buffer<int, device_accessible>>);
  static_assert(
    std::is_same_v<host_buffer<int>, gcxx::buffer<int, host_accessible>>);

  // Same type across allocators (pool call sits in an unevaluated decltype).
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

// Ctor contract: resource must advertise ⊇ the buffer's Properties.
TEST(PropertyTest, RejectsMismatchedResourceTrait) {
  using gcxx::device_accessible;
  using gcxx::resource_has_all_v;

  // host_mock_resource (host-only) cannot back a device_accessible buffer.
  static_assert(!resource_has_all_v<host_mock_resource, device_accessible>);
  static_assert(resource_has_all_v<host_mock_resource, gcxx::host_accessible>);
}

// CCCL __properties_match: managed→host/device only, never host↔device.
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

TEST(BufferCopyCtorTest, SameTypeDeepCopyZeroSize) {
  buf<int, gcxx::host_accessible> src(gcxx::StreamView::Null(),
                                      host_mock_resource{}, std::size_t{0},
                                      gcxx::no_init);
  buf<int, gcxx::host_accessible> dst(src);  // copy ctor
  EXPECT_EQ(dst.size(), 0);
}
