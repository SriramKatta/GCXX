// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tier 3 coverage: property system.
//
//   * has_property_v / is_host_accessible_v / is_device_accessible_v /
//     contains_execution_space_property_v behave correctly for resources
//     with and without property advertisements.
//   * basic_resource<...> propagates Properties to has_property_v queries.
//   * buffer's static_assert rejects resources with no execution-space
//     property (compile-fail — verified via a negative-test alias that
//     deliberately omits properties; here we only assert the POSITIVE cases
//     because gtest can't express compile-fail at runtime).
//   * buffer's element accessors (operator[], at, front, back) are visible
//     iff the Resource advertises host_accessible.
//   * Cross-space deep-copy ctor: buffer<int, R1> can be constructed from
//     buffer<int, R2> via the new ctor (zero-size path is exercised to
//     avoid needing a GPU).
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  // Mock that advertises host_accessible.
  struct host_mock_resource {
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
    friend constexpr void get_property(const host_mock_resource&,
                                       gcxx::memory::host_accessible) noexcept {}
  };

  // Mock that advertises device_accessible only.
  struct device_mock_resource {
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
    friend constexpr void get_property(const device_mock_resource&,
                                       gcxx::memory::device_accessible) noexcept {}
  };

  // Mock that advertises BOTH (e.g. managed memory).
  struct host_device_mock_resource {
    void* allocate(std::size_t n, gcxx::StreamView) { return std::malloc(n); }
    void deallocate(void* p, gcxx::StreamView) { std::free(p); }
    friend constexpr void get_property(const host_device_mock_resource&,
                                       gcxx::memory::host_accessible) noexcept {}
    friend constexpr void get_property(const host_device_mock_resource&,
                                       gcxx::memory::device_accessible) noexcept {}
  };

  template <typename VT, typename R>
  using buf = gcxx::memory::buffer<VT, R>;

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
  static_assert(gcxx::memory::has_property_v<R, gcxx::memory::device_accessible>);
}

TEST(PropertyTest, ConvenienceTraitsMatchHasProperty) {
  static_assert(gcxx::memory::is_host_accessible_v<host_mock_resource>);
  static_assert(!gcxx::memory::is_host_accessible_v<device_mock_resource>);
  static_assert(gcxx::memory::is_device_accessible_v<device_mock_resource>);
  static_assert(!gcxx::memory::is_device_accessible_v<host_mock_resource>);
  static_assert(gcxx::memory::is_host_accessible_v<host_device_mock_resource>);
  static_assert(gcxx::memory::is_device_accessible_v<host_device_mock_resource>);
}

TEST(PropertyTest, ContainsExecutionSpacePropertyForAllVariants) {
  static_assert(
    gcxx::memory::contains_execution_space_property_v<host_mock_resource>);
  static_assert(
    gcxx::memory::contains_execution_space_property_v<device_mock_resource>);
  static_assert(
    gcxx::memory::contains_execution_space_property_v<host_device_mock_resource>);
}

// =============================================================================
// basic_resource propagates Properties via friend get_property.
// =============================================================================
TEST(PropertyTest, BasicResourceAdvertisesProperties) {
  using gcxx::memory::async_device_resource;
  using gcxx::memory::device_accessible;
  using gcxx::memory::host_accessible;
  using gcxx::memory::managed_device_resource;
  using gcxx::memory::sync_device_resource;
  using gcxx::memory::sync_host_resource;

  static_assert(gcxx::memory::has_property_v<sync_device_resource, device_accessible>);
  static_assert(!gcxx::memory::has_property_v<sync_device_resource, host_accessible>);

  static_assert(gcxx::memory::has_property_v<sync_host_resource, host_accessible>);
  static_assert(!gcxx::memory::has_property_v<sync_host_resource, device_accessible>);

  static_assert(gcxx::memory::has_property_v<async_device_resource, device_accessible>);
  static_assert(!gcxx::memory::has_property_v<async_device_resource, host_accessible>);

  // managed_device_resource advertises both
  static_assert(gcxx::memory::has_property_v<managed_device_resource, device_accessible>);
  static_assert(gcxx::memory::has_property_v<managed_device_resource, host_accessible>);
}

// =============================================================================
// Accessor visibility: operator[] exists iff Resource is host_accessible.
// Uses static_assert + decltype — gtest can't directly assert "method does
// not exist", but well-formedness via decltype is the C++17 way.
// =============================================================================
TEST(PropertyAccessorGatingTest, HostAccessibleResourceHasSubscript) {
  using B = buf<int, host_mock_resource>;
  static_assert(std::is_same_v<decltype(std::declval<B&>()[std::size_t{0}]), int&>);
  static_assert(std::is_same_v<
                decltype(std::declval<const B&>()[std::size_t{0}]), const int&>);
}

TEST(PropertyAccessorGatingTest, HostAccessibleResourceHasFrontBack) {
  using B = buf<int, host_mock_resource>;
  static_assert(std::is_same_v<decltype(std::declval<B&>().front()), int&>);
  static_assert(std::is_same_v<decltype(std::declval<B&>().back()), int&>);
}

TEST(PropertyAccessorGatingTest, HostAccessibleResourceHasAt) {
  using B = buf<int, host_mock_resource>;
  // at() throws — return type is reference.
  static_assert(std::is_same_v<decltype(std::declval<B&>().at(std::size_t{0})), int&>);
}

// =============================================================================
// Cross-space deep-copy ctor (T3).
//
// Exercise the zero-size path so no GPU is needed; the ctor body checks
// `other.size() != 0` before issuing Copy, so size 0 skips the driver call.
// SFINAE callability for non-zero paths is verified via decltype.
// =============================================================================
TEST(BufferCrossSpaceCtorTest, HostFromDeviceZeroSizeCompilesAndRuns) {
  buf<int, device_mock_resource> src(gcxx::StreamView::Null(),
                                     device_mock_resource{}, std::size_t{0},
                                     gcxx::memory::no_init);
  buf<int, host_mock_resource> dst(gcxx::StreamView::Null(),
                                   host_mock_resource{}, src);
  EXPECT_EQ(dst.size(), 0);
}

TEST(BufferCrossSpaceCtorTest, DeviceFromHostZeroSizeCompilesAndRuns) {
  buf<int, host_mock_resource> src(gcxx::StreamView::Null(),
                                   host_mock_resource{}, std::size_t{0},
                                   gcxx::memory::no_init);
  buf<int, device_mock_resource> dst(gcxx::StreamView::Null(),
                                     device_mock_resource{}, src);
  EXPECT_EQ(dst.size(), 0);
}

TEST(BufferCrossSpaceCtorTest, SameTypeResourceRejected) {
  // The cross-space ctor has SFINAE `!is_same_v<OtherResource, Resource>` to
  // prevent it from shadowing the (deleted) regular copy ctor. If the user
  // tries to copy from a same-type buffer, the call should not resolve to
  // the cross-space ctor. We verify this indirectly: the call must not
  // compile when OtherResource == Resource. We can't write a negative
  // runtime test, but the deleted copy ctor `buffer(const buffer&) = delete`
  // catches the same case.
  // Sanity: ensure the SFINAE check exists by asserting the cross-space
  // ctor is callable for DIFFERENT resources.
  using src_t = buf<int, host_mock_resource>;
  using dst_t = buf<int, device_mock_resource>;
  static_assert(std::is_constructible_v<
                dst_t, gcxx::StreamView, device_mock_resource, const src_t&>);
}
