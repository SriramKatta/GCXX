// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Phase 1 coverage: property-system foundation (TypeSet, properties_list,
// accessibility traits, pack traits). Compile-time only; no GPU needed.
#include "tests_common.hpp"

#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  struct host_only_resource {
    using properties = gcxx::TypeSet<gcxx::host_accessible>;
  };
  struct host_device_resource {
    using properties =
      gcxx::TypeSet<gcxx::host_accessible, gcxx::device_accessible>;
  };

  template <typename... Ts>
  struct rebind_target {};

}  // namespace

TEST(TypeSetTest, ContainsAndSize) {
  using gcxx::TypeSet;
  using S = TypeSet<int, float, double>;
  static_assert(S::contains<int>);
  static_assert(S::contains<float>);
  static_assert(!S::contains<char>);
  static_assert(S::size == 3);
}

TEST(TypeSetTest, EmptySetIsValid) {
  using gcxx::TypeSet;
  using E = TypeSet<>;
  static_assert(E::size == 0);
  static_assert(!E::contains<int>);
}

TEST(TypeSetTest, AllUniqueTrait) {
  using gcxx::all_unique_v;
  static_assert(all_unique_v<>);
  static_assert(all_unique_v<int>);
  static_assert(all_unique_v<int, float, double>);
  static_assert(!all_unique_v<int, int>);
  static_assert(!all_unique_v<int, float, int>);
}

TEST(PropertiesListTest, RebindAppendsPropertiesAfterExtraArgs) {
  using gcxx::device_accessible;
  using gcxx::host_accessible;
  using gcxx::properties_list;
  using L = properties_list<device_accessible, host_accessible>;
  using R = L::rebind<rebind_target, int>;
  static_assert(
    std::is_same_v<R, rebind_target<int, device_accessible, host_accessible>>);
}

TEST(PropertiesListTest, HasPropertyQuery) {
  using gcxx::device_accessible;
  using gcxx::host_accessible;
  using gcxx::properties_list;
  using L = properties_list<device_accessible>;
  static_assert(L::has_property(device_accessible{}));
  static_assert(!L::has_property(host_accessible{}));
}

TEST(MemoryAccessibilityTest, ReducesStaticFlags) {
  using gcxx::accessibility_from_static_properties;
  using gcxx::memory_accessibility;
  static_assert(accessibility_from_static_properties<true, true>() ==
                memory_accessibility::host_device);
  static_assert(accessibility_from_static_properties<false, true>() ==
                memory_accessibility::device);
  static_assert(accessibility_from_static_properties<true, false>() ==
                memory_accessibility::host);
  static_assert(accessibility_from_static_properties<false, false>() ==
                memory_accessibility::unknown);
}

TEST(MemoryAccessibilityTest, DynamicPropertyValueType) {
  static_assert(std::is_same_v<gcxx::dynamic_accessibility_property::value_type,
                               gcxx::memory_accessibility>);
}

TEST(PackTraitsTest, IsHostAccessible) {
  using gcxx::device_accessible;
  using gcxx::host_accessible;
  using gcxx::is_host_accessible;
  static_assert(is_host_accessible<host_accessible>);
  static_assert(is_host_accessible<host_accessible, device_accessible>);
  static_assert(!is_host_accessible<device_accessible>);
  static_assert(!is_host_accessible<>);
}

TEST(PackTraitsTest, IsDeviceAccessible) {
  using gcxx::device_accessible;
  using gcxx::host_accessible;
  using gcxx::is_device_accessible;
  static_assert(is_device_accessible<device_accessible>);
  static_assert(is_device_accessible<host_accessible, device_accessible>);
  static_assert(!is_device_accessible<host_accessible>);
  static_assert(!is_device_accessible<>);
}

TEST(PackTraitsTest, ContainsExecutionSpaceProperty) {
  using gcxx::contains_execution_space_property;
  using gcxx::device_accessible;
  using gcxx::host_accessible;
  static_assert(contains_execution_space_property<device_accessible>);
  static_assert(contains_execution_space_property<host_accessible>);
  static_assert(
    contains_execution_space_property<host_accessible, device_accessible>);
  static_assert(!contains_execution_space_property<>);
}

// resource_has_all_v: R's properties must be a superset of {Ps...}.
TEST(ResourceHasKeyAllTest, MatchesAdvertisedProperties) {
  using gcxx::device_accessible;
  using gcxx::host_accessible;
  using gcxx::resource_has_all_v;

  static_assert(resource_has_all_v<host_only_resource, host_accessible>);
  static_assert(!resource_has_all_v<host_only_resource, device_accessible>);
  static_assert(!resource_has_all_v<host_only_resource, host_accessible,
                                    device_accessible>);

  static_assert(resource_has_all_v<host_device_resource, host_accessible>);
  static_assert(resource_has_all_v<host_device_resource, device_accessible>);
  static_assert(resource_has_all_v<host_device_resource, host_accessible,
                                   device_accessible>);
}
