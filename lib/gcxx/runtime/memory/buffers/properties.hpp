// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_PROPERTIES_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_PROPERTIES_HPP_

#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// has_property<R,P>: true iff R exposes a `properties` TypeSet containing P.
GCXX_NAMESPACE_DETAILS_BEGIN()

template <typename R, typename P, typename = void>
struct has_property : std::false_type {};

template <typename R, typename P>
struct has_property<R, P, std::void_t<typename R::properties>>
    : std::bool_constant<R::properties::template contains<P>> {};

GCXX_NAMESPACE_DETAILS_END()

template <typename... Ts>
struct all_unique : std::true_type {};  // Empty pack is trivially unique.

template <typename T, typename... Rest>
struct all_unique<T, Rest...>
    : std::bool_constant<(!std::is_same_v<T, Rest> && ...) &&
                         all_unique<Rest...>::value> {};

template <typename... Ts>
inline constexpr bool all_unique_v = all_unique<Ts...>::value;

template <typename... Ts>
struct TypeSet {
  static_assert(all_unique_v<Ts...>, "TypeSet requires unique types");

  template <typename T>
  static constexpr bool contains = (std::is_same_v<T, Ts> || ...);

  static constexpr std::size_t size = sizeof...(Ts);
};

struct device_accessible {};

struct host_accessible {};

// Four accessibility states; mirrors CCCL's __memory_accessibility.
enum class memory_accessibility {
  unknown,
  host,
  device,
  host_device,
};

template <bool HostAccessible, bool DeviceAccessible>
constexpr memory_accessibility accessibility_from_static_properties() noexcept {
  if (HostAccessible && DeviceAccessible) {
    return memory_accessibility::host_device;
  }
  if (DeviceAccessible) {
    return memory_accessibility::device;
  }
  if (HostAccessible) {
    return memory_accessibility::host;
  }
  return memory_accessibility::unknown;
}

// CCCL-parity runtime-queryable property; answered from static properties.
struct dynamic_accessibility_property {
  using value_type = memory_accessibility;
};

// Type-list of properties; mirrors CCCL's properties_list (make_buffer).
template <typename... Properties>
struct properties_list {
  template <template <typename...> class Fn, typename... ExtraArgs>
  using rebind = Fn<ExtraArgs..., Properties...>;

  template <typename QueryProperty>
  static constexpr bool has_property(QueryProperty) noexcept {
    return TypeSet<Properties...>::template contains<QueryProperty>;
  }
};

// Resource-keyed traits: read R::properties (a TypeSet).
template <typename Resource, typename Property>
inline constexpr bool has_property_v =
  details_::has_property<Resource, Property>::value;

template <typename Resource>
inline constexpr bool is_host_accessible_v =
  has_property_v<Resource, host_accessible>;

template <typename Resource>
inline constexpr bool is_device_accessible_v =
  has_property_v<Resource, device_accessible>;

// At least one execution-space property; buffer's static_assert body.
template <typename Resource>
inline constexpr bool contains_execution_space_property_v =
  is_host_accessible_v<Resource> || is_device_accessible_v<Resource>;

// True iff Resource's properties ⊇ {Ps...} — the buffer ctor contract.
template <typename Resource, typename... Ps>
inline constexpr bool resource_has_all_v =
  (has_property_v<Resource, Ps> && ...);

// Static answer to CCCL's "dynamic" accessibility query (from R::properties).
template <typename R>
inline constexpr memory_accessibility resource_accessibility_v =
  accessibility_from_static_properties<is_host_accessible_v<R>,
                                       is_device_accessible_v<R>>();

// Pack-keyed traits: drive accessor gating and cross-properties ctor SFINAE.
template <typename... Ps>
inline constexpr bool is_host_accessible =
  TypeSet<Ps...>::template contains<host_accessible>;

template <typename... Ps>
inline constexpr bool is_device_accessible =
  TypeSet<Ps...>::template contains<device_accessible>;

template <typename... Ps>
inline constexpr bool contains_execution_space_property =
  is_host_accessible<Ps...> || is_device_accessible<Ps...>;


GCXX_NAMESPACE_MAIN_END()

#endif
