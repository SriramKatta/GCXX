// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_PROPERTIES_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_PROPERTIES_HPP_

#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// has_property<R, P>: std::true_type iff R exposes a `using properties =
// TypeSet<...>` member containing P. This is the Option-B mechanism —
// properties are carried as a template parameter (exposed via the `properties`
// member), NOT via ADL get_property. A type with no `properties` member yields
// false.
//
// Lives in gcxx::details_ (kept here for consistency with the rest of the
// type-detection traits; no ADL concern remains).
// ─────────────────────────────────────────────────────────────────────────────
GCXX_NAMESPACE_DETAILS_BEGIN()

template <typename R, typename P, typename = void>
struct has_property : std::false_type {};

template <typename R, typename P>
struct has_property<R, P, std::void_t<typename R::properties>>
    : std::bool_constant<R::properties::template contains<P>> {};

GCXX_NAMESPACE_DETAILS_END()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// TypeSet<Ts...>: a set of unique types with a `contains<T>` membership query.
// Replaces std::tuple as the property carrier: uniqueness is enforced at the
// point of definition (catches `TypeSet<device_accessible,
// device_accessible>`), mirroring CCCL's __make_type_set dedup. `TypeSet<>` is
// valid (empty set); `contains<T>` on the empty set is false.
// ─────────────────────────────────────────────────────────────────────────────

// all_unique<Ts...>: true iff no two types in Ts... are equal. Recursive
// (head-vs-tail) so each check compares the head against the FULL remaining
// tail. A fold of the form `count_occurrences_v<Ts, Ts...>` is WRONG here:
// both packs share the name `Ts`, so the fold expands them together and each
// per-element check sees only one element. The head/tail split keeps the
// searched pack distinct from the head.
template <typename... Ts>
struct all_unique : std::true_type {};  // empty pack is trivially unique

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

// ─────────────────────────────────────────────────────────────────────────────
// Property tags + detection traits (CCCL cuda::mr parity, adapted to GCXX).
//
// Property mechanism: every property-aware type exposes
//   `using properties = TypeSet<...>;`
// and has_property_v<R, P> reads `R::properties::contains<P>`. No ADL
// get_property. Resources carry Properties as a template parameter; the buffer
// validates (static_assert) that the resource's properties ⊇ the buffer's.
// ─────────────────────────────────────────────────────────────────────────────

/// Tag signalling that the allocated memory is reachable from the device.
struct device_accessible {};

/// Tag signalling that the allocated memory is reachable from the host.
struct host_accessible {};

// ─────────────────────────────────────────────────────────────────────────────
// memory_accessibility: the four accessibility states a property set can imply.
// Mirrors CCCL's __memory_accessibility.
// ─────────────────────────────────────────────────────────────────────────────
enum class memory_accessibility {
  unknown,
  host,
  device,
  host_device,
};

/// Reduce two static access flags to a single memory_accessibility value.
template <bool HostAccessible, bool DeviceAccessible>
constexpr memory_accessibility accessibility_from_static_properties() noexcept {
  return HostAccessible && DeviceAccessible ? memory_accessibility::host_device
         : DeviceAccessible                 ? memory_accessibility::device
         : HostAccessible                   ? memory_accessibility::host
                                            : memory_accessibility::unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_accessibility_property: a runtime-queryable property whose value is
// the resource's memory_accessibility. Carried for CCCL parity
// (property_with_value); answered from a type's static `properties` set (Phase
// 3, any_resource).
// ─────────────────────────────────────────────────────────────────────────────
struct dynamic_accessibility_property {
  using value_type = memory_accessibility;
};

// ─────────────────────────────────────────────────────────────────────────────
// properties_list<Properties...>: a type-list of properties with a `rebind`
// alias-template (appends Properties after ExtraArgs) and a static
// `has_property(P)` query. Mirrors CCCL's properties_list; used by make_buffer.
// ─────────────────────────────────────────────────────────────────────────────
template <typename... Properties>
struct properties_list {
  template <template <typename...> class Fn, typename... ExtraArgs>
  using rebind = Fn<ExtraArgs..., Properties...>;

  template <typename QueryProperty>
  static constexpr bool has_property(QueryProperty) noexcept {
    return TypeSet<Properties...>::template contains<QueryProperty>;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Resource-keyed traits (query a concrete resource/buffer TYPE's properties).
// has_property_v reads `R::properties::contains<P>` (the `properties` member is
// a TypeSet).
// ─────────────────────────────────────────────────────────────────────────────
template <typename Resource, typename Property>
inline constexpr bool has_property_v =
  details_::has_property<Resource, Property>::value;

/// True iff Resource advertises host_accessible.
template <typename Resource>
inline constexpr bool is_host_accessible_v =
  has_property_v<Resource, host_accessible>;

/// True iff Resource advertises device_accessible.
template <typename Resource>
inline constexpr bool is_device_accessible_v =
  has_property_v<Resource, device_accessible>;

/// True iff Resource advertises at least one execution-space property.
/// Use as the body of buffer's static_assert.
template <typename Resource>
inline constexpr bool contains_execution_space_property_v =
  is_host_accessible_v<Resource> || is_device_accessible_v<Resource>;

/// True iff Resource's properties ⊇ {Ps...} — the buffer ctor contract.
template <typename Resource, typename... Ps>
inline constexpr bool resource_has_all_v =
  (has_property_v<Resource, Ps> && ...);

/// The memory_accessibility implied by a type's static `properties` set. The
/// Option-B answer to CCCL's dynamic_accessibility_property query — under
/// Option-B accessibility is always known at compile time from R::properties,
/// so the "dynamic" query is a static fold.
template <typename R>
inline constexpr memory_accessibility resource_accessibility_v =
  accessibility_from_static_properties<is_host_accessible_v<R>,
                                       is_device_accessible_v<R>>();

// ─────────────────────────────────────────────────────────────────────────────
// Pack-keyed traits (over a property PACK, e.g. the buffer's own
// Properties...). Drive accessor gating and cross-properties ctor SFINAE.
// Constructing the TypeSet also enforces uniqueness of the pack.
// ─────────────────────────────────────────────────────────────────────────────
template <typename... Ps>
inline constexpr bool is_host_accessible =
  TypeSet<Ps...>::template contains<host_accessible>;

template <typename... Ps>
inline constexpr bool is_device_accessible =
  TypeSet<Ps...>::template contains<device_accessible>;

template <typename... Ps>
inline constexpr bool contains_execution_space_property =
  is_host_accessible<Ps...> || is_device_accessible<Ps...>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
