// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_PROPERTIES_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_PROPERTIES_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// has_property<R, P>: std::true_type iff ADL `get_property(const R&, P)` is
// callable. Implements the standard SFINAE detection idiom (n4502).
//
// Lives in gcxx::details_ (NOT gcxx::memory::details_) — the memory namespace
// already has type-detection traits in gcxx::details_ and a nested details_
// here would shadow it for the rest of the memory namespace (see the comment
// in fill.hpp). Defining has_property FIRST, before opening namespace memory,
// lets the traits below reference details_::has_property via outer-namespace
// lookup.
// ─────────────────────────────────────────────────────────────────────────────
GCXX_NAMESPACE_DETAILS_BEGIN()

template <typename R, typename P, typename = void>
struct has_property : std::false_type {};

template <typename R, typename P>
struct has_property<R, P,
                    std::void_t<decltype(get_property(std::declval<const R&>(),
                                                      std::declval<P>()))>>
    : std::true_type {};

GCXX_NAMESPACE_DETAILS_END()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// Property tags + detection traits.
//
// Mirror of CCCL's cuda::mr::device_accessible / host_accessible and the
// get_property customization point — adapted to GCXX conventions. Lazy
// variant (per refactoring plan T3): no any_resource type-erasure, no
// properties_list/rebind machinery. Resources advertise their properties via
// ADL-found `get_property(const R&, P)` overloads; the buffer template
// queries them via has_property_v<Resource, P>.
//
// Usage on a resource type:
//   struct my_resource {
//     friend constexpr void get_property(const my_resource&,
//                                        gcxx::memory::device_accessible)
//                                        noexcept {}
//   };
// ─────────────────────────────────────────────────────────────────────────────

/// Tag signalling that the allocated memory is reachable from the device.
struct device_accessible {};

/// Tag signalling that the allocated memory is reachable from the host.
struct host_accessible {};

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

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
