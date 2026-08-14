// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_RESRICT_ACCESSOR_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_RESRICT_ACCESSOR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// restrict_accessor wraps any mdspan accessor and overrides data_handle_type to
// carry the backend restrict qualifier (__restrict / __restrict__), enabling
// restrict-qualified loads/stores without changing the access/offset logic.
//
// It composes with the memory-space accessors: wrapping a device view keeps it
// a device view (gcxx::is_device_view_v propagates through any
// single-parameter accessor wrapper).
template <class Accessor>
struct restrict_accessor : public Accessor {
  static_assert(std::is_object_v<typename Accessor::element_type>,
                "Accessor::element_type must be an object type");
  static_assert(std::is_pointer_v<typename Accessor::data_handle_type>,
                "Accessor::data_handle_type must be a raw pointer");

 public:
  using offset_policy    = restrict_accessor<typename Accessor::offset_policy>;
  using element_type     = typename Accessor::element_type;
  using reference        = typename Accessor::reference;
  using data_handle_type = element_type* GCXX_RESTRICT_KEYWORD();

  constexpr restrict_accessor() noexcept = default;

  GCXX_TEMPLATE(class Other)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const Other&>)
  constexpr restrict_accessor(const Other& other) noexcept : Accessor(other) {}

  constexpr reference access(data_handle_type p, std::size_t i) const noexcept {
    return Accessor::access(p, i);
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    std::size_t i) const noexcept {
    return Accessor::offset(p, i);
  }
};

// Identity trait, mirroring is_device_accessor_v / is_managed_accessor_v.
template <class>
GCXX_CXPR inline bool is_restrict_accessor_v = false;

template <class Accessor>
GCXX_CXPR inline bool is_restrict_accessor_v<restrict_accessor<Accessor>> =
  true;

GCXX_NAMESPACE_MAIN_END()


#endif
