// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_SCALED_ACCESSOR_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_SCALED_ACCESSOR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// Falls back to Reference when factor*ref is ill-formed (device scalars).
template <class ScalingFactor, class Reference, class = void>
struct scaled_reference {
  using type = Reference;
};

template <class ScalingFactor, class Reference>
struct scaled_reference<ScalingFactor, Reference,
                        std::void_t<decltype(std::declval<ScalingFactor>() *
                                             std::declval<Reference>())>> {
  using type =
    decltype(std::declval<ScalingFactor>() * std::declval<Reference>());
};

template <class ScalingFactor, class Reference, class = void>
struct is_host_multiplicable : std::false_type {};

template <class ScalingFactor, class Reference>
struct is_host_multiplicable<
  ScalingFactor, Reference,
  std::void_t<decltype(std::declval<ScalingFactor>() *
                       std::declval<Reference>())>> : std::true_type {};

template <class ScalingFactor, class Reference>
GCXX_CXPR inline bool is_host_multiplicable_v =
  is_host_multiplicable<ScalingFactor, Reference>::value;

GCXX_NAMESPACE_MAIN_DETAILS_END()

GCXX_NAMESPACE_MAIN_BEGIN()

// P1673-style scaled(alpha,x) view; blas unwraps the factor at dispatch.
template <class ScalingFactor, class NestedAccessor>
struct scaled_accessor : public NestedAccessor {
  static_assert(std::is_object_v<typename NestedAccessor::element_type>,
                "NestedAccessor::element_type must be an object type");

 public:
  using offset_policy =
    scaled_accessor<ScalingFactor, typename NestedAccessor::offset_policy>;
  using element_type = typename NestedAccessor::element_type;
  using reference    = typename details_::scaled_reference<
       ScalingFactor, typename NestedAccessor::reference>::type;
  using data_handle_type = typename NestedAccessor::data_handle_type;

  constexpr scaled_accessor() noexcept = default;

  constexpr scaled_accessor(const ScalingFactor& s,
                            const NestedAccessor& a) noexcept
      : NestedAccessor(a), scaling_factor_(s) {}

  constexpr reference access(data_handle_type p, std::size_t i) const {
    static_assert(
      details_::is_host_multiplicable_v<ScalingFactor,
                                        typename NestedAccessor::reference>,
      "element access on a scaled view whose factor is not host-multipliable "
      "(e.g. a device-resident scalar factor) is not available; consumers "
      "such as gcxx::blas resolve the factor at dispatch time instead");
    return scaling_factor_ * NestedAccessor::access(p, i);
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    std::size_t i) const noexcept {
    return NestedAccessor::offset(p, i);
  }

  constexpr const ScalingFactor& scaling_factor() const noexcept {
    return scaling_factor_;
  }

  constexpr const NestedAccessor& nested_accessor() const noexcept {
    return *this;
  }

 private:
  ScalingFactor scaling_factor_{};
};

// Identity trait, mirroring is_restrict_accessor_v.
template <class>
GCXX_CXPR inline bool is_scaled_accessor_v = false;

template <class ScalingFactor, class NestedAccessor>
GCXX_CXPR inline bool
  is_scaled_accessor_v<scaled_accessor<ScalingFactor, NestedAccessor>> = true;

// Elements read as alpha*x_i; mapping/handle unchanged, alpha in accessor.
GCXX_TEMPLATE(class ScalingFactor, class T, class Extents, class Layout,
              class Accessor)
GCXX_REQUIRES(std::is_object_v<T>)
constexpr auto scaled(ScalingFactor alpha,
                      const gcxx::mdspan<T, Extents, Layout, Accessor>& x) {
  return gcxx::mdspan(
    x.data_handle(), x.mapping(),
    scaled_accessor<ScalingFactor, Accessor>{alpha, x.accessor()});
}

// Removes one scaled layer; identity for unscaled views.
template <class T, class Extents, class Layout, class ScalingFactor,
          class Nested>
constexpr auto strip_scaled(
  const gcxx::mdspan<T, Extents, Layout,
                     scaled_accessor<ScalingFactor, Nested>>& x) {
  return gcxx::mdspan(x.data_handle(), x.mapping(),
                      x.accessor().nested_accessor());
}

GCXX_TEMPLATE(class T, class Extents, class Layout, class Accessor)
GCXX_REQUIRES(!is_scaled_accessor_v<Accessor>)
constexpr auto strip_scaled(
  const gcxx::mdspan<T, Extents, Layout, Accessor>& x) {
  return gcxx::mdspan(x.data_handle(), x.mapping(), x.accessor());
}

GCXX_NAMESPACE_MAIN_END()

// Generic trait propagation only fits 1-param wrappers, hence these.
GCXX_NAMESPACE_MAIN_BEGIN()

template <class ScalingFactor, class NestedAccessor>
GCXX_CXPR inline bool
  is_device_view_v<scaled_accessor<ScalingFactor, NestedAccessor>> =
    gcxx::is_device_view_v<NestedAccessor>;

template <class ScalingFactor, class NestedAccessor>
GCXX_CXPR inline bool
  is_host_view_v<scaled_accessor<ScalingFactor, NestedAccessor>> =
    gcxx::is_host_view_v<NestedAccessor>;

GCXX_NAMESPACE_MAIN_END()

#endif
