// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_SCALED_ACCESSOR_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_SCALED_ACCESSOR_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// ScalingFactor * Reference when that expression is well-formed, else the
// nested reference unchanged. The fallback keeps
// scaled_accessor<DeviceFactor, ...> instantiable when the factor is not
// host-multipliable (e.g. a device-resident scalar, whose value cannot be
// read on the host).
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

// Whether ScalingFactor * Reference is a well-formed expression (i.e.
// scaled_reference did NOT fall back).
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

// scaled_accessor<ScalingFactor, NestedAccessor> wraps an mdspan accessor and
// scales every element it hands out, implementing P1673R13's scaled(alpha, x)
// view. Consumers that hand operands to element-wise backends (the gcxx::blas
// operations) unwrap the factor at dispatch time instead of reading elements
// through it, so no extra pass is launched.
//
// Differences from the P1673 spec, both forced by gcxx::mdspan (the vendored
// Kokkos reference implementation static-asserts that the mdspan's element
// type equals accessor::element_type):
//   - element_type stays the NESTED accessor's element type instead of the
//     alpha*element product type; the factor is only virtual.
//   - reference falls back to the nested reference when ScalingFactor *
//     nested-reference is ill-formed (e.g. a device-resident factor that
//     cannot be applied on the host). Element access through such a view is
//     static_asserted away in access().
//
// It composes with the memory-space accessors: wrapping a device view keeps
// it a device view (via the gcxx::is_device_view_v specializations below).
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

// scaled(alpha, x) returns a view of x whose elements read as alpha * x_i.
// The mapping and the underlying data handle are forwarded unchanged, so the
// view has the same problem geometry as x and consumers simply recover alpha
// from the accessor. alpha may be any host-multipliable scalar, or a
// non-host-multipliable factor type (e.g. a device-resident scalar) when the
// consumer resolves it itself — see the accessor notes above.
//
// Example:
//   gcxx::blas::matrix_vector_product(h, gcxx::scaled(2.0, A), x, y);
GCXX_TEMPLATE(class ScalingFactor, class T, class Extents, class Layout,
              class Accessor)
GCXX_REQUIRES(std::is_object_v<T>)
constexpr auto scaled(ScalingFactor alpha,
                      const gcxx::mdspan<T, Extents, Layout, Accessor>& x) {
  return gcxx::mdspan(
    x.data_handle(), x.mapping(),
    scaled_accessor<ScalingFactor, Accessor>{alpha, x.accessor()});
}

// strip_scaled(x) removes ONE scaled_accessor layer, returning the unmodified
// base view (identity for views that were never scaled). Consumers that
// forward a scaled addend's factor as a separate scalar argument use this to
// hand the base view to a second call.
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

// is_device_view_v / is_host_view_v must see through the (two-parameter)
// scaled_accessor wrapper: the generic propagation partial specialization in
// host_device_accessor.hpp only matches single-parameter wrappers, so without
// these specializations every device-view gate (e.g. the gcxx::blas operand
// checks) would reject scaled views. Declared at gcxx scope, the nearest
// enclosing namespace of the primary templates.
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
