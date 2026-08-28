// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_HOST_DEVICE_ACCESSOR_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_HOST_DEVICE_ACCESSOR_HPP_

#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// CCCL-port accessor tags marking memory space; cross-space ctors deleted.

template <class Accessor>
class host_accessor;

template <class Accessor>
class device_accessor;

template <class Accessor>
class managed_accessor;

// Accessor identity traits.

template <class>
GCXX_CXPR inline bool is_host_accessor_v = false;

template <class>
GCXX_CXPR inline bool is_device_accessor_v = false;

template <class>
GCXX_CXPR inline bool is_managed_accessor_v = false;

template <class Accessor>
GCXX_CXPR inline bool is_host_accessor_v<host_accessor<Accessor>> = true;

template <class Accessor>
GCXX_CXPR inline bool is_device_accessor_v<device_accessor<Accessor>> = true;

template <class Accessor>
GCXX_CXPR inline bool is_managed_accessor_v<managed_accessor<Accessor>> = true;

template <class Accessor>
GCXX_CXPR inline bool is_host_device_managed_accessor_v =
  is_host_accessor_v<Accessor> || is_device_accessor_v<Accessor> ||
  is_managed_accessor_v<Accessor>;

// Host accessor: memory reachable from host code only.

template <class Accessor>
class host_accessor : public Accessor {
  static_assert(!is_host_device_managed_accessor_v<Accessor>,
                "host_accessor/device_accessor/managed_accessor cannot be "
                "nested");

 public:
  using offset_policy    = host_accessor<typename Accessor::offset_policy>;
  using data_handle_type = typename Accessor::data_handle_type;
  using reference        = typename Accessor::reference;
  using element_type     = typename Accessor::element_type;

  GCXX_TEMPLATE(class Base = Accessor)
  GCXX_REQUIRES(std::is_default_constructible_v<Base>)
  constexpr host_accessor() noexcept(
    std::is_nothrow_default_constructible_v<Base>)
      : Accessor{} {}

  constexpr host_accessor(const Accessor& acc) noexcept(
    std::is_nothrow_copy_constructible_v<Accessor>)
      : Accessor{acc} {}

  // Same-space conversion; implicit iff the base conversion is implicit.
  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                  std::is_convertible_v<const OtherAccessor&, Accessor>)
  constexpr host_accessor(const host_accessor<OtherAccessor>& acc) noexcept(
    std::is_nothrow_constructible_v<Accessor, const OtherAccessor&>)
      : Accessor{acc} {}

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(
    std::is_constructible_v<Accessor, const OtherAccessor&>
      GCXX_AND !std::is_convertible_v<const OtherAccessor&, Accessor>)
  explicit constexpr host_accessor(
    const host_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  // Cross-space conversions.
  template <class OtherAccessor>
  host_accessor(const device_accessor<OtherAccessor>&) = delete;

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                  std::is_convertible_v<const OtherAccessor&, Accessor>)
  constexpr host_accessor(const managed_accessor<OtherAccessor>& acc) noexcept(
    std::is_nothrow_constructible_v<Accessor, const OtherAccessor&>)
      : Accessor{acc} {}

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(
    std::is_constructible_v<Accessor, const OtherAccessor&>
      GCXX_AND !std::is_convertible_v<const OtherAccessor&, Accessor>)
  explicit constexpr host_accessor(
    const managed_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  constexpr reference access(data_handle_type p, std::size_t i) const {
    return Accessor::access(p, i);
  }

  constexpr data_handle_type offset(data_handle_type p, std::size_t i) const {
    return Accessor::offset(p, i);
  }
};

// Device accessor: dereferenceable from device code only (cu/hipBLAS).

template <class Accessor>
class device_accessor : public Accessor {
  static_assert(!is_host_device_managed_accessor_v<Accessor>,
                "host_accessor/device_accessor/managed_accessor cannot be "
                "nested");

 public:
  using offset_policy    = device_accessor<typename Accessor::offset_policy>;
  using data_handle_type = typename Accessor::data_handle_type;
  using reference        = typename Accessor::reference;
  using element_type     = typename Accessor::element_type;

  GCXX_TEMPLATE(class Base = Accessor)
  GCXX_REQUIRES(std::is_default_constructible_v<Base>)
  constexpr device_accessor() noexcept(
    std::is_nothrow_default_constructible_v<Base>)
      : Accessor{} {}

  constexpr device_accessor(const Accessor& acc) noexcept(
    std::is_nothrow_copy_constructible_v<Accessor>)
      : Accessor{acc} {}

  // Same-space conversion; implicit iff the base conversion is implicit.
  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                  std::is_convertible_v<const OtherAccessor&, Accessor>)
  constexpr device_accessor(const device_accessor<OtherAccessor>& acc) noexcept(
    std::is_nothrow_constructible_v<Accessor, const OtherAccessor&>)
      : Accessor{acc} {}

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(
    std::is_constructible_v<Accessor, const OtherAccessor&>
      GCXX_AND !std::is_convertible_v<const OtherAccessor&, Accessor>)
  explicit constexpr device_accessor(
    const device_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  // Cross-space conversions.
  template <class OtherAccessor>
  device_accessor(const host_accessor<OtherAccessor>&) = delete;

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                  std::is_convertible_v<const OtherAccessor&, Accessor>)
  constexpr device_accessor(
    const managed_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(
    std::is_constructible_v<Accessor, const OtherAccessor&>
      GCXX_AND !std::is_convertible_v<const OtherAccessor&, Accessor>)
  explicit constexpr device_accessor(
    const managed_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  constexpr reference access(data_handle_type p, std::size_t i) const {
    return Accessor::access(p, i);
  }

  constexpr data_handle_type offset(data_handle_type p, std::size_t i) const {
    return Accessor::offset(p, i);
  }
};

// Managed accessor: reachable from both host and device.

template <class Accessor>
class managed_accessor : public Accessor {
  static_assert(!is_host_device_managed_accessor_v<Accessor>,
                "host_accessor/device_accessor/managed_accessor cannot be "
                "nested");

 public:
  using offset_policy    = managed_accessor<typename Accessor::offset_policy>;
  using data_handle_type = typename Accessor::data_handle_type;
  using reference        = typename Accessor::reference;
  using element_type     = typename Accessor::element_type;

  GCXX_TEMPLATE(class Base = Accessor)
  GCXX_REQUIRES(std::is_default_constructible_v<Base>)
  constexpr managed_accessor() noexcept(
    std::is_nothrow_default_constructible_v<Base>)
      : Accessor{} {}

  constexpr managed_accessor(const Accessor& acc) noexcept(
    std::is_nothrow_copy_constructible_v<Accessor>)
      : Accessor{acc} {}

  // Same-space conversion; implicit iff the base conversion is implicit.
  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                  std::is_convertible_v<const OtherAccessor&, Accessor>)
  constexpr managed_accessor(
    const managed_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(
    std::is_constructible_v<Accessor, const OtherAccessor&>
      GCXX_AND !std::is_convertible_v<const OtherAccessor&, Accessor>)
  explicit constexpr managed_accessor(
    const managed_accessor<OtherAccessor>&
      acc) noexcept(std::is_nothrow_constructible_v<Accessor,
                                                    const OtherAccessor&>)
      : Accessor{acc} {}

  // Cross-space conversions.
  template <class OtherAccessor>
  managed_accessor(const host_accessor<OtherAccessor>&) = delete;

  template <class OtherAccessor>
  managed_accessor(const device_accessor<OtherAccessor>&) = delete;

  constexpr reference access(data_handle_type p, std::size_t i) const {
    return Accessor::access(p, i);
  }

  constexpr data_handle_type offset(data_handle_type p, std::size_t i) const {
    return Accessor::offset(p, i);
  }
};

// Traits named *_view_v to avoid clashing with gcxx buffer-property traits.

// A view is device-accessible iff it carries the device or managed accessor.
template <class>
GCXX_CXPR inline bool is_device_view_v = false;

template <class>
GCXX_CXPR inline bool is_host_view_v = false;

template <class Accessor>
GCXX_CXPR inline bool is_device_view_v<device_accessor<Accessor>> = true;

template <class Accessor>
GCXX_CXPR inline bool is_device_view_v<managed_accessor<Accessor>> = true;

template <class Accessor>
GCXX_CXPR inline bool is_host_view_v<host_accessor<Accessor>> = true;

template <class Accessor>
GCXX_CXPR inline bool is_host_view_v<managed_accessor<Accessor>> = true;

// Wrapper accessors propagate the wrapped accessor's memory space.
template <template <class> class Wrapper, class Accessor>
GCXX_CXPR inline bool is_device_view_v<Wrapper<Accessor>> =
  is_device_view_v<Accessor>;

template <template <class> class Wrapper, class Accessor>
GCXX_CXPR inline bool is_host_view_v<Wrapper<Accessor>> =
  is_host_view_v<Accessor>;

GCXX_NAMESPACE_MAIN_END()

#endif
