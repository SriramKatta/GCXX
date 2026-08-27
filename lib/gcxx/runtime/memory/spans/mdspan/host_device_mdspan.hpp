// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_HOST_DEVICE_MDSPAN_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_HOST_DEVICE_MDSPAN_HPP_

#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/host_device_accessor.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// CCCL port; (ptr, ints...)/array/span deduction guides are omitted.

// Device-memory view; usable by device libraries and in device code.

template <class ElementType, class Extents,
          class LayoutPolicy   = gcxx::layout_right,
          class AccessorPolicy = gcxx::default_accessor<ElementType>>
class device_mdspan
    : public gcxx::mdspan<ElementType, Extents, LayoutPolicy,
                          gcxx::device_accessor<AccessorPolicy>> {
 public:
  using base_t = gcxx::mdspan<ElementType, Extents, LayoutPolicy,
                              gcxx::device_accessor<AccessorPolicy>>;
  using base_t::base_t;

  friend constexpr void swap(device_mdspan& x, device_mdspan& y) noexcept {
    swap(static_cast<base_t&>(x), static_cast<base_t&>(y));
  }
};

GCXX_TEMPLATE(class Pointer)
GCXX_REQUIRES(std::is_pointer_v<std::remove_reference_t<Pointer>>)
device_mdspan(Pointer&&)
  -> device_mdspan<std::remove_pointer_t<std::remove_reference_t<Pointer>>,
                   gcxx::extents<std::size_t>>;

template <class ElementType, class OtherIndexType, std::size_t... ExtentsPack>
device_mdspan(ElementType*,
              const gcxx::extents<OtherIndexType, ExtentsPack...>&)
  -> device_mdspan<ElementType, gcxx::extents<OtherIndexType, ExtentsPack...>>;

template <class ElementType, class MappingType>
device_mdspan(ElementType*, const MappingType&)
  -> device_mdspan<ElementType, typename MappingType::extents_type,
                   typename MappingType::layout_type>;

template <class MappingType, class AccessorPolicy>
device_mdspan(typename gcxx::device_accessor<AccessorPolicy>::data_handle_type,
              const MappingType&, const gcxx::device_accessor<AccessorPolicy>&)
  -> device_mdspan<typename AccessorPolicy::element_type,
                   typename MappingType::extents_type,
                   typename MappingType::layout_type, AccessorPolicy>;

// Host-memory view; dereferenceable from host code only.

template <class ElementType, class Extents,
          class LayoutPolicy   = gcxx::layout_right,
          class AccessorPolicy = gcxx::default_accessor<ElementType>>
class host_mdspan : public gcxx::mdspan<ElementType, Extents, LayoutPolicy,
                                        gcxx::host_accessor<AccessorPolicy>> {
 public:
  using base_t = gcxx::mdspan<ElementType, Extents, LayoutPolicy,
                              gcxx::host_accessor<AccessorPolicy>>;
  using base_t::base_t;

  friend constexpr void swap(host_mdspan& x, host_mdspan& y) noexcept {
    swap(static_cast<base_t&>(x), static_cast<base_t&>(y));
  }
};

GCXX_TEMPLATE(class Pointer)
GCXX_REQUIRES(std::is_pointer_v<std::remove_reference_t<Pointer>>)
host_mdspan(Pointer&&)
  -> host_mdspan<std::remove_pointer_t<std::remove_reference_t<Pointer>>,
                 gcxx::extents<std::size_t>>;

template <class ElementType, class OtherIndexType, std::size_t... ExtentsPack>
host_mdspan(ElementType*, const gcxx::extents<OtherIndexType, ExtentsPack...>&)
  -> host_mdspan<ElementType, gcxx::extents<OtherIndexType, ExtentsPack...>>;

template <class ElementType, class MappingType>
host_mdspan(ElementType*, const MappingType&)
  -> host_mdspan<ElementType, typename MappingType::extents_type,
                 typename MappingType::layout_type>;

template <class MappingType, class AccessorPolicy>
host_mdspan(typename gcxx::host_accessor<AccessorPolicy>::data_handle_type,
            const MappingType&, const gcxx::host_accessor<AccessorPolicy>&)
  -> host_mdspan<typename AccessorPolicy::element_type,
                 typename MappingType::extents_type,
                 typename MappingType::layout_type, AccessorPolicy>;

// Managed-memory view; reachable from both host and device.

template <class ElementType, class Extents,
          class LayoutPolicy   = gcxx::layout_right,
          class AccessorPolicy = gcxx::default_accessor<ElementType>>
class managed_mdspan
    : public gcxx::mdspan<ElementType, Extents, LayoutPolicy,
                          gcxx::managed_accessor<AccessorPolicy>> {
 public:
  using base_t = gcxx::mdspan<ElementType, Extents, LayoutPolicy,
                              gcxx::managed_accessor<AccessorPolicy>>;
  using base_t::base_t;

  friend constexpr void swap(managed_mdspan& x, managed_mdspan& y) noexcept {
    swap(static_cast<base_t&>(x), static_cast<base_t&>(y));
  }
};

GCXX_TEMPLATE(class Pointer)
GCXX_REQUIRES(std::is_pointer_v<std::remove_reference_t<Pointer>>)
managed_mdspan(Pointer&&)
  -> managed_mdspan<std::remove_pointer_t<std::remove_reference_t<Pointer>>,
                    gcxx::extents<std::size_t>>;

template <class ElementType, class OtherIndexType, std::size_t... ExtentsPack>
managed_mdspan(ElementType*,
               const gcxx::extents<OtherIndexType, ExtentsPack...>&)
  -> managed_mdspan<ElementType, gcxx::extents<OtherIndexType, ExtentsPack...>>;

template <class ElementType, class MappingType>
managed_mdspan(ElementType*, const MappingType&)
  -> managed_mdspan<ElementType, typename MappingType::extents_type,
                    typename MappingType::layout_type>;

template <class MappingType, class AccessorPolicy>
managed_mdspan(
  typename gcxx::managed_accessor<AccessorPolicy>::data_handle_type,
  const MappingType&, const gcxx::managed_accessor<AccessorPolicy>&)
  -> managed_mdspan<typename AccessorPolicy::element_type,
                    typename MappingType::extents_type,
                    typename MappingType::layout_type, AccessorPolicy>;

GCXX_NAMESPACE_MAIN_END()

#endif
