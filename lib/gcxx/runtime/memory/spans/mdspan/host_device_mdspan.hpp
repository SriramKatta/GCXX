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

// ─────────────────────────────────────────────────────────────────────────────
// host_mdspan / device_mdspan / managed_mdspan.
//
// Port of CCCL libcudacxx's cuda/__mdspan/host_device_mdspan.h: vocabulary
// types for multi-dimensional views of the respective CUDA/HIP memory
// spaces. Each derives gcxx::mdspan with the memory-space accessor
// (host_device_accessor.hpp) substituted as the accessor policy; the data
// handle stays a raw pointer, so they are constructed from ordinary
// pointers exactly like gcxx::mdspan. The memory-space guarantee is
// enforced where the view is consumed — e.g. every gcxx::blas operation
// requires a device view at compile time — and by the accessors' deleted
// cross-space conversions (host_mdspan -> device_mdspan does not compile).
//
// Deviation from CCCL: the (ptr, ints...) / array / span deduction guides
// are omitted (they need the vendored mdspan's internal maybe-static-extent
// helpers); construct from an extents object, a mapping, or with explicit
// template arguments.
// ─────────────────────────────────────────────────────────────────────────────

// ─── device_mdspan ───────────────────────────────────────────────────────────

// View of device memory: passable to device libraries (gcxx::blas) and
// dereferenceable in device code.
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
device_mdspan(
  const typename gcxx::device_accessor<AccessorPolicy>::data_handle_type,
  const MappingType&, const gcxx::device_accessor<AccessorPolicy>&)
  -> device_mdspan<typename AccessorPolicy::element_type,
                   typename MappingType::extents_type,
                   typename MappingType::layout_type, AccessorPolicy>;

// ─── host_mdspan ─────────────────────────────────────────────────────────────

// View of host memory: dereferenceable from host code only.
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
host_mdspan(
  const typename gcxx::host_accessor<AccessorPolicy>::data_handle_type,
  const MappingType&, const gcxx::host_accessor<AccessorPolicy>&)
  -> host_mdspan<typename AccessorPolicy::element_type,
                 typename MappingType::extents_type,
                 typename MappingType::layout_type, AccessorPolicy>;

// ─── managed_mdspan ──────────────────────────────────────────────────────────

// View of managed memory: reachable from both host and device.
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
  const typename gcxx::managed_accessor<AccessorPolicy>::data_handle_type,
  const MappingType&, const gcxx::managed_accessor<AccessorPolicy>&)
  -> managed_mdspan<typename AccessorPolicy::element_type,
                    typename MappingType::extents_type,
                    typename MappingType::layout_type, AccessorPolicy>;

GCXX_NAMESPACE_MAIN_END()

#endif
