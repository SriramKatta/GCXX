// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SCALARS_SCALAR_HPP_
#define GCXX_RUNTIME_MEMORY_SCALARS_SCALAR_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/buffers/uninitialized_buffer.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime/memory/memset.hpp>
#include <gcxx/runtime/memory/scalars/device_scalar_view.hpp>
#include <gcxx/runtime/memory/spans/mdspan/scaled_accessor.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// RMM device_scalar adapted to GCXX's property model: an owning,
// single-element container parameterized like buffer and backed by an
// uninit_buffer (RMM: device_scalar over device_uvector). Move-only; deep
// copies are the explicit-stream constructors. The backing resource carries
// the allocation kind (device / pinned / managed pool), exactly as buffer
// resolves it — no per-kind deleter state lives here.
//
// Stream ordering (RMM parity): allocation and the async setters never
// synchronize; value() is the one synchronizing API (D2H copy + stream sync).
// host_accessible storage (pinned/managed) is read/written directly by the
// host thread — a caller with pending device writes on another stream must
// order them first.
template <typename VT, typename... Properties>
class scalar {
  static_assert(std::is_trivially_copyable_v<VT>,
                "scalar requires trivially copyable VT");
  static_assert(contains_execution_space_property<Properties...>,
                "scalar requires device_accessible or host_accessible");

 public:
  // The scalar's accessibility contract (uniform with resources/buffer).
  using properties    = TypeSet<Properties...>;
  using storage_t     = uninit_buffer<VT, Properties...>;
  using value_type    = VT;
  using size_type     = std::size_t;
  using pointer       = value_type*;
  using const_pointer = const value_type*;

  // Duck-type marker for details_::is_owning_scalar_v (device_scalar_view.hpp)
  // — [[maybe_unused]] keeps nvcc #177-D quiet on TUs that never probe it.
  [[maybe_unused]] static constexpr bool is_gcxx_scalar = true;

  // Resource-taking ctors: resource properties must ⊇ scalar's Properties.
  // Exactly one element is allocated; uninitialized (RMM: no sync).

  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, scalar>)
  explicit scalar(gcxx::StreamView stream, Resource&& resource)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)), 1) {
    details_::validate_resource<Resource, Properties...>();
  }

  // Allocate one element and copy `value` into it (async, no sync).
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, scalar>)
  scalar(gcxx::StreamView stream, Resource&& resource, const value_type& value)
      : scalar(stream, std::forward<Resource>(resource)) {
    copy_element_(stream, data(), &value);
  }

  // Deep copy on an explicit stream, re-allocating from `resource`
  // (RMM: device_scalar(other, stream, mr)).
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, scalar>)
  scalar(const scalar& other, gcxx::StreamView stream, Resource&& resource)
      : scalar(stream, std::forward<Resource>(resource)) {
    copy_from_(other, stream);
  }

  // Deep copy on an explicit stream, borrowing the source's resource
  // (buffer's copy-ctor pattern).
  scalar(const scalar& other, gcxx::StreamView stream)
      : m_storage(stream, other.m_storage.borrow_resource(), 1) {
    copy_from_(other, stream);
  }

  scalar(const scalar&)                = delete;
  scalar& operator=(const scalar&)     = delete;
  scalar(scalar&&) noexcept            = default;
  scalar& operator=(scalar&&) noexcept = default;
  ~scalar()                            = default;

  // Raw access (exactly one element).
  GCXX_FHDC auto data() noexcept -> pointer { return m_storage.data(); }
  GCXX_FHDC auto data() const noexcept -> const_pointer {
    return m_storage.data();
  }

  // Observers.
  GCXX_FHDC auto size() const noexcept -> size_type { return 1; }
  GCXX_FHDC auto empty() const noexcept -> bool { return false; }
  GCXX_FHDC auto size_bytes() const noexcept -> size_type { return sizeof(VT); }
  GCXX_FH auto memory_resource() const noexcept -> const any_resource& {
    return m_storage.memory_resource();
  }
  GCXX_FHDC auto stream() const noexcept -> gcxx::StreamView {
    return m_storage.stream();
  }
  GCXX_FH auto set_stream(gcxx::StreamView new_stream) -> void {
    m_storage.set_stream(new_stream);
  }
  // Caller must ensure stream order going forward after this rebind.
  GCXX_FH auto set_stream_unsynchronized(gcxx::StreamView new_stream) noexcept
    -> void {
    m_storage.set_stream_unsynchronized(new_stream);
  }

  // Operations.
  GCXX_FH auto destroy() -> void { m_storage.destroy(); }
  GCXX_FH auto destroy(gcxx::StreamView s) -> void { m_storage.destroy(s); }

  // Value access.
  // device storage: async D2H on the captured stream, then sync (the one
  // synchronizing accessor, RMM parity). host_accessible storage: direct
  // read — ordered w.r.t. the host thread only; sync device writers first.
  GCXX_FH auto value() const -> value_type {
    if constexpr (is_host_accessible<Properties...>) {
      return *data();
    } else {
      value_type tmp{};
      copy_element_(stream(), &tmp, data());
      stream().sync();
      return tmp;
    }
  }

  // host_accessible storage: direct write. device storage: async H2D + sync.
  GCXX_FH auto set_value(const value_type& v) -> void {
    copy_element_(stream(), data(), &v);
    if constexpr (!is_host_accessible<Properties...>) {
      stream().sync();
    }
  }

  // Async H2D; never syncs (RMM set_value_async). The host `v` must not be
  // modified or destroyed until the stream completes. host_accessible
  // storage takes the direct write (the stream is unused).
  GCXX_FH auto set_value_async(const value_type& v,
                               gcxx::StreamView s) -> void {
    copy_element_(s, data(), &v);
  }
  GCXX_FH auto set_value_async(const value_type& v) -> void {
    set_value_async(v, stream());
  }
  // A temporary host source could dangle under the in-flight async copy.
  auto set_value_async(value_type&&, gcxx::StreamView) -> void = delete;
  auto set_value_async(value_type&&) -> void                   = delete;

  // Zeroes the element (async memset / direct store; never syncs).
  GCXX_FH auto set_value_to_zero_async(gcxx::StreamView s) -> void {
    if constexpr (is_host_accessible<Properties...>) {
      (void)s;
      *data() = value_type{};
    } else {
      Memset(s, data(), 0, 1);
    }
  }
  GCXX_FH auto set_value_to_zero_async() -> void {
    set_value_to_zero_async(stream());
  }

 private:
  // One-element host<->device copy: the single home of the Copy same-cv
  // const_cast workaround and the host/device store split.
  GCXX_FH auto copy_element_(gcxx::StreamView s, pointer dst,
                             const value_type* src) const -> void {
    if constexpr (is_host_accessible<Properties...>) {
      (void)s;
      *dst = *src;
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): Copy requires
      // same-cv pointers; `src` is only read.
      Copy(s, dst, const_cast<value_type*>(src), 1);
    }
  }

  GCXX_FH auto copy_from_(const scalar& other,
                          gcxx::StreamView stream) -> void {
    copy_element_(stream, data(), other.data());
  }

  storage_t m_storage{};
};

// Properties are explicit: accessibility is a claim, not inferable
// (make_buffer parity).
template <typename VT, typename... Properties, typename Resource,
          typename... Args>
GCXX_FH auto make_scalar(gcxx::StreamView stream, Resource&& resource,
                         Args&&... args) -> scalar<VT, Properties...> {
  return scalar<VT, Properties...>{stream, std::forward<Resource>(resource),
                                   std::forward<Args>(args)...};
}

// The allocation-kind tags make the four scalar kinds DISTINCT types and
// carry pointer-attribute provenance statically: host pointer-mode
// positions require host_allocated (unified_allocated cannot name them),
// device pointer-mode requires device_accessible (satisfied by device,
// unified, or pinned-UVA memory).
template <typename VT>
using device_scalar = scalar<VT, device_accessible, device_allocated>;

// Plain host storage (stack/heap): the only kind valid for host pointer
// mode.
template <typename VT>
using host_scalar = scalar<VT, host_accessible, host_allocated>;

// Page-locked host memory (cudaMallocHost/host pool): also
// device-accessible via its UVA mapping — pinned_allocated, not
// host_allocated (pageable).
template <typename VT>
using pinned_scalar =
  scalar<VT, host_accessible, device_accessible, pinned_allocated>;

// Managed memory: unified_allocated — a different type from pinned.
template <typename VT>
using managed_scalar =
  scalar<VT, host_accessible, device_accessible, unified_allocated>;

// scaled() bridge: owning scalars decay to a copyable factor at the boundary
// because scaled_accessor stores its factor by value inside a freely copied
// mdspan. The mdspan does NOT keep the scalar alive — the caller must keep it
// alive for the lifetime of every BLAS call consuming the view.

// Device-accessible owner (device/pinned/managed) -> device_scalar_view
// factor; BLAS dispatch resolves it to device pointer mode.
GCXX_TEMPLATE(typename VT, typename... Properties, typename T, typename Extents,
              typename Layout, typename Accessor)
GCXX_REQUIRES(is_device_accessible<Properties...>)
constexpr auto scaled(const scalar<VT, Properties...>& alpha,
                      const gcxx::mdspan<T, Extents, Layout, Accessor>& x) {
  // Reuse the generic factor overload: device_scalar_view is not an owner,
  // so it takes the plain path.
  return scaled(device_scalar_view<VT>{alpha.data()}, x);
}

// Host-only owner -> resolved to a host value factor here (host storage is
// directly readable).
GCXX_TEMPLATE(typename VT, typename... Properties, typename T, typename Extents,
              typename Layout, typename Accessor)
GCXX_REQUIRES(!is_device_accessible<Properties...> GCXX_AND
                is_host_accessible<Properties...>)
constexpr auto scaled(const scalar<VT, Properties...>& alpha,
                      const gcxx::mdspan<T, Extents, Layout, Accessor>& x) {
  return scaled(alpha.value(), x);
}

// A temporary owner would dangle inside the copied accessor; pass the scalar
// by lvalue instead. (Plain template<> on purpose: this overload must not
// pair GCXX_TEMPLATE with GCXX_REQUIRES.)
template <typename VT, typename... Properties, typename T, typename Extents,
          typename Layout, typename Accessor>
constexpr auto scaled(scalar<VT, Properties...>&& alpha,
                      const gcxx::mdspan<T, Extents, Layout, Accessor>& x) =
  delete;

GCXX_NAMESPACE_MAIN_END()

#endif
