// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_UNINITIALIZED_BUFFER_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_UNINITIALIZED_BUFFER_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/iterators/heterogeneous_iterator.hpp>
#include <gcxx/iterators/reverse_iterator.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_resource/any_resource.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// ─────────────────────────────────────────────────────────────────────────────
// gcxx::uninit_buffer<VT, Properties...>
//
// Everything buffer<VT, Properties...> does — stream-ordered allocation,
// deallocation, stream/resource bookkeeping, RAII — except filling the data:
// elements are never written or initialized, so users must construct them
// themselves.
//
// Properties check execution-space constraints at compile time and gate the
// launch integration; the resource itself is taken type-erased (any_resource),
// so property validation of a concrete resource happens in the composing
// buffer's ctors (resource_has_all_v before erasure).
//
// Splitting raw ownership from typed initialization guarantees that if a
// composing object's constructor throws after storage is allocated, the
// storage destructor runs and returns the memory — no leak.
//
// Resource concept (the any_resource interface):
//   void* allocate(gcxx::StreamView, std::size_t num_bytes)
//   void  deallocate(gcxx::StreamView, void* ptr)
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename... Properties>
class uninit_buffer {
  static_assert(contains_execution_space_property<Properties...>,
                "uninit_buffer requires device_accessible or "
                "host_accessible");

  template <typename, typename...>
  friend class uninit_buffer;

 public:
  /// The buffer's accessibility contract (uniform with resources/buffer).
  using properties      = TypeSet<Properties...>;
  using value_type      = VT;
  using reference       = VT&;
  using const_reference = const VT&;
  using size_type       = std::size_t;
  using pointer         = VT*;
  using const_pointer   = const VT*;
  /// Space-restricted iterators
  using iterator         = heterogeneous_iterator<VT, Properties...>;
  using const_iterator   = heterogeneous_iterator<const VT, Properties...>;
  using reverse_iterator = gcxx::reverse_iterator<iterator>;
  using const_reverse_iterator = gcxx::reverse_iterator<const_iterator>;
  using resource_type          = any_resource;

  /// Empty storage (no allocation, no resource).
  uninit_buffer() noexcept = default;

  /// Empty storage bound to a stream + type-erased resource (no allocation).
  uninit_buffer(gcxx::StreamView stream, resource_type res) noexcept
      : m_resource(std::move(res)), m_stream(stream) {}

  /// Allocate num_elems elements from the type-erased resource on stream.
  /// Storage is uninitialized. A zero-size request allocates nothing.
  uninit_buffer(gcxx::StreamView stream, resource_type res, size_type num_elems)
      : m_resource(std::move(res)),
        m_stream(stream),
        m_ptr(num_elems != 0
                ? m_resource.allocate(m_stream, sizeof(VT) * num_elems)
                : nullptr),
        m_num_elems(num_elems) {}

  ~uninit_buffer() { release(); }

  uninit_buffer(const uninit_buffer&)            = delete;
  uninit_buffer& operator=(const uninit_buffer&) = delete;

  uninit_buffer(uninit_buffer&& other) noexcept
      // Resources (unlike the allocation) are cheap, shareable handles that
      // may be reused to allocate further buffers — copy, don't move, so the
      // moved-from storage retains a valid resource for reallocation (e.g.
      // buffer::resize on a moved-from buffer).
      : m_resource(other.m_resource),
        m_stream(other.m_stream),
        m_ptr(other.m_ptr),
        m_num_elems(other.m_num_elems) {
    other.m_ptr       = nullptr;
    other.m_stream    = gcxx::StreamView::Null();
    other.m_num_elems = 0;
  }

  /// Cross-properties move ctor (CCCL __properties_match: other's Properties
  /// ⊇ this buffer's). Steals the source's allocation.
  GCXX_TEMPLATE(typename... OtherProperties)
  GCXX_REQUIRES((TypeSet<OtherProperties...>::template contains<Properties> &&
                 ...))
  uninit_buffer(uninit_buffer<VT, OtherProperties...>&& other) noexcept
      : m_resource(other.m_resource),
        m_stream(other.m_stream),
        m_ptr(other.m_ptr),
        m_num_elems(other.m_num_elems) {
    other.m_ptr       = nullptr;
    other.m_stream    = gcxx::StreamView::Null();
    other.m_num_elems = 0;
  }

  uninit_buffer& operator=(uninit_buffer&& other) noexcept {
    if (this != &other) {
      release();
      // Copy the resource (see the move constructor): the moved-from storage
      // keeps a valid resource for reallocation.
      m_resource        = other.m_resource;
      m_stream          = other.m_stream;
      m_ptr             = other.m_ptr;
      m_num_elems       = other.m_num_elems;
      other.m_ptr       = nullptr;
      other.m_stream    = gcxx::StreamView::Null();
      other.m_num_elems = 0;
    }
    return *this;
  }

  // ───────────────────────────── raw access ─────────────────────────────────
  /// Pointer to the first element. The storage is uninitialized — reading
  /// through this pointer before writing is undefined behavior.
  GCXX_FHDC auto data() noexcept -> pointer {
    return static_cast<pointer>(m_ptr);
  }
  GCXX_FHDC auto data() const noexcept -> const_pointer {
    return static_cast<const_pointer>(m_ptr);
  }

  GCXX_FHDC auto begin() noexcept -> iterator { return iterator(data()); }
  GCXX_FHDC auto begin() const noexcept -> const_iterator {
    return const_iterator(data());
  }
  GCXX_FHDC auto end() noexcept -> iterator {
    return iterator(data() + size());
  }
  GCXX_FHDC auto end() const noexcept -> const_iterator {
    return const_iterator(data() + size());
  }

  /// Reverse iteration: rbegin references the last element, rend the slot
  /// before the first (std::reverse_iterator semantics).
  GCXX_FHDC auto rbegin() noexcept -> reverse_iterator {
    return reverse_iterator(end());
  }
  GCXX_FHDC auto rbegin() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(end());
  }
  GCXX_FHDC auto rend() noexcept -> reverse_iterator {
    return reverse_iterator(begin());
  }
  GCXX_FHDC auto rend() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(begin());
  }

  // ─────────────────────────────── observers ────────────────────────────────
  /// Number of elements the storage was allocated for.
  GCXX_FHDC auto size() const noexcept -> size_type { return m_num_elems; }

  GCXX_FHDC auto empty() const noexcept -> bool { return m_num_elems == 0; }

  /// Size of the allocated block in bytes (0 for empty storage).
  GCXX_FHDC auto size_bytes() const noexcept -> size_type {
    return m_num_elems * sizeof(VT);
  }

  /// The type-erased memory resource the storage was allocated from.
  GCXX_FH auto memory_resource() const noexcept -> const resource_type& {
    return m_resource;
  }

  /// Clone the stored resource (for resize / cross-properties copy). Each call
  /// heap-allocates a copy of the erased resource.
  GCXX_FH auto borrow_resource() const -> resource_type { return m_resource; }

  /// The stream associated with this storage (used for deallocation).
  GCXX_FHDC auto stream() const noexcept -> gcxx::StreamView {
    return m_stream;
  }

  /// Rebind the storage to a new stream (synchronizes the old stream first).
  GCXX_FH auto set_stream(gcxx::StreamView new_stream) -> void {
    m_stream.Synchronize();
    m_stream = new_stream;
  }

  /// Rebind the storage to a new stream without synchronizing. It is the
  /// user's responsibility to ensure proper stream order going forward.
  GCXX_FH auto set_stream_unsynchronized(gcxx::StreamView new_stream) noexcept
    -> void {
    m_stream = new_stream;
  }

  // ─────────────────────────────── operations ───────────────────────────────
  /// Deallocate using the stored stream; storage becomes empty.
  GCXX_FH auto destroy() -> void { release(); }

  /// Deallocate using an explicit stream; storage becomes empty.
  GCXX_FH auto destroy(gcxx::StreamView s) -> void {
    if (m_ptr != nullptr) {
      m_resource.deallocate(s, m_ptr);
      m_ptr       = nullptr;
      m_stream    = gcxx::StreamView::Null();
      m_num_elems = 0;
    }
  }

  /// Reallocate to num_elems elements, discarding the current contents and
  /// reusing the stored resource. The old block is deallocated on the OLD
  /// stream (its allocation order), then the new one is allocated on
  /// new_stream. (CCCL __replace_allocation_discard parity; used by
  /// buffer::resize.)
  GCXX_FH auto replace_allocation_discard(gcxx::StreamView new_stream,
                                          size_type num_elems) -> void {
    if (m_ptr != nullptr) {
      m_resource.deallocate(m_stream, m_ptr);
      m_ptr = nullptr;
    }
    m_stream    = new_stream;
    m_num_elems = num_elems;
    if (num_elems != 0) {
      m_ptr = m_resource.allocate(m_stream, sizeof(VT) * num_elems);
    }
  }

  // ──────────────────── launch integration (device-gated) ────────────────────
  // Lets a device_accessible uninit_buffer be treated as a span when
  // passed to a launch customization point (CCCL cuda::launch parity).
  GCXX_TEMPLATE(bool D = is_device_accessible<Properties...>)
  GCXX_REQUIRES(D)
  GCXX_FH friend auto transform_launch_argument(
    gcxx::StreamView, uninit_buffer& self) noexcept -> gcxx::span<value_type> {
    return {self.data(), self.size()};
  }
  GCXX_TEMPLATE(bool D = is_device_accessible<Properties...>)
  GCXX_REQUIRES(D)
  GCXX_FH friend auto transform_launch_argument(
    gcxx::StreamView,
    const uninit_buffer& self) noexcept -> gcxx::span<const value_type> {
    return {self.data(), self.size()};
  }

 private:
  resource_type m_resource;
  gcxx::StreamView m_stream{gcxx::StreamView::Null()};
  void* m_ptr{nullptr};
  size_type m_num_elems{0};

  auto release() noexcept -> void {
    if (m_ptr != nullptr) {
      m_resource.deallocate(m_stream, m_ptr);
      m_ptr       = nullptr;
      m_stream    = gcxx::StreamView::Null();
      m_num_elems = 0;
    }
  }
};


// ─────────────────────────────────────────────────────────────────────────────
// Convenience aliases.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT>
using uninit_device_buffer = uninit_buffer<VT, device_accessible>;

template <typename VT>
using uninit_host_buffer = uninit_buffer<VT, host_accessible>;


GCXX_NAMESPACE_MAIN_END()

#endif
