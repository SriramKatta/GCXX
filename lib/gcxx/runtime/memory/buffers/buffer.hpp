// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/buffer_storage.hpp>
#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/fill.hpp>
#include <gcxx/runtime/memory/memory_resource/pooled_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/resources.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::buffer<VT, Resource>
//
// A typed raw-storage owner allocated from a memory resource in stream order.
// Interface modelled on cuda::buffer (CCCL/libcudacxx): the resource is a
// compile-time template argument (zero overhead; one per backend), while the
// stream and resource value are passed at construction. Elements are NOT
// constructed or destroyed — VT must be trivially destructible.
//
// Responsibilities are split across two layers:
//   * buffer_storage<Resource>  — owns the raw byte block (allocation,
//                                 deallocation, byte size); RAII with the
//                                 memory resource.
//   * buffer<VT, Resource>      — composes buffer_storage and provides typed
//                                 data()/iterator access, deriving the element
//                                 count from the stored byte size.
//
// Because raw ownership (including size) lives in the buffer_storage
// subobject, any future initialization performed by buffer's constructor is
// exception-safe: if that initialization throws after storage is allocated,
// the storage subobject's destructor runs and returns the memory to the
// resource — no leak.
//
// Resource concept (duck-typed):
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename Resource>
class buffer {
  static_assert(std::is_trivially_copyable_v<VT>,
                "buffer<VT, Resource> requires trivially copyable VT");

  // T3: Resource must advertise at least one execution-space property
  // (host_accessible or device_accessible) via friend get_property. Catches
  // misuse early instead of silently disabling all element accessors.
  static_assert(contains_execution_space_property_v<Resource>,
                "buffer<VT, Resource> requires Resource to advertise "
                "device_accessible or host_accessible");

 public:
  using buffer_t               = buffer_storage<VT, Resource>;
  using value_type             = typename buffer_t::value_type;
  using reference              = value_type&;
  using const_reference        = const value_type&;
  using size_type              = typename buffer_t::size_type;
  using pointer                = typename buffer_t::pointer;
  using const_pointer          = typename buffer_t::const_pointer;
  using iterator               = typename buffer_t::iterator;
  using const_iterator         = typename buffer_t::const_iterator;
  using reverse_iterator       = typename buffer_t::reverse_iterator;
  using const_reverse_iterator = typename buffer_t::const_reverse_iterator;

  /// Empty buffer (no allocation). Resource and stream are default/unset.
  buffer() noexcept(noexcept(Resource{})) : m_storage{} {}

  /// Empty buffer, explicit lazy-intent tag.
  explicit buffer(no_init_t) noexcept(noexcept(Resource{})) : m_storage{} {}

  /// Empty buffer bound to an explicit stream + resource (no allocation).
  buffer(gcxx::StreamView stream, Resource resource) noexcept(
    std::is_nothrow_move_constructible<Resource>::value)
      : m_storage(stream, std::move(resource)) {}

  /// Allocate n elements from resource on stream (CCCL ctor order:
  /// stream, resource, size). Storage is uninitialized.
  buffer(gcxx::StreamView stream, Resource resource, size_type n)
      : m_storage(stream, std::move(resource), n) {}

  /// Allocate n elements from resource on stream; storage is left
  /// uninitialized (explicit no-init tag).
  buffer(gcxx::StreamView stream, Resource resource, size_type n, no_init_t)
      : m_storage(stream, std::move(resource), n) {}

  /// Allocate n elements from resource on stream and initialize every element
  /// to value. Dispatches to memset when value is zero, otherwise to a fill
  /// kernel. Exception-safe: if initialization throws after the storage
  /// subobject has allocated, m_storage's destructor reclaims the memory.
  buffer(gcxx::StreamView stream, Resource resource, size_type n,
         const value_type& value)
      : m_storage(stream, std::move(resource), n) {
    pointer p = data();
    Fill(stream, p, value, n);
  }

  /// Allocate il.size() elements and copy from the initializer list.
  /// Uses the existing gcxx::memory::Copy (async device-side copy).
  // ponytail: const_cast because gcxx::memory::Copy's static_assert requires
  // same-type pointers (int* vs const int* fail). The destination is freshly
  // allocated uninitialized storage owned by this buffer; writing through it
  // is safe. Loosen Copy's assert (remove_cv on both sides) if more const
  // source use cases appear.
  buffer(gcxx::StreamView stream, Resource resource,
         std::initializer_list<value_type> il)
      : m_storage(stream, std::move(resource), il.size()) {
    if (il.size() != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      Copy(stream, data(), const_cast<value_type*>(il.begin()), il.size());
    }
  }
  /// Allocate rng.size() elements and copy from a sized range.
  /// SFINAE on non-integral Range to avoid shadowing the (stream, resource, n)
  /// ctor.
  GCXX_TEMPLATE(typename Range)
  GCXX_REQUIRES(!std::is_integral_v<std::decay_t<Range>>)
  buffer(gcxx::StreamView stream, Resource resource, Range&& rng)
      : m_storage(stream, std::move(resource),
                  static_cast<size_type>(rng.size())) {
    if (rng.size() != 0) {
      // ponytail: same const_cast story as the initializer_list ctor above.
      // std::data(rng) returns a raw pointer (int* for mutable vector,
      // const int* for const vector / initializer_list); std::begin(rng)
      // would return a wrapped iterator that const_cast can't handle.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      Copy(stream, data(), const_cast<value_type*>(std::data(rng)), rng.size());
    }
  }

  /// Cross-space deep copy ctor (T3). Allocate other.size() elements from
  /// resource on stream and copy from `other` (which may use a different
  /// Resource with different accessibility properties). SFINAE rejects
  /// same-type buffers (use the regular copy ctor — which is deleted, so this
  /// prevents accidental implicit copies too).
  /// ponytail: const_cast because gcxx::memory::Copy's static_assert requires
  /// same-type pointers; other.data() returns const_pointer. See the
  /// initializer_list ctor for the same workaround.
  GCXX_TEMPLATE(typename OtherResource)
  GCXX_REQUIRES(!std::is_same_v<OtherResource, Resource>)
  buffer(gcxx::StreamView stream, Resource resource,
         const buffer<value_type, OtherResource>& other)
      : m_storage(stream, std::move(resource), other.size()) {
    if (other.size() != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      Copy(stream, data(), const_cast<value_type*>(other.data()), other.size());
    }
  }

  /// Destructor is defaulted: raw memory is released by m_storage's RAII.
  ~buffer() = default;

  buffer(const buffer&)            = delete;
  buffer& operator=(const buffer&) = delete;

  buffer(buffer&&) noexcept = default;

  buffer& operator=(buffer&& other) noexcept {
    if (this != &other) {
      m_storage = std::move(other.m_storage);
    }
    return *this;
  }

  // ───────────────────────────── element access ─────────────────────────────
  GCXX_FHDC auto data() noexcept -> pointer {
    return static_cast<pointer>(m_storage.get());
  }
  GCXX_FHDC auto data() const noexcept -> const_pointer {
    return static_cast<const_pointer>(m_storage.get());
  }

  GCXX_FHDC auto begin() noexcept -> iterator { return data(); }
  GCXX_FHDC auto begin() const noexcept -> const_iterator { return data(); }
  GCXX_FHDC auto end() noexcept -> iterator { return data() + size(); }
  GCXX_FHDC auto end() const noexcept -> const_iterator {
    return data() + size();
  }
  GCXX_FHDC auto cbegin() const noexcept -> const_iterator { return data(); }
  GCXX_FHDC auto cend() const noexcept -> const_iterator {
    return data() + size();
  }
  GCXX_FHDC auto rbegin() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(end());
  }
  GCXX_FHDC auto rend() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(begin());
  }

  // ─────────────────────────── element access (T3) ──────────────────────────
  // Gated on host_accessible: dereferencing device-only memory from the host
  // is UB, so the accessors are SFINAE-removed unless Resource advertises
  // host_accessible via friend get_property. Mirrors CCCL buffer.h:484-565.
  // first/last/subspan below remain un-gated — they return spans (views),
  // which is safe to construct without touching memory.
  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto operator[](size_type i) noexcept -> reference {
    GCXX_RUNTIME_EXPECT(i < size(), "buffer::operator[] index out of range");
    return data()[i];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto operator[](size_type i) const noexcept -> const_reference {
    GCXX_RUNTIME_EXPECT(i < size(), "buffer::operator[] index out of range");
    return data()[i];
  }

  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FH auto at(size_type i) -> reference {
    if (i >= size())
      throw std::out_of_range{"buffer::at"};
    return data()[i];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FH auto at(size_type i) const -> const_reference {
    if (i >= size())
      throw std::out_of_range{"buffer::at"};
    return data()[i];
  }

  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto front() noexcept -> reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::front on empty buffer");
    return data()[0];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto front() const noexcept -> const_reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::front on empty buffer");
    return data()[0];
  }

  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto back() noexcept -> reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::back on empty buffer");
    return data()[size() - 1];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible_v<Resource>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto back() const noexcept -> const_reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::back on empty buffer");
    return data()[size() - 1];
  }

  // ─────────────────────────────── slicing (T1) ─────────────────────────────
  // Returns gcxx::span (the project's own span — spans/span/span.hpp).
  GCXX_FHDC auto first(size_type n) noexcept -> gcxx::span<value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::first count out of range");
    return {data(), n};
  }
  GCXX_FHDC auto first(size_type n) const noexcept
    -> gcxx::span<const value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::first count out of range");
    return {data(), n};
  }

  GCXX_FHDC auto last(size_type n) noexcept -> gcxx::span<value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::last count out of range");
    return {data() + size() - n, n};
  }
  GCXX_FHDC auto last(size_type n) const noexcept
    -> gcxx::span<const value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::last count out of range");
    return {data() + size() - n, n};
  }

  GCXX_FHDC auto subspan(size_type offset,
                         size_type count = gcxx::dynamic_extent) noexcept
    -> gcxx::span<value_type> {
    GCXX_RUNTIME_EXPECT(offset <= size(),
                        "buffer::subspan offset out of range");
    return count == gcxx::dynamic_extent
             ? gcxx::span<value_type>{data() + offset, size() - offset}
             : gcxx::span<value_type>{data() + offset, count};
  }
  GCXX_FHDC auto subspan(size_type offset,
                         size_type count = gcxx::dynamic_extent) const noexcept
    -> gcxx::span<const value_type> {
    GCXX_RUNTIME_EXPECT(offset <= size(),
                        "buffer::subspan offset out of range");
    return count == gcxx::dynamic_extent
             ? gcxx::span<const value_type>{data() + offset, size() - offset}
             : gcxx::span<const value_type>{data() + offset, count};
  }

  // ─────────────────────────────── observers ────────────────────────────────
  /// Number of elements (derived from the stored byte size).
  GCXX_FHDC auto size() const noexcept -> size_type {
    return m_storage.size_bytes() / sizeof(VT);
  }

  /// Size of the allocated block in bytes.
  GCXX_FHDC auto size_bytes() const noexcept -> size_type {
    return m_storage.size_bytes();
  }

  GCXX_FHDC auto empty() const noexcept -> bool { return m_storage.empty(); }

  /// The memory resource the storage was allocated from.
  GCXX_FH auto memory_resource() const noexcept -> const Resource& {
    return m_storage.memory_resource();
  }

  /// The stream associated with this buffer (used for deallocation).
  GCXX_FHDC auto stream() const noexcept -> gcxx::StreamView {
    return m_storage.stream();
  }

  /// Rebind the buffer to a new stream (synchronizes the old stream first).
  GCXX_FH auto set_stream(gcxx::StreamView new_stream) -> void {
    m_storage.set_stream(new_stream);
  }

  // ─────────────────────────────── operations ───────────────────────────────
  /// Deallocate using the stored stream; buffer becomes empty.
  GCXX_FH auto destroy() -> void { m_storage.destroy(); }

  /// Deallocate using an explicit stream; buffer becomes empty.
  GCXX_FH auto destroy(gcxx::StreamView s) -> void { m_storage.destroy(s); }

  /// Reallocate to n elements (discards contents) using the stored stream.
  /// A failed resize does not destroy the current allocation.
  GCXX_FH auto resize(size_type n) -> void {
    *this = buffer(m_storage.stream(), m_storage.memory_resource(), n);
  }

  /// Access the underlying raw storage (e.g. for resource-level operations).
  GCXX_FH auto storage() noexcept -> buffer_t& { return m_storage; }
  GCXX_FH auto storage() const noexcept -> const buffer_t& { return m_storage; }

 private:
  buffer_t m_storage{};
};

// ─────────────────────────────────────────────────────────────────────────────
// make_buffer: CTAD-friendly factory. Single variadic template — perfect
// forwarding picks the right ctor overload. ponytail: not 10 overloads like
// CCCL; add overloads only if a call site needs them.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename Resource, typename... Args>
GCXX_FH auto make_buffer(gcxx::StreamView stream, Resource resource,
                         Args&&... args) -> buffer<VT, Resource> {
  return buffer<VT, Resource>{stream, std::move(resource),
                              std::forward<Args>(args)...};
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience aliases.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT>
using device_buffer = buffer<VT, sync_device_resource>;

template <typename VT>
using host_buffer = buffer<VT, sync_host_resource>;

template <typename VT>
using device_buffer_async = buffer<VT, async_device_resource>;

template <typename VT>
using device_buffer_pooled = buffer<VT, pooled_device_resource>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
