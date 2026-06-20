// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/buffer_storage.hpp>
#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/memory_resource/async_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/pooled_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/sync_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/sync_host_resource.hpp>
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
  static_assert(std::is_trivially_constructible_v<VT>,
                "buffer<VT, Resource> requires trivially constructable VT");
  static_assert(std::is_trivially_destructible_v<VT>,
                "buffer<VT, Resource> requires trivially destructible VT");

 public:
  using buffer_t               = buffer_storage<VT, Resource>;
  using value_type             = typename buffer_t::value_type;
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

  /// Allocate n elements from resource on stream (CCCL ctor order:
  /// stream, resource, size). Storage is uninitialized.
  buffer(gcxx::StreamView stream, Resource resource, size_type n)
      : m_storage(stream, std::move(resource), n) {}

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
