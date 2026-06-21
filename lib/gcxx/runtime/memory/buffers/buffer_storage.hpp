// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_STORAGE_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_STORAGE_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::buffer_storage<Resource>
//
// RAII owner of a raw, untyped byte block obtained from a memory resource in
// stream order. Its sole responsibility is to allocate on construction and
// deallocate on destruction (or via explicit destroy()). It deliberately holds
// no typed pointer and no initialization logic — those concerns belong to
// higher-level containers (e.g. buffer<VT, Resource>), which compose this for
// memory lifetime while managing typed access themselves.
//
// Splitting raw ownership from typed initialization guarantees that if a
// composing object's constructor throws after the storage subobject has been
// fully constructed, the storage destructor runs and returns the memory to the
// resource. Without this split, a partially-constructed object would leak the
// allocation because its own destructor never runs.
//
// Resource concept (duck-typed):
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename Resource>
class buffer_storage {
 public:
  using value_type             = VT;
  using size_type              = std::size_t;
  using pointer                = VT*;
  using const_pointer          = const VT*;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  /// Empty storage (no allocation). Resource is default-constructed.
  buffer_storage() noexcept(noexcept(Resource{})) : m_resource{} {}

  /// Empty storage bound to an explicit stream + resource (no allocation).
  /// Useful for buffers that want a ready stream/resource but defer the
  /// allocation to a later resize().
  buffer_storage(gcxx::StreamView stream, Resource resource) noexcept(
    std::is_nothrow_move_constructible<Resource>::value)
      : m_resource(std::move(resource)),
        m_stream(stream),
        m_ptr(nullptr),
        m_num_elems(0) {}

  /// Allocate num_bytes from resource on stream. Storage is uninitialized.
  /// If allocation throws, m_ptr is left null and the resource/stream members
  /// are cleaned up by their own destructors — no leak.
  buffer_storage(gcxx::StreamView stream, Resource resource,
                 size_type num_elems)
      : m_resource(std::move(resource)),
        m_stream(stream),
        m_ptr(m_resource.allocate(sizeof(VT) * num_elems, m_stream)),
        m_num_elems(num_elems) {}

  ~buffer_storage() { release(); }

  buffer_storage(const buffer_storage&)            = delete;
  buffer_storage& operator=(const buffer_storage&) = delete;

  buffer_storage(buffer_storage&& other) noexcept
      : m_resource(other.m_resource),
        m_stream(other.m_stream),
        m_ptr(other.m_ptr),
        m_num_elems(other.m_num_elems) {
    other.m_ptr       = nullptr;
    other.m_stream    = gcxx::StreamView::Null();
    other.m_num_elems = 0;
  }

  buffer_storage& operator=(buffer_storage&& other) noexcept {
    if (this != &other) {
      release();
      // Resources (e.g. pooled_device_resource) are cheap, shareable handles
      // that may be reused to allocate further buffers — copy, don't move, so
      // the moved-from storage retains a valid resource for reallocation.
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
  GCXX_FHD auto get() noexcept -> void* { return m_ptr; }
  GCXX_FHDC auto get() const noexcept -> const void* { return m_ptr; }

  // ─────────────────────────────── observers ────────────────────────────────
  GCXX_FHDC auto empty() const noexcept -> bool { return m_ptr == nullptr; }

  /// Size of the allocated block in bytes (0 for empty storage).
  GCXX_FHDC auto size_bytes() const noexcept -> size_type {
    return m_num_elems * sizeof(VT);
  }

  /// The memory resource the storage was allocated from.
  GCXX_FH auto memory_resource() const noexcept -> const Resource& {
    return m_resource;
  }

  /// The stream associated with this storage (used for deallocation).
  GCXX_FHDC auto stream() const noexcept -> gcxx::StreamView {
    return m_stream;
  }

  /// Rebind the storage to a new stream (synchronizes the old stream first).
  GCXX_FH auto set_stream(gcxx::StreamView new_stream) -> void {
    m_stream.Synchronize();
    m_stream = new_stream;
  }

  // ─────────────────────────────── operations ───────────────────────────────
  /// Deallocate using the stored stream; storage becomes empty.
  GCXX_FH auto destroy() -> void { release(); }

  /// Deallocate using an explicit stream; storage becomes empty.
  GCXX_FH auto destroy(gcxx::StreamView s) -> void {
    if (m_ptr != nullptr) {
      m_resource.deallocate(m_ptr, s);
      m_ptr       = nullptr;
      m_stream    = gcxx::StreamView::Null();
      m_num_elems = 0;
    }
  }

 private:
  Resource m_resource;
  gcxx::StreamView m_stream{gcxx::StreamView::Null()};
  void* m_ptr{nullptr};
  size_type m_num_elems{0};

  auto release() noexcept -> void {
    if (m_ptr != nullptr) {
      m_resource.deallocate(m_ptr, m_stream);
      m_ptr       = nullptr;
      m_stream    = gcxx::StreamView::Null();
      m_num_elems = 0;
    }
  }
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
