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
// Resource concept (duck-typed):
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
//   bool  is_device()  // static
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename Resource>
class buffer {
  static_assert(std::is_trivially_destructible_v<VT>,
                "buffer<VT, Resource> requires trivially destructible VT");

 public:
  using value_type             = VT;
  using size_type              = std::size_t;
  using pointer                = VT*;
  using const_pointer          = const VT*;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  /// Empty buffer (no allocation). Resource and stream are default/unset.
  buffer() noexcept(noexcept(Resource{})) : resource_{} {}

  /// Empty buffer, explicit lazy-intent tag.
  explicit buffer(no_init_t) noexcept(noexcept(Resource{})) : resource_{} {}

  /// Allocate n elements from resource on stream (CCCL ctor order:
  /// stream, resource, size). Storage is uninitialized.
  buffer(gcxx::StreamView stream, Resource resource, size_type n)
      : resource_(std::move(resource)),
        stream_(stream),
        ptr_(static_cast<VT*>(resource_.allocate(n * sizeof(VT), stream_))),
        size_(n) {}

  ~buffer() { release(); }

  buffer(const buffer&)            = delete;
  buffer& operator=(const buffer&) = delete;

  buffer(buffer&& other) noexcept
      : resource_(std::move(other.resource_)),
        stream_(other.stream_),
        ptr_(other.ptr_),
        size_(other.size_) {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }

  buffer& operator=(buffer&& other) noexcept {
    if (this != &other) {
      release();
      resource_   = std::move(other.resource_);
      stream_     = other.stream_;
      ptr_        = other.ptr_;
      size_       = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // ───────────────────────────── element access ─────────────────────────────
  GCXX_FHDC auto data() noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto data() const noexcept -> const_pointer { return ptr_; }

  GCXX_FHDC auto begin() noexcept -> iterator { return ptr_; }
  GCXX_FHDC auto begin() const noexcept -> const_iterator { return ptr_; }
  GCXX_FHDC auto end() noexcept -> iterator { return ptr_ + size_; }
  GCXX_FHDC auto end() const noexcept -> const_iterator { return ptr_ + size_; }
  GCXX_FHDC auto cbegin() const noexcept -> const_iterator { return ptr_; }
  GCXX_FHDC auto cend() const noexcept -> const_iterator {
    return ptr_ + size_;
  }
  GCXX_FHDC auto rbegin() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(end());
  }
  GCXX_FHDC auto rend() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(begin());
  }

  // ─────────────────────────────── observers ────────────────────────────────
  GCXX_FHDC auto size() const noexcept -> size_type { return size_; }
  GCXX_FHDC auto empty() const noexcept -> bool { return ptr_ == nullptr; }

  /// The memory resource the storage was allocated from.
  GCXX_FH auto memory_resource() const noexcept -> const Resource& {
    return resource_;
  }

  /// The stream associated with this buffer (used for deallocation).
  GCXX_FHDC auto stream() const noexcept -> gcxx::StreamView { return stream_; }

  /// Rebind the buffer to a new stream (synchronizes the old stream first).
  GCXX_FH auto set_stream(gcxx::StreamView new_stream) -> void {
    stream_.Synchronize();
    stream_ = new_stream;
  }

  // ─────────────────────────────── operations ───────────────────────────────
  /// Deallocate using the stored stream; buffer becomes empty.
  GCXX_FH auto destroy() -> void { release(); }

  /// Deallocate using an explicit stream; buffer becomes empty.
  GCXX_FH auto destroy(gcxx::StreamView s) -> void {
    if (ptr_ != nullptr) {
      resource_.deallocate(static_cast<void*>(ptr_), s);
      ptr_  = nullptr;
      size_ = 0;
    }
  }

  /// Reallocate to n elements (discards contents) using the stored stream.
  /// A failed resize does not destroy the current allocation.
  GCXX_FH auto resize(size_type n) -> void {
    *this = buffer(stream_, resource_, n);
  }

 private:
  Resource resource_;
  gcxx::StreamView stream_{gcxx::StreamView::Null()};
  VT* ptr_{nullptr};
  size_type size_{0};

  auto release() noexcept -> void {
    if (ptr_ != nullptr) {
      resource_.deallocate(static_cast<void*>(ptr_), stream_);
      ptr_  = nullptr;
      size_ = 0;
    }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// make_buffer: factory deducing the resource type (cf. cuda::make_buffer).
// ─────────────────────────────────────────────────────────────────────────────
template <typename T, typename Resource>
GCXX_FH auto make_buffer(gcxx::StreamView stream, Resource resource,
                         std::size_t n) -> buffer<T, Resource> {
  return buffer<T, Resource>(stream, std::move(resource), n);
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
