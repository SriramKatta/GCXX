// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_

#include <cstddef>
#include <memory>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/memory_resource/async_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/sync_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/sync_host_resource.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// Type-erased resource holder backing the runtime-dispatch buffer specialization.
// Lives here (in buffer's details_) so no separate type_erased_resource header
// is needed: the runtime_resource_t tag triggers a partial specialization that
// owns one of these.
// ─────────────────────────────────────────────────────────────────────────────
class runtime_resource_base {
 public:
  runtime_resource_base() = default;
  virtual ~runtime_resource_base() = default;
  runtime_resource_base(const runtime_resource_base&) = delete;
  runtime_resource_base& operator=(const runtime_resource_base&) = delete;
  runtime_resource_base(runtime_resource_base&&) = delete;
  runtime_resource_base& operator=(runtime_resource_base&&) = delete;
  virtual auto allocate(std::size_t num_bytes, gcxx::StreamView sv) -> void* = 0;
  virtual auto deallocate(void* ptr, gcxx::StreamView sv) -> void = 0;
  virtual auto is_device() const -> bool = 0;
};

template <typename Resource>
class runtime_resource_impl : public runtime_resource_base {
  Resource resource_;

 public:
  explicit runtime_resource_impl(Resource r) : resource_(std::move(r)) {}

  auto allocate(std::size_t num_bytes, gcxx::StreamView sv) -> void* override {
    return resource_.allocate(num_bytes, sv);
  }

  auto deallocate(void* ptr, gcxx::StreamView sv) -> void override {
    resource_.deallocate(ptr, sv);
  }

  auto is_device() const -> bool override { return Resource::is_device(); }
};

GCXX_NAMESPACE_DETAILS_END()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// Tag selecting runtime (type-erased) resource dispatch. Passing this as the
// Resource template argument activates the buffer<VT, runtime_resource_t>
// partial specialization below.
// ─────────────────────────────────────────────────────────────────────────────
struct runtime_resource_t {
  explicit runtime_resource_t() = default;
};

inline constexpr runtime_resource_t runtime_resource{};

// ─────────────────────────────────────────────────────────────────────────────
// Primary template: compile-time resource, zero overhead.
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
  using value_type = VT;
  using size_type  = std::size_t;
  using pointer    = VT*;

  buffer() noexcept(noexcept(Resource{})) : resource_{} {}

  explicit buffer(no_init_t) noexcept(noexcept(Resource{})) : resource_{} {}

  explicit buffer(size_type n, Resource r = {},
         gcxx::StreamView sv = gcxx::StreamView::Null())
      : resource_(std::move(r)),
        stream_(sv),
        ptr_(static_cast<VT*>(resource_.allocate(n * sizeof(VT), stream_))),
        size_(n) {}

  ~buffer() { release(); }

  buffer(const buffer&) = delete;
  buffer& operator=(const buffer&) = delete;

  buffer(buffer&& other) noexcept
      : resource_(std::move(other.resource_)),
        stream_(other.stream_),
        ptr_(other.ptr_),
        size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  buffer& operator=(buffer&& other) noexcept {
    if (this != &other) {
      release();
      resource_ = std::move(other.resource_);
      stream_   = other.stream_;
      ptr_      = other.ptr_;
      size_     = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  GCXX_FHDC auto data() noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto data() const noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto size() const noexcept -> size_type { return size_; }
  GCXX_FHDC auto is_empty() const noexcept -> bool { return ptr_ == nullptr; }

  // Pointer-range access for span construction (gcxx::span(buf)) and standard
  // algorithms. Does not construct/destroy elements; safe because VT is
  // trivially destructible (static_assert above).
  GCXX_FHDC auto begin() noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto begin() const noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto end() noexcept -> pointer { return ptr_ + size_; }
  GCXX_FHDC auto end() const noexcept -> pointer { return ptr_ + size_; }

  GCXX_FH auto destroy() -> void { release(); }
  GCXX_FH auto clear() -> void { release(); }

  /// Allocate a fresh block of n elements; discards any existing contents.
  /// A failed resize does not destroy the current allocation: the new block is
  /// secured first, then the old one is freed via move-assignment.
  GCXX_FH auto resize(size_type n,
                      gcxx::StreamView sv = gcxx::StreamView::Null()) -> void {
    *this = buffer(n, resource_, sv);
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
// Partial specialization: runtime resource dispatch via the runtime_resource_t
// tag. The concrete resource is type-erased into a details_::runtime_resource_*
// holder at construction. Constrained members (e.g. prefetch/advise) are not
// available here: the compiler cannot know at compile time whether the held
// resource models a capability, so only the common surface is exposed.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT>
class buffer<VT, runtime_resource_t> {
  static_assert(std::is_trivially_destructible_v<VT>,
                "buffer<VT, runtime_resource_t> requires trivially destructible VT");

 public:
  using value_type = VT;
  using size_type  = std::size_t;
  using pointer    = VT*;

  buffer() noexcept = default;
  explicit buffer(no_init_t) noexcept {}

  template <typename R>
  buffer(size_type n, R r,
         gcxx::StreamView sv = gcxx::StreamView::Null())
      : res_(static_cast<details_::runtime_resource_base*>(
          std::make_unique<details_::runtime_resource_impl<R>>(std::move(r))
            .release())),
        stream_(sv),
        ptr_(static_cast<VT*>(res_->allocate(n * sizeof(VT), stream_))),
        size_(n) {}

  ~buffer() { release(); }

  buffer(const buffer&) = delete;
  buffer& operator=(const buffer&) = delete;

  buffer(buffer&& other) noexcept
      : res_(std::move(other.res_)),
        stream_(other.stream_),
        ptr_(other.ptr_),
        size_(other.size_) {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }

  buffer& operator=(buffer&& other) noexcept {
    if (this != &other) {
      release();
      res_    = std::move(other.res_);
      stream_ = other.stream_;
      ptr_    = other.ptr_;
      size_   = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  GCXX_FHDC auto data() noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto data() const noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto size() const noexcept -> size_type { return size_; }
  GCXX_FHDC auto is_empty() const noexcept -> bool { return ptr_ == nullptr; }

  // Pointer-range access for span construction (gcxx::span(buf)) and standard
  // algorithms. Does not construct/destroy elements; safe because VT is
  // trivially destructible (static_assert above).
  GCXX_FHDC auto begin() noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto begin() const noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto end() noexcept -> pointer { return ptr_ + size_; }
  GCXX_FHDC auto end() const noexcept -> pointer { return ptr_ + size_; }

  GCXX_FH auto destroy() -> void { release(); }
  GCXX_FH auto clear() -> void { release(); }

 private:
  std::unique_ptr<details_::runtime_resource_base> res_;
  gcxx::StreamView stream_{gcxx::StreamView::Null()};
  VT* ptr_{nullptr};
  size_type size_{0};

  auto release() noexcept -> void {
    if (ptr_ != nullptr && res_ != nullptr) {
      res_->deallocate(static_cast<void*>(ptr_), stream_);
      ptr_  = nullptr;
      size_ = 0;
    }
  }
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
using runtime_buffer = buffer<VT, runtime_resource_t>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
