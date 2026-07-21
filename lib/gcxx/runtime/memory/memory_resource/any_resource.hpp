// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_ANY_RESOURCE_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_ANY_RESOURCE_HPP_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::any_resource
//
// A non-templated, type-erased, owning, copyable holder for ANY resource
// matching the concept:
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
//
// Under the Option-B property mechanism, accessibility (Properties) lives on
// the BUFFER type (buffer<VT, Properties...>), validated at construction via
// resource_has_all_v. The type-erased allocator does not carry Properties — it
// is just an opaque allocate/deallocate handle. This keeps buffer_storage<VT>
// (and thus the cross-properties copy ctor) independent of the property pack:
// a buffer<int, host_accessible> and buffer<int, device_accessible> share the
// same any_resource / buffer_storage<int> erasure, differing only in their
// Properties.
//
// The concrete resource is held behind a virtual interface (model<R>), copied
// via clone(). ponytail (deferred): SBO — the current implementation
// heap-allocates one model<R> per resource (and one per clone, e.g. on resize /
// cross-copy). Negligible next to the driver allocate/free it wraps; a
// small-buffer optimization for stateless resources is a future refinement.
// ─────────────────────────────────────────────────────────────────────────────
class any_resource {
 public:
  /// Empty (no resource). allocate/deallocate on an empty any_resource assert.
  any_resource() noexcept = default;

  /// Type-erase a concrete resource. The resource must be copy constructible
  /// (any_resource owns a copy, cloned on copy/resize). Property validation is
  /// the caller's responsibility (the buffer ctor enforces it via
  /// resource_has_all_v before erasure).
  template <typename Resource, typename = std::enable_if_t<!std::is_same_v<
                                 std::decay_t<Resource>, any_resource>>>
  any_resource(Resource&& r) {
    static_assert(std::is_copy_constructible_v<std::decay_t<Resource>>,
                  "any_resource owns a copy of the resource; it must be copy "
                  "constructible");
    impl_.reset(new model<std::decay_t<Resource>>(std::forward<Resource>(r)));
  }

  any_resource(const any_resource& other)
      : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

  any_resource(any_resource&&) noexcept = default;

  any_resource& operator=(const any_resource& other) {
    impl_.reset(other.impl_ ? other.impl_->clone() : nullptr);
    return *this;
  }
  any_resource& operator=(any_resource&&) noexcept = default;

  ~any_resource() = default;

  GCXX_FH auto allocate(std::size_t num_bytes,
                        gcxx::StreamView sv) const -> void* {
    GCXX_RUNTIME_EXPECT(impl_ != nullptr,
                        "any_resource::allocate on empty resource");
    return impl_->allocate(num_bytes, sv);
  }

  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView sv) const -> void {
    GCXX_RUNTIME_EXPECT(impl_ != nullptr,
                        "any_resource::deallocate on empty resource");
    impl_->deallocate(ptr, sv);
  }

  explicit operator bool() const noexcept { return impl_ != nullptr; }

 private:
  struct interface {
    virtual ~interface()                                          = default;
    virtual auto allocate(std::size_t, gcxx::StreamView) -> void* = 0;
    virtual auto deallocate(void*, gcxx::StreamView) -> void      = 0;
    virtual auto clone() const -> interface*                      = 0;
  };

  template <typename R>
  struct model final : interface {
    R resource;
    explicit model(R r) : resource(std::move(r)) {}
    auto allocate(std::size_t b, gcxx::StreamView sv) -> void* override {
      return resource.allocate(b, sv);
    }
    auto deallocate(void* p, gcxx::StreamView sv) -> void override {
      resource.deallocate(p, sv);
    }
    auto clone() const -> interface* override { return new model(resource); }
  };

  std::unique_ptr<interface> impl_;
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
