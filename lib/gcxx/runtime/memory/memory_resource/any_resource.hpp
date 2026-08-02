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


// ─────────────────────────────────────────────────────────────────────────────
// gcxx::any_resource
//
// A non-templated, type-erased, owning, copyable holder for ANY resource
// matching the concept:
//   void* allocate(gcxx::StreamView, std::size_t num_bytes)
//   void  deallocate(gcxx::StreamView, void* ptr)
// ─────────────────────────────────────────────────────────────────────────────
class any_resource {
 public:
  /// Empty (no resource). allocate/deallocate on an empty any_resource assert.
  any_resource() noexcept = default;

  /// Type-erase a concrete resource. The resource must be copy constructible
  /// (any_resource owns a copy, cloned on copy/resize). Property validation is
  /// the caller's responsibility (the buffer ctor enforces it via
  /// resource_has_all_v before erasure).
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, any_resource>)
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

  GCXX_FH auto allocate(gcxx::StreamView sv,
                        std::size_t num_bytes) const -> void* {
    GCXX_RUNTIME_EXPECT(impl_ != nullptr,
                        "any_resource::allocate on empty resource");
    return impl_->allocate(sv, num_bytes);
  }

  GCXX_FH auto deallocate(gcxx::StreamView sv, void* ptr) const -> void {
    GCXX_RUNTIME_EXPECT(impl_ != nullptr,
                        "any_resource::deallocate on empty resource");
    impl_->deallocate(sv, ptr);
  }

  explicit operator bool() const noexcept { return impl_ != nullptr; }

 private:
  struct interface {
    virtual ~interface()                                          = default;
    virtual auto allocate(gcxx::StreamView, std::size_t) -> void* = 0;
    virtual auto deallocate(gcxx::StreamView, void*) -> void      = 0;
    virtual auto clone() const -> interface*                      = 0;
  };

  template <typename R>
  struct model final : interface {
    R resource;
    explicit model(R r) : resource(std::move(r)) {}
    auto allocate(gcxx::StreamView sv, std::size_t b) -> void* override {
      return resource.allocate(sv, b);
    }
    auto deallocate(gcxx::StreamView sv, void* p) -> void override {
      resource.deallocate(sv, p);
    }
    auto clone() const -> interface* override { return new model(resource); }
  };

  std::unique_ptr<interface> impl_;
};


GCXX_NAMESPACE_MAIN_END()

#endif
