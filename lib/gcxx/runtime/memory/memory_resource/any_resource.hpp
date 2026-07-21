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
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::any_resource<Properties...>
//
// A type-erased, owning, copyable holder for any resource whose advertised
// properties ⊇ Properties... . Lets a buffer<VT, Properties...> store ONE
// resource type regardless of the concrete allocator the user passed at
// construction (the decoupling that makes `device_buffer<T>` a single type).
//
// The concrete resource is held behind a virtual interface (model<R>), copied
// via clone(). Properties are carried by the any_resource<Properties...> TYPE
// (not a runtime property vtable) and exposed as `using properties`; the
// construction-time `static_assert` enforces the resource matches. This is the
// Option-B adaptation of CCCL's any_resource — no virtual get_property, no
// dynamic_accessibility_property vtable; accessibility is always known
// statically from Properties (see resource_accessibility_v below).
//
// ponytail (deferred): SBO. The current implementation heap-allocates one
// model<R> per resource (and one per clone, e.g. on resize/cross-copy). That is
// negligible next to the driver allocate/free it wraps, but a small-buffer
// optimization for stateless resources is a future refinement. Correctness and
// clarity first.
// ─────────────────────────────────────────────────────────────────────────────
template <typename... Properties>
class any_resource {
 public:
  /// Advertised accessibility — read by has_property_v / resource_has_all_v.
  using properties = TypeSet<Properties...>;

  /// Empty (no resource). allocate/deallocate on an empty any_resource assert.
  any_resource() noexcept = default;

  /// Type-erase a concrete resource. The resource must be copy constructible
  /// (any_resource owns a copy, cloned on copy/resize) and must advertise every
  /// one of Properties... via its `using properties` member.
  template <typename Resource, typename = std::enable_if_t<!std::is_same_v<
                                 std::decay_t<Resource>, any_resource>>>
  any_resource(Resource&& r) {
    static_assert(std::is_copy_constructible_v<std::decay_t<Resource>>,
                  "any_resource owns a copy of the resource; it must be copy "
                  "constructible");
    static_assert(
      resource_has_all_v<std::decay_t<Resource>, Properties...>,
      "resource properties do not satisfy this any_resource's Properties "
      "(e.g. a host_accessible resource cannot back a device_accessible "
      "buffer)");
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

  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView sv) const
    -> void* {
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

// ─────────────────────────────────────────────────────────────────────────────
// resource_accessibility_v<R>: the memory_accessibility implied by a type's
// static `properties` set. The Option-B answer to CCCL's
// dynamic_accessibility_property query — accessibility is always known at
// compile time from R::properties, so the "dynamic" query is a static fold.
// ─────────────────────────────────────────────────────────────────────────────
template <typename R>
inline constexpr memory_accessibility resource_accessibility_v =
  accessibility_from_static_properties<is_host_accessible_v<R>,
                                       is_device_accessible_v<R>>();

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
