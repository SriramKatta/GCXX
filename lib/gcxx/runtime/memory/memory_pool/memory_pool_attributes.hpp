// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/memory_pool_base.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Typed memory-pool attribute descriptors. A pool attribute (cudaMemPoolAttr)
// is a (value-type, storage-type, settable) triple: CUDA stores boolean reuse
// attributes as `int` and size attributes as `size_t`, so each descriptor
// carries the storage type it reads/writes through the void* driver API. This
// mirrors CCCL's __pool_attr / namespace memory_pool_attributes.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_ATTRIBUTES_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_ATTRIBUTES_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// pool_attr_impl<Attr, ValueType, StorageType, Settable>: a descriptor for one
// cudaMemPoolAttr. ValueType is what the user sees (bool for the reuse flags,
// std::size_t for the thresholds/watermarks); StorageType is the C type CUDA
// reads/writes through the void* get/set API (int for booleans, std::size_t for
// sizes). Settable=false marks a read-only attribute: set() is a no-op (CCCL
// throws std::invalid_argument; gcxx silently ignores to stay usable in builds
// with exceptions disabled).
template <flags::MemPoolAttr Attr, typename ValueType, typename StorageType, bool Settable>
struct pool_attr_impl {
  using type = ValueType;

  static constexpr flags::MemPoolAttr attribute = Attr;

  /// Implicit conversion to the underlying driver attribute enum.
  GCXX_FH constexpr operator flags::MemPoolAttr() const noexcept {  // NOLINT(google-explicit-constructor)
    return Attr;
  }

  /// Read the attribute value for the given pool.
  GCXX_FH auto operator()(driver::deviceMemPool_t pool) const -> type {
    StorageType storage{};
    driver::deviceMemPoolGetAttribute(
      pool, static_cast<driver::deviceMemPoolAttr_t>(Attr),
      static_cast<void*>(&storage));
    return static_cast<type>(storage);
  }

  /// Write the attribute value for the given pool. No-op for read-only attrs.
  GCXX_FH static auto set(driver::deviceMemPool_t pool, type value) -> void {
    if constexpr (Settable) {
      StorageType storage = static_cast<StorageType>(value);
      driver::deviceMemPoolSetAttribute(
        pool, static_cast<driver::deviceMemPoolAttr_t>(Attr),
        static_cast<void*>(&storage));
    }
    // else: read-only attribute — set is intentionally a no-op.
  }
};

// Default: a settable size_t attribute.
template <flags::MemPoolAttr Attr>
struct pool_attr : pool_attr_impl<Attr, std::size_t, std::size_t, true> {};

// Reuse flags: stored as int, exposed as bool.
template <>
struct pool_attr<flags::MemPoolAttr::FollowEventDependencies>
    : pool_attr_impl<flags::MemPoolAttr::FollowEventDependencies, bool, int, true> {};
template <>
struct pool_attr<flags::MemPoolAttr::AllowOpportunistic>
    : pool_attr_impl<flags::MemPoolAttr::AllowOpportunistic, bool, int, true> {};
template <>
struct pool_attr<flags::MemPoolAttr::AllowInternalDependencies>
    : pool_attr_impl<flags::MemPoolAttr::AllowInternalDependencies, bool, int, true> {};

// Read-only current-usage attributes.
template <>
struct pool_attr<flags::MemPoolAttr::ReservedMemCurrent>
    : pool_attr_impl<flags::MemPoolAttr::ReservedMemCurrent, std::size_t, std::size_t,
                     false> {};
template <>
struct pool_attr<flags::MemPoolAttr::UsedMemCurrent>
    : pool_attr_impl<flags::MemPoolAttr::UsedMemCurrent, std::size_t, std::size_t,
                     false> {};

// memory_pool_attributes: the named, typed attribute objects passed to
// memory_pool_base::attribute() / set_attribute(). Each is a constexpr instance
// of its descriptor type.
namespace memory_pool_attributes {

/// Threshold at which the pool releases unused memory back to the driver.
using release_threshold_t = pool_attr<flags::MemPoolAttr::ReleaseThreshold>;
static constexpr release_threshold_t release_threshold{};

/// Reuse memory across streams linked by event dependencies.
using reuse_follow_event_dependencies_t =
  pool_attr<flags::MemPoolAttr::FollowEventDependencies>;
static constexpr reuse_follow_event_dependencies_t
  reuse_follow_event_dependencies{};

/// Reuse already-completed frees even without a stream dependency.
using reuse_allow_opportunistic_t = pool_attr<flags::MemPoolAttr::AllowOpportunistic>;
static constexpr reuse_allow_opportunistic_t reuse_allow_opportunistic{};

/// Insert internal stream dependencies to enable cross-stream reuse.
using reuse_allow_internal_dependencies_t =
  pool_attr<flags::MemPoolAttr::AllowInternalDependencies>;
static constexpr reuse_allow_internal_dependencies_t
  reuse_allow_internal_dependencies{};

/// Current amount of memory reserved by the pool (read-only).
using reserved_mem_current_t = pool_attr<flags::MemPoolAttr::ReservedMemCurrent>;
static constexpr reserved_mem_current_t reserved_mem_current{};

/// High-water mark of reserved memory (settable only to 0, to reset).
using reserved_mem_high_t = pool_attr<flags::MemPoolAttr::ReservedMemHigh>;
static constexpr reserved_mem_high_t reserved_mem_high{};

/// Current amount of memory used by the pool (read-only).
using used_mem_current_t = pool_attr<flags::MemPoolAttr::UsedMemCurrent>;
static constexpr used_mem_current_t used_mem_current{};

/// High-water mark of used memory (settable only to 0, to reset).
using used_mem_high_t = pool_attr<flags::MemPoolAttr::UsedMemHigh>;
static constexpr used_mem_high_t used_mem_high{};

}  // namespace memory_pool_attributes

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
