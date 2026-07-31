// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/memory_pool_base.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_ATTRIBUTES_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_ATTRIBUTES_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// ── pool_attr: typed memory-pool attribute descriptor ───────────────────────
// Attr        — the flags::MemPoolAttr this describes.
// ValueType   — what the user sees (bool for the reuse flags, std::size_t for
//               the thresholds/watermarks).
// StorageType — the C type CUDA reads/writes through the void* get/set API
//               (int for booleans, std::size_t for sizes).
// Settable=false marks a read-only attribute: set() is a no-op (CCCL throws
// std::invalid_argument; gcxx silently ignores to stay usable in builds with
// exceptions disabled).
template <flags::MemPoolAttr Attr, typename ValueType, typename StorageType,
          bool Settable>
struct pool_attr_impl {
  using type = ValueType;

  static inline constexpr flags::MemPoolAttr attribute = Attr;

  GCXX_FH constexpr operator flags::MemPoolAttr()
    const noexcept {  // NOLINT(google-explicit-constructor)
    return Attr;
  }

  GCXX_FH auto operator()(driver::deviceMemPool_t pool) const -> type {
    StorageType storage{};
    driver::deviceMemPoolGetAttribute(
      pool, static_cast<driver::deviceMemPoolAttr_t>(Attr),
      static_cast<void*>(&storage));
    return static_cast<type>(storage);
  }

  // Write the attribute value for the given pool. No-op for read-only attrs.
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


#define GCXX_POOL_ATTR_SPECIALIZATION(ATTR_FLAG, VALUE_TYPE, STORAGE_TYPE, \
                                      SETTABLE)                            \
  template <>                                                              \
  struct pool_attr<flags::MemPoolAttr::ATTR_FLAG>                          \
      : pool_attr_impl<flags::MemPoolAttr::ATTR_FLAG, VALUE_TYPE,          \
                       STORAGE_TYPE, SETTABLE> {}

// Reuse flags: CUDA stores 0/1, expose as bool. Each is settable.
GCXX_POOL_ATTR_SPECIALIZATION(FollowEventDependencies, bool, int, true);
GCXX_POOL_ATTR_SPECIALIZATION(AllowOpportunistic, bool, int, true);
GCXX_POOL_ATTR_SPECIALIZATION(AllowInternalDependencies, bool, int, true);

// Current-usage watermarks: read-only (set is a no-op).
GCXX_POOL_ATTR_SPECIALIZATION(ReservedMemCurrent, std::size_t, std::size_t,
                              false);
GCXX_POOL_ATTR_SPECIALIZATION(UsedMemCurrent, std::size_t, std::size_t, false);

#undef GCXX_POOL_ATTR_SPECIALIZATION


// memory_pool_attributes: the named, typed, constexpr attribute objects passed
// to MemPoolView::attribute() / set_attribute(). Each is a constexpr instance
// of its descriptor type — the pool analogue of dev_attr on the device surface
// (see runtime/device/device_attributes.hpp).
namespace memory_pool_attributes {

#define GCXX_POOL_ATTR_DEFINE(ATTR_MEMBER, NAME)               \
  using NAME##_t = pool_attr<flags::MemPoolAttr::ATTR_MEMBER>; \
  static inline constexpr NAME##_t NAME {}

  /// Threshold at which the pool releases unused memory back to the driver.
  GCXX_POOL_ATTR_DEFINE(ReleaseThreshold, release_threshold);

  /// Reuse memory across streams linked by event dependencies.
  GCXX_POOL_ATTR_DEFINE(FollowEventDependencies,
                        reuse_follow_event_dependencies);

  /// Reuse already-completed frees even without a stream dependency.
  GCXX_POOL_ATTR_DEFINE(AllowOpportunistic, reuse_allow_opportunistic);

  /// Insert internal stream dependencies to enable cross-stream reuse.
  GCXX_POOL_ATTR_DEFINE(AllowInternalDependencies,
                        reuse_allow_internal_dependencies);

  /// Current amount of memory reserved by the pool (read-only).
  GCXX_POOL_ATTR_DEFINE(ReservedMemCurrent, reserved_mem_current);

  /// High-water mark of reserved memory (settable only to 0, to reset).
  GCXX_POOL_ATTR_DEFINE(ReservedMemHigh, reserved_mem_high);

  /// Current amount of memory used by the pool (read-only).
  GCXX_POOL_ATTR_DEFINE(UsedMemCurrent, used_mem_current);

  /// High-water mark of used memory (settable only to 0, to reset).
  GCXX_POOL_ATTR_DEFINE(UsedMemHigh, used_mem_high);

#undef GCXX_POOL_ATTR_DEFINE

}  // namespace memory_pool_attributes


GCXX_NAMESPACE_MAIN_END()

#endif
