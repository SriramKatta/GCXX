// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Default memory pools: non-owning refs to the CUDA-managed default pools for
// each memory type. Default pools are created automatically by CUDA, are shared
// process-wide, and are never destroyed — so the returned *_ref views do not
// own their handles. Mirrors CCCL's device/managed/pinned_default_memory_pool.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_DEFAULT_MEMORY_POOLS_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_DEFAULT_MEMORY_POOLS_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/device_memory_pool.hpp>
#include <gcxx/runtime/memory/mempool/managed_memory_pool.hpp>
#include <gcxx/runtime/memory/mempool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/memory/mempool/pinned_memory_pool.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


GCXX_FH auto device_default_memory_pool(const gcxx::DeviceHandle& device)
  -> DeviceMemPoolView {
  return DeviceMemPoolView(driver::deviceGetDefaultMemoryPool(device.id()));
}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
GCXX_FH auto managed_default_memory_pool() -> ManagedMemPoolView {
  MemAccessDesc location{flags::MemLocation::None, 0,
                         flags::MemAccessFlags::None};
  auto rawLoc  = location.getRawMemLocation();
  auto rawType = static_cast<driver::deviceMemAllocationType_t>(
    flags::MemAllocation::Managed);
  return ManagedMemPoolView(
    driver::deviceMemPoolGetDefaultMemPool(&rawLoc, rawType));
}
#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

// Pre-13.0 this creates a fresh pool; cache it for default-pool semantics.
inline auto get_default_mem_pool(flags::MemLocation locationType,
                                 int locationId, flags::MemAllocation type)
  -> driver::deviceMemPool_t {
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  MemAccessDesc loc{locationType, locationId, flags::MemAccessFlags::None};
  auto rawLoc = loc.getRawMemLocation();
  return driver::deviceMemPoolGetDefaultMemPool(
    &rawLoc, static_cast<driver::deviceMemAllocationType_t>(type));
#else
  return create_memory_pool(locationType, locationId, type);
#endif
}

// Deliberately uses the generic Host location (NUMA pinning breaks allocs).
GCXX_FH auto pinned_default_memory_pool() -> PinnedMemPoolView {
#if GCXX_HIP_MODE()
  // ROCm has no host-location memory pool, so the default pinned pool is the
  // hipMallocHost-backed shim (see PinnedMemPoolView). Built once per process
  // (magic-static init is thread-safe); trivially destructible, releases
  // nothing at exit — there is no pool handle to destroy.
  static PinnedMemPoolView pool = [] {
    return PinnedMemPoolView(PinnedMemPoolView::hip_shim_handle_t{});
  }();
  return pool;
#else
  // Lazily build the access-enabled view ONCE (magic-static init is
  // thread-safe). Caching the prepared view — not just the handle — means the
  // O(num-devices) enable_access_from_all() runs once, not on every call
  // (matching CCCL's call_once init). The handle lives for the process (default
  // pools are never destroyed) and PinnedMemPoolView is trivially destructible,
  // so the static releases nothing at exit.
  // TODO: Absolutely an stupid thing since on alex we dont know what is the.
  static PinnedMemPoolView pool = [] {
    PinnedMemPoolView ref(get_default_mem_pool(flags::MemLocation::Host, 0,
                                               flags::MemAllocation::Pinned));
    // Host-location pool allocations are only GPU-mapped after access is
    // granted — do it here so the cached view is ready to allocate from.
    ref.enable_access_from_all();
    return ref;
  }();
  return pool;
#endif
}


GCXX_NAMESPACE_MAIN_END()

#endif
