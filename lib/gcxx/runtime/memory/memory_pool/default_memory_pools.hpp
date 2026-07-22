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
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/memory/memory_pool/device_memory_pool.hpp>
#include <gcxx/runtime/memory/memory_pool/managed_memory_pool.hpp>
#include <gcxx/runtime/memory/memory_pool/pinned_memory_pool.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

/// Non-owning ref to the default device memory pool for `device`. The default
/// device pool exists on all CUDA versions that support memory pools.
GCXX_FH auto device_default_memory_pool(gcxx::DeviceHandle device)
  -> device_memory_pool_ref {
  return device_memory_pool_ref(
    driver::deviceGetDefaultMemoryPool(device.id()));
}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
/// Non-owning ref to the default managed (unified) memory pool (CUDA 13.0+).
GCXX_FH auto managed_default_memory_pool() -> managed_memory_pool_ref {
  MemAccessDesc location{flags::MemLocation::None, 0, flags::MemAccessFlags::None};
  auto rawLoc  = location.getRawMemLocation();
  auto rawType = static_cast<driver::deviceMemAllocationType_t>(
    flags::MemAllocation::Managed);
  return managed_memory_pool_ref(
    driver::deviceMemPoolGetDefaultMemPool(&rawLoc, rawType));
}
#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)
/// Non-owning ref to the default pinned (page-locked) host memory pool.
/// On CUDA 13.0+ it queries the runtime default pinned pool and grants access
/// from every device; pre-13.0 it creates a NUMA-0 pinned pool and adopts its
/// handle (matching CCCL's fallback).
GCXX_FH auto pinned_default_memory_pool() -> pinned_memory_pool_ref {
#  if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  MemAccessDesc location{flags::MemLocation::Host, 0, flags::MemAccessFlags::None};
  auto rawLoc  = location.getRawMemLocation();
  auto rawType = static_cast<driver::deviceMemAllocationType_t>(
    flags::MemAllocation::Pinned);
  pinned_memory_pool_ref ref(
    driver::deviceMemPoolGetDefaultMemPool(&rawLoc, rawType));
  ref.enable_access_from_all();
  return ref;
#  else
  pinned_memory_pool pool{0};  // NUMA node 0 pinned pool
  return pinned_memory_pool_ref(pool.release());
#  endif
}
#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
