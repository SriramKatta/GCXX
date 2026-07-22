// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/memory_pool_base.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// memory_pool_properties: creation-time options for a memory pool (initial size,
// release threshold, IPC handle type, max size). Unlike attributes, properties
// can only be set when the pool is created. Mirrors CCCL's
// cuda::memory_pool_properties / __create_cuda_mempool.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_PROPERTIES_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_PROPERTIES_HPP_

#include <cstddef>
#include <limits>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

/// Creation-time options for a memory pool. Properties are fixed at creation;
/// runtime tuning uses memory_pool_attributes instead.
struct memory_pool_properties {
  /// Bytes to reserve up-front (primed via an allocate/free on the default
  /// stream). 0 means no pre-allocation.
  std::size_t initial_pool_size = 0;

  /// Upper bound on bytes the pool may keep reserved; above this, unused memory
  /// is released to the driver at the next sync.
  std::size_t release_threshold = std::numeric_limits<std::size_t>::max();

  /// Handle type for inter-process sharing of the pool (None = no IPC).
  flags::MemAllocationHandle allocation_handle_type = flags::MemAllocationHandle::None;

  /// Hard maximum size of the pool (0 = no limit).
  std::size_t max_pool_size = 0;
};

/// Create a cudaMemPool_t at `location`/`alloc_type` honouring `props`.
///
/// Builds a CUmemPoolProps (via the shared MemPoolProps descriptor), creates the
/// pool, sets the release-threshold attribute, and primes `initial_pool_size`.
/// Mirrors CCCL's __create_cuda_mempool. The returned handle is owned by the
/// caller (the owning device/managed/pinned pools destroy it in their dtor).
GCXX_FH auto create_memory_pool(flags::MemLocation location_type, int location_id,
                                flags::MemAllocation alloc_type,
                                memory_pool_properties props = {})
  -> driver::deviceMemPool_t {
  MemPoolProps raw;
  raw.allocType   = alloc_type;
  raw.handleTypes = static_cast<details_::flag_t>(props.allocation_handle_type);
  raw.locationType = location_type;
  raw.locationId   = location_id;
#if GCXX_CUDA_MODE()
  raw.maxSize = props.max_pool_size;
#endif

  auto pool = driver::deviceMemPoolCreate(raw.getRawMemPoolProps());

  // Apply the release threshold (a creation-time property).
  std::size_t release_threshold = props.release_threshold;
  driver::deviceMemPoolSetAttribute(
    pool, static_cast<driver::deviceMemPoolAttr_t>(flags::MemPoolAttr::ReleaseThreshold),
    static_cast<void*>(&release_threshold));

  // Prime the requested initial size on the default stream so the pool reserves
  // it immediately, then free it back — the reservation stays.
  if (props.initial_pool_size != 0) {
    void* ptr = driver::deviceMallocFromPoolAsync(props.initial_pool_size, pool,
                                                  nullptr);
    driver::deviceFreeAsync(ptr, nullptr);
  }
  return pool;
}

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
